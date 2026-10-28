import torch
import torch.nn as nn
import torch.nn.functional as F

from .ConvLSTM import ConvLSTM
from .ConvSelfAttention import ConvSelfAttention

import numpy as np
import random


class TimeDistributedRL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, initial_layers, window):
        super(TimeDistributedRL, self).__init__()

        # Create a list of convolutional layers. The weights are shared across
        # time steps and depths: depth d at any time step uses the first d
        # layers of this list (prefix sharing).
        self.conv_layers = nn.ModuleList()
        for i in range(initial_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.conv_layers.append(nn.Conv2d(in_channels=in_ch, out_channels=out_channels,
                                              kernel_size=kernel_size, padding=padding))

    def forward(self, x, dynamic_layer_counts):
        # x: input tensor of shape (batch, window, channels, height, width)
        # dynamic_layer_counts: number of layers to apply at each time step
        batch_size, window, channels, height, width = x.shape
        output = []
        for i in range(window):
            x_i = x[:, i]
            for j in range(dynamic_layer_counts[i]):
                x_i = F.relu(self.conv_layers[j](x_i))
            output.append(x_i)
        return torch.stack(output, dim=1)


class AdDepth(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, in_channels, out_channels,
                 tensor_shape, window, initial_layers, attention=False, pooling=None, convdeconv=None):
        super(AdDepth, self).__init__()

        if attention:
            self.conv_lstm = ConvSelfAttention(input_dim, hidden_dim, kernel_size, num_layers,
                                               in_channels, out_channels, tensor_shape, window,
                                               pooling=pooling, convdeconv=convdeconv)
        else:
            self.conv_lstm = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers,
                                      in_channels, out_channels, tensor_shape, window)
        self.time_distributed_cnn = TimeDistributedRL(in_channels=in_channels, out_channels=out_channels,
                                                      kernel_size=3, padding=1,
                                                      initial_layers=initial_layers, window=window)
        self.final_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=3, padding=1)

        # Default to uniform depth so forward() works before the RL loop has
        # made its first update.
        self.dynamic_layer_counts = [initial_layers] * window

    def update_dynamic_layer_counts(self, dynamic_layer_counts):
        self.dynamic_layer_counts = dynamic_layer_counts

    def forward(self, input_tensor, input_standalone=None):
        # Pass the input through the time-distributed CNN with dynamic layer counts
        x = self.time_distributed_cnn(input_tensor, self.dynamic_layer_counts)
        # pass the output + input_standalone through the ConvLSTM
        x = self.conv_lstm(x, input_standalone)
        # Pass the output through the final convolutional layer
        x = self.final_conv(x)
        return x


class QLearningAgent:
    def __init__(self, window_size, init_loss, discrete_states=100, learning_rate=0.01,
                 discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        """
        Tabular Q-learning agent over a loss-based state space.

        The state space has a FIXED number of bins (`discrete_states`)
        spanning the loss range [0, max_loss], where max_loss is derived from
        the initial validation loss. This keeps the Q-table size independent
        of the loss scale (an init_loss < 1 previously produced a degenerate
        or empty table).
        """
        self.action_size = window_size * 2 + 1  # +/- one layer per position, plus no-op
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.discrete_states = discrete_states
        self.previous_layers = None
        # Loss ceiling for binning; losses above this are clamped to the top bin.
        self.max_loss = max(float(init_loss), 1e-8)
        self.q_table = np.zeros((discrete_states, self.action_size))

    def discretize_state(self, loss):
        # Map loss in [0, max_loss] to a bin in [0, discrete_states - 1],
        # clamping losses that exceed the ceiling.
        state = int(loss / self.max_loss * self.discrete_states)
        return min(max(state, 0), self.discrete_states - 1)

    def store_previous_layers(self, layers):
        self.previous_layers = layers.copy()

    def choose_action(self, loss):
        state = self.discretize_state(loss)
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randrange(self.action_size)
        else:
            return int(np.argmax(self.q_table[state, :]))

    def learn(self, current_loss, action, reward, new_loss):
        current_state = self.discretize_state(current_loss)
        next_state = self.discretize_state(new_loss)
        predict = self.q_table[current_state, action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state, :])
        self.q_table[current_state, action] += self.learning_rate * (target - predict)
        self.exploration_rate *= self.exploration_decay


def translate_action_to_dynamic_layer_counts(action, current_layers, min_depth=1,
                                             max_depth=5, window_size=7):
    """Apply an action index (0 .. 2*window_size) to a depth configuration.

    Returns a NEW list; the input configuration is not mutated.
    """
    new_layers = list(current_layers)

    action = action - window_size  # Map to the range [-window_size, window_size]

    if action == 0:
        # Zero action means no change
        return new_layers

    action_position = abs(action) - 1  # Convert action to 0-based index

    if action < 0:
        # Decrease the layer count at the specified position
        new_layers[action_position] = max(min_depth, new_layers[action_position] - 1)
    else:
        # Increase the layer count at the specified position
        new_layers[action_position] = min(max_depth, new_layers[action_position] + 1)

    return new_layers


def get_initial_state(initial_value, window_size):
    initial_dynamic_layer_counts = [initial_value] * window_size  # e.g. [3,3,3,3,3,3,3]
    # No action has been taken yet. (Previously this returned action index 0,
    # which is NOT a no-op — index 0 maps to "decrease depth at the last
    # position" — so the first Q-update credited an action that never ran.)
    init_action = None
    return init_action, initial_dynamic_layer_counts


def calculate_reward(val_losses):
    """Reward = improvement in validation loss between consecutive evaluations.

    The caller is responsible for gating when RL updates begin (Rl_start_step);
    this function no longer hard-codes a warm-up length.
    """
    if len(val_losses) < 2:
        return 0.0
    return val_losses[-2][1] - val_losses[-1][1]
