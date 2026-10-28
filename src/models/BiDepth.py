
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ConvLSTM import ConvLSTM
from .ConvSelfAttention import ConvSelfAttention


class TimeDistributedConv2d2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, initial_layers, window, reverse=False):
        super(TimeDistributedConv2d2, self).__init__()

        layer_decrement = (initial_layers - 1) / (window - 1)

        # Create a list of convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(initial_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.conv_layers.append(nn.Conv2d(in_channels=in_ch, out_channels=out_channels, kernel_size=kernel_size, padding=padding))

        # For each timestamp, create a list of indices into self.conv_layers
        self.conv_indices = []
        for i in range(window):
            layers = initial_layers - int(round(i * layer_decrement))
            layers = max(1, layers)
            self.conv_indices.append(list(range(layers)))

        # Reverse the order of conv_indices if the reverse parameter is True
        if reverse:
            self.conv_indices = self.conv_indices[::-1]

    def forward(self, x):
        batch_size, window, channels, height, width = x.shape
        output = []
        for i in range(window):
            x_i = x[:, i]
            for j in self.conv_indices[i]:
                x_i = F.relu(self.conv_layers[j](x_i))
            output.append(x_i)
        return torch.stack(output, dim=1)


class DeepShallowConvLSTM2(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, in_channels, out_channels , tensor_shape , window, initial_layers, reverse=False, attention=False, pooling=None, convdeconv= None):
        super(DeepShallowConvLSTM2, self).__init__()

        if attention:
          self.conv_lstm = ConvSelfAttention(input_dim, hidden_dim, kernel_size, num_layers, in_channels, out_channels, tensor_shape, window, pooling = pooling, convdeconv=convdeconv)
        else:
          self.conv_lstm = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, in_channels,  out_channels, tensor_shape , window)
        self.time_distributed_cnn = TimeDistributedConv2d2(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, initial_layers=initial_layers, window=window , reverse=reverse)
        self.final_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, input_tensor, input_standalone = None):
        # Pass the input through the time-distributed CNN
        x = self.time_distributed_cnn(input_tensor)
        # pass the output + input_standalone through the ConvLSTM
        x = self.conv_lstm(x, input_standalone)
        # Pass the output through the final convolutional layer
        x = self.final_conv(x)
        return x



class BidirectionalDeepShallow(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, in_channels, out_channels, tensor_shape, window, initial_layers, attention = False, pooling=None, convdeconv=None):
        super(BidirectionalDeepShallow, self).__init__()

        # DeepShallow path
        self.deep_shallow_cnn = TimeDistributedConv2d2(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, initial_layers=initial_layers, window=window, reverse=False)
        # ShallowDeep path
        self.shallow_deep_cnn = TimeDistributedConv2d2(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, initial_layers=initial_layers, window=window, reverse=True)

        if attention:
          self.conv_lstm = ConvSelfAttention(input_dim*2, hidden_dim, kernel_size, num_layers, in_channels, out_channels, tensor_shape, window, pooling=pooling, convdeconv=convdeconv)
        else:
          self.conv_lstm = ConvLSTM(input_dim*2, hidden_dim, kernel_size, num_layers, in_channels,  out_channels, tensor_shape, window)


        # Fusion layer to combine DeepShallow and ShallowDeep paths
        # self.fusion_layer = nn.Conv2d(in_channels=out_channels*2*window, out_channels=out_channels*window, kernel_size=3, padding=1)

        # Final convolutional layer
        self.final_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, input_tensor, input_standalone= None):
        # DeepShallow path
        deep_shallow_output = self.deep_shallow_cnn(input_tensor)

        # ShallowDeep path
        shallow_deep_output = self.shallow_deep_cnn(input_tensor)

        # Combine the outputs
        combined_output = torch.cat([deep_shallow_output, shallow_deep_output], dim=2)
        # print(combined_output.shape)

        # Reshape for the fusion layer: merge batch and sequence dimensions
        # bs, seq, ch, h, w = combined_output.shape
        # combined_output = combined_output.view(bs, seq * ch, h, w)

        # # Pass the reshaped output through the fusion layer
        # combined_output = self.fusion_layer(combined_output)

        # # Reshape back to separate batch and sequence dimensions
        # combined_output = combined_output.view(bs, -1, ch, h, w

        # Pass the combined output + input_standalone through the ConvLSTM
        x = self.conv_lstm(combined_output, input_standalone)

        # Pass the output through the final convolutional layer
        x = self.final_conv(x)
        return x