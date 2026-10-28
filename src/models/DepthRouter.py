"""Input-conditioned differentiable depth routing (AdDepth v2).

Replaces the epoch-level tabular Q-learning of AdaptiveDepth.py with a
lightweight router that, for EACH input window, predicts a processing depth
for each historical time step. Trained end-to-end with the task loss via
Gumbel-softmax, plus an optional compute-budget penalty.

Design (Mixture-of-Depths style, adapted to spatio-temporal frames):
  - The shared-weight prefix conv stack from AdDepth is kept unchanged:
    depth d means "apply the first d layers of the shared stack".
  - Training uses a SOFT MIXTURE over prefix outputs (fully differentiable):
        out_i = sum_d p(d | X_{t-i}, i) * stack_d(X_{t-i})
  - Evaluation uses hard argmax routing; computation runs only to the
    deepest depth selected in the batch for that time step.
  - The router sees both the frame content (conv features) and the time-step
    position (learned embedding), so it can express position priors (weekly /
    diurnal patterns) AND input-dependent adjustments (e.g. storm frames).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ConvLSTM import ConvLSTM
from .ConvSelfAttention import ConvSelfAttention


class DepthRouter(nn.Module):
    """Predicts a depth distribution per time step from frame content + position."""

    def __init__(self, in_channels, window, min_depth, max_depth,
                 feat_channels=16, pos_dim=8, hidden_dim=64, hard_routing=False,
                 positional=False):
        super(DepthRouter, self).__init__()
        if not (1 <= min_depth <= max_depth):
            raise ValueError(f"Need 1 <= min_depth <= max_depth, got [{min_depth}, {max_depth}]")
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_options = max_depth - min_depth + 1
        self.window = window
        # POSITIONAL mode: the depth distribution is a learnable function of
        # time-step POSITION ONLY (one logit row per position), identical for
        # every input. This is the input-CONDITIONING ablation turned into a
        # method: experiments showed the optimal depth on precipitation is a
        # positional (recency) property, not a per-sample one, and a fixed
        # recency schedule beat the input-conditioned router on both accuracy
        # and compute. Positional mode keeps the differentiable schedule-search
        # (Gumbel + STE + budget penalty) but drops the frame-content input, so
        # it cannot overfit spurious per-input features (the suspected cause of
        # the router's val→test generalization gap).
        self.positional = bool(positional)
        # Straight-through HARD Gumbel during training: the forward pass uses a
        # real discrete (one-hot) depth — matching the hard eval path — while
        # gradients still flow to all logits via the STE. Fixes the train/eval
        # mismatch (training was a soft mixture the model never runs at eval).
        self.hard_routing = bool(hard_routing)
        # Gumbel-softmax temperature; annealed by the training loop via
        # set_temperature(). Higher = softer routing.
        self.temperature = 1.0
        # Warmup mode: route uniformly so every depth's layers receive equal
        # gradient before the router starts choosing (prevents rich-get-richer
        # collapse onto whichever depth happens to train fastest).
        self.force_uniform = False
        # Softmax of the latest logits (the router's actual distribution, as
        # opposed to the Gumbel sample) — used for the entropy bonus.
        self.last_dist = None

        if self.positional:
            # One learnable logit vector per position; no content input.
            self.pos_logits = nn.Parameter(torch.zeros(window, self.num_options))
        else:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels, feat_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(feat_channels, feat_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.pos_embedding = nn.Embedding(window, pos_dim)
            self.mlp = nn.Sequential(
                nn.Linear(feat_channels + pos_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_options),
            )

    def set_temperature(self, tau):
        self.temperature = float(tau)

    def forward(self, x):
        """
        :param x: (batch, window, channels, height, width)
        :return: depth probabilities (batch, window, num_options).
                 Training: Gumbel-softmax samples (soft, differentiable).
                 Eval: exact one-hot argmax.
        """
        b, window = x.shape[0], x.shape[1]

        if self.positional:
            # Depth depends on position only; broadcast the learned table over
            # the batch. Independent of frame content x.
            logits = self.pos_logits.unsqueeze(0).expand(b, -1, -1)                 # (b, T, K)
        else:
            c, h, w = x.shape[2], x.shape[3], x.shape[4]
            feats = self.feature_extractor(x.reshape(b * window, c, h, w)).flatten(1)  # (b*T, F)
            pos = self.pos_embedding(torch.arange(window, device=x.device))            # (T, P)
            pos = pos.unsqueeze(0).expand(b, -1, -1).reshape(b * window, -1)            # (b*T, P)
            logits = self.mlp(torch.cat([feats, pos], dim=1)).view(b, window, self.num_options)

        self.last_dist = F.softmax(logits, dim=-1)

        if self.training and self.force_uniform:
            return torch.full_like(logits, 1.0 / self.num_options)

        if self.training:
            probs = F.gumbel_softmax(logits, tau=self.temperature, hard=self.hard_routing, dim=-1)
        else:
            probs = F.one_hot(logits.argmax(dim=-1), self.num_options).float()
        return probs


class TimeDistributedRouted(nn.Module):
    """Shared-weight prefix conv stack executed as a mixture over depths.

    Identical weight-sharing scheme to TimeDistributedRL: layer l has one set
    of weights regardless of time step or total depth (prefix sharing).
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, min_depth, max_depth):
        super(TimeDistributedRouted, self).__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.conv_layers = nn.ModuleList()
        for i in range(max_depth):
            in_ch = in_channels if i == 0 else out_channels
            self.conv_layers.append(nn.Conv2d(in_channels=in_ch, out_channels=out_channels,
                                              kernel_size=kernel_size, padding=padding))

    def forward(self, x, depth_probs):
        """
        :param x: (batch, window, channels, height, width)
        :param depth_probs: (batch, window, num_options) over depths
                            [min_depth, ..., max_depth]
        """
        batch_size, window, channels, height, width = x.shape
        output = []
        for i in range(window):
            x_i = x[:, i]
            p_i = depth_probs[:, i]  # (b, K)

            if self.training:
                needed = self.max_depth
            else:
                # Hard routing at eval: probabilities beyond the deepest
                # selected depth in the batch are exactly zero, so the stack
                # can be truncated without changing the result.
                needed = self.min_depth + int(p_i.argmax(dim=-1).max().item())

            prefix_outputs = []
            cur = x_i
            for l in range(needed):
                cur = F.relu(self.conv_layers[l](cur))
                if l + 1 >= self.min_depth:
                    prefix_outputs.append(cur)

            stacked = torch.stack(prefix_outputs, dim=1)             # (b, K', C, H, W)
            weights = p_i[:, :stacked.size(1)]                       # (b, K')
            out_i = (stacked * weights.view(batch_size, -1, 1, 1, 1)).sum(dim=1)
            output.append(out_i)

        return torch.stack(output, dim=1)


class AdDepthRouted(nn.Module):
    """AdDepth with input-conditioned differentiable depth routing.

    Drop-in replacement for AdDepth: same two-stage architecture (per-step
    encoder -> TimeSeries encoder), same forward interface, but the per-step
    depth is predicted by DepthRouter per input window instead of being a
    single global schedule searched by RL.

    With positional=True the router ignores frame content and learns a global
    per-position depth schedule (shared across all inputs) — the recommended
    mode given the empirical finding that the optimal depth allocation on
    these datasets is positional, not per-sample. It keeps the differentiable
    schedule-search machinery (Gumbel + STE + budget penalty) so the schedule
    is discovered automatically and the lambda_budget compute dial still works.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, in_channels, out_channels,
                 tensor_shape, window, max_depth, min_depth=1, attention=False,
                 pooling=None, convdeconv=None, router_feat_channels=16,
                 router_pos_dim=8, router_hidden_dim=64, hard_routing=False,
                 positional=False):
        super(AdDepthRouted, self).__init__()

        if attention:
            self.conv_lstm = ConvSelfAttention(input_dim, hidden_dim, kernel_size, num_layers,
                                               in_channels, out_channels, tensor_shape, window,
                                               pooling=pooling, convdeconv=convdeconv)
        else:
            self.conv_lstm = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers,
                                      in_channels, out_channels, tensor_shape, window)

        self.router = DepthRouter(in_channels, window, min_depth, max_depth,
                                  feat_channels=router_feat_channels,
                                  pos_dim=router_pos_dim,
                                  hidden_dim=router_hidden_dim,
                                  hard_routing=hard_routing,
                                  positional=positional)
        self.positional = bool(positional)
        self.time_distributed_cnn = TimeDistributedRouted(in_channels=in_channels,
                                                          out_channels=out_channels,
                                                          kernel_size=3, padding=1,
                                                          min_depth=min_depth, max_depth=max_depth)
        self.final_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=3, padding=1)

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.register_buffer("depth_values",
                             torch.arange(min_depth, max_depth + 1, dtype=torch.float32))
        self.last_depth_probs = None

    def set_temperature(self, tau):
        self.router.set_temperature(tau)

    def set_router_warmup(self, on):
        self.router.force_uniform = bool(on)

    def router_entropy(self):
        """Mean entropy of the router's distribution, normalized to [0, 1].

        1 = uniform over depths, 0 = fully collapsed. Subtract
        `entropy_weight * router_entropy()` from the loss to discourage
        premature collapse.
        """
        dist = self.router.last_dist
        if dist is None:
            raise RuntimeError("router_entropy() called before any forward pass.")
        eps = 1e-8
        ent = -(dist * (dist + eps).log()).sum(dim=-1).mean()
        return ent / torch.log(torch.tensor(float(self.router.num_options), device=dist.device))

    def forward(self, input_tensor, input_standalone=None):
        probs = self.router(input_tensor)          # (b, window, K)
        self.last_depth_probs = probs
        x = self.time_distributed_cnn(input_tensor, probs)
        x = self.conv_lstm(x, input_standalone)
        x = self.final_conv(x)
        return x

    def expected_depth(self):
        """Differentiable expected depth per time step, (batch, window).

        Uses the probabilities from the most recent forward pass.
        """
        if self.last_depth_probs is None:
            raise RuntimeError("expected_depth() called before any forward pass.")
        return (self.last_depth_probs * self.depth_values).sum(dim=-1)

    def depth_budget(self):
        """Scalar in (0, 1]: mean expected depth normalized by max_depth.

        Add `lambda_budget * model.depth_budget()` to the task loss to
        penalize compute. Differentiable w.r.t. the router.
        """
        return self.expected_depth().mean() / self.max_depth
