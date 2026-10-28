"""Minimal PredRNN (ST-LSTM, dual-memory zigzag) with VARIABLE per-position
encode depth — for the HANDOFF3 Phase-0 go/no-go probe.

Phase 0 question: on a strong autoregressive backbone (proper seq2seq decoder),
does skipping depth on old input frames preserve accuracy? We compare uniform
full-depth encode vs a hand-set reduced schedule like [1,1,...,1,L]. No router
yet (that's Phase 2) — encode_depths is set explicitly.

ST-LSTM (PredRNN V1) per layer l, step t:
  temporal cell  C_t^l = f * C_{t-1}^l + i * g          (gates from X, H_{t-1})
  spatiotemporal M_t^l = f' * M_below + i' * g'         (gates from X, M_below)
  H_t^l = o * tanh(conv1x1([C_t, M_t])),  o from X,H,C,M
M zigzags: up through layers within a step, then top->bottom across steps.

Variable depth (HANDOFF3 §2.3): a depth-d step runs only cells 1..d; deeper
cells retain their (H,C); the zigzag continues from the deepest ACTIVE layer.
Decode is full-depth (v1).
"""
import torch
import torch.nn as nn

from .DepthRouter import DepthRouter


class STLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden, ksize=5):
        super().__init__()
        self.hidden = hidden
        pad = ksize // 2
        self.conv_x = nn.Conv2d(in_ch, hidden * 7, ksize, padding=pad)
        self.conv_h = nn.Conv2d(hidden, hidden * 4, ksize, padding=pad)
        self.conv_m = nn.Conv2d(hidden, hidden * 3, ksize, padding=pad)
        self.conv_o = nn.Conv2d(hidden * 2, hidden, 1)     # C,M contribution to output gate
        self.conv_last = nn.Conv2d(hidden * 2, hidden, 1)  # [C,M] -> H

    def forward(self, x, h, c, m):
        H = self.hidden
        xi, xf, xg, xi2, xf2, xg2, xo = torch.split(self.conv_x(x), H, dim=1)
        hi, hf, hg, ho = torch.split(self.conv_h(h), H, dim=1)
        mi, mf, mg = torch.split(self.conv_m(m), H, dim=1)
        i = torch.sigmoid(xi + hi); f = torch.sigmoid(xf + hf); g = torch.tanh(xg + hg)
        c_new = f * c + i * g
        i2 = torch.sigmoid(xi2 + mi); f2 = torch.sigmoid(xf2 + mf); g2 = torch.tanh(xg2 + mg)
        m_new = f2 * m + i2 * g2
        mem = torch.cat([c_new, m_new], dim=1)
        o = torch.sigmoid(xo + ho + self.conv_o(mem))
        h_new = o * torch.tanh(self.conv_last(mem))
        return h_new, c_new, m_new


class VarDepthPredRNN(nn.Module):
    """Prefix-shared ST-LSTM stack with variable per-position encode depth.

    encode_depths: list length T_in of ints in [1..max_depth]; None => uniform
    (all max_depth). Set via set_encode_depths(). forward(X) returns the next
    T_out frames so it drops into train_model / evaluate_mse unchanged.
    """
    def __init__(self, in_ch, hidden, max_depth, t_out, ksize=5, tensor_shape=(64, 64),
                 collapse_out=False):
        super().__init__()
        self.in_ch, self.hidden, self.max_depth, self.t_out = in_ch, hidden, max_depth, t_out
        self.height, self.width = tensor_shape
        # collapse_out: for single-shot predict (t_out=1) where in_ch encodes the
        # output frames as channels (precip/taxi), drop the singleton time dim so
        # the output (B, in_ch, H, W) matches a target with no time axis.
        self.collapse_out = collapse_out
        cells = []
        for l in range(max_depth):
            cells.append(STLSTMCell(in_ch if l == 0 else hidden, hidden, ksize))
        self.cells = nn.ModuleList(cells)
        self.out_conv = nn.Conv2d(hidden, in_ch, 1)
        self.encode_depths = None

    def set_encode_depths(self, depths):
        self.encode_depths = list(depths)

    def _init_states(self, b, dev):
        H = [torch.zeros(b, self.hidden, self.height, self.width, device=dev) for _ in range(self.max_depth)]
        C = [torch.zeros(b, self.hidden, self.height, self.width, device=dev) for _ in range(self.max_depth)]
        M = torch.zeros(b, self.hidden, self.height, self.width, device=dev)
        return H, C, M

    def forward(self, X, input_standalone=None):
        # X: (B, T_in, C, H, W)
        b, t_in = X.shape[0], X.shape[1]
        depths = self.encode_depths or [self.max_depth] * t_in
        assert len(depths) == t_in, f"encode_depths {len(depths)} != T_in {t_in}"
        H, C, M = self._init_states(b, X.device)

        # ---- ENCODE: variable depth ----
        for i in range(t_in):
            inp = X[:, i]
            for l in range(depths[i]):           # run only first depths[i] cells
                H[l], C[l], M = self.cells[l](inp, H[l], C[l], M)
                inp = H[l]
            # deeper cells retain (H,C); M = M at deepest active layer, zigzags on

        # ---- DECODE: autoregressive, full depth (v1) ----
        gen = X[:, -1]
        preds = []
        for _ in range(self.t_out):
            inp = gen
            for l in range(self.max_depth):
                H[l], C[l], M = self.cells[l](inp, H[l], C[l], M)
                inp = H[l]
            gen = self.out_conv(H[self.max_depth - 1])
            preds.append(gen)
        out = torch.stack(preds, dim=1)          # (B, T_out, C, H, W)
        return out[:, 0] if self.collapse_out else out

    def encode_compute_frac(self, t_in):
        """compute% (HANDOFF3 §4): (sum d_i + Lmax*T_out)/(Lmax*(T_in+T_out))."""
        depths = self.encode_depths or [self.max_depth] * t_in
        return (sum(depths) + self.max_depth * self.t_out) / (self.max_depth * (t_in + self.t_out))


class PredRNNRouted(nn.Module):
    """VarDepthPredRNN with a LEARNED positional depth router (ICLR method).

    The router picks the per-encode-position depth; unlike the hand-set
    VarDepthPredRNN, the schedule is learned end-to-end. Mirrors AdDepthRouted's
    surface so `train_model_routed` is reused unchanged.

    TRAINING: run the full max_depth chain per encode position (candidates are
    depth-independent up to where the chain stops), then propagate STRAIGHT-THROUGH
    keep-gated mixtures of (H, C, M) — the forward value equals the selected discrete
    depth's state (router probs are one-hot), while gradient reaches every depth logit.
    EVAL: discrete fast path (only the argmax depth's cells run -> inference saving).
    Decode is full-depth (v1). See plan/HANDOFF3.
    """

    def __init__(self, in_ch, hidden, max_depth, t_out, window, min_depth=1,
                 ksize=5, tensor_shape=(64, 64), hard_routing=True, positional=True,
                 collapse_out=False, route_decode=False):
        super().__init__()
        self.in_ch, self.hidden, self.max_depth, self.t_out = in_ch, hidden, max_depth, t_out
        self.min_depth, self.window = min_depth, window
        self.height, self.width = tensor_shape
        self.collapse_out = collapse_out         # see VarDepthPredRNN.collapse_out
        # route_decode (v2): also learn per-decode-step depth. The router table is
        # extended to window + t_out positions; positions [0, window) route the
        # encode steps, [window, window+t_out) route the decode steps. The output
        # head reads the deepest ACTIVE layer at each decode step, so pruning decode
        # depth actually saves compute (v1 always ran full depth on decode).
        self.route_decode = bool(route_decode)
        if self.route_decode and not positional:
            raise ValueError("route_decode requires positional=True (decode frames "
                             "are generated autoregressively -> no content to condition on).")
        router_window = window + (t_out if self.route_decode else 0)
        # same layout/names as VarDepthPredRNN so a VarDepth checkpoint loads (strict=False)
        self.cells = nn.ModuleList([STLSTMCell(in_ch if l == 0 else hidden, hidden, ksize)
                                    for l in range(max_depth)])
        self.out_conv = nn.Conv2d(hidden, in_ch, 1)
        self.router = DepthRouter(in_ch, router_window, min_depth, max_depth,
                                  hard_routing=hard_routing, positional=positional)
        self.register_buffer("depth_values",
                             torch.arange(min_depth, max_depth + 1, dtype=torch.float32))
        self.last_depth_probs = None

    # ---- surface mirrored from AdDepthRouted (consumed by train_model_routed) ----
    def set_temperature(self, tau):
        self.router.set_temperature(tau)

    def set_router_warmup(self, on):
        self.router.force_uniform = bool(on)

    def router_entropy(self):
        dist = self.router.last_dist
        if dist is None:
            raise RuntimeError("router_entropy() called before any forward pass.")
        eps = 1e-8
        ent = -(dist * (dist + eps).log()).sum(dim=-1).mean()
        return ent / torch.log(torch.tensor(float(self.router.num_options), device=dist.device))

    def expected_depth(self):
        if self.last_depth_probs is None:
            raise RuntimeError("expected_depth() called before any forward pass.")
        return (self.last_depth_probs * self.depth_values).sum(dim=-1)  # (b, window)

    def depth_budget(self):
        return self.expected_depth().mean() / self.max_depth

    def _init_states(self, b, dev):
        H = [torch.zeros(b, self.hidden, self.height, self.width, device=dev) for _ in range(self.max_depth)]
        C = [torch.zeros(b, self.hidden, self.height, self.width, device=dev) for _ in range(self.max_depth)]
        M = torch.zeros(b, self.hidden, self.height, self.width, device=dev)
        return H, C, M

    def forward(self, X, input_standalone=None):
        b, t_in = X.shape[0], X.shape[1]
        probs = self.router(X)                       # (b, t_in, K); train: STE one-hot, eval: argmax
        self.last_depth_probs = probs
        K = self.router.num_options
        H, C, M = self._init_states(b, X.device)

        if self.training:
            # ENCODE: full chain per position -> STE keep-gated state mixture
            for i in range(t_in):
                inp, m = X[:, i], M
                h_cand, c_cand, m_after = [], [], []
                for l in range(self.max_depth):
                    h, c, m = self.cells[l](inp, H[l], C[l], m)
                    h_cand.append(h); c_cand.append(c); m_after.append(m)
                    inp = h
                w = probs[:, i]                                  # (b, K)
                for l in range(self.max_depth):
                    kmin = max(0, l + 1 - self.min_depth)        # active iff depth >= l+1
                    g = w[:, kmin:].sum(-1).view(b, 1, 1, 1)     # STE 0/1 forward, differentiable
                    H[l] = g * h_cand[l] + (1 - g) * H[l]
                    C[l] = g * c_cand[l] + (1 - g) * C[l]
                m_stack = torch.stack([m_after[self.min_depth - 1 + k] for k in range(K)], dim=1)
                M = (m_stack * w.view(b, K, 1, 1, 1)).sum(dim=1)  # deepest-active memory (STE-weighted)
        else:
            # ENCODE eval fast path: only the selected depth's cells run (positional => shared)
            depths = (self.min_depth + probs.argmax(-1))         # (b, t_in)
            d = depths[0].tolist()
            for i in range(t_in):
                inp = X[:, i]
                for l in range(int(d[i])):
                    H[l], C[l], M = self.cells[l](inp, H[l], C[l], M)
                    inp = H[l]

        # ---- DECODE ----
        if not self.route_decode:
            # v1: full depth, autoregressive
            gen, preds = X[:, -1], []
            for _ in range(self.t_out):
                inp = gen
                for l in range(self.max_depth):
                    H[l], C[l], M = self.cells[l](inp, H[l], C[l], M)
                    inp = H[l]
                gen = self.out_conv(H[self.max_depth - 1])
                preds.append(gen)
        else:
            # v2: routed depth per decode step. Same STE keep-gate mixture as encode;
            # the output head reads the deepest ACTIVE layer (out_conv of the selected
            # depth's candidate), so a pruned decode step runs (and pays for) only its
            # selected cells at eval.
            dec_probs = probs[:, t_in:t_in + self.t_out]     # (b, t_out, K)
            gen, preds = X[:, -1], []
            if self.training:
                for t in range(self.t_out):
                    inp, m = gen, M
                    h_cand, c_cand, m_after = [], [], []
                    for l in range(self.max_depth):
                        h, c, m = self.cells[l](inp, H[l], C[l], m)
                        h_cand.append(h); c_cand.append(c); m_after.append(m)
                        inp = h
                    w = dec_probs[:, t]                              # (b, K)
                    for l in range(self.max_depth):
                        kmin = max(0, l + 1 - self.min_depth)        # active iff depth >= l+1
                        g = w[:, kmin:].sum(-1).view(b, 1, 1, 1)     # STE 0/1 fwd, differentiable
                        H[l] = g * h_cand[l] + (1 - g) * H[l]
                        C[l] = g * c_cand[l] + (1 - g) * C[l]
                    m_stack = torch.stack([m_after[self.min_depth - 1 + k] for k in range(K)], dim=1)
                    M = (m_stack * w.view(b, K, 1, 1, 1)).sum(dim=1)
                    out_stack = torch.stack([self.out_conv(h_cand[self.min_depth - 1 + k])
                                             for k in range(K)], dim=1)   # (b, K, C, H, W)
                    gen = (out_stack * w.view(b, K, 1, 1, 1)).sum(dim=1)   # STE: selected depth's output
                    preds.append(gen)
            else:
                dec_d = (self.min_depth + dec_probs.argmax(-1))[0].tolist()  # per-step depth
                for t in range(self.t_out):
                    inp, dt = gen, int(dec_d[t])
                    for l in range(dt):
                        H[l], C[l], M = self.cells[l](inp, H[l], C[l], M)
                        inp = H[l]
                    gen = self.out_conv(H[dt - 1])           # deepest active layer
                    preds.append(gen)
        out = torch.stack(preds, dim=1)
        return out[:, 0] if self.collapse_out else out
