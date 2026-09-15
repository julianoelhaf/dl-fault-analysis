import torch
from torch import nn, Tensor


class RNNClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,  # features per timestep
        output_size: int,  # #classes
        seq_len: int,  # sequence length
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        nonlinearity: str = "tanh",  # or "relu"
    ):
        super().__init__()
        self.input_dim = input_size
        self.seq_len = seq_len
        self.num_dirs = 2 if bidirectional else 1

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_size * self.num_dirs, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, T, F) or flattened (B, T*F)
        if x.ndim == 2:
            bsz, flat = x.shape
            expected = self.seq_len * self.input_dim
            if flat != expected:
                raise ValueError(
                    f"Flattened size {flat} != expected {expected} (= seq_len*input_dim)"
                )
            x = x.view(bsz, self.seq_len, self.input_dim)

        out, _ = self.rnn(x)  # (B, T, H*num_dirs)
        last = out[:, -1, :]  # (B, H*num_dirs)
        return self.fc(last)


class RNNRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,  # features per timestep
        seq_len: int,  # sequence length
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = False,
        out_dim: int = 1,  # regression output size
        nonlinearity: str = "tanh",  # or "relu"
        # --- Top-3 stability helpers only ---
        sanitize_input: bool = True,  # replace NaN/Inf; optional clipping
        input_clip: float | None = None,  # e.g., 10.0 to clamp |x|<=10
        use_layer_norm: bool = True,  # LayerNorm on last hidden state
        ln_eps: float = 1e-5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_dirs = 2 if bidirectional else 1

        self.sanitize_input = sanitize_input
        self.input_clip = input_clip

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        feat = hidden_dim * self.num_dirs
        self.norm = nn.LayerNorm(feat, eps=ln_eps) if use_layer_norm else nn.Identity()
        self.head = nn.Linear(feat, out_dim)

        self._init_weights()

    def _init_weights(self):
        # Safer initialization to avoid early explosions
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.kaiming_uniform_(param, a=0.0, nonlinearity="relu")
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.kaiming_uniform_(self.head.weight, a=0.0, nonlinearity="linear")
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def _prep_input(self, x: Tensor) -> Tensor:
        # Accept (B, T, F) or flattened (B, T*F)
        if x.ndim == 2:
            bsz, flat = x.shape
            expected = self.seq_len * self.input_dim
            if flat != expected:
                raise ValueError(
                    f"Flattened size {flat} != expected {expected} (= seq_len*input_dim)"
                )
            x = x.view(bsz, self.seq_len, self.input_dim)
        if self.sanitize_input:
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            if self.input_clip is not None:
                x = x.clamp_(-self.input_clip, self.input_clip)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._prep_input(x)  # (B, T, F)
        out, _ = self.rnn(x)  # (B, T, H*num_dirs)
        last = out[:, -1, :]  # (B, H*num_dirs)
        last = self.norm(last)  # stabilize features
        y = self.head(last)  # (B, out_dim)

        # Keep (B,) if out_dim == 1 to match typical MAE expectations
        if y.shape[-1] == 1:
            y = y.squeeze(-1)
        return y
