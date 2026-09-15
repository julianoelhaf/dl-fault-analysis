import torch
from torch import nn


# --- Shared helper (same as in your other models) ---
def reshape_input(x, seq_len, feat_dim):
    if x.ndim == 2:
        bsz, flat = x.shape
        if flat != seq_len * feat_dim:
            raise ValueError(f"Expected {seq_len*feat_dim}, got {flat}")
        return x.view(bsz, seq_len, feat_dim)
    if x.ndim != 3:
        raise ValueError(f"Expected 2D or 3D input, got {x.shape}")
    return x


# --- TCN building blocks ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size == 0:
            return x.contiguous()
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = self.downsample(x)
        return self.relu(out + res)


# --- Classifier ---
class TCNClassifier(nn.Module):
    """
    Temporal Convolutional Network (causal dilated Conv1d blocks) for classification.
    Accepts (B, T, F) or flattened (B, T*F). Returns (B, C) or (B,) if C==1.
    """
    def __init__(
        self,
        input_size: int,      # features per timestep
        output_size: int,     # #classes
        seq_len: int,         # sequence length
        hidden_size: int = 128,
        num_layers: int = 2,
        kernel_size: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.output_size = output_size

        # Channel plan: [hidden_size]*num_layers
        channels = [hidden_size] * max(1, num_layers)
        layers = []
        for i in range(len(channels)):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else channels[i - 1]
            out_ch = channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

        self.fc = nn.Linear(channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = reshape_input(x, self.seq_len, self.input_size)
        x = x.permute(0, 2, 1)       # (B, F, T) for Conv1d
        y = self.network(x)          # (B, C, T)
        y_last = y[:, :, -1]         # last timestep features
        out = self.fc(y_last)        # (B, C)
        return out.squeeze(-1) if self.output_size == 1 else out


# --- Regressor ---
class TCNRegressor(nn.Module):
    """
    Temporal Convolutional Network for regression.
    Accepts (B, T, F) or flattened (B, T*F). Returns (B, D) or (B,) if D==1.
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int,
        out_features: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        kernel_size: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.out_features = out_features

        channels = [hidden_size] * max(1, num_layers)
        layers = []
        for i in range(len(channels)):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else channels[i - 1]
            out_ch = channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

        self.fc = nn.Linear(channels[-1], out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = reshape_input(x, self.seq_len, self.input_size)
        x = x.permute(0, 2, 1)
        y = self.network(x)
        y_last = y[:, :, -1]
        out = self.fc(y_last)
        return out.squeeze(-1) if self.out_features == 1 else out
