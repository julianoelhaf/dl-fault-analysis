import torch
from torch import nn


def reshape_input(x: torch.Tensor, seq_len: int, feat_dim: int) -> torch.Tensor:
    """Reshape input to (B, T, F) if needed."""
    if x.ndim == 2:
        bsz, flat = x.shape
        if flat != seq_len * feat_dim:
            raise ValueError(f"Expected {seq_len*feat_dim}, got {flat}")
        return x.view(bsz, seq_len, feat_dim)
    if x.ndim != 3:
        raise ValueError(f"Expected 2D or 3D, got {x.shape}")
    return x

class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, output_size: int, seq_len: int,
                 hidden_size: int=128, num_layers:int =2, dropout: float =0.1, bidirectional=False):
        super().__init__()
        self.input_size, self.seq_len = input_size, seq_len
        dirs = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_size * dirs, output_size)

    def forward(self, x):
        x = reshape_input(x, self.seq_len, self.input_size)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,  # features per timestep
        seq_len: int,  # sequence length
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = False,
        out_dim: int = 1,  # output size (e.g., 1 for scalar)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_dirs = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_dim * self.num_dirs, out_dim)

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

        out, _ = self.lstm(x)  # (B, T, H*num_dirs)
        last = out[:, -1, :]  # (B, H*num_dirs)
        return self.fc(last)
