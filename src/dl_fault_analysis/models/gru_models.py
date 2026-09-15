import torch
from torch import nn


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,   # features per timestep
        output_size: int,  # #classes
        seq_len: int,      # sequence length
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_dim = input_size
        self.seq_len = seq_len
        self.num_dirs = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
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

        out, _ = self.gru(x)   # (B, T, H*num_dirs)
        last = out[:, -1, :]   # (B, H*num_dirs)
        return self.fc(last)


class GRURegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,   # features per timestep
        seq_len: int,     # sequence length
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = False,
        out_dim: int = 1,  # regression output size
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_dirs = 2 if bidirectional else 1

        self.gru = nn.GRU(
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

        out, _ = self.gru(x)   # (B, T, H*num_dirs)
        last = out[:, -1, :]   # (B, H*num_dirs)
        return self.fc(last)
