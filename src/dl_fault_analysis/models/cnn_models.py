import torch
from torch import nn


class CNNClassifier(nn.Module):
    """
    One-layer 1D CNN for classification.
    Accepts (B, T, F), (B, F, T), or flattened (B, T*F).
    """

    def __init__(
        self,
        input_size: int,  # features per timestep
        output_size: int,  # #classes
        seq_len: int,  # sequence length
        channels: int = 128,
        kernel_size: int = 7,
        stride: int = 1,
        padding: int = 3,
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.output_size = output_size

        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.act = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1)  # (B, C, 1)
        self.fc = nn.Linear(channels, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to (B, F, T)
        if x.dim() == 2:  # (B, T*F)
            B = x.size(0)
            x = x.view(B, self.seq_len, self.input_size)
        if x.dim() != 3:
            raise ValueError(f"Expected 3D or 2D input, got {x.shape}")

        if x.shape[1] == self.seq_len and x.shape[2] == self.input_size:
            x = x.permute(0, 2, 1)  # (B, T, F) -> (B, F, T)

        x = self.conv(x)
        x = self.act(x)
        x = self.gap(x).squeeze(-1)  # (B, channels)
        out = self.fc(x)

        # --- Fix for BCE targets ---
        # If output_size == 1, squeeze to (B,) so it matches target shape
        if self.output_size == 1:
            out = out.squeeze(-1)

        return out


class CNNRegressor(nn.Module):
    """
    One-layer 1D CNN for regression.
    Accepts (B, T, F), (B, F, T), or flattened (B, T*F).
    Returns (B,) if out_features == 1, else (B, out_features).
    """

    def __init__(
        self,
        input_size: int,  # features per timestep
        seq_len: int,  # sequence length
        out_features: int = 1,
        channels: int = 128,
        kernel_size: int = 7,
        stride: int = 1,
        padding: int = 3,
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.out_features = out_features

        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.act = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1)  # (B, C, 1)
        self.fc = nn.Linear(channels, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to (B, F, T)
        if x.dim() == 2:  # (B, T*F)
            B = x.size(0)
            x = x.view(B, self.seq_len, self.input_size)
        if x.dim() != 3:
            raise ValueError(f"Expected 3D or 2D input, got {x.shape}")

        # If (B, T, F), permute to (B, F, T). If already (B, F, T), leave as is.
        if x.shape[1] == self.seq_len and x.shape[2] == self.input_size:
            x = x.permute(0, 2, 1)

        x = self.conv(x)
        x = self.act(x)
        x = self.gap(x).squeeze(-1)  # (B, channels)
        out = self.fc(x)  # (B, out_features)

        # If single-output regression, squeeze to (B,)
        if self.out_features == 1:
            out = out.squeeze(-1)
        return out


def reshape_input(x, seq_len, feat_dim):
    if x.ndim == 2:
        bsz, flat = x.shape
        if flat != seq_len * feat_dim:
            raise ValueError(f"Expected {seq_len*feat_dim}, got {flat}")
        return x.view(bsz, seq_len, feat_dim)
    if x.ndim != 3:
        raise ValueError(f"Expected 2D or 3D input, got {x.shape}")
    return x


class CNNLSTMClassifier(nn.Module):
    """
    Hybrid CNN–LSTM Classifier.
    Accepts (B, T, F) or flattened (B, T*F).
    """

    def __init__(
        self,
        input_size: int,  # features per timestep
        output_size: int,  # number of classes
        seq_len: int,  # sequence length
        cnn_channels: int = 64,
        kernel_size: int = 7,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.output_size = output_size
        self.num_dirs = 2 if bidirectional else 1

        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(lstm_hidden * self.num_dirs, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = reshape_input(x, self.seq_len, self.input_size)

        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.drop(self.act(self.conv(x)))
        x = x.permute(0, 2, 1)  # (B, T, F)

        out, _ = self.lstm(x)
        last = out[:, -1, :]  # last timestep
        out = self.fc(last)

        if self.output_size == 1:  # BCE-style target
            out = out.squeeze(-1)
        return out


class CNNLSTMRegressor(nn.Module):
    """
    Hybrid CNN–LSTM Regressor.
    Accepts (B, T, F) or flattened (B, T*F).
    """

    def __init__(
        self,
        input_size: int,  # features per timestep
        seq_len: int,  # sequence length
        out_features: int = 1,
        cnn_channels: int = 64,
        kernel_size: int = 7,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.out_features = out_features
        self.num_dirs = 2 if bidirectional else 1

        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(lstm_hidden * self.num_dirs, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = reshape_input(x, self.seq_len, self.input_size)

        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.drop(self.act(self.conv(x)))
        x = x.permute(0, 2, 1)  # (B, T, F)

        out, _ = self.lstm(x)
        last = out[:, -1, :]
        out = self.fc(last)

        if self.out_features == 1:  # scalar regression
            out = out.squeeze(-1)
        return out






class DilatedCNNClassifier(nn.Module):
    """
    Dilated CNN Classifier.
    Accepts (B, T, F) or flattened (B, T*F).
    """

    def __init__(
        self,
        input_size: int,  # features per timestep
        output_size: int,  # number of classes
        seq_len: int,  # sequence length
        hidden_size: int = 64,  # number of filters
        kernel_size: int = 3,
        dilation_base: int = 2,
        num_layers: int = 4,
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.output_size = output_size

        layers = []
        in_channels = input_size
        for i in range(num_layers):
            dilation = dilation_base**i
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=((kernel_size - 1) // 2) * dilation,
                )
            )
            layers.append(nn.ReLU())
            in_channels = hidden_size
        self.conv_layers = nn.Sequential(*layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = reshape_input(x, self.seq_len, self.input_size)

        # (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)  # (B, hidden_size)
        out = self.fc(x)

        if self.output_size == 1:
            out = out.squeeze(-1)
        return out


class DilatedCNNRegressor(nn.Module):
    """
    Dilated CNN Regressor.
    Accepts (B, T, F) or flattened (B, T*F).
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        out_features: int = 1,
        hidden_size: int = 64,
        kernel_size: int = 3,
        dilation_base: int = 2,
        num_layers: int = 4,
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.out_features = out_features

        layers = []
        in_channels = input_size
        for i in range(num_layers):
            dilation = dilation_base**i
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=((kernel_size - 1) // 2) * dilation,
                )
            )
            layers.append(nn.ReLU())
            in_channels = hidden_size
        self.conv_layers = nn.Sequential(*layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = reshape_input(x, self.seq_len, self.input_size)

        x = x.permute(0, 2, 1)  # (B, T, F) -> (B, F, T)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)
        out = self.fc(x)

        if self.out_features == 1:
            out = out.squeeze(-1)
        return out
