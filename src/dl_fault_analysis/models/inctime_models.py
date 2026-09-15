import torch
from torch import nn


def reshape_input(x, seq_len, feat_dim):
    if x.ndim == 2:
        bsz, flat = x.shape
        if flat != seq_len * feat_dim:
            raise ValueError(f"Expected {seq_len*feat_dim}, got {flat}")
        return x.view(bsz, seq_len, feat_dim)
    if x.ndim != 3:
        raise ValueError(f"Expected 2D or 3D input, got {x.shape}")
    return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes, bottleneck_channels: int):
        super().__init__()
        self.use_bottleneck = in_channels > 1
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if self.use_bottleneck else nn.Identity()
        )

        self.conv_list = nn.ModuleList()
        for ks in kernel_sizes:
            padding = ks // 2
            self.conv_list.append(
                nn.Conv1d(
                    bottleneck_channels if self.use_bottleneck else in_channels,
                    out_channels,
                    kernel_size=ks,
                    padding=padding,
                    bias=False,
                )
            )

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        )

        self.bn = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bottleneck = self.bottleneck(x)
        conv_outputs = [conv(x_bottleneck) for conv in self.conv_list]
        conv_outputs.append(self.maxpool_conv(x))
        x = torch.cat(conv_outputs, dim=1)
        x = self.bn(x)
        return self.relu(x)


class InceptionTimeClassifier(nn.Module):
    """
    InceptionTime for classification.
    Accepts (B, T, F) or flattened (B, T*F). Returns (B, C) or (B,) if C==1.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        seq_len: int,
        num_blocks: int = 3,
        out_channels: int = 32,
        bottleneck_channels: int = 32,
        kernel_sizes=None,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [9, 19, 39]
        self.input_size = input_size
        self.seq_len = seq_len
        self.output_size = output_size

        block_out = out_channels * (len(kernel_sizes) + 1)
        self.blocks = nn.Sequential(*[
            InceptionBlock(
                in_channels=input_size if i == 0 else block_out,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
            )
            for i in range(num_blocks)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(block_out, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = reshape_input(x, self.seq_len, self.input_size)
        x = x.permute(0, 2, 1)              # (B, F, T)
        x = self.blocks(x)
        x = self.global_pool(x).squeeze(-1) # (B, block_out)
        out = self.fc(x)
        return out.squeeze(-1) if self.output_size == 1 else out


class InceptionTimeRegressor(nn.Module):
    """
    InceptionTime for regression.
    Accepts (B, T, F) or flattened (B, T*F). Returns (B, D) or (B,) if D==1.
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int,
        out_features: int = 1,
        num_blocks: int = 3,
        out_channels: int = 32,
        bottleneck_channels: int = 32,
        kernel_sizes=None,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [9, 19, 39]
        self.input_size = input_size
        self.seq_len = seq_len
        self.out_features = out_features

        block_out = out_channels * (len(kernel_sizes) + 1)
        self.blocks = nn.Sequential(*[
            InceptionBlock(
                in_channels=input_size if i == 0 else block_out,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
            )
            for i in range(num_blocks)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(block_out, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = reshape_input(x, self.seq_len, self.input_size)
        x = x.permute(0, 2, 1)
        x = self.blocks(x)
        x = self.global_pool(x).squeeze(-1)
        out = self.fc(x)
        return out.squeeze(-1) if self.out_features == 1 else out
