import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# --- Shared helper (as in your other models) ---
def reshape_input(x, seq_len, feat_dim):
    if x.ndim == 2:
        bsz, flat = x.shape
        if flat != seq_len * feat_dim:
            raise ValueError(f"Expected {seq_len*feat_dim}, got {flat}")
        return x.view(bsz, seq_len, feat_dim)
    if x.ndim != 3:
        raise ValueError(f"Expected 2D or 3D input, got {x.shape}")
    return x


# --- Base TFT block (projection + positional embedding + encoder) ---
class _TFTBackbone(nn.Module):
    def __init__(
        self, input_size, seq_len, hidden_size, num_layers, n_heads, dropout=0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_emb = nn.Embedding(seq_len, hidden_size)  # learned positional encoding
        enc_layer = TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, batch_first=True, dropout=dropout
        )
        self.encoder = TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, F)
        bsz, T, _ = x.shape
        if T != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {T}")
        h = self.input_proj(x)  # (B, T, H)
        pos_ids = torch.arange(T, device=h.device).unsqueeze(0).expand(bsz, T)
        h = h + self.pos_emb(pos_ids)  # (B, T, H)
        h = self.encoder(h)  # (B, T, H)
        h = self.dropout(h.mean(dim=1))  # GAP over time -> (B, H)
        return h


class TFTClassifierSimple(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_layers: int,
        hidden_size: int,
        n_heads: int,
        output_size: int,
    ):
        super(TFTClassifierSimple, self).__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        x = self.input_projection(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)


# --- Classifier ---
class TFTClassifier(nn.Module):
    """
    Transformer-based (TFT-style) classifier.
    Accepts (B, T, F) or flattened (B, T*F). Returns (B, C) or (B,) if C==1.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        seq_len: int,
        hidden_size: int = 64,
        num_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.output_size = output_size

        self.backbone = _TFTBackbone(
            input_size=input_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = reshape_input(x, self.seq_len, self.input_size)
        h = self.backbone(x)
        out = self.fc(h)
        return out.squeeze(-1) if self.output_size == 1 else out


# --- Regressor ---
class TFTRegressor(nn.Module):
    """
    Transformer-based (TFT-style) regressor.
    Accepts (B, T, F) or flattened (B, T*F). Returns (B, D) or (B,) if D==1.
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        out_features: int = 1,
        hidden_size: int = 64,
        num_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.out_features = out_features

        self.backbone = _TFTBackbone(
            input_size=input_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = reshape_input(x, self.seq_len, self.input_size)
        h = self.backbone(x)
        out = self.fc(h)
        return out.squeeze(-1) if self.out_features == 1 else out
