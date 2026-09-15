from typing import Callable, Dict, Optional

import torch
from torch import nn

from dl_fault_analysis.config import MainConfig
from dl_fault_analysis.constants import FAULT_TARGET_TO_OUTPUT_DIM
from dl_fault_analysis.models.gru_models import GRUClassifier, GRURegressor
from dl_fault_analysis.models.lstm_models import LSTMClassifier, LSTMRegressor
from dl_fault_analysis.models.rnn_models import RNNClassifier, RNNRegressor
from dl_fault_analysis.models.cnn_models import (
    CNNClassifier,
    CNNRegressor,
    CNNLSTMClassifier,
    CNNLSTMRegressor,
    DilatedCNNClassifier,
    DilatedCNNRegressor,
)
from dl_fault_analysis.models.inctime_models import InceptionTimeClassifier, InceptionTimeRegressor
from dl_fault_analysis.models.tcn_models import TCNClassifier, TCNRegressor
from dl_fault_analysis.models.tft_models import TFTClassifier, TFTRegressor, TFTClassifierSimple

# ---- Model Registry ----
_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model constructor under a name (case-insensitive)."""
    key = name.lower()

    def _wrap(fn: Callable[..., nn.Module]):
        _MODEL_REGISTRY[key] = fn
        return fn

    return _wrap


# ---- Adapters (handle regressor arg names) ----
def _make_cls(Cls, **kw) -> nn.Module:
    return Cls(
        input_size=kw["input_size"],
        output_size=kw["output_size"],
        seq_len=kw["seq_len"],
        hidden_size=kw["hidden_size"],
        num_layers=kw["num_layers"],
        dropout=kw["dropout"],
        bidirectional=kw["bidirectional"],
    )


def _make_reg(Reg, **kw) -> nn.Module:
    return Reg(
        input_dim=kw["input_size"],
        out_dim=kw["output_size"],
        seq_len=kw["seq_len"],
        hidden_dim=kw["hidden_size"],
        num_layers=kw["num_layers"],
        dropout=kw["dropout"],
        bidirectional=kw["bidirectional"],
    )


# ---- Register built-ins (aliases allowed) ----
@register_model("lstm_classifier")
@register_model("lstm-cls")
def _lstm_cls(**kwargs) -> nn.Module:
    return _make_cls(LSTMClassifier, **kwargs)


@register_model("lstm_regressor")
@register_model("lstm-reg")
def _lstm_reg(**kwargs) -> nn.Module:
    return _make_reg(LSTMRegressor, **kwargs)


@register_model("rnn_classifier")
@register_model("rnn-cls")
def _rnn_cls(**kwargs) -> nn.Module:
    return _make_cls(RNNClassifier, **kwargs)


@register_model("rnn_regressor")
@register_model("rnn-reg")
def _rnn_reg(**kwargs) -> nn.Module:
    return _make_reg(RNNRegressor, **kwargs)


@register_model("gru_classifier")
@register_model("gru-cls")
def _gru_cls(**kwargs) -> nn.Module:
    return _make_cls(GRUClassifier, **kwargs)


@register_model("gru_regressor")
@register_model("gru-reg")
def _gru_reg(**kwargs) -> nn.Module:
    return _make_reg(GRURegressor, **kwargs)


# --- CNN Models ---
@register_model("cnn_classifier")
@register_model("cnn-cls")
def _cnn_cls(**kwargs) -> nn.Module:
    return CNNClassifier(
        input_size=kwargs["input_size"],
        output_size=kwargs["output_size"],
        seq_len=kwargs["seq_len"],
        channels=kwargs.get("hidden_size", 128),
        kernel_size=kwargs.get("kernel_size", 7),
        stride=kwargs.get("stride", 1),
        padding=kwargs.get("padding", 3),
    )


@register_model("cnn_regressor")
@register_model("cnn-reg")
def _cnn_reg(**kwargs) -> nn.Module:
    return CNNRegressor(
        input_size=kwargs["input_size"],
        seq_len=kwargs["seq_len"],
        out_features=kwargs["output_size"],
        channels=kwargs.get("hidden_size", 128),
        kernel_size=kwargs.get("kernel_size", 7),
        stride=kwargs.get("stride", 1),
        padding=kwargs.get("padding", 3),
    )


# ---- CNN–LSTM Models ----
@register_model("cnn_lstm_classifier")
@register_model("cnn-lstm-cls")
def _cnn_lstm_cls(**kwargs) -> nn.Module:
    return CNNLSTMClassifier(
        input_size=kwargs["input_size"],
        output_size=kwargs["output_size"],
        seq_len=kwargs["seq_len"],
        cnn_channels=kwargs.get("hidden_size", 128),
        kernel_size=kwargs.get("kernel_size", 7),
        lstm_hidden=kwargs.get("lstm_hidden", 128),
        lstm_layers=kwargs.get("num_layers", 2),
        dropout=kwargs.get("dropout", 0.1),
        bidirectional=kwargs.get("bidirectional", False),
    )


@register_model("cnn_lstm_regressor")
@register_model("cnn-lstm-reg")
def _cnn_lstm_reg(**kwargs) -> nn.Module:
    return CNNLSTMRegressor(
        input_size=kwargs["input_size"],
        out_features=kwargs["output_size"],
        seq_len=kwargs["seq_len"],
        cnn_channels=kwargs.get("hidden_size", 128),
        kernel_size=kwargs.get("kernel_size", 7),
        lstm_hidden=kwargs.get("lstm_hidden", 128),
        lstm_layers=kwargs.get("num_layers", 2),
        dropout=kwargs.get("dropout", 0.1),
        bidirectional=kwargs.get("bidirectional", False),
    )


# ---- Dilated CNN Models ----
@register_model("dilated_cnn_classifier")
@register_model("dcnn-cls")
def _dcnn_cls(**kwargs) -> nn.Module:
    return DilatedCNNClassifier(
        input_size=kwargs["input_size"],
        output_size=kwargs["output_size"],
        seq_len=kwargs["seq_len"],
        hidden_size=kwargs.get("hidden_size", 128),
        kernel_size=kwargs.get("kernel_size", 3),
        dilation_base=kwargs.get("dilation_base", 2),
        num_layers=kwargs.get("num_layers", 2),
    )


@register_model("dilated_cnn_regressor")
@register_model("dcnn-reg")
def _dcnn_reg(**kwargs) -> nn.Module:
    return DilatedCNNRegressor(
        input_size=kwargs["input_size"],
        out_features=kwargs["output_size"],
        seq_len=kwargs["seq_len"],
        hidden_size=kwargs.get("hidden_size", 128),
        kernel_size=kwargs.get("kernel_size", 3),
        dilation_base=kwargs.get("dilation_base", 2),
        num_layers=kwargs.get("num_layers", 2),
    )


# ---- InceptionTime Models ----
@register_model("inceptiontime_classifier")
@register_model("inctime-cls")
def _inctime_cls(**kwargs) -> nn.Module:
    return InceptionTimeClassifier(
        input_size=kwargs["input_size"],
        output_size=kwargs["output_size"],
        seq_len=kwargs["seq_len"],
        num_blocks=kwargs.get("num_blocks", 3),
        out_channels=kwargs.get("hidden_size", 32),
        bottleneck_channels=kwargs.get("bottleneck_channels", 32),
        kernel_sizes=kwargs.get("kernel_sizes", [9, 19, 39]),
    )


@register_model("inceptiontime_regressor")
@register_model("inctime-reg")
def _inctime_reg(**kwargs) -> nn.Module:
    return InceptionTimeRegressor(
        input_size=kwargs["input_size"],
        out_features=kwargs["output_size"],
        seq_len=kwargs["seq_len"],
        num_blocks=kwargs.get("num_blocks", 3),
        out_channels=kwargs.get("hidden_size", 32),
        bottleneck_channels=kwargs.get("bottleneck_channels", 32),
        kernel_sizes=kwargs.get("kernel_sizes", [9, 19, 39]),
    )


# ---- TCN Models ----
@register_model("tcn_classifier")
@register_model("tcn-cls")
def _tcn_cls(**kwargs) -> nn.Module:
    return TCNClassifier(
        input_size=kwargs["input_size"],
        output_size=kwargs["output_size"],
        seq_len=kwargs["seq_len"],
        hidden_size=kwargs.get("hidden_size", 128),
        num_layers=kwargs.get("num_layers", 2),
        kernel_size=kwargs.get("kernel_size", 2),
        dropout=kwargs.get("dropout", 0.05),
    )


@register_model("tcn_regressor")
@register_model("tcn-reg")
def _tcn_reg(**kwargs) -> nn.Module:
    return TCNRegressor(
        input_size=kwargs["input_size"],
        out_features=kwargs["output_size"],
        seq_len=kwargs["seq_len"],
        hidden_size=kwargs.get("hidden_size", 128),
        num_layers=kwargs.get("num_layers", 2),
        kernel_size=kwargs.get("kernel_size", 2),
        dropout=kwargs.get("dropout", 0.05),
    )


# ---- TFT / Transformer Models ----
@register_model("tft_classifier")
@register_model("tft-cls")
def _tft_cls(**kwargs) -> nn.Module:
    return TFTClassifier(
        input_size=kwargs["input_size"],
        output_size=kwargs["output_size"],
        seq_len=kwargs["seq_len"],
        hidden_size=kwargs.get("hidden_size", 64),
        num_layers=kwargs.get("num_layers", 3),
        n_heads=kwargs.get("n_heads", 4),
        dropout=kwargs.get("dropout", 0.1),
    )


@register_model("tft_regressor")
@register_model("tft-reg")
def _tft_reg(**kwargs) -> nn.Module:
    return TFTRegressor(
        input_size=kwargs["input_size"],
        out_features=kwargs["output_size"],
        seq_len=kwargs["seq_len"],
        hidden_size=kwargs.get("hidden_size", 64),
        num_layers=kwargs.get("num_layers", 3),
        n_heads=kwargs.get("n_heads", 4),
        dropout=kwargs.get("dropout", 0.1),
    )


@register_model("tft_classifier_simple")
@register_model("tft-cls-spl")
def _tft_cls_spl(**kwargs) -> nn.Module:
    return TFTClassifierSimple(
        input_size=kwargs["input_size"],
        num_layers=kwargs.get("num_layers", 3),
        hidden_size=kwargs.get("hidden_size", 64),
        n_heads=kwargs.get("n_heads", 4),
        output_size=kwargs["output_size"],
    )


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- State-dict compatibility helpers ---


def _infer_prefix_from_model(model: nn.Module) -> str:
    """Return the primary recurrent module prefix present in model.state_dict()."""
    keys = list(model.state_dict().keys())
    for prefix in ("gru.", "lstm.", "rnn."):
        if any(k.startswith(prefix) for k in keys):
            return prefix
    return ""  # models without a single named recurrent block


def _infer_prefix_from_state_dict(state_dict: dict) -> str:
    keys = list(state_dict.keys())
    for prefix in ("gru.", "lstm.", "rnn."):
        if any(k.startswith(prefix) for k in keys):
            return prefix
    return ""


def remap_recurrent_prefix(state_dict: dict, src_prefix: str, dst_prefix: str) -> dict:
    if not src_prefix or not dst_prefix or src_prefix == dst_prefix:
        return state_dict
    return {
        (k.replace(src_prefix, dst_prefix, 1) if k.startswith(src_prefix) else k): v
        for k, v in state_dict.items()
    }


def detect_model_prefix(model: nn.Module) -> str:
    return _infer_prefix_from_model(model)


def load_state_dict_compatible(model: nn.Module, state_dict: dict, strict: bool = True):
    """Load a checkpoint into `model`, remapping rnn/gru/lstm prefixes if necessary."""
    dst_prefix = _infer_prefix_from_model(model)
    src_prefix = _infer_prefix_from_state_dict(state_dict)
    state_dict = remap_recurrent_prefix(state_dict, src_prefix, dst_prefix)
    model.load_state_dict(state_dict, strict=strict)


# ---- The Factory ----
def create_model_from_name(
    config: MainConfig,
    *,
    n_features: Optional[int] = None,  # F
    flattened_dim: Optional[int] = None,  # T*F
    out_dim: Optional[int] = None,
) -> nn.Module:
    model_name = config.model.model_name
    if model_name is None:
        raise ValueError("Model name not specified in config.model.model_name")

    key = model_name.lower()
    if key not in _MODEL_REGISTRY:
        known = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Known: {known}")

    hidden_size = config.model.hidden_size
    num_layers = config.model.num_layers
    bidirectional = config.model.bidirectional
    dropout = config.model.dropout

    if out_dim is None:
        # fallback: only for fixed-output targets
        out_dim = FAULT_TARGET_TO_OUTPUT_DIM.get(config.training.target_label)
        if out_dim is None:
            raise KeyError(
                f"Unknown target_label='{config.training.target_label}'. "
                "Pass out_dim explicitly for dynamic targets (e.g., y_fault_line)."
            )
    out_features = int(out_dim)

    seq_len = int(
        round(
            config.dataset.sampling_frequency * config.window_extraction.window_length
        )
    )

    # ------------------------------------------------------------------
    # INPUT INTERPRETATION
    # - Most time-series models here expect input_size == F (n_features)
    # - Only flat/MLP-style models should use flattened_dim == T*F
    # ------------------------------------------------------------------
    feature_dim_models = {
        # RNN family
        "lstm_classifier",
        "lstm-cls",
        "lstm_regressor",
        "lstm-reg",
        "gru_classifier",
        "gru-cls",
        "gru_regressor",
        "gru-reg",
        "rnn_classifier",
        "rnn-cls",
        "rnn_regressor",
        "rnn-reg",
        # CNN family
        "cnn_classifier",
        "cnn-cls",
        "cnn_regressor",
        "cnn-reg",
        "dilated_cnn_classifier",
        "dcnn-cls",
        "dilated_cnn_regressor",
        "dcnn-reg",
        # CNN-LSTM family
        "cnn_lstm_classifier",
        "cnn-lstm-cls",
        "cnn_lstm_regressor",
        "cnn-lstm-reg",
        # InceptionTime
        "inceptiontime_classifier",
        "inctime-cls",
        "inceptiontime_regressor",
        "inctime-reg",
        # TCN
        "tcn_classifier",
        "tcn-cls",
        "tcn_regressor",
        "tcn-reg",
        # TFT / Transformer
        "tft_classifier",
        "tft-cls",
        "tft_regressor",
        "tft-reg",
        "tft_classifier_simple",
        "tft-cls-spl",
    }

    if key in feature_dim_models:
        if n_features is None:
            raise ValueError(f"{model_name} requires n_features (F) for input_size.")
        input_size = int(n_features)
    else:
        # Flat-input models (only if you actually have any registered like this)
        if flattened_dim is None and getattr(config.model, "input_size", None) is None:
            raise ValueError(
                f"{model_name} appears to be a flat-input model; provide flattened_dim or config.model.input_size."
            )
        input_size = (
            int(flattened_dim)
            if flattened_dim is not None
            else int(config.model.input_size)
        )

    ctor = _MODEL_REGISTRY[key]
    return ctor(
        input_size=input_size,
        output_size=out_features,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )
