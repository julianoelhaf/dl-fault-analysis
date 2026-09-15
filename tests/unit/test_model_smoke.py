"""CPU smoke tests for all 9 published architectures.

For each architecture, constructs both the classification and regression
head via the same factory (`create_model_from_name`) used by the training
pipeline, runs one forward pass on a small synthetic batch, and checks the
output shape. This validates that every registered model can be built and
executed -- it does not train or assert on prediction quality.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dl_fault_analysis.models.model_utils import create_model_from_name

ARCHITECTURES = [
    "lstm",
    "rnn",
    "gru",
    "cnn",
    "cnn_lstm",
    "dilated_cnn",
    "inceptiontime",
    "tcn",
    "tft",
]

BATCH_SIZE = 4
N_FEATURES = 12
SAMPLING_FREQUENCY = 6400
WINDOW_LENGTH = 0.01  # -> seq_len = 64
N_CLASSES = 3


def _config(model_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            model_name=model_name,
            hidden_size=8,
            num_layers=1,
            bidirectional=False,
            dropout=0.0,
        ),
        dataset=SimpleNamespace(sampling_frequency=SAMPLING_FREQUENCY),
        window_extraction=SimpleNamespace(window_length=WINDOW_LENGTH),
        training=SimpleNamespace(target_label="y_fault_present"),
    )


def _seq_len() -> int:
    return int(round(SAMPLING_FREQUENCY * WINDOW_LENGTH))


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_classifier_forward_shape(arch: str):
    model = create_model_from_name(
        _config(f"{arch}_classifier"),
        n_features=N_FEATURES,
        out_dim=N_CLASSES,
    )
    x = torch.randn(BATCH_SIZE, _seq_len(), N_FEATURES)
    out = model(x)
    assert out.shape == (BATCH_SIZE, N_CLASSES)


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_regressor_forward_shape(arch: str):
    model = create_model_from_name(
        _config(f"{arch}_regressor"),
        n_features=N_FEATURES,
        out_dim=1,
    )
    x = torch.randn(BATCH_SIZE, _seq_len(), N_FEATURES)
    out = model(x)
    assert out.shape[0] == BATCH_SIZE
    assert out.numel() == BATCH_SIZE  # single regression output per sample
