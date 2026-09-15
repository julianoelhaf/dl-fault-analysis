"""Round-trip checks for the revision runners' checkpoint helpers.

`save_checkpoint` stores a `meta` dict alongside the state dict, so these
checkpoints are deliberately not weights-only. PyTorch changed `torch.load`'s
default to `weights_only=True` in 2.6, and this repository declares only
`torch>=2.1`, so a checkpoint written by `save_checkpoint` must stay loadable
by `load_into` regardless of which torch the user resolved.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dl_fault_analysis.utils.ckpt import load_into, save_checkpoint


def _model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6, 4), nn.ReLU(), nn.Linear(4, 1))


def test_checkpoint_round_trip_restores_weights_and_meta(tmp_path):
    src = _model(0)
    # Mirrors run_revision_observability.py's meta, whose test_metrics values
    # come from evaluate() and are typically numpy scalars -- exactly what the
    # weights-only unpickler refuses.
    meta = {
        "model_name": "gru_regressor",
        "target_label": "y_fault_location",
        "n_features": 48,
        "seed": 42,
        "test_metrics": {"mae": np.float64(0.0834), "rmse": np.float32(0.115)},
    }
    path = str(tmp_path / "checkpoints" / "fold0.pt")
    save_checkpoint(path, src, meta)

    dst = _model(1)
    # Sanity: the two models genuinely differ before loading.
    assert not torch.allclose(
        src[0].weight, dst[0].weight
    ), "test setup is degenerate -- models must differ before load"

    restored_meta = load_into(dst, path, device="cpu")

    for a, b in zip(src.state_dict().values(), dst.state_dict().values()):
        assert torch.allclose(a, b)
    assert restored_meta["model_name"] == "gru_regressor"
    assert restored_meta["n_features"] == 48
    assert float(restored_meta["test_metrics"]["mae"]) == float(np.float64(0.0834))


def test_load_into_does_not_depend_on_the_torch_weights_only_default(tmp_path):
    """Pin the actual regression: numpy scalars in meta break weights_only=True.

    If a future change drops the explicit weights_only=False, this fails on
    torch >= 2.6 (and documents why the explicit argument is there on older
    versions, where the default still happens to work).
    """
    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, _model(0), {"test_metrics": {"mae": np.float64(0.5)}})

    # The payload is genuinely not weights-only loadable ...
    try:
        torch.load(path, map_location="cpu", weights_only=True)
        weights_only_rejects_payload = False
    except Exception:
        weights_only_rejects_payload = True
    assert weights_only_rejects_payload, (
        "meta no longer contains anything the weights-only unpickler rejects; "
        "if meta was deliberately reduced to plain Python scalars, load_into "
        "can switch to weights_only=True and this test should be updated"
    )

    # ... yet load_into still works, because it does not rely on the default.
    assert load_into(_model(1), path, device="cpu")["test_metrics"]["mae"] == 0.5
