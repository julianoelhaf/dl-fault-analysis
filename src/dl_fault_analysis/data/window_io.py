"""Loading precomputed PROTECT-90 window tensors from disk.

Vendored (and trimmed to what this repository needs -- reading an already
windowed dataset, not building one) from the private `psp_helper` package's
`windows_helper.generate_paths`/`open_raw_memmap`, `io.feature_meta`, and
`windows.global_loader.GlobalWindowLoader` -- see docs/PROVENANCE.md.

On-disk layout expected under
``<dataset.dataset_directory>/<window_extraction.windows_local_dir>``:
    X_<topology>_W<window_length>_S<step_length>.raw            (memmap, float32, shape (N, L, F))
    X_<topology>_W<window_length>_S<step_length>.raw.meta.json  (feature names/groups)
    y_<topology>_W<window_length>_S<step_length>.parquet        (per-window labels)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from time import time
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from dl_fault_analysis.config import MainConfig
from dl_fault_analysis.utils.logging import get_logger

logger = get_logger(__name__)

_CHAN_SUFFIX = re.compile(
    r"^(?P<cubicle>.+)_(?P<qty>Isec|Usec|cur|vol|I0|U0|V0)_(?P<phase>L[0-3])_(?P<unit>[AV])$"
)
_CHAN_CUBICLE = re.compile(
    r"^Bus_(?P<bus>[^_]+)_(?P<type>Line|Transformer|Load|ExtGrid|Ibr|IBR|Wind|CapBank|Generator)_(?P<id>.+)$"
)
_QUANTITY_NORM = {
    "Isec": "I", "cur": "I", "I": "I",
    "Usec": "V", "vol": "V", "V": "V",
    "I0": "I0", "U0": "V0", "V0": "V0",
}
_ETYPE_NORM = {"IBR": "Ibr"}


def build_channels_descriptor(feature_names: Iterable[str]) -> list[dict[str, Any]]:
    """Derive a structured per-channel descriptor from ``feature_names``."""
    channels: list[dict[str, Any]] = []
    for i, name in enumerate(feature_names):
        m = _CHAN_SUFFIX.match(name)
        if m is None:
            raise ValueError(
                f"feature name {name!r} does not match the expected grammar "
                "'<cubicle>_<qty>_<phase>_<unit>' (qty in Isec/Usec/cur/vol/I0/U0/V0)"
            )
        cubicle = m.group("cubicle")
        entry: dict[str, Any] = {
            "index": i,
            "cubicle": cubicle,
            "bus": None,
            "element_type": None,
            "element_id": None,
            "quantity": _QUANTITY_NORM[m.group("qty")],
            "phase": m.group("phase"),
            "unit": m.group("unit"),
        }
        cm = _CHAN_CUBICLE.match(cubicle)
        if cm is not None:
            etype = cm.group("type")
            entry["bus"] = cm.group("bus")
            entry["element_type"] = _ETYPE_NORM.get(etype, etype)
            entry["element_id"] = cm.group("id")
        channels.append(entry)
    return channels


def load_feature_metadata(path: str | Path) -> Dict[str, Any]:
    """Load the ``<X_path>.meta.json`` sidecar for a windowed-tensor file."""
    p = Path(path)
    meta_path = p if p.suffix.endswith(".json") else p.with_suffix(p.suffix + ".meta.json")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_paths(
    config: MainConfig,
    data_file_ext: str = "raw",
    labels_file_ext: str = "parquet",
) -> Tuple[str, str]:
    """Generate the (X, y) file paths for a windowed dataset, given ``config``."""
    output_dir = os.path.join(
        config.dataset.dataset_directory,
        config.window_extraction.windows_local_dir,
    )

    def fmt(x: Any) -> str:
        if isinstance(x, str):
            return x.replace(".", "p")
        return f"{x:.3f}".replace(".", "p")

    W = fmt(config.window_extraction.window_length)
    S = fmt(config.window_extraction.step_length_seconds)
    topo = config.dataset.topology

    X_path = os.path.join(output_dir, f"X_{topo}_W{W}_S{S}.{data_file_ext}")
    y_path = os.path.join(output_dir, f"y_{topo}_W{W}_S{S}.{labels_file_ext}")
    return X_path, y_path


def open_raw_memmap(
    path: str, N: int, L: int, dtype: Any = np.float32, mode: str = "r"
) -> np.memmap:
    """Open a ``(N, L, F)`` raw memmap, inferring ``F`` from the file size."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw memmap file not found: {path}")
    if N <= 0 or L <= 0:
        raise ValueError(f"N and L must be positive, got N={N}, L={L}")

    size_bytes = os.path.getsize(path)
    itemsize = np.dtype(dtype).itemsize
    denom = N * L * itemsize
    if size_bytes % denom != 0:
        raise ValueError(
            f"Inconsistent memmap size: file has {size_bytes} bytes, "
            f"expected multiple of {denom} (N={N}, L={L}, dtype={dtype})"
        )

    F = size_bytes // denom
    if F <= 0:
        raise ValueError(f"Inferred F={F} invalid for memmap {path}")

    logger.info(
        "[memmap-open] shape=(%d, %d, %d), dtype=%s, mode=%s, path=%s",
        N, L, F, dtype, mode, path,
    )
    return np.memmap(path, dtype=dtype, mode=mode, shape=(N, L, F), order="C")  # type: ignore[misc]


class GlobalWindowLoader:
    """Loader for precomputed, windowed data in the *global view*.

    - X: np.memmap, shape (N, L, F)
    - y: pd.DataFrame, length N
    """

    def __init__(self, config: MainConfig) -> None:
        self._cfg = config

    def load(self) -> Tuple[np.memmap, pd.DataFrame, Dict[str, Any]]:
        load_time = time()
        logger.info("Loading windowed X and labels y (global view)...")

        X_path, y_path = generate_paths(self._cfg)

        y = self._load_labels(y_path)
        N = len(y)
        if N == 0:
            raise ValueError("Labels Parquet contains zero rows; cannot load X.")

        L = self._compute_window_length()

        feature_meta = load_feature_metadata(X_path)
        feature_names = feature_meta["feature_names"]
        feature_groups = feature_meta.get("feature_groups", {})

        X = self._load_X(X_path, N=N, L=L)

        if X.ndim != 3:
            raise ValueError(f"Expected X with 3 dims (N, L, F), got shape={X.shape!r}")
        if len(X) != len(y):
            raise ValueError(
                f"Number of samples in X ({len(X)}) and y ({len(y)}) do not match."
            )

        _, _, F = X.shape
        if len(feature_names) != F:
            raise ValueError(
                f"Mismatch between feature dimension and feature_names length: F={F}, "
                f"len(feature_names)={len(feature_names)}"
            )

        logger.info(
            "Loaded X: %s, y: %s. Time taken: %.2f s.", X.shape, y.shape, time() - load_time
        )

        loader_meta: Dict[str, Any] = {
            "feature_meta": feature_meta,
            "feature_names": feature_names,
            "feature_groups": feature_groups,
        }
        return X, y, loader_meta

    def _load_labels(self, y_path: str | Path) -> pd.DataFrame:
        try:
            y = pd.read_parquet(y_path)
            logger.info("Loaded labels DataFrame with shape %s", y.shape)
        except Exception as e:
            logger.error("Failed to read Parquet labels at %s: %s", y_path, e)
            raise
        return y

    def _compute_window_length(self) -> int:
        W = float(self._cfg.window_extraction.window_length)
        fs = float(self._cfg.dataset.sampling_frequency)
        L = int(round(W * fs))
        if L <= 0:
            raise ValueError(
                f"Computed window length L={L} from window_length={W} and "
                f"sampling_frequency={fs} is not positive."
            )
        return L

    def _load_X(self, X_path: str | Path, N: int, L: int) -> np.memmap:
        X_path = str(X_path)
        if not Path(X_path).exists():
            raise FileNotFoundError(f"Raw memmap file not found: {X_path}")
        try:
            return open_raw_memmap(X_path, N=N, L=L)
        except Exception as e:
            raise RuntimeError(f"Failed to open raw memmap {X_path}: {e}") from e
