from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dl_fault_analysis.config import MainConfig
from dl_fault_analysis.data.window_io import GlobalWindowLoader, generate_paths
from dl_fault_analysis.utils.logging import get_logger

logger = get_logger(__name__)


# ----------------------------
# Dataset for memory-mapped features (X) and labels (y)
# ----------------------------
class XYMemmapDataset(Dataset):
    """
    Memory-efficient dataset for (windows, labels) backed by a memmap.

    X: np.memmap, shape (N, T, F)
    y: labels, shape (N,) (numeric or strings depending on task_type)

    Notes
    -----
    - Row subsetting is done via `indices` (no global slicing of X).
    - Feature subsetting is done lazily via `feature_indices` (per-sample), to avoid
      materializing huge arrays.
    - By default, samples are converted to contiguous, writable float32 arrays to
      avoid PyTorch warnings/undefined behavior with non-writable memmaps.
    """

    def __init__(
        self,
        windows_mem: np.memmap,
        y_array: Union[np.ndarray, Sequence[Any]],
        task_type: str,
        indices: Optional[np.ndarray] = None,
        class_to_idx: Optional[Dict[Any, int]] = None,
        feature_indices: Optional[Sequence[int]] = None,
        ensure_writable: bool = True,
    ):
        if not isinstance(windows_mem, np.memmap):
            raise TypeError(f"windows_mem must be np.memmap, got {type(windows_mem)}")

        self.X = windows_mem
        self.ensure_writable = ensure_writable

        # --- row indices (subset) ---
        if indices is None:
            self.indices = np.arange(self.X.shape[0], dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

        # --- feature indices (subset), applied per sample ---
        if feature_indices is None:
            self.feature_indices = None
        else:
            fi = np.asarray(feature_indices, dtype=np.int64)
            if fi.ndim != 1:
                raise ValueError("feature_indices must be 1D")
            # basic bounds check (helps catch config mistakes early)
            if fi.size > 0 and (fi.min() < 0 or fi.max() >= self.X.shape[-1]):
                raise ValueError(
                    f"feature_indices out of bounds: valid [0, {self.X.shape[-1]-1}]"
                )
            self.feature_indices = fi

        # --- labels to numpy 1D ---
        y = np.asarray(y_array)
        if y.ndim > 1 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        if y.ndim != 1:
            raise ValueError(f"y_array must be 1D after squeeze, got shape {y.shape}")

        if self.X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Length mismatch: X has {self.X.shape[0]}, y has {y.shape[0]}."
            )

        # --- encode labels per task ---
        task_type = task_type.lower().strip()
        self.task_type = task_type

        if task_type == "multiclass":
            if class_to_idx is None:
                classes = np.unique(y)
                self.class_to_idx = {c: i for i, c in enumerate(classes)}
            else:
                self.class_to_idx = dict(class_to_idx)

            try:
                y_idx = np.array([self.class_to_idx[v] for v in y], dtype=np.int64)
            except KeyError as e:
                raise KeyError(f"Label {e} not found in class_to_idx mapping.") from e

            self.target_dtype = torch.long
            self.num_classes = len(self.class_to_idx)
            self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

            # store as torch tensor once (faster than torch.tensor per item)
            self.y_t = torch.from_numpy(y_idx)

        elif task_type == "binary":
            if y.dtype.kind in {"U", "S", "O"}:
                uniq = np.unique(y)
                if len(uniq) != 2:
                    raise ValueError(
                        f"Binary task requires exactly 2 unique labels, got {uniq}."
                    )
                bin_map = {uniq[0]: 0.0, uniq[1]: 1.0}
                y_bin = np.array([bin_map[v] for v in y], dtype=np.float32)
            else:
                y_bin = y.astype(np.float32, copy=False)

            self.target_dtype = torch.float32
            self.num_classes = 2
            self.class_to_idx = {0.0: 0, 1.0: 1}
            self.idx_to_class = {0: 0.0, 1: 1.0}

            self.y_t = torch.from_numpy(y_bin)

        elif task_type == "regression":
            y_reg = y.astype(np.float32, copy=False)

            self.target_dtype = torch.float32
            self.num_classes = None
            self.class_to_idx = None
            self.idx_to_class = None

            self.y_t = torch.from_numpy(y_reg)

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        i = int(self.indices[idx])

        # memmap view of one sample: (T, F)
        x_np = self.X[i]

        # optional feature filtering (per sample) - may allocate for this sample only
        if self.feature_indices is not None:
            x_np = x_np[:, self.feature_indices]

        # guarantee float32 (cheap if already float32)
        if x_np.dtype != np.float32:
            x_np = x_np.astype(np.float32, copy=False)

        # avoid torch warning/undefined behavior with non-writable memmap views
        if self.ensure_writable:
            # contiguous copy only if needed
            x_np = np.ascontiguousarray(x_np)
            if not x_np.flags.writeable:
                x_np = x_np.copy()

        x = torch.from_numpy(x_np)

        # fast scalar/tensor indexing
        y = self.y_t[i]
        # ensure correct dtype (y_t is already correct dtype for most cases)
        if y.dtype != self.target_dtype:
            y = y.to(self.target_dtype)

        return x, y


def log_memmap_file_size(config: MainConfig, X: np.ndarray) -> None:
    """Logs on-disk size of the memmap file + expected size from X shape/dtype."""
    try:
        X_path, _ = generate_paths(config)
        X_path = str(X_path)

        size_bytes = os.path.getsize(X_path)
        size_gib = size_bytes / (1024**3)

        expected_bytes = int(np.prod(X.shape)) * np.dtype(X.dtype).itemsize
        expected_gib = expected_bytes / (1024**3)

        logger.info(
            "Memmap file: %s | size=%.2f GiB | dtype=%s | shape=%s | expected≈%.2f GiB",
            X_path,
            size_gib,
            X.dtype,
            X.shape,
            expected_gib,
        )

        if size_bytes != expected_bytes:
            logger.warning(
                "Memmap size mismatch: file=%d bytes vs expected=%d bytes (shape=%s, dtype=%s)",
                size_bytes,
                expected_bytes,
                X.shape,
                X.dtype,
            )
    except Exception as e:
        logger.warning("Could not determine memmap file size: %s", e)


# =============================================================================
# Data loading (global view)
# =============================================================================
def load_windowed_dataset(
    config: MainConfig,
) -> Tuple[np.memmap, pd.DataFrame, Dict[str, Any]]:
    logger.info("Loading windowed dataset (global view)...")
    global_loader = GlobalWindowLoader(config)
    X, y, loader_meta = global_loader.load()

    log_memmap_file_size(config, X)

    feature_meta = loader_meta.get("feature_meta", {}) or {}
    meta: Dict[str, Any] = {
        "topology": str(config.dataset.topology),
        "sampling_frequency": float(config.dataset.sampling_frequency),
        "window_length": float(config.window_extraction.window_length),
        "view": "global",
        "n_samples": int(len(y)),
        "feature_names": loader_meta.get("feature_names", []),
        "feature_meta": feature_meta,
        "feature_groups": feature_meta.get("feature_groups", {}) or {},
    }

    logger.info(
        "Loaded X shape=%s | labels_df shape=%s | topology=%s",
        getattr(X, "shape", None),
        y.shape,
        meta["topology"],
    )
    # logger.debug("First 25 labels:\n%s", y.head(25))
    return X, y, meta


class LocalXYMemmapDataset(Dataset):
    """
    Local-view dataset backed by the GLOBAL memmap (N, L, F).

    Each dataset item corresponds to (global_window_index=gw, local_index=li)
    and returns:
        x: (L, C) where C = channels_per_local
        y: encoded label for gw (same label repeated for all locals)
        gw, li (optional via return_indices=True)

    IMPORTANT: This does NOT materialize X_local. It slices the memmap per item.
    """

    def __init__(
        self,
        X_global: np.ndarray,  # np.memmap recommended, shape (N, L, F)
        y_global: np.ndarray,  # length N (window-level labels)
        task_type: str,
        channels_per_local: int,
        local_indices: Optional[
            np.ndarray
        ] = None,  # subset in *local* space [0..N*n_locals)
        class_to_idx: Optional[Dict[Any, int]] = None,
        return_indices: bool = False,  # if True, returns (x, y, gw, li)
        copy_x: bool = True,
    ) -> None:
        if X_global.ndim != 3:
            raise ValueError(
                f"X_global must be 3D (N,L,F), got shape={X_global.shape!r}"
            )
        self.X = X_global

        y = np.asarray(y_global)
        if y.ndim > 1 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        if len(y) != self.X.shape[0]:
            raise ValueError(
                f"Length mismatch: len(y)={len(y)} vs X_global N={self.X.shape[0]}"
            )

        self.task_type = task_type.lower()
        self.cpl = int(channels_per_local)
        if self.cpl <= 0:
            raise ValueError(f"channels_per_local must be >0, got {self.cpl}")

        N, _L, F = self.X.shape
        if F % self.cpl != 0:
            raise ValueError(f"F={F} not divisible by channels_per_local={self.cpl}")
        self.n_locals = F // self.cpl
        self.N = N

        # --- label encoding (per window gw) ---
        if self.task_type == "multiclass":
            if class_to_idx is None:
                classes = np.unique(y)
                self.class_to_idx = {c: i for i, c in enumerate(classes)}
            else:
                self.class_to_idx = dict(class_to_idx)
            try:
                self.y = np.array([self.class_to_idx[v] for v in y], dtype=np.int64)
            except KeyError as e:
                raise KeyError(f"Label {e} not found in class_to_idx mapping.") from e
            self.target_dtype = torch.long
            self.num_classes = len(self.class_to_idx)

        elif self.task_type == "binary":
            if y.dtype.kind in {"U", "S", "O"}:
                uniq = np.unique(y)
                if len(uniq) != 2:
                    raise ValueError(
                        f"Binary task requires exactly 2 unique labels, got {uniq}."
                    )
                bin_map = {uniq[0]: 0.0, uniq[1]: 1.0}
                y = np.array([bin_map[v] for v in y], dtype=np.float32)
            else:
                y = y.astype(np.float32, copy=False)
            self.y = y
            self.target_dtype = torch.float32
            self.num_classes = 2
            self.class_to_idx = {0.0: 0, 1.0: 1}

        elif self.task_type == "regression":
            self.y = y.astype(np.float32, copy=False)
            self.target_dtype = torch.float32
            self.num_classes = None
            self.class_to_idx = None

        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        self.return_indices = bool(return_indices)

        # --- indexing in local space ---
        total = self.N * self.n_locals
        if local_indices is None:
            self.indices = np.arange(total, dtype=np.int64)
        else:
            idx = np.asarray(local_indices, dtype=np.int64)
            if idx.ndim != 1:
                raise ValueError("local_indices must be a 1D array")
            if idx.size == 0:
                raise ValueError("local_indices is empty")
            if idx.min() < 0 or idx.max() >= total:
                raise IndexError(f"local_indices out of range [0, {total})")
            self.indices = idx

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _map_local(self, local_flat_idx: int) -> Tuple[int, int]:
        """Map flat local index -> (gw, li)."""
        gw = local_flat_idx // self.n_locals
        li = local_flat_idx % self.n_locals
        return int(gw), int(li)

    def __getitem__(self, idx: int):
        local_flat = int(self.indices[idx])
        gw, li = self._map_local(local_flat)

        s = slice(li * self.cpl, (li + 1) * self.cpl)

        # Per-sample copy (safe, writable, warning-free)
        x = torch.tensor(self.X[gw, :, s], dtype=torch.float32)

        y = torch.as_tensor(self.y[gw], dtype=self.target_dtype)

        if self.return_indices:
            return x, y, gw, li
        return x, y


# =============================================================================
# Per-sample scaling (-1, 1)
# =============================================================================
def scale_sample_to_minus1_1(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x_min = np.min(x)
    x_max = np.max(x)
    denom = x_max - x_min
    if denom < eps:
        return np.zeros_like(x, dtype=np.float32)
    x01 = (x - x_min) / denom
    return (x01 * 2.0 - 1.0).astype(np.float32, copy=False)


# =============================================================================
# Dataset (memmap-friendly + feature indices)
# =============================================================================
class WindowedDataset(Dataset):
    """
    Dataset for windowed time-series data with optional feature selection and row mapping.

    Supports memmap arrays for memory efficiency.
    """

    def __init__(
        self,
        X: np.ndarray,  # memmap or ndarray, (N,T,F)
        y: np.ndarray,  # (N_filtered,)
        indices: np.ndarray,  # indices into filtered set
        task_type: str,  # "binary" | "multiclass" | "regression"
        feature_indices: Optional[Sequence[int]] = None,
        ensure_writable: bool = True,
        row_indices: Optional[np.ndarray] = None,
    ):
        self.X = X
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.task_type = task_type
        self.ensure_writable = bool(ensure_writable)

        # ---- task_type ----
        if self.task_type not in {"binary", "multiclass", "regression"}:
            raise ValueError(f"Invalid task_type: {self.task_type}")

        # ---- X shape ----
        if not hasattr(self.X, "shape") or self.X.ndim != 3:
            raise ValueError(
                f"X must be 3D (N,T,F), got shape={getattr(self.X, 'shape', None)}"
            )

        # ---- y ----
        self.y = np.asarray(self.y)
        if self.y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape={self.y.shape}")
        n_filtered = int(self.y.shape[0])
        if n_filtered <= 0:
            raise ValueError("y must be non-empty")

        # ---- row_indices mapping ----
        self.row_indices: Optional[np.ndarray] = None
        if row_indices is not None:
            ri = np.asarray(row_indices, dtype=np.int64)
            if ri.ndim != 1:
                raise ValueError("row_indices must be 1D")
            if ri.shape[0] != n_filtered:
                raise ValueError(
                    f"Length mismatch: row_indices has {ri.shape[0]}, y has {n_filtered}"
                )
            if ri.min() < 0 or ri.max() >= self.X.shape[0]:
                raise ValueError("row_indices contains out-of-range entries for X")
            self.row_indices = ri
        else:
            # Without mapping, filtered space must match X rows
            if self.X.shape[0] != n_filtered:
                raise ValueError(
                    f"Length mismatch: X has {self.X.shape[0]} rows, y has {n_filtered}"
                )

        # ---- indices (subset into filtered space) ----
        if self.indices.ndim != 1:
            raise ValueError("indices must be 1D")
        if self.indices.size == 0:
            raise ValueError("indices must be non-empty")
        if self.indices.min() < 0:
            raise ValueError("indices contains negative values")
        if self.indices.max() >= n_filtered:
            raise ValueError(
                f"indices out of range: max={self.indices.max()} >= n_filtered={n_filtered}"
            )

        # ---- feature selection ----
        self.feature_indices: Optional[np.ndarray] = None
        if feature_indices is not None:
            fi = np.asarray(feature_indices, dtype=np.int64)
            if fi.ndim != 1:
                raise ValueError("feature_indices must be 1D")
            if fi.size == 0:
                raise ValueError("feature_indices must be non-empty")
            if fi.min() < 0 or fi.max() >= self.X.shape[2]:
                raise ValueError("feature_indices out of range for X feature dimension")
            self.feature_indices = fi

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        j = int(self.indices[idx])  # index into filtered set

        # map to original memmap row
        i = int(self.row_indices[j]) if self.row_indices is not None else j

        x_np = self.X[i]  # (T,F) view into memmap
        if self.feature_indices is not None:
            x_np = x_np[:, self.feature_indices]  # per-sample allocation only

        x_np = scale_sample_to_minus1_1(x_np)

        if self.ensure_writable:
            x_np = np.ascontiguousarray(x_np)
            if not x_np.flags.writeable:
                x_np = x_np.copy()

        x = torch.from_numpy(x_np)

        if self.task_type == "binary":
            y = torch.tensor(float(self.y[j]), dtype=torch.float32)
        elif self.task_type == "multiclass":
            y = torch.tensor(int(self.y[j]), dtype=torch.long)
        else:
            y = torch.tensor(float(self.y[j]), dtype=torch.float32)

        return x, y
