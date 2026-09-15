from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from dl_fault_analysis.utils.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# Feature-group selection
# =============================================================================
_GROUP_ALIASES: Dict[str, str] = {
    "line": "lines",
    "lines": "lines",
    "voltage": "voltages",
    "voltages": "voltages",
    "current": "currents",
    "currents": "currents",
    "extgrid": "ext_grid",
    "ext_grid": "ext_grid",
    "load": "loads",
    "loads": "loads",
    "wind": "winds",
    "winds": "winds",
}


def _normalize_group_name(name: str) -> str:
    return _GROUP_ALIASES.get(name.strip().lower(), name.strip().lower())


def select_feature_indices_from_groups(
    feature_groups: Dict[str, List[int]],
    include_groups: Sequence[str],
) -> List[int]:
    """
    Build a deduped, order-preserving list of feature indices from group names.
    Empty groups are allowed. Unknown groups are warned and ignored.
    """
    if not include_groups:
        return []

    normalized_groups = {_normalize_group_name(k): v for k, v in feature_groups.items()}

    selected: List[int] = []
    missing: List[str] = []

    for g in include_groups:
        g_norm = _normalize_group_name(g)
        idxs = normalized_groups.get(g_norm)
        if idxs is None:
            missing.append(g)
            continue
        selected.extend(idxs)

    if missing:
        logger.warning(
            "Requested groups not found: %s. Available: %s",
            missing,
            sorted(normalized_groups.keys()),
        )

    seen: set[int] = set()
    selected = [i for i in selected if not (i in seen or seen.add(i))]
    return selected


def maybe_filter_features(
    X: np.memmap,
    meta: Dict[str, Any],
    include_groups: Sequence[str],
    materialize: bool,
) -> Tuple[np.ndarray, Optional[Sequence[int]]]:
    """Return (X_used, feature_indices_for_ds)."""
    feature_groups: Dict[str, List[int]] = meta.get("feature_groups", {}) or {}
    selected = select_feature_indices_from_groups(feature_groups, list(include_groups))

    if selected:
        logger.info(
            "Feature filter: include_groups=%s -> selected %d/%d features",
            list(include_groups),
            len(selected),
            X.shape[-1],
        )
    else:
        logger.info("Feature filter: none (using all features)")

    if materialize and selected:
        logger.warning(
            "Materializing X[:, :, selected_indices] (allocates). Use only for small demos."
        )
        return X[:, :, selected], None  # dataset doesn't need per-sample slicing
    return X, (selected if selected else None)
