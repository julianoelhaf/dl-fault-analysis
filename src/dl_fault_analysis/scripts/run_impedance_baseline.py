"""Classical impedance-based fault-location baseline (Reviewer #2.1).

Ported from the private development repository's EPSR reviewer-response
branch -- see docs/PROVENANCE.md for exact source commit and integration
notes (this port genericizes the original's hardcoded cluster paths into
CLI args / environment variables; the scientific logic and archived result
values in results/paper/impedance/ are unchanged). Manuscript Section 3.7 /
Table 10.

Evaluates two physically grounded fault locators on the 90 kV double-line FL
data using the *true* per-episode line parameters from the scenario-metadata
CSV (``hv_double_line_90kv_labels.csv``) -- parameters the data-driven models
never see:

* double-ended synchronized (positive-sequence) -- needs both line terminals;
* single-ended reactance -- needs one terminal (reported per side, mean+worst).

Two evaluation protocols are written so the comparison to the DL models is
honest:

* ``per_window``   -- every ``fault_start`` window (the exact sample selection
  the DL FL pipeline uses); apples-to-apples with the reported DL MAE.
* ``per_episode``  -- one *developed*-fault window per episode (latest
  ``in_fault``); the proper benchmark for a classical locator that assumes a
  settled post-fault phasor.

Outputs to ``results/paper/impedance/`` by default:
``aggregate_metrics.csv``, ``per_line.csv``, ``sensitivity.csv``,
``predictions.parquet``, ``summary.md``, ``run_config.yaml``, ``dataset_hash.txt``.

This is an analytic baseline -- no training, no folds, no learnable parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dl_fault_analysis.data.line_parameters import faulted_line_params, load_line_parameters
from dl_fault_analysis.models.impedance_locator import (
    double_ended_location,
    fundamental_phasors,
    positive_sequence,
    single_ended_reactance,
)

# Terminal channel layout per faulted line: (S_cur, S_vol, R_cur, R_vol).
# S = line_from (lower bus, the reference end for sc_location); R = line_to.
# Channel order is current-first then voltage within each measurement point.
_TERMINALS: Dict[str, Tuple[List[int], List[int], List[int], List[int]]] = {
    "Line_1_2_a": ([0, 1, 2], [3, 4, 5], [12, 13, 14], [15, 16, 17]),
    "Line_1_2_b": ([6, 7, 8], [9, 10, 11], [18, 19, 20], [21, 22, 23]),
    "Line_2_3_a": ([24, 25, 26], [27, 28, 29], [36, 37, 38], [39, 40, 41]),
    "Line_2_3_b": ([30, 31, 32], [33, 34, 35], [42, 43, 44], [45, 46, 47]),
}

# No hardcoded cluster path: base/CSV default to env vars, overridable via CLI.
_DEFAULT_BASE = os.environ.get("PROTECT90_WINDOWS_DIR", "")
_DEFAULT_STEM = "X_hv_double_line_90kv_W0p050_S0p005"
_DEFAULT_CSV = os.environ.get("PROTECT90_LABELS_CSV", "")
_LINE_F0_HZ = 50.0  # 90 kV European grid


@dataclass
class ImpedanceConfig:
    base_dir: str = _DEFAULT_BASE
    stem: str = _DEFAULT_STEM
    csv_path: str = _DEFAULT_CSV
    out_dir: str = "results/paper/impedance"
    f0_hz: float = _LINE_F0_HZ


def _load_windows(cfg: ImpedanceConfig):
    meta = json.load(open(os.path.join(cfg.base_dir, f"{cfg.stem}.raw.meta.json")))
    shape = tuple(meta["shape"])
    X = np.memmap(
        os.path.join(cfg.base_dir, f"{cfg.stem}.raw"),
        dtype=meta["dtype"],
        mode="r",
        shape=shape,
    )
    y = pd.read_parquet(
        os.path.join(cfg.base_dir, f"{cfg.stem.replace('X_', 'y_')}.parquet")
    ).reset_index(drop=True)
    win_seconds = float(meta["extra"]["window_length"])
    spc = int(round(shape[1] / win_seconds / cfg.f0_hz))  # samples per cycle
    return X, y, shape, spc, meta


def _faulted_phases(row) -> List[int]:
    return [k for k, f in enumerate((row.y_phase_A, row.y_phase_B, row.y_phase_C)) if f]


def _estimate_one(window: np.ndarray, line: str, lp, spc: int):
    """Return (double_ended fraction, S/R phasors, z1, z0) for one window."""
    s_cur, s_vol, r_cur, r_vol = _TERMINALS[line]
    vsp = fundamental_phasors(window[:, s_vol], spc)
    isp = fundamental_phasors(window[:, s_cur], spc)
    vrp = fundamental_phasors(window[:, r_vol], spc)
    irp = fundamental_phasors(window[:, r_cur], spc)
    z1, z0 = lp.z1_total, lp.z0_total
    de = double_ended_location(
        positive_sequence(vsp),
        positive_sequence(isp),
        positive_sequence(vrp),
        positive_sequence(irp),
        z1,
    )
    return de, vsp, isp, vrp, irp, z1, z0


def _run_protocol(
    X, sub: pd.DataFrame, params_df, spc: int, label: str
) -> pd.DataFrame:
    recs = []
    for i, row in zip(sub.index.values, sub.itertuples(index=False)):
        line = row.y_fault_line
        if line not in _TERMINALS:
            continue
        lp = faulted_line_params(params_df, int(row.sample_id), line)
        win = np.asarray(X[i])
        de, vsp, isp, vrp, irp, z1, z0 = _estimate_one(win, line, lp, spc)
        fp = _faulted_phases(row)
        grounded = bool(row.y_is_grounded)
        se_s = single_ended_reactance(vsp, isp, z1, z0, fp, grounded)
        se_r = single_ended_reactance(vrp, irp, z1, z0, fp, grounded)
        # se_r is distance from R; express from S for a common frame.
        se_r_fromS = 1.0 - se_r if np.isfinite(se_r) else float("nan")
        true = float(row.y_fault_location)  # % from S
        # fault_resistance lives in the scenario CSV (not the windowed labels).
        rf = float(params_df.loc[int(row.sample_id), "fault_resistance"])
        recs.append(
            {
                "protocol": label,
                "sample_id": int(row.sample_id),
                "line": line,
                "true_pct": true,
                "double_ended_pct": 100.0 * de,
                "single_ended_S_pct": 100.0 * se_s,
                "single_ended_R_pct": 100.0 * se_r_fromS,
                "fault_resistance": rf,
                "n_faulted_phases": len(fp),
                "grounded": grounded,
            }
        )
    df = pd.DataFrame(recs)
    df["de_abs_err"] = (df["double_ended_pct"] - df["true_pct"]).abs()
    se_s_err = (df["single_ended_S_pct"] - df["true_pct"]).abs()
    se_r_err = (df["single_ended_R_pct"] - df["true_pct"]).abs()
    df["se_mean_abs_err"] = np.nanmean(np.vstack([se_s_err, se_r_err]), axis=0)
    df["se_worst_abs_err"] = np.nanmax(np.vstack([se_s_err, se_r_err]), axis=0)
    return df


def _agg(err: pd.Series) -> dict:
    e = err.dropna().to_numpy()
    if e.size == 0:
        return {
            k: float("nan")
            for k in (
                "n",
                "mae",
                "median",
                "rmse",
                "p90",
                "p99",
                "hit_1pct",
                "hit_3pct",
                "hit_5pct",
            )
        }
    return {
        "n": int(e.size),
        "mae": float(np.mean(e)),
        "median": float(np.median(e)),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "p90": float(np.percentile(e, 90)),
        "p99": float(np.percentile(e, 99)),
        "hit_1pct": float(np.mean(e < 1.0)),
        "hit_3pct": float(np.mean(e < 3.0)),
        "hit_5pct": float(np.mean(e < 5.0)),
    }


def _dataset_hash(cfg: ImpedanceConfig) -> str:
    h = hashlib.sha256()
    for path in (
        os.path.join(cfg.base_dir, f"{cfg.stem}.raw.manifest.json"),
        cfg.csv_path,
    ):
        if os.path.exists(path):
            with open(path, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def run_impedance_baseline(cfg: ImpedanceConfig) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    X, y, shape, spc, meta = _load_windows(cfg)
    params_df = load_line_parameters(cfg.csv_path)

    fault = y[(y.y_fault_present == 1) & (y.y_fault_line.isin(_TERMINALS))].copy()
    # per_window: every fault_start window (DL FL sample selection)
    per_window = fault[fault.status == "fault_start"]
    # per_episode: latest developed (in_fault) window, else latest fault_start
    fault["_dev"] = (fault.status == "in_fault").astype(int)
    per_episode = (
        fault.sort_values(["_dev", "window_start_time"]).groupby("sample_id").tail(1)
    )

    dfa = _run_protocol(X, per_window, params_df, spc, "per_window")
    dfb = _run_protocol(X, per_episode, params_df, spc, "per_episode")
    preds = pd.concat([dfa, dfb], ignore_index=True)
    preds.to_parquet(os.path.join(cfg.out_dir, "predictions.parquet"), index=False)

    # aggregate per protocol x method
    rows = []
    for proto, d in (("per_window", dfa), ("per_episode", dfb)):
        for method, col in (
            ("double_ended", "de_abs_err"),
            ("single_ended_mean", "se_mean_abs_err"),
            ("single_ended_worst", "se_worst_abs_err"),
        ):
            rows.append({"protocol": proto, "method": method, **_agg(d[col])})
    agg = pd.DataFrame(rows)
    agg.to_csv(os.path.join(cfg.out_dir, "aggregate_metrics.csv"), index=False)

    # per-line (double-ended, per_episode)
    per_line = (
        dfb.groupby("line")
        .apply(lambda g: pd.Series(_agg(g["de_abs_err"])), include_groups=False)
        .reset_index()
    )
    per_line.to_csv(os.path.join(cfg.out_dir, "per_line.csv"), index=False)

    # sensitivity of double-ended error (per_episode) vs Rf / distance / fault type
    sens = []
    d = dfb.copy()
    d["rf_bin"] = pd.cut(d["fault_resistance"], [0, 1, 3, 6, 10], include_lowest=True)
    for b, g in d.groupby("rf_bin", observed=True):
        sens.append(
            {
                "factor": "fault_resistance_ohm",
                "bucket": str(b),
                **_agg(g["de_abs_err"]),
            }
        )
    d["dist_bin"] = pd.cut(
        d["true_pct"], [0, 20, 80, 100], labels=["near(<20%)", "mid", "far(>80%)"]
    )
    for b, g in d.groupby("dist_bin", observed=True):
        sens.append(
            {"factor": "true_distance", "bucket": str(b), **_agg(g["de_abs_err"])}
        )
    for b, g in d.groupby("n_faulted_phases", observed=True):
        sens.append(
            {"factor": "n_faulted_phases", "bucket": str(b), **_agg(g["de_abs_err"])}
        )
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv(os.path.join(cfg.out_dir, "sensitivity.csv"), index=False)

    _write_summary(cfg, agg, per_line, sens_df, shape, spc)
    with open(os.path.join(cfg.out_dir, "dataset_hash.txt"), "w") as fh:
        fh.write(_dataset_hash(cfg) + "\n")
    with open(os.path.join(cfg.out_dir, "run_config.yaml"), "w") as fh:
        for k, v in vars(cfg).items():
            fh.write(f"{k}: {v}\n")
        fh.write(f"samples_per_cycle: {spc}\n")

    return {"aggregate": agg, "per_line": per_line, "sensitivity": sens_df}


def _write_summary(cfg, agg, per_line, sens, shape, spc) -> None:
    def g(proto, method, k):
        r = agg[(agg.protocol == proto) & (agg.method == method)]
        return float(r[k].iloc[0]) if len(r) else float("nan")

    lines = []
    lines.append("# Impedance-based fault-location baseline (true parameters)\n")
    lines.append(
        "Classical physically grounded baselines evaluated on the 90 kV double-line "
        "50 ms FL data with the **true per-episode line parameters** "
        "(R'/X', length, sequence impedances) from the scenario metadata. These "
        "parameters are NOT available to the data-driven models.\n"
    )
    lines.append(
        f"- Window tensor: {shape}; samples/cycle at {cfg.f0_hz:.0f} Hz: {spc}"
    )
    lines.append(
        "- Distance error in **% of line length** (= MAE units of the DL FL table).\n"
    )
    lines.append("## Headline (double-ended synchronized, true parameters)\n")
    lines.append(
        f"- per_window (same fault_start windows as DL): MAE "
        f"{g('per_window','double_ended','mae'):.2f}% · median "
        f"{g('per_window','double_ended','median'):.2f}% · "
        f"<1% in {100*g('per_window','double_ended','hit_1pct'):.0f}% of windows"
    )
    lines.append(
        f"- per_episode (developed window): MAE "
        f"{g('per_episode','double_ended','mae'):.2f}% · median "
        f"{g('per_episode','double_ended','median'):.3f}% · "
        f"<1% in {100*g('per_episode','double_ended','hit_1pct'):.0f}% of episodes\n"
    )
    lines.append("## Single-ended reactance (one terminal)\n")
    lines.append(
        f"- per_episode mean-side MAE {g('per_episode','single_ended_mean','mae'):.1f}% "
        f"· median {g('per_episode','single_ended_mean','median'):.2f}% "
        f"(unstable tail under load/infeed/Rf -> one terminal is insufficient)\n"
    )
    lines.append("## Aggregate table\n")
    lines.append(agg.to_string(index=False))
    lines.append("\n\n## Per-line (double-ended, per_episode)\n")
    lines.append(per_line.to_string(index=False))
    lines.append("\n\n## Double-ended error sensitivity (per_episode)\n")
    lines.append(sens.to_string(index=False))
    lines.append(
        "\n\n## Interpretation (Reviewer #2.1)\n"
        "With known line parameters, synchronized two-ended measurements, and a "
        "developed post-fault phasor, the classical double-ended method localizes "
        "essentially exactly (sub-1% for the large majority of cases) -- far below "
        "the DL error (~8% of line length at 50 ms). This CONFIRMS the reviewer: "
        "impedance methods are more accurate AND interpretable when their "
        "assumptions hold. The contribution of the DL approach is orthogonal: it "
        "achieves its accuracy **without any line parameters, without knowing the "
        "faulted segment a priori, and from a single forward pass on raw "
        "waveforms** -- and degrades gracefully, whereas the single-ended classical "
        "estimate is unstable. The two are complementary, not competing.\n"
    )
    with open(os.path.join(cfg.out_dir, "summary.md"), "w") as fh:
        fh.write("\n".join(lines))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Impedance-based FL baseline")
    ap.add_argument(
        "--base-dir",
        default=_DEFAULT_BASE,
        help="directory containing the windowed PROTECT-90 .raw/.parquet files "
        "(default: $PROTECT90_WINDOWS_DIR)",
    )
    ap.add_argument("--stem", default=_DEFAULT_STEM)
    ap.add_argument(
        "--csv",
        dest="csv_path",
        default=_DEFAULT_CSV,
        help="path to hv_double_line_90kv_labels.csv "
        "(default: $PROTECT90_LABELS_CSV)",
    )
    ap.add_argument("--out-dir", default="results/paper/impedance")
    args = ap.parse_args(argv)
    if not args.base_dir or not args.csv_path:
        raise SystemExit(
            "--base-dir/--csv (or $PROTECT90_WINDOWS_DIR/$PROTECT90_LABELS_CSV) "
            "must be set to a local PROTECT-90 windowed-dataset directory."
        )
    cfg = ImpedanceConfig(
        base_dir=args.base_dir,
        stem=args.stem,
        csv_path=args.csv_path,
        out_dir=args.out_dir,
    )
    out = run_impedance_baseline(cfg)
    print(out["aggregate"].to_string(index=False))


if __name__ == "__main__":
    main()
