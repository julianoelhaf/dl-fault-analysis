# Classical Impedance-Based Fault-Location Baseline (Reviewer #2, comment 1)

**Answers Reviewer #2(1):** *"traditional impedance-based methods can achieve
significantly lower localization errors … grounded in well-established physical
principles … In contrast, the deep learning approach … is purely data-driven and
lacks physical interpretability."*

We implemented the classical physically grounded fault locators and evaluated
them on the **same 90 kV double-line 50 ms FL data**, using the **true
per-episode line parameters** recovered from the PROTECT-90 scenario metadata
(`hv_double_line_90kv_labels.csv`). These parameters — per-line length and
positive/zero-sequence series impedance (R′, X′) — are **not available to the
deep-learning models**, which localize from raw waveforms alone.

- Code: [`run_impedance_baseline.py`](../src/dl_fault_analysis/scripts/run_impedance_baseline.py),
  [`impedance_locator.py`](../src/dl_fault_analysis/models/impedance_locator.py),
  [`line_parameters.py`](../src/dl_fault_analysis/data/line_parameters.py); unit-tested (exact recovery on synthetic lines).
- Results: `results/paper/impedance/`.
- Method: fundamental phasors via full-cycle DFT (6400 Hz → 128 samples/cycle at 50 Hz);
  symmetrical components; **double-ended synchronized** (positive-sequence) and
  **single-ended reactance** (fault-loop selected) locators. Analytic — no training, no folds.

## Provenance of the parameters
The scenario CSV joins **exactly** to our windowed data: all 9022 episode
`sample_id`s match, `fault_target == y_fault_line` for every episode, and
`sc_location` is identical to `y_fault_location` (% of line length from the
`line_from` terminal). Line lengths span 10–60 km; R′ 0.01–0.20 Ω/km;
X′ 0.35–0.45 Ω/km — i.e. the parameters are **randomized per episode**, which is
precisely why a classical method cannot be applied without this table.

## Headline results — error in % of line length

| Method (true params) | needs line params | needs both terminals | needs known faulted line | MAE | median | < 1% |
|---|:--:|:--:|:--:|--:|--:|--:|
| **Double-ended synchronized** — per-window¹ | yes | yes | yes | **3.62** | **0.54** | 60% |
| **Double-ended synchronized** — per-episode² | yes | yes | yes | 5.08 | **0.028** | **86%** |
| Single-ended reactance — per-episode (mean side) | yes | no | yes | ~217 | ~17 | 10% |
| — | | | | | | |
| **DL GRU (50 ms, full obs)** — paper tuned | **no** | **no** | **no** | 7.57 | — | — |
| DL GRU / CNN-LSTM / InceptionTime (our fixed-opt full) | no | no | no | 7.94 / 10.70 / 9.89 | — | — |

¹ *per-window* = every `fault_start` window (the exact samples the DL FL pipeline scores) — apples-to-apples with the DL MAE.
² *per-episode* = one developed (`in_fault`) window per episode — the proper setting for a classical locator that assumes a settled post-fault phasor.

### Reading the numbers
- **Double-ended, with true parameters, localizes essentially exactly.** Per
  episode the **median error is 0.03%** of line length and **86% of episodes are
  within 1%** — roughly two orders of magnitude below the DL error (~8%). This
  **confirms the reviewer**: when its assumptions hold, the impedance method is
  far more accurate and is physically interpretable by construction.
- Even **per-window, apples-to-apples with the DL evaluation, double-ended (MAE
  3.62%) beats the DL models (7.6–10.7%)** by ~2×, despite a heavier tail from
  windows where the fault is not yet developed.
- The **MAE > median** gap (5.08 vs 0.03 per-episode) is a long tail from a
  minority of degenerate windows (fault inception inside the window, currents
  near breaker clearing). Classical locators require careful phasor/window
  conditioning; we report median and hit-rates as the primary metrics, as is
  standard for fault locators.

## Fault-resistance independence (directly addresses the reviewer's caveat)
The reviewer notes impedance methods excel "without … high-impedance grounding."
The double-ended positive-sequence method is **fault-resistance independent by
construction**, and the data confirm it — median error is flat across the full
0–10 Ω fault-resistance range:

| fault resistance | median err | MAE |
|---|--:|--:|
| 0–1 Ω | 0.036% | 5.52% |
| 1–3 Ω | 0.030% | 4.91% |
| 3–6 Ω | 0.027% | 5.48% |
| 6–10 Ω | 0.028% | 4.77% |

Error is likewise flat across fault types (SLG/LL/3φ median 0.06/0.03/0.01%) and
mildly higher only for faults very near the remote terminal (>80% from S: p90 27%).

## Single-ended is unreliable on this topology
The **single-ended reactance method is unstable** here (median ~17%, MAE ~200%,
heavy tail). Two physical reasons: (i) **zero-sequence mutual coupling** between
the two parallel circuits of the double-line corridor, which we did not
compensate; (ii) remote infeed and load. Notably, **the DL single-ended /
single-terminal result (~18% MAE, stable) is more robust than the classical
single-ended estimate** — i.e. when restricted to one terminal the learned model
degrades gracefully where the textbook reactance method does not. This is
consistent with the reduced-observability finding (E1) that fault localization
fundamentally needs **double-ended** observation of the faulted line.

## What this means for the manuscript (rebuttal of #2.1)
1. **We agree, and now quantify it:** with known line parameters, synchronized
   two-ended measurements, a developed post-fault phasor, and ideal instrument
   transformers (the EMT dataset models no CT saturation — the regime *most
   favorable* to impedance methods), the classical double-ended locator is
   near-exact (median 0.03%), far below the DL error.
2. **The contributions are complementary, not competing.** The DL models reach
   ~8% **without any line parameters, without knowing the faulted segment a
   priori, and from a single forward pass on raw waveforms** — the setting where
   classical methods cannot be applied at all (parameters here are randomized and
   withheld). The value proposition is parameter-agnostic, topology-general,
   multi-task inference, not beating a fully-informed impedance relay on its home
   turf.
3. **Interpretability:** the reduced-observability analysis (E1) shows the
   learned model relies on exactly the **two terminals of the faulted line** that
   the physical double-ended method uses — and collapses to one terminal — i.e.
   the data-driven solution is physically consistent, not exploiting spurious
   global cues.

## Honest caveats
- EMT simulation uses **ideal instrument transformers**; CT saturation (which the
  reviewer notes degrades impedance methods) is therefore **not** exercised — the
  comparison is deliberately in the impedance method's best case.
- Single-ended results are a **lower bound**: no parallel-line mutual-coupling
  compensation was applied.
- The double-ended MAE tail reflects un-conditioned analysis windows, not the
  method's intrinsic accuracy (see median / hit-rates).
