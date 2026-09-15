# Runtime benchmark protocol (Table 11)

No runtime-measurement code exists in the current `dl_fault_analysis` (or
`dl_psp`) codebase -- it lived in a script (`src/dl_psp/models/run_model.py`)
that was deleted from the private development repository's history well
before the commit this public release is based on. The implementation was
recovered by inspecting the private repository's git history one commit
before its removal, and is **not** re-added here (see the "no code change"
decision in `docs/PROVENANCE.md`): this file documents the recovered
protocol rather than shipping an executable reproduction of it.

**Recovered implementation** (`measure_runtime`, private-repo git history
only): 50 warm-up iterations, 5000 timed iterations, `torch.cuda.synchronize()`
called once after warm-up and once per timed iteration, `model.eval()` +
`torch.no_grad()`, fp32, host-to-device data transfer performed once before
the timed loop (excluded from the timed region).

**Two discrepancies against the manuscript's stated protocol** (Section 3.9),
found while cross-referencing this implementation against raw per-model
result data:
1. The timed batch is `min(config.batch_size, len(val_set))` (up to 256 in
   the default config), not literally a single sample as the text states.
2. A raw per-model overview CSV in the private repository
   (`reports/csv/overview_by_model_50ms.csv`) records `inference_hardware`
   varying across models (Tesla V100, RTX 2080 Ti, RTX 3080) rather than a
   single fixed GPU. Table 11's own caption ("aggregated over the latest
   completed runs") suggests the published numbers already reflect this
   rolling, mixed-hardware measurement practice rather than one controlled
   sweep.

**Disposition:** archived Table 11 values are preserved as published; no
code was changed and no re-measurement was performed. A fresh benchmark on
different hardware would be a new measurement, not a reproduction of the
published numbers, and is out of scope for this release.
