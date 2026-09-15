# Publication cross-validation split: what is and is not recoverable

The paper reports an episode-grouped 5-fold cross-validation protocol,
applied consistently across FD, FC, FLI, and FL:

- Outer split: 5 folds, grouped by episode (`sample_id`) so that no episode's
  windows appear in both the train and test portion of a fold.
- Multiclass tasks (FC, FLI): `sklearn.model_selection.StratifiedGroupKFold`,
  so each fold has a similar class distribution.
- Binary/regression tasks (FD, FL): `sklearn.model_selection.GroupKFold`
  (unstratified).
- A single fixed seed, `42`, is used throughout (see
  `config/training/default.yaml` and `docs/PROVENANCE.md` for the related
  seed-list cleanup).

**What is frozen here:** the algorithm and the seed above, implemented in
`src/dl_fault_analysis/utils/cv_utils.py::build_cv_splits_stratified`.

**What is *not* recoverable:** the exact realized episode-to-fold assignment
used to produce the published Tables 4-10. No split manifest (an
episode-id-to-fold-id mapping) was ever committed to either the public or the
private development repository, and the archived MLflow run artifacts from
the original training runs contain no saved fold-split files or checkpoints
to reconstruct it from. See `docs/PROVENANCE.md` for the full evidence trail.

**Consequence:** re-running `build_cv_splits_stratified` with the same
inputs, seed, and scikit-learn version will deterministically reproduce *a*
valid 5-fold split satisfying the same invariants the paper's protocol
guarantees (each episode is a test episode exactly once, no train/test
episode overlap, all folds represented -- see
`tests/unit/test_cv_utils.py`). It is not guaranteed, and should not be
assumed, to be *byte-identical* to the specific fold assignment used for the
published numbers -- a different scikit-learn version, in particular, can
produce a different (but equally valid) `StratifiedGroupKFold`/`GroupKFold`
partition from the same seed.

No new split was fabricated to paper over this gap, and no regenerated split
is presented as "the publication split".

The Weights & Biases export behind Tables 4-7 is now archived locally
(`results/paper/main/raw/`, see `docs/PROVENANCE.md` §8), and it does not
close this gap: it records per-fold *metrics*, but no episode-to-fold
mapping. The split-generation method and seed are known; the exact
historical episode-to-fold assignment is not recoverable.
