"""Hydra config-composition smoke test.

Confirms main-config.yaml composes without error and every config group
referenced by the codebase (including the analysis group added to fix the
config.analysis.run_subgroup crash -- see docs/PROVENANCE.md) is actually
present and resolvable, without needing the real dataset.
"""

from __future__ import annotations

import os

from hydra import compose, initialize_config_dir

CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config")
)


def test_main_config_composes():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="main-config.yaml",
            overrides=["dataset.dataset_directory=/tmp/does-not-need-to-exist"],
        )

    # Groups referenced by src/dl_fault_analysis at runtime.
    assert cfg.dataset.topology == "hv_double_line_90kv"
    assert cfg.model.model_name
    assert cfg.window_extraction.window_length > 0
    assert cfg.training.n_splits == 5
    assert cfg.training.seeds == [42]
    assert cfg.tracking.use_wandb is False

    # This group did not exist in the originally exported repository, which
    # took it from psp_helper; its absence used to crash run_dl_experiment.py.
    assert cfg.analysis.run_subgroup is False


def test_paper_split_seed_matches_frozen_documentation():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="main-config.yaml",
            overrides=["dataset.dataset_directory=/tmp/does-not-need-to-exist"],
        )
    # config/paper/splits/SPLIT_LIMITATION.md documents seed 42 as frozen.
    assert int(cfg.training.split_seed) == 42
