"""Structured-config schema for the Hydra pipeline.

Vendored (and trimmed to the fields this repository's `config/` tree actually
uses) from the private `psp_helper` package's `config.MainConfig` -- see
docs/PROVENANCE.md. Note this is used only as a static type annotation on
`run_dl_experiment.main(config: MainConfig)`; it is never registered with
Hydra's ConfigStore, so at runtime `config` is a plain `omegaconf.DictConfig`
composed from the YAML files under `config/`, not an instance of this
dataclass. Keeping the schema here documents the intended shape and gives
IDE/static-analysis support without reintroducing the external dependency.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

WandbMode = Literal["disabled", "offline", "online"]


@dataclass
class DatasetConfig:
    topology: str
    dataset_directory: str = ""
    windows_local_dir: str = ""
    sampling_frequency: int = 0
    duration: float = 0.0
    expected_start_time: float = 0.0
    event_start_column: str = ""


@dataclass
class ModelConfig:
    model_name: str = ""
    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = False
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    target_label: str = ""
    n_splits: int = 5
    split_seed: int = 42
    stratify: bool = True

    test_size: float = 0.15
    val_size: float = 0.15

    seeds: List[int] = field(default_factory=lambda: [42])

    epochs: int = 500
    batch_size: int = 256
    learning_rate: float = 1e-4

    binary_threshold: float = 0.5
    weight_decay: float = 1e-4

    feature_groups_include: List[str] = field(default_factory=list)
    materialize_feature_filters: bool = False

    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True


@dataclass
class WindowExtractionConfig:
    period_of_interest: float = 0.08
    step_length_seconds: float = 0.005
    window_length: float = 0.01
    windows_local_dir: str = ""
    fault_start_only: bool = False


@dataclass
class TrackingConfig:
    use_wandb: bool = False
    mode: WandbMode = "disabled"
    project: str = ""
    entity: str = ""


@dataclass
class AnalysisConfig:
    run_subgroup: bool = False


@dataclass
class MainConfig:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    window_extraction: WindowExtractionConfig

    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def copy(self) -> "MainConfig":
        return deepcopy(self)
