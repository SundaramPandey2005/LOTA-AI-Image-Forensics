"""YAML configuration with defaults and dotted command-line overrides."""
import copy
import os
from typing import Any, Dict, Iterable, Optional

import yaml

DEFAULTS: Dict[str, Any] = {
    "name": "lota",
    "seed": 0,
    "data": {
        "root": "./data/GenImage",
        "train_generator": "sdv5",
        "val_fraction": 0.1,          # held out from the train split for checkpoint selection
        "max_train_per_class": None,  # None = use everything
        "max_test_per_class": None,
        "num_workers": 4,
    },
    "forensic": {
        "bit_mode": "scaling",        # scaling (official code) | thresholding (Eq. 4)
        "num_bits": 3,
        "patch_size": 32,
        "num_candidates": 64,
        "selection": "max",           # max (MGPS) | min | random
        "image_size": 256,
        "resize_before_bits": False,  # True reproduces the earlier, incorrect ordering
    },
    "model": {
        "backbone": "resnet50",
        "pretrained": True,
        "num_patches": 1,             # K: 1 = original LOTA, >1 = LOTA-MP
        "aggregation": "logit_mean",
    },
    "train": {
        "epochs": 10,
        "batch_size": 32,             # images per step (each contributes K patches)
        "lr": 1.0e-4,
        "lr_power": 0.9,              # poly schedule of the official code
        "instance_loss_weight": 0.0,
        "amp": True,
    },
    "eval": {
        "max_patches": 8,             # patches extracted per test image
        "k_values": [1, 2, 4, 8],
        "batch_size": 32,
    },
}


def _merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def apply_overrides(config: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Apply ``section.key=value`` overrides; values are parsed as YAML."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like section.key=value")
        dotted, raw = item.split("=", 1)
        node = config
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return config


def load_config(path: Optional[str] = None, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULTS)
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            _merge(config, yaml.safe_load(handle) or {})
    return apply_overrides(config, overrides)
