"""Configuration: one YAML file with every hyperparameter (configs/default.yaml).

A run can override any subset of keys with its own YAML file; nested sections are merged.
"""
import copy
import os

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "default.yaml")


def merge(base, extra):
    """Recursively merge dict ``extra`` into dict ``base`` (in place) and return ``base``."""
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path=None, overrides=None):
    """Default configuration, optionally updated by a YAML file and/or a dict."""
    if not os.path.exists(DEFAULT_CONFIG):
        raise FileNotFoundError("%s not found: run divtrack from the repository (python scripts/...) or install it "
                                "in editable mode (pip install -e .)" % DEFAULT_CONFIG)
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path:
        with open(path, encoding="utf-8") as f:
            merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        merge(cfg, overrides)
    return cfg


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
