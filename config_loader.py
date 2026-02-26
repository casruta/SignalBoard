"""Load and validate configuration from config.yaml."""

import os
from pathlib import Path

import yaml


_CONFIG = None


def load_config(path: str | None = None) -> dict:
    """Load config from YAML file. Caches after first load."""
    global _CONFIG
    if _CONFIG is not None and path is None:
        return _CONFIG

    if path is None:
        path = os.environ.get(
            "SIGNALBOARD_CONFIG",
            str(Path(__file__).parent / "config.yaml"),
        )

    with open(path) as f:
        config = yaml.safe_load(f)

    _CONFIG = config
    return config


def get_config() -> dict:
    """Return already-loaded config, loading defaults if needed."""
    if _CONFIG is None:
        return load_config()
    return _CONFIG
