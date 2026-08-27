import os
from typing import Any, Dict
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse a YAML experiment configuration file.

    Args:
        config_path (str): Path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
