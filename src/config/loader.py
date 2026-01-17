"""YAML configuration loading utilities."""

import yaml


def load_yaml(path: str) -> dict:
    """Load a YAML configuration file.
    
    Args:
        path: Path to the YAML file.
        
    Returns:
        Dictionary containing the parsed configuration.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
