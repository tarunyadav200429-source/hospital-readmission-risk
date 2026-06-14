"""
config.py  --  loads config/config.yaml into a plain Python dictionary.

Every other script does:
    from src.config import load_config
    cfg = load_config()
    url = cfg["data"]["source_url"]

This keeps all settings in ONE file (config/config.yaml) instead of scattered
across the code -- the same pattern real data-science teams use.
"""

from pathlib import Path          # safe, OS-independent file paths
import yaml                       # reads .yaml files into Python dicts

# PROJECT_ROOT = the top folder of the repo (this file is in <root>/src/).
# Path(__file__) is this file; .resolve() makes it absolute; .parents[1] goes
# up two levels: src/config.py -> src/ -> <repo root>.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | None = None) -> dict:
    """Read config/config.yaml and return it as a dictionary.

    `path` lets tests point at a different file; normally we use the default.
    """
    cfg_path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, "r") as f:        # open the file for reading
        cfg = yaml.safe_load(f)           # parse YAML text into a dict
    return cfg
