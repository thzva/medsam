"""Training utilities: seed, config I/O."""

import os
import random
import yaml
import numpy as np
import torch
from typing import Dict
from pathlib import Path


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> Dict:
    """Load YAML config file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_config(cfg: Dict, path: str):
    """Save config dict to YAML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
