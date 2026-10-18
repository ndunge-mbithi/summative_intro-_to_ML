"""Reproducibility and environment helpers."""
from __future__ import annotations

import os
import platform
import random
from pathlib import Path

import numpy as np

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_global_seed(seed: int = SEED, deterministic_tf: bool = True) -> None:
    """Seed Python, NumPy and (if installed) TensorFlow for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
        if deterministic_tf:
            tf.config.experimental.enable_op_determinism()
    except ImportError:  # TensorFlow is optional for the traditional-ML modules
        pass


def environment_report() -> dict:
    """Collect package versions and hardware details for the reproducibility record."""
    import importlib

    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil  # optional

        report["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        try:
            with open("/proc/meminfo") as fh:
                kb = int(fh.readline().split()[1])
            report["ram_gb"] = round(kb / 1e6, 1)
        except OSError:
            report["ram_gb"] = None
    for pkg in ["numpy", "pandas", "sklearn", "tensorflow", "keras", "scipy",
                "matplotlib", "seaborn", "shap", "pyarrow"]:
        try:
            report[pkg] = importlib.import_module(pkg).__version__
        except ImportError:
            report[pkg] = "not installed"
    try:
        import tensorflow as tf

        report["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except ImportError:
        report["gpus"] = []
    return report
