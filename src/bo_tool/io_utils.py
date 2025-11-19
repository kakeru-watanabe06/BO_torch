# src/BO_torch/io_utils.py
from __future__ import annotations
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_history_excel(history: List[Dict], path: str | Path) -> None:
    df = pd.DataFrame(history)
    ensure_dir(Path(path).parent)
    df.to_excel(path, index=False)


def save_metrics_excel(metrics_df: pd.DataFrame, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    metrics_df.to_excel(path, index=False)
