from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd
import torch


def _read_table(path: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        return pd.read_csv(path)


def _resolve_x_cols(
    all_df: pd.DataFrame,
    x_cols: Optional[Sequence[str]],
    x_col_start: Optional[int | str],
    x_col_end: Optional[int | str],
) -> list[str]:
    if (x_col_start is None) and (x_col_end is None):
        return list(x_cols or [])

    start = x_col_start if x_col_start is not None else 0
    end = x_col_end if x_col_end is not None else len(all_df.columns)
    cols = all_df.columns

    if isinstance(start, str):
        start = cols.get_loc(start)
    if isinstance(end, str):
        end = cols.get_loc(end) + 1

    resolved = list(cols[start:end])
    print(f"[INFO] X_cols resolved by index [{start}:{end}) -> {len(resolved)} columns")
    return resolved


def to_tensor(df: pd.DataFrame, cols: Sequence[str], dtype: torch.dtype = torch.double) -> torch.Tensor:
    """DataFrame の列を torch.Tensor に変換する。"""
    return torch.tensor(df[list(cols)].to_numpy(), dtype=dtype)


def load_offline_data(
    train_path: str,
    all_path: str,
    id_col: str,
    X_cols: Optional[Sequence[str]],
    y_cols: Sequence[str],
    x_col_start: Optional[int | str] = None,
    x_col_end: Optional[int | str] = None,
):
    """
    既知空間のデータから offline BO 用の初期集合とプールを組み立てる。
    """
    train_df = _read_table(train_path)
    all_df = _read_table(all_path)

    resolved_x_cols = _resolve_x_cols(all_df, X_cols, x_col_start, x_col_end)
    resolved_y_cols = list(y_cols)

    train_ids = set(train_df[id_col].astype(str))

    used_df = all_df.loc[
        all_df[id_col].astype(str).isin(train_ids),
        [id_col] + resolved_x_cols + resolved_y_cols,
    ].reset_index(drop=True)

    pool_df = all_df.loc[
        ~all_df[id_col].astype(str).isin(train_ids),
        [id_col] + resolved_x_cols + resolved_y_cols,
    ].reset_index(drop=True)

    X_train = to_tensor(used_df, resolved_x_cols, dtype=torch.double)
    Y_train_raw = to_tensor(used_df, resolved_y_cols, dtype=torch.double)

    return used_df, pool_df, X_train, Y_train_raw, resolved_x_cols


def load_online_data(
    train_path: str,
    all_path: str,
    id_col: str,
    smiles_col: str,
    X_cols: Optional[Sequence[str]],
    y_cols: Sequence[str],
    x_col_start: Optional[int | str] = None,
    x_col_end: Optional[int | str] = None,
):
    """
    オンライン BO 用のデータローダー。
    - train: 初期点（ID + X + y）
    - all: 候補全集合（ID + SMILES + X）
    """
    train_df = _read_table(train_path)
    all_df = _read_table(all_path)

    resolved_x_cols = _resolve_x_cols(all_df, X_cols, x_col_start, x_col_end)
    resolved_y_cols = list(y_cols)

    used_df = train_df[[id_col] + resolved_x_cols + resolved_y_cols].reset_index(drop=True)

    train_ids = set(train_df[id_col].astype(str))
    pool_df = all_df.loc[
        ~all_df[id_col].astype(str).isin(train_ids),
        [id_col, smiles_col] + resolved_x_cols,
    ].reset_index(drop=True)

    X_train = torch.tensor(used_df[resolved_x_cols].to_numpy(), dtype=torch.double)
    Y_train = torch.tensor(used_df[resolved_y_cols].to_numpy(), dtype=torch.double)

    return used_df, pool_df, X_train, Y_train, resolved_x_cols


@dataclass
class FixedScaler:
    mean: torch.Tensor
    std: torch.Tensor
    cols: list[str]

    @classmethod
    def from_config(cls, cols: Sequence[str], mean: Sequence[float], std: Sequence[float]) -> "FixedScaler":
        t_mean = torch.tensor(mean, dtype=torch.double)
        t_std = torch.tensor(std, dtype=torch.double)
        return cls(mean=t_mean, std=t_std, cols=list(cols))

    def transform(self, y_raw_vec: Sequence[float]) -> torch.Tensor:
        y = torch.tensor(y_raw_vec, dtype=torch.double)
        return (y - self.mean) / self.std
