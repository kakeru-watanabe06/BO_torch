# src/BO_torch/data_utils.py
from __future__ import annotations
from typing import List, Optional, Tuple
import pandas as pd
import torch

def to_tensor(df: pd.DataFrame, cols: List[str], dtype=torch.double) -> torch.Tensor:
    """DataFrame の列を torch.Tensor(double) に変換"""
    return torch.tensor(df[cols].to_numpy(), dtype=dtype)

def load_offline_data(
    train_path: str,
    all_path: str,
    id_col: str,
    X_cols: List[str],
    y_cols: List[str],
    x_col_start: Optional[int] = None,
    x_col_end: Optional[int] = None,
):
    """
    既知データ空間（all）と、学習済み初期点（train）から
      - used_df（学習に使う行）
      - pool_df（未使用プール）
      - X_train, Y_train_raw (Tensor)
    を返すユーティリティ。
    """
    train_df = pd.read_excel(train_path, engine="openpyxl")
    all_df   = pd.read_excel(all_path,   engine="openpyxl")

    # ★ ここで start/end による自動決定をする
    if (x_col_start is not None) or (x_col_end is not None):
        start = x_col_start if x_col_start is not None else 0
        end   = x_col_end   if x_col_end   is not None else len(all_df.columns)
        X_cols = list(all_df.columns[start:end])
        print(f"[INFO] X_cols resolved by index [{start}:{end}) → {len(X_cols)} columns")

        if len(X_cols) == 0:
            raise ValueError(f"No X columns selected by range [{start}:{end}). Check x_col_start / x_col_end.")

    train_ids = set(train_df[id_col].astype(str))

    used_df = all_df.loc[
        all_df[id_col].astype(str).isin(train_ids),
        [id_col] + X_cols + y_cols,
    ].reset_index(drop=True)

    pool_df = all_df.loc[
        ~all_df[id_col].astype(str).isin(train_ids),
        [id_col] + X_cols + y_cols,
    ].reset_index(drop=True)

    X_train = to_tensor(used_df, X_cols, dtype=torch.double)
    Y_train_raw = to_tensor(used_df, y_cols, dtype=torch.double)

    print("[DEBUG] X_train.shape =", X_train.shape)  # 一回だけ様子を見る用
    return used_df, pool_df, X_train, Y_train_raw, X_cols