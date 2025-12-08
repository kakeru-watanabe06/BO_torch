# src/BO_torch/data_utils.py
from __future__ import annotations
from typing import List, Optional, Tuple
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Sequence, List
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
    try:
        train_df = pd.read_excel(train_path, engine="openpyxl")
    except Exception as e:
        train_df = pd.read_csv(train_path)
    try:
        all_df = pd.read_excel(all_path, engine="openpyxl")
    except Exception as e:
        all_df = pd.read_csv(all_path)

    # ★ ここで start/end による自動決定をする
    if (x_col_start is not None) or (x_col_end is not None):
        start = x_col_start if x_col_start is not None else 0
        end   = x_col_end   if x_col_end   is not None else len(all_df.columns)

        cols = all_df.columns

        # 追加: strなら列名→位置へ（endは“まで”なので含める）
        if isinstance(start, str):
            start = cols.get_loc(start)
        if isinstance(end, str):
            end = cols.get_loc(end) + 1

        X_cols = list(cols[start:end])
        print(f"[INFO] X_cols resolved by index [{start}:{end}) → {len(X_cols)} columns")

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

def load_online_data(
    train_path: str,
    all_path: str,
    id_col: str,
    smiles_col: str,
    X_cols: List,
    y_cols: List,
    x_col_start: Optional[int] = None,
    x_col_end: Optional[int] = None,
):
    """
    オンライン BO 用のデータローダー。
    - train.xlsx: ID + X + y（標準化済み）
    - all.xlsx  : ID + SMILES + X（y なし）
    """
    try:
        train_df = pd.read_excel(train_path, engine="openpyxl")
    except Exception as e:
        train_df = pd.read_csv(train_path)
    try:
        all_df = pd.read_excel(all_path, engine="openpyxl")
    except Exception as e:
        all_df = pd.read_csv(all_path)

    # 必要なら all 側から X_cols を index指定で決める
    if (x_col_start is not None) or (x_col_end is not None):
        start = x_col_start if x_col_start is not None else 0
        end   = x_col_end   if x_col_end   is not None else len(all_df.columns)

        cols = all_df.columns

        # 追加: strなら列名→位置へ（endは“まで”なので含める）
        if isinstance(start, str):
            start = cols.get_loc(start)
        if isinstance(end, str):
            end = cols.get_loc(end) + 1

        X_cols = list(cols[start:end])
        print(f"[INFO] X_cols resolved by index [{start}:{end}) → {len(X_cols)} columns")

    print("[DEBUG] X_cols number from config/load_online_data:", len(X_cols))
    print("[DEBUG] all_df.columns:", list(all_df.columns))
    print("[DEBUG] smiles_col:", smiles_col)  # ★ これを追加

    # --- used_df（初期点）は train.xlsx からそのまま ---
    used_df = train_df[[id_col] + X_cols + y_cols].reset_index(drop=True)

    # --- pool_df（候補）は all.xlsx から「train にいないID」だけ ---
    train_ids = set(train_df[id_col].astype(str))
    pool_df = all_df.loc[
        ~all_df[id_col].astype(str).isin(train_ids),
        [id_col, smiles_col] + X_cols,   # y はまだない
    ].reset_index(drop=True)

    X_train = torch.tensor(used_df[X_cols].to_numpy(), dtype=torch.double)
    Y_train = torch.tensor(used_df[y_cols].to_numpy(), dtype=torch.double)


    return used_df, pool_df, X_train, Y_train, X_cols


@dataclass
class FixedScaler:
    mean: torch.Tensor  # (m,)
    std: torch.Tensor   # (m,)
    cols: List[str]     # 対応する y_cols の順番

    @classmethod
    def from_config(cls, cols: List[str], mean: Sequence[float], std: Sequence[float]):
        t_mean = torch.tensor(mean, dtype=torch.double)
        t_std  = torch.tensor(std, dtype=torch.double)
        return cls(mean=t_mean, std=t_std, cols=list(cols))

    def transform(self, y_raw_vec: Sequence[float]) -> torch.Tensor:
        y = torch.tensor(y_raw_vec, dtype=torch.double)
        return (y - self.mean) / self.std
