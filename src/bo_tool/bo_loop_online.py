# src/BO_torch/optimization/bo_loop_online.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable
import torch
import pandas as pd
from pathlib import Path

from bo_tool.models import build_models
from bo_tool.acquisition import pick_next
from bo_tool.objectives import ObjectiveSpec
from bo_tool.metrics import brute_force_loocv_metrics
from bo_tool.io_utils import save_history_excel  # まだ import してなければ



@torch.no_grad()
def _compute_target_dists(
    y_vec: torch.Tensor,
    targets: Optional[List[float]],
) -> Tuple[Optional[List[float]], Optional[float]]:
    """
    target_distance / mixed_multi 用:
    1 点分の y_vec と targets の絶対差 |y - t| を次元ごと / 合計で返す。
    """
    if targets is None:
        return None, None
    t = torch.tensor(targets, dtype=y_vec.dtype, device=y_vec.device)
    per_dim = (y_vec - t).abs().tolist()
    total = float(sum(per_dim))
    return per_dim, total


def online_bo_loop(
    X_cols: List[str],
    y_cols: List[str],
    id_col: str,
    smiles_col: str,
    pool_df: pd.DataFrame,
    X_train: torch.Tensor,
    Y_train_raw: torch.Tensor,
    spec: ObjectiveSpec,
    model_cfg,
    observe_func: Callable[[pd.Series], torch.Tensor],
    max_iters: int = 64,
    num_mc_samples: int = 512,
    acq_type: str = "auto",
    ucb_beta: float = 2.0,
    eval_cfg: Optional[Dict] = None,
    save_history_dir: Optional[str] = None,
) -> List[Dict]:
    """
    オンライン BO ループ（真値は外部計算で取得する版）。

    - モデル構築・獲得関数・LOOCV ロジックは offline_bo_loop と同じ。
    - 「真値観測」の部分だけを、pool_df から y を読む → observe_func(picked)
      に置き換えている。

    Parameters
    ----------
    X_cols:
        説明変数として使う列名のリスト。
    y_cols:
        BO が最適化する目的変数（ここでは既に標準化済みのスケール想定）。
    id_col:
        行を一意に識別する ID 列。
    smiles_col:
        外部計算で使う SMILES が入った列名（ログにも残す）。
    pool_df:
        まだ評価していない候補プール。
        必須カラム: [id_col, smiles_col] + X_cols
    X_train:
        これまでに評価済みの候補の説明変数 (N, d) Tensor。
    Y_train_raw:
        これまでに評価済みの候補の目的変数 (N, m) Tensor。
        「プロジェクト内で統一したスケール」（固定標準化後）の値。
    spec:
        ObjectiveSpec。to_object_space などで使用。
    model_cfg:
        ModelConfig 相当。build_models にそのまま渡す。
    observe_func:
        picked_row: pd.Series → newY: torch.Tensor を返すコールバック。
        - picked_row: pool_df.iloc[idx] がそのまま渡される。
        - newY: shape (m,) or (1, m) の Tensor（標準化済み）。
    max_iters:
        最大反復回数。
    num_mc_samples:
        MC サンプル数（qEI / qEHVI 用）。
    eval_cfg:
        LOOCV など評価の設定。例: {"loocv": True, "min_points": 5}

    Returns
    -------
    history: List[Dict]
        各イテレーションでのログ（offline_bo_loop と同じテイスト）。
    """
    history: List[Dict] = []

    device = X_train.device
    X_train = X_train.to(device=device, dtype=torch.double)
    Y_train_raw = Y_train_raw.to(device=device, dtype=torch.double)

    for it in range(1, max_iters + 1):
        # 候補が尽きたら終了
        if len(pool_df) == 0:
            break

        # ===== 1) モデル学習 =====
        model = build_models(X_train, Y_train_raw, model_cfg)

        # ===== 2) 候補選択 =====
        X_pool = torch.tensor(pool_df[X_cols].to_numpy(), dtype=torch.double, device=device)
        best_idx, vals, is_multi = pick_next(
            model=model,
            X_pool=X_pool,
            Y_train_raw=Y_train_raw,
            spec=spec,
            num_mc_samples=num_mc_samples,
            acq_type=acq_type,
            ucb_beta=ucb_beta,
        )

        # ===== 3) 真値観測（オンライン計算） =====
        picked = pool_df.iloc[best_idx]
        print("========== ONLINE BO DEBUG ==========")
        print(f"[iter {it}] best_idx: {best_idx}")
        print(f"[iter {it}] acq values of best_idx: {vals[best_idx]}")
        print(f"[iter {it}] picked ID: {picked[id_col]}")
        if "SMILES" in pool_df.columns:
            print(f"[iter {it}] picked SMILES: {picked['SMILES']}")
        print(f"[iter {it}] picked X values:")
        print(picked[X_cols])
        print(f"[iter {it}] picked X dtypes:")
        print(picked[X_cols].dtypes)
        print("=====================================")


        # X はこれまで通りプールの特徴量をそのまま使う
        newX_vals = pd.to_numeric(picked[X_cols], errors="coerce").to_numpy(dtype="float64").reshape(1, -1)
        newX = torch.tensor(newX_vals, dtype=torch.double)
        # ★ observe_func: SMILES などを使って外部計算 → y_raw → 固定スケールで newY を返す
        newY = observe_func(picked)  # -> torch.Tensor

        if newY.ndim == 1:
            newY = newY.view(1, -1)
        newY = newY.to(dtype=torch.double, device=device)
        assert newY.shape[1] == len(y_cols), (
            f"observe_func が返した newY の列数 {newY.shape[1]} が "
            f"y_cols (m={len(y_cols)}) と一致しません"
        )

        # ===== 4) 学習集合に追加 =====
        X_train = torch.cat([X_train, newX], dim=0)
        Y_train_raw = torch.cat([Y_train_raw, newY], dim=0)

        # ===== 5) ログ記録 =====
        y_now = newY[0]
        per_dim_dists, sum_dist = _compute_target_dists(y_now, spec.targets)

        rec: Dict = {
            "iter": it,
            "id": picked[id_col],
            "smiles": picked[smiles_col],
            "acq_value": float(vals[best_idx]),
            "is_multiobjective": bool(is_multi),
        }

        for j, col in enumerate(y_cols):
            rec[f"y[{j}]_{col}"] = float(y_now[j])

        if per_dim_dists is not None:
            for j, dj in enumerate(per_dim_dists):
                rec[f"target_absdiff[{j}]"] = float(dj)
            rec["target_absdiff_sum"] = float(sum_dist)

        # ===== 6) LOOCV 評価（任意） =====
        if eval_cfg is not None and getattr(eval_cfg, "loocv", False):
            min_pts = eval_cfg.get("min_points", 5)
            if X_train.shape[0] >= min_pts:
                loocv_res = brute_force_loocv_metrics(
                    X_train=X_train,
                    Y_train_raw=Y_train_raw,
                    y_names=y_cols,
                    model_cfg=model_cfg,
                )
                for name, m in loocv_res.items():
                    rec[f"loocv_{name}_rmse"] = m["rmse"]
                    rec[f"loocv_{name}_mae"] = m["mae"]
                    rec[f"loocv_{name}_r2"] = m["r2"]

        history.append(rec)

        # ===== 7) プールから削除 =====
        pool_df = pool_df.drop(pool_df.index[best_idx]).reset_index(drop=True)

        # ===== 8) 履歴保存（任意） =====
                # ===== 8) 各 iter ごとの履歴 Excel を保存 (オプション) =====
        if save_history_dir is not None:
            out_dir = Path(save_history_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            hist_path = out_dir / f"history_iter{it:03d}.xlsx"
            # history[:] で「ここまでの累積」を渡す
            save_history_excel(history, hist_path)

    return history
