# src/BO_torch/optimization/bo_loop_offline.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import torch
import pandas as pd

from bo_tool.models import build_models
from bo_tool.acquisition import pick_next
from bo_tool.objectives import ObjectiveSpec
from bo_tool.metrics import brute_force_loocv_metrics




@torch.no_grad()
def _compute_target_dists(
    y_vec: torch.Tensor,
    targets: Optional[List[float]],
) -> Tuple[Optional[List[float]], Optional[float]]:
    if targets is None:
        return None, None
    t = torch.tensor(targets, dtype=y_vec.dtype, device=y_vec.device)
    per_dim = (y_vec - t).abs().tolist()
    total = float(sum(per_dim))
    return per_dim, total


def offline_bo_loop(
    X_cols: List[str],
    y_cols: List[str],
    id_col: str,
    pool_df: pd.DataFrame,
    X_train: torch.Tensor,
    Y_train_raw: torch.Tensor,
    spec: ObjectiveSpec,
    model_cfg,                # ← ★ 追加：model 設定
    max_iters: int = 64,
    num_mc_samples: int = 512,
    eval_cfg: Optional[Dict] = None,
) -> List[Dict]:
    """
    既知空間での BO ループ（プールから選んで真値をそのまま観測）。
    戻り値: history の list[dict]
    """
    history: List[Dict] = []

    for it in range(1, max_iters + 1):
        if len(pool_df) == 0:
            break

        # ===== 1) モデル学習 =====
        model = build_models(X_train, Y_train_raw, model_cfg)

        # ===== 2) 候補選択 =====
        X_pool = torch.tensor(pool_df[X_cols].to_numpy(), dtype=torch.double)
        best_idx, vals, is_multi = pick_next(
            model=model,
            X_pool=X_pool,
            Y_train_raw=Y_train_raw,
            spec=spec,
            num_mc_samples=num_mc_samples,
        )

         # ===== 3) 真値観測 =====
        picked = pool_df.iloc[best_idx]
        newX = torch.tensor(picked[X_cols].to_numpy().reshape(1, -1), dtype=torch.double)
        newY = torch.tensor(picked[y_cols].to_numpy().reshape(1, -1), dtype=torch.double)

        # ===== 4) 学習集合に追加 =====
        X_train = torch.cat([X_train, newX], dim=0)
        Y_train_raw = torch.cat([Y_train_raw, newY], dim=0)

        # ===== 5) ログ記録 =====
        y_now = newY[0]
        per_dim_dists, sum_dist = _compute_target_dists(y_now, spec.targets)

        rec: Dict = {
            "iter": it,
            "id": picked[id_col],
            "acq_value": float(vals[best_idx]),
            "is_multiobjective": bool(is_multi),
        }

        for j, col in enumerate(y_cols):
            rec[f"y[{j}]_{col}"] = float(y_now[j])

        if per_dim_dists is not None:
            for j, dj in enumerate(per_dim_dists):
                rec[f"target_absdiff[{j}]"] = float(dj)
            rec["target_absdiff_sum"] = float(sum_dist)

        # ===== 6) LOOCV 評価 =====
        if eval_cfg is not None and eval_cfg.get("loocv", False):
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

    return history
