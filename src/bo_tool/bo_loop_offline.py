from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

from bo_tool.acquisition import pick_next
from bo_tool.metrics import brute_force_loocv_metrics
from bo_tool.models import ModelConfig, build_models
from bo_tool.objectives import ObjectiveSpec


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
    model_cfg: ModelConfig,
    max_iters: int = 64,
    num_mc_samples: int = 512,
    acq_type: str = "auto",
    ucb_beta: float = 2.0,
    eval_cfg: Optional[Dict] = None,
    n_init: Optional[int] = None,
) -> List[Dict]:
    """
    既知空間での BO ループ（プールから選んで真値をそのまま観測）。
    戻り値は `history: list[dict]`。
    """
    history: List[Dict] = []

    for it in range(1, max_iters + 1):
        print(f"[Offline BO] Iteration {it}/{max_iters}")
        if len(pool_df) == 0:
            break

        model = build_models(X_train, Y_train_raw, model_cfg)

        X_pool = torch.tensor(pool_df[X_cols].to_numpy(), dtype=torch.double)
        best_idx, vals, is_multi = pick_next(
            model=model,
            X_pool=X_pool,
            Y_train_raw=Y_train_raw,
            spec=spec,
            num_mc_samples=num_mc_samples,
            acq_type=acq_type,
            ucb_beta=ucb_beta,
        )

        picked = pool_df.iloc[best_idx]
        newX = torch.tensor(picked[X_cols].to_numpy().reshape(1, -1), dtype=torch.double)
        newY = torch.tensor(picked[y_cols].to_numpy().reshape(1, -1), dtype=torch.double)

        X_train = torch.cat([X_train, newX], dim=0)
        Y_train_raw = torch.cat([Y_train_raw, newY], dim=0)

        y_now = newY[0]
        per_dim_dists, sum_dist = _compute_target_dists(y_now, spec.targets)

        rec: Dict[str, object] = {
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

        if eval_cfg is not None and getattr(eval_cfg, "loocv", False):
            print("  Performing LOOCV evaluation...")
            min_pts = getattr(eval_cfg, "min_points", 5)
            if X_train.shape[0] >= min_pts:
                loocv_res = brute_force_loocv_metrics(
                    X_train=X_train,
                    Y_train_raw=Y_train_raw,
                    y_names=y_cols,
                    model_cfg=model_cfg,
                )

                N = X_train.shape[0]
                for j, name in enumerate(y_cols):
                    m = loocv_res[name]
                    rec[f"loocv_{name}_rmse"] = m["rmse"]
                    rec[f"loocv_{name}_mae"] = m["mae"]
                    rec[f"loocv_{name}_r2"] = m["r2"]

                    if n_init is not None and "errors" in m and n_init < N:
                        errs = torch.tensor(m["errors"], dtype=torch.double, device=Y_train_raw.device)
                        explore_errs = errs[n_init:]
                        y_true_ex = Y_train_raw[n_init:, j]

                        rmse2 = float(torch.sqrt((explore_errs**2).mean()))
                        mae2 = float(explore_errs.abs().mean())

                        ss_res = float((explore_errs**2).sum())
                        ss_tot = float(((y_true_ex - y_true_ex.mean()) ** 2).sum())
                        r2_2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

                        rec[f"loocv_explore_{name}_rmse"] = rmse2
                        rec[f"loocv_explore_{name}_mae"] = mae2
                        rec[f"loocv_explore_{name}_r2"] = r2_2

            print("  LOOCV evaluation done.")

        history.append(rec)
        pool_df = pool_df.drop(pool_df.index[best_idx]).reset_index(drop=True)

    return history
