from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch

from bo_tool.acquisition import pick_next
from bo_tool.io_utils import save_history_excel
from bo_tool.metrics import brute_force_loocv_metrics
from bo_tool.models import ModelConfig, build_models
from bo_tool.objectives import ObjectiveSpec


@torch.no_grad()
def _compute_target_dists(
    y_vec: torch.Tensor,
    targets: Optional[List[float]],
) -> Tuple[Optional[List[float]], Optional[float]]:
    """
    target_distance / mixed_multi 用:
    1点分の y_vec と targets の絶対差 |y - t| を次元ごと / 合計で返す。
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
    model_cfg: ModelConfig,
    observe_func: Callable[[pd.Series], torch.Tensor],
    max_iters: int = 64,
    num_mc_samples: int = 512,
    acq_type: str = "auto",
    ucb_beta: float = 2.0,
    eval_cfg: Optional[Dict] = None,
    save_history_dir: Optional[str] = None,
) -> List[Dict]:
    """
    オンライン BO ループ（真値は observe_func で外部計算して取得）。
    """
    history: List[Dict] = []

    device = X_train.device
    X_train = X_train.to(device=device, dtype=torch.double)
    Y_train_raw = Y_train_raw.to(device=device, dtype=torch.double)

    for it in range(1, max_iters + 1):
        print(f"[Online BO] Iteration {it}/{max_iters}")
        if len(pool_df) == 0:
            break

        model = build_models(X_train, Y_train_raw, model_cfg)

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

        picked = pool_df.iloc[best_idx]

        newX_vals = pd.to_numeric(picked[X_cols], errors="coerce").to_numpy(dtype="float64").reshape(1, -1)
        newX = torch.tensor(newX_vals, dtype=torch.double, device=device)

        newY = observe_func(picked)
        if newY.ndim == 1:
            newY = newY.view(1, -1)
        newY = newY.to(dtype=torch.double, device=device)
        assert newY.shape[1] == len(y_cols), (
            f"observe_func が返した newY の列数 {newY.shape[1]} が "
            f"y_cols (m={len(y_cols)}) と一致しません"
        )

        X_train = torch.cat([X_train, newX], dim=0)
        Y_train_raw = torch.cat([Y_train_raw, newY], dim=0)

        y_now = newY[0]
        per_dim_dists, sum_dist = _compute_target_dists(y_now, spec.targets)

        rec: Dict[str, object] = {
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

        if eval_cfg is not None and getattr(eval_cfg, "loocv", False):
            min_pts = getattr(eval_cfg, "min_points", 5)
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
        pool_df = pool_df.drop(pool_df.index[best_idx]).reset_index(drop=True)

        if save_history_dir is not None:
            out_dir = Path(save_history_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            hist_path = out_dir / f"history_iter{it:03d}.xlsx"
            save_history_excel(history, hist_path)

    return history
