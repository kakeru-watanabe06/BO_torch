from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.utils.multi_objective.hypervolume import Hypervolume

from bo_tool.models import build_models
from bo_tool.objectives import ObjectiveSpec, build_scalar_objective, to_object_space


def fixed_ref_point(
    all_Y_raw: torch.Tensor,
    spec: ObjectiveSpec,
    eps: float | List[float] = 0.1,
) -> torch.Tensor:
    """
    解析全体で使い回す固定参照点 `ref_point` を作る。
    """
    Y_all_obj = to_object_space(all_Y_raw, spec)
    assert Y_all_obj.ndim == 2 and Y_all_obj.shape[-1] == spec.dim(), (
        "HVはm次元の目的が必要（linear_scalarizationなどの1次元は対象外）"
    )

    if isinstance(eps, float):
        eps = [eps] * spec.dim()
    eps_t = torch.tensor(eps, dtype=Y_all_obj.dtype, device=Y_all_obj.device)
    return Y_all_obj.min(dim=0).values - eps_t


def hypervolume_of(Y_raw: torch.Tensor, spec: ObjectiveSpec, ref_point: torch.Tensor) -> float:
    """
    与えた観測集合（Y_raw: (k, m)）の非支配集合に対する HV を返す。
    """
    Y_obj = to_object_space(Y_raw, spec)
    hv = Hypervolume(ref_point=ref_point)
    return float(hv.compute(Y_obj))


def hypervolume_curve(
    initial_Y: torch.Tensor,
    appended_Ys: List[torch.Tensor],
    spec: ObjectiveSpec,
    ref_point: torch.Tensor,
) -> List[float]:
    """
    反復ごとの HV を返す。
    """
    Ys = [initial_Y.clone()]
    hv_values: List[float] = []
    for y in appended_Ys:
        Ys.append(torch.cat([Ys[-1], y.view(1, -1)], dim=0))
        hv_values.append(hypervolume_of(Ys[-1], spec, ref_point))
    return hv_values


def hypervolume_gap_curve(
    initial_Y: torch.Tensor,
    appended_Ys: List[torch.Tensor],
    all_Y_raw: torch.Tensor,
    spec: ObjectiveSpec,
    ref_point: Optional[torch.Tensor] = None,
) -> List[float]:
    """
    全データに対する HV 上限からのギャップ（上限 - 現在HV）を返す。
    """
    if ref_point is None:
        ref_point = fixed_ref_point(all_Y_raw, spec)

    hv_star = hypervolume_of(all_Y_raw, spec, ref_point)
    hv_seq = hypervolume_curve(initial_Y, appended_Ys, spec, ref_point)
    return [hv_star - hv for hv in hv_seq]


def scalar_best_curve(
    initial_Y: torch.Tensor,
    appended_Ys: List[torch.Tensor],
    spec: ObjectiveSpec,
) -> List[float]:
    scalar_obj = build_scalar_objective(spec)

    def _scalarize(Y: torch.Tensor) -> torch.Tensor:
        return scalar_obj(Y)

    cur = _scalarize(initial_Y)
    best = float(torch.max(cur))

    seq: List[float] = []
    for y in appended_Ys:
        val = float(_scalarize(y.view(1, -1))[0])
        if val > best:
            best = val
        seq.append(best)
    return seq


def history_to_appended_Ys(history: List[Dict], y_cols: List[str]) -> List[torch.Tensor]:
    """
    history（list[dict]）から各反復で追加された y を (m,) Tensor として返す。
    """
    ys: List[torch.Tensor] = []
    for rec in history:
        row = [rec[f"y[{j}]_{col}"] for j, col in enumerate(y_cols)]
        ys.append(torch.tensor(row, dtype=torch.double))
    return ys


def make_metrics_dataframe(
    hv_curve_vals: Optional[List[float]] = None,
    hv_gap_vals: Optional[List[float]] = None,
    scalar_best_vals: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    指標曲線を iter 軸の DataFrame にまとめる。
    """
    iters = None
    data: Dict[str, List[float]] = {}

    if hv_curve_vals is not None:
        iters = list(range(1, len(hv_curve_vals) + 1))
        data["HV"] = hv_curve_vals
    if hv_gap_vals is not None:
        if iters is None:
            iters = list(range(1, len(hv_gap_vals) + 1))
        data["HV_gap"] = hv_gap_vals
    if scalar_best_vals is not None:
        if iters is None:
            iters = list(range(1, len(scalar_best_vals) + 1))
        data["ScalarBest"] = scalar_best_vals

    if iters is None:
        return pd.DataFrame()

    return pd.DataFrame({"iter": iters, **data})


# ===== LOOCV =====

def _regression_metrics_1d(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    """
    1次元系列に対する RMSE / MAE / R²。
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    residual = y_pred - y_true
    rmse = torch.sqrt(torch.mean(residual**2))
    mae = torch.mean(torch.abs(residual))

    ss_res = torch.sum(residual**2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def brute_force_loocv_metrics(
    X_train: torch.Tensor,
    Y_train_raw: torch.Tensor,
    y_names: List[str],
    model_cfg,
) -> Dict[str, Dict[str, object]]:
    """
    完全愚直版 LOOCV:
    各点を1つずつ抜いて再学習し、その点の予測誤差を集計する。
    """
    device = X_train.device
    dtype = X_train.dtype

    N = X_train.shape[0]
    Y_train_raw = Y_train_raw.to(device=device, dtype=dtype)

    if Y_train_raw.ndim == 1:
        Y_train_raw = Y_train_raw.view(N, 1)
    N, M = Y_train_raw.shape
    assert M == len(y_names), "y_names の長さと Y_train_raw の列数が一致していません"

    Y_pred = torch.empty_like(Y_train_raw)

    for i in range(N):
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[i] = False

        X_i = X_train[mask]
        Y_i = Y_train_raw[mask]

        model_i = build_models(X_i, Y_i, model_cfg)
        model_i.eval()

        x_test = X_train[i : i + 1]

        if isinstance(model_i, SingleTaskGP):
            with torch.no_grad():
                post = model_i.posterior(x_test)
                mean = post.mean.view(1, -1)
            Y_pred[i] = mean[0]

        elif isinstance(model_i, ModelListGP):
            assert len(model_i.models) == M, "ModelListGP のモデル数と Y の次元が一致していません"
            preds = []
            with torch.no_grad():
                for sub_model in model_i.models:
                    assert isinstance(sub_model, SingleTaskGP)
                    post = sub_model.posterior(x_test)
                    preds.append(post.mean.view(-1)[0])
            Y_pred[i] = torch.stack(preds)

        else:
            raise TypeError(f"Unsupported model type in brute_force_loocv_metrics: {type(model_i)}")

    results: Dict[str, Dict[str, object]] = {}
    for j, name in enumerate(y_names):
        y_true_j = Y_train_raw[:, j]
        y_pred_j = Y_pred[:, j]

        metrics = _regression_metrics_1d(y_true_j, y_pred_j)
        residual = (y_pred_j - y_true_j).detach().cpu()
        metrics["errors"] = residual.tolist()
        results[name] = metrics

    return results
