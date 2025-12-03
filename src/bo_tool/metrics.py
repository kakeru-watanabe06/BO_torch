# src/BO_torch/evaluation/metrics.py
from __future__ import annotations
from typing import List, Optional, Dict

import torch
import pandas as pd
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.models import SingleTaskGP, ModelListGP
from bo_tool.models import build_models  # まだ import してなければ追加
from botorch.models import SingleTaskGP, ModelListGP
from bo_tool.objectives import (
    ObjectiveSpec,
    to_object_space,
    build_scalar_objective,
)


def fixed_ref_point(all_Y_raw: torch.Tensor, spec: ObjectiveSpec, eps: float | List[float] = 0.1) -> torch.Tensor:
    """
    解析全体で使い回す固定 ref_point を作る。
    all_Y_raw（＝データセット全体など）を目的空間に写像して min - eps とする。
    """
    Y_all_obj = to_object_space(all_Y_raw, spec)  # (N, m) or (N, 1)
    assert Y_all_obj.ndim == 2 and Y_all_obj.shape[-1] == spec.dim(), \
        "HVはm次元の目的が必要（linear_scalarizationなどの1次元は対象外）"

    if isinstance(eps, float):
        eps = [eps] * spec.dim()
    eps_t = torch.tensor(eps, dtype=Y_all_obj.dtype, device=Y_all_obj.device)
    return Y_all_obj.min(dim=0).values - eps_t


def hypervolume_of(Y_raw: torch.Tensor, spec: ObjectiveSpec, ref_point: torch.Tensor) -> float:
    """
    与えた観測集合（Y_raw: (k, m)）の非支配集合に対する HV を返す。
    """
    Y_obj = to_object_space(Y_raw, spec)  # (k, m)
    hv = Hypervolume(ref_point=ref_point)
    return float(hv.compute(Y_obj))


def hypervolume_curve(
    initial_Y: torch.Tensor,      # (n0, m)
    appended_Ys: List[torch.Tensor],  # 各反復で新規観測した y（(m,) を想定）
    spec: ObjectiveSpec,
    ref_point: torch.Tensor,
) -> List[float]:
    """
    反復ごとの HV を返す。各ステップで観測集合を累積して HV を計算。
    """
    Ys = [initial_Y.clone()]
    hv_values: List[float] = []
    for y in appended_Ys:
        Ys.append(torch.cat([Ys[-1], y.view(1, -1)], dim=0))
        hv_values.append(hypervolume_of(Ys[-1], spec, ref_point))
    return hv_values


def hypervolume_gap_curve(
    initial_Y: torch.Tensor,       # (n0, m)
    appended_Ys: List[torch.Tensor],
    all_Y_raw: torch.Tensor,       # 既知空間の “全観測”（ベンチマーク用にHVの上限を作る）
    spec: ObjectiveSpec,
    ref_point: Optional[torch.Tensor] = None,
) -> List[float]:
    """
    全データの非支配集合が持つ HV（上限）からのギャップ（上限 − 現在HV）を出す。
    ベンチマーク比較に使いやすい。
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
        # build_scalar_objective の中で to_object_space するので、
        # ここでは「生の Y_raw」をそのまま渡す
        return scalar_obj(Y)

    cur = _scalarize(initial_Y)  # (n0,)
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
    offline_bo_loop の history（list[dict]）から、各反復で追加された y を (m,) Tensor にして返す。
    """
    ys = []
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
    曲線群を 1 本の DataFrame（iter 対 column）にまとめるユーティリティ。
    """
    iters = None
    data = {}

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

    df = pd.DataFrame({"iter": iters, **data})
    return df

# ===== ここから LOOCV 用ユーティリティ =====

def _regression_metrics_1d(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> Dict[str, float]:
    """
    1 次元系列に対して RMSE / MAE / R² を計算。
    （すでにあればそれを使ってOK）
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    residual = y_pred - y_true
    rmse = torch.sqrt(torch.mean(residual ** 2))
    mae = torch.mean(torch.abs(residual))

    ss_res = torch.sum(residual ** 2)
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
) -> Dict[str, Dict[str, float]]:
    """
    完全愚直版 LOOCV：
      各 i について
        - i を抜いたデータで build_models(...) して GP をフィット
        - x_i を予測
      を N 回まわし、
      各出力次元ごとに RMSE / MAE / R² を返す。

    戻り値:
      {
        "S1_energy_eV_scaled": {"rmse": ..., "mae": ..., "r2": ...},
        "Oscillator_strength_scaled": {...},
        ...
      }
    """
    device = X_train.device
    dtype = X_train.dtype

    N = X_train.shape[0]
    Y_train_raw = Y_train_raw.to(device=device, dtype=dtype)

    # 予測値を貯めるバッファ (N, M)
    if Y_train_raw.ndim == 1:
        Y_train_raw = Y_train_raw.view(N, 1)
    N, M = Y_train_raw.shape
    assert M == len(y_names), "y_names の長さと Y_train_raw の列数が一致していません"

    Y_pred = torch.empty_like(Y_train_raw)

    # 各 i について「i を抜いて再フィット → x_i を予測」
    for i in range(N):
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[i] = False

        X_i = X_train[mask]        # (N-1, d)
        Y_i = Y_train_raw[mask]    # (N-1, M)

        # build_models は X, Y, model_cfg から SingleTaskGP or ModelListGP を返す想定
        model_i = build_models(X_i, Y_i, model_cfg)
        model_i.eval()

        x_test = X_train[i : i + 1]  # (1, d)

        if isinstance(model_i, SingleTaskGP):
            # 多出力 SingleTaskGP の場合は mean の shape が (1, M) のイメージ
            with torch.no_grad():
                post = model_i.posterior(x_test)
                mean = post.mean.view(1, -1)  # (1, M)
            Y_pred[i] = mean[0]

        elif isinstance(model_i, ModelListGP):
            # 各 output ごとに 1 つの SingleTaskGP
            assert len(model_i.models) == M, "ModelListGP のモデル数と Y の次元が一致していません"
            preds = []
            with torch.no_grad():
                for sub_model in model_i.models:
                    assert isinstance(sub_model, SingleTaskGP)
                    post = sub_model.posterior(x_test)
                    # (1,1) を想定
                    preds.append(post.mean.view(-1)[0])
            Y_pred[i] = torch.stack(preds)

        else:
            raise TypeError(f"Unsupported model type in brute_force_loocv_metrics: {type(model_i)}")

    # 各次元ごとに指標を計算
    results: Dict[str, Dict[str, float]] = {}
    for j, name in enumerate(y_names):
        y_true_j = Y_train_raw[:, j]
        y_pred_j = Y_pred[:, j]

        # 既存のメトリクス計算を使う
        metrics = _regression_metrics_1d(y_true_j, y_pred_j)

        # ★ ここを追加：各サンプルの誤差も持たせる
        residual = (y_pred_j - y_true_j).detach().cpu()
        metrics["errors"] = residual.tolist()  # 長さ N

        results[name] = metrics

    return results