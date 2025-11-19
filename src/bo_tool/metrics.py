# src/BO_torch/evaluation/metrics.py
from __future__ import annotations
from typing import List, Optional, Dict

import torch
import pandas as pd
from botorch.utils.multi_objective.hypervolume import Hypervolume

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
