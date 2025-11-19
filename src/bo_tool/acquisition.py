# src/bo_tool/acquisition.py
from __future__ import annotations
from typing import Tuple

import torch
from botorch.models import ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement

from .objectives import (
    ObjectiveSpec,
    build_multiobjective,
    build_scalar_objective,
    to_object_space,
    compute_ref_point,
)

DEFAULT_NUM_MC = 512


def build_acquisition(
    model: ModelListGP,
    Y_train_raw: torch.Tensor,
    spec: ObjectiveSpec,
    num_mc_samples: int = DEFAULT_NUM_MC,
):
    """
    目的次元に応じて qEHVI (m>=2) / qEI (m==1) を構成して返す。
    戻り値: (acqf, is_multiobjective)
    """
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_mc_samples]))

    m = spec.dim()
    if m >= 2 and spec.kind != "linear_scalarization":
        # --- qEHVI (真の多目的最適化) ---
        objective = build_multiobjective(spec)
        Y_obj_train = to_object_space(Y_train_raw, spec)  # (N, m)
        ref_point = compute_ref_point(Y_train_raw, spec)  # (m,)
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=Y_obj_train)

        acqf = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            sampler=sampler,
            objective=objective,
        )
        return acqf, True

    else:
        # --- qEI（一目的）：scalar objective を使う ---
        objective = build_scalar_objective(spec)  # (…,) を返す
        # best_f は学習データを目的空間に写像してスカラー化したものの最大値
        y_scalar = objective(to_object_space(Y_train_raw, spec)).detach()  # (N,) が返るように
        best_f = torch.max(y_scalar)

        acqf = qExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
        )
        return acqf, False


def pick_next(
    model: ModelListGP,
    X_pool: torch.Tensor,
    Y_train_raw: torch.Tensor,
    spec: ObjectiveSpec,
    num_mc_samples: int = DEFAULT_NUM_MC,
) -> Tuple[int, torch.Tensor, bool]:
    """
    プールから 1 件選ぶ。qEHVI or qEI を自動選択。
    Returns:
      best_idx, vals, is_multiobjective
    """
    acqf, is_multi = build_acquisition(model, Y_train_raw, spec, num_mc_samples)
    with torch.no_grad():
        vals = acqf(X_pool.unsqueeze(1)).reshape(-1)
    best_idx = int(torch.argmax(vals).item())
    return best_idx, vals, is_multi
