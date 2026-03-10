from __future__ import annotations

from typing import Callable, Tuple

import torch
from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.models import ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning

from .objectives import (
    ObjectiveSpec,
    build_multiobjective,
    build_scalar_objective,
    build_scalar_objective_for_aq,
    compute_ref_point,
    to_object_space,
)

DEFAULT_NUM_MC = 512


def _build_standardized_ucb(
    model: ModelListGP,
    spec: ObjectiveSpec,
    num_mc_samples: int,
    ucb_beta: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    sUCB:
    1) 目的次元ごとに MC サンプルから UCB(mean + beta * std) を計算
    2) 候補集合内で目的ごとに標準化し、次元方向に合算
    """
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_mc_samples]))

    def _acqf(X: torch.Tensor) -> torch.Tensor:
        posterior = model.posterior(X)
        samples = sampler(posterior)  # (mc, batch, q, m)

        obj_samples = to_object_space(samples, spec)
        if obj_samples.shape[-2] == 1:
            obj_samples = obj_samples.squeeze(-2)  # (mc, batch, m)
        else:
            # 本プロジェクトでは q=1 を想定。q>1 の場合は平均で集約。
            obj_samples = obj_samples.mean(dim=-2)

        per_obj_mean = obj_samples.mean(dim=0)  # (batch, m)
        per_obj_std = obj_samples.std(dim=0, unbiased=False)
        per_obj_ucb = per_obj_mean + ucb_beta * per_obj_std

        if per_obj_ucb.ndim == 1:
            per_obj_ucb = per_obj_ucb.unsqueeze(-1)
        elif per_obj_ucb.ndim > 2:
            per_obj_ucb = per_obj_ucb.reshape(-1, per_obj_ucb.shape[-1])

        mean_over_inputs = per_obj_ucb.mean(dim=0, keepdim=True)
        std_over_inputs = per_obj_ucb.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-12)
        standardized = (per_obj_ucb - mean_over_inputs) / std_over_inputs

        return standardized.sum(dim=-1)  # (batch,)

    return _acqf


def build_acquisition(
    model: ModelListGP,
    Y_train_raw: torch.Tensor,
    spec: ObjectiveSpec,
    num_mc_samples: int = DEFAULT_NUM_MC,
    acq_type: str = "auto",
    ucb_beta: float = 2.0,
):
    """
    `acq_type` と目的次元に応じて獲得関数を構築し、
    `(acqf, is_multiobjective)` を返す。
    """
    acq_mode = acq_type.lower()
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_mc_samples]))
    m = spec.dim()

    if acq_mode in ("sucb", "s_ucb"):
        return _build_standardized_ucb(
            model=model,
            spec=spec,
            num_mc_samples=num_mc_samples,
            ucb_beta=ucb_beta,
        ), False

    if acq_mode in ("qucb", "q_ucb", "ucb"):
        objective = build_scalar_objective_for_aq(spec)
        acqf = qUpperConfidenceBound(
            model=model,
            beta=ucb_beta,
            sampler=sampler,
            objective=objective,
        )
        return acqf, False

    if acq_mode in ("auto", "qehvi") and m >= 2 and spec.kind != "linear_scalarization":
        objective = build_multiobjective(spec)
        Y_obj_train = to_object_space(Y_train_raw, spec)
        ref_point = compute_ref_point(Y_train_raw, spec)
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=Y_obj_train)

        acqf = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            sampler=sampler,
            objective=objective,
        )
        return acqf, True

    objective = build_scalar_objective(spec)
    y_scalar = objective(Y_train_raw).detach()
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
    acq_type: str = "auto",
    ucb_beta: float = 2.0,
) -> Tuple[int, torch.Tensor, bool]:
    """
    プール `X_pool` から 1 件選ぶ。
    """
    acqf, is_multi = build_acquisition(
        model=model,
        Y_train_raw=Y_train_raw,
        spec=spec,
        num_mc_samples=num_mc_samples,
        acq_type=acq_type,
        ucb_beta=ucb_beta,
    )
    with torch.no_grad():
        vals = acqf(X_pool.unsqueeze(1)).reshape(-1)
    best_idx = int(torch.argmax(vals).item())
    return best_idx, vals, is_multi
