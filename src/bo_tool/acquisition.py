# src/bo_tool/acquisition.py
from __future__ import annotations
from typing import Tuple

import torch
from botorch.models import ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound

from .objectives import (
    ObjectiveSpec,
    build_multiobjective,
    build_scalar_objective,
    build_scalar_objective_for_aq,
    to_object_space,
    compute_ref_point,
)

DEFAULT_NUM_MC = 512


def build_acquisition(
    model: ModelListGP,
    Y_train_raw: torch.Tensor,
    spec: ObjectiveSpec,
    num_mc_samples: int = DEFAULT_NUM_MC,
    acq_type: str = "auto",      # "auto", "qei", "qehvi", "qucb"
    ucb_beta: float = 2.0,       # qUCB 用 β
):
    """
    目的次元と acq_type に応じて
      - "qucb"       : スカラー目的に対する qUCB
      - "auto"/"qehvi": m>=2 & 非線形 → qEHVI
      - それ以外     : qEI（スカラー目的）
    を構成して返す。
    戻り値: (acqf, is_multiobjective)
    """
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_mc_samples]))

    m = spec.dim()

    # -----------------
    # 1) qUCB モード
    # -----------------
    if acq_type.lower() in ("qucb", "q_ucb", "ucb"):
        # ObjectiveSpec から「スカラー目的 f(x)」を作る
        # build_scalar_objective 内で to_object_space を呼ぶので、
        # ここでは「生の Y_raw」を渡すだけでよい。
        objective = build_scalar_objective_for_aq(spec)  # Y_raw -> scalar

        acqf = qUpperConfidenceBound(
            model=model,
            beta=ucb_beta,
            sampler=sampler,
            objective=objective,
        )
        # HV ベースではないので is_multiobjective=False
        return acqf, False
    
    # if acq_type.lower() not in ("pqucb"):


    # -----------------
    # 2) qEHVI（真の多目的）
    # -----------------
    if acq_type.lower() in ("auto", "qehvi") and m >= 2 and spec.kind != "linear_scalarization":
        objective = build_multiobjective(spec)              # Y_raw -> Y_obj(m次元)
        Y_obj_train = to_object_space(Y_train_raw, spec)    # (N, m)
        ref_point = compute_ref_point(Y_train_raw, spec)    # (m,)
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=Y_obj_train)

        acqf = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            sampler=sampler,
            objective=objective,
        )
        return acqf, True

    # -----------------
    # 3) qEI（一目的）
    # -----------------
    # スカラー目的：build_scalar_objective が内部で to_object_space を呼ぶ
    objective = build_scalar_objective(spec)  # Y_raw -> scalar

    # best_f は「観測済み Y をスカラー目的に通したものの最大値」
    y_scalar = objective(Y_train_raw).detach()  # (N,)
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
    acq_type: str = "auto",      # "auto", "qei", "qehvi", "qucb"
    ucb_beta: float = 2.0,
) -> Tuple[int, torch.Tensor, bool]:
    """
    プールから 1 件選ぶ。

      acq_type:
        - "auto"        : m>=2 & non-linear_scalarization → qEHVI, else qEI
        - "qehvi"       : 強制 qEHVI（条件満たさない場合は qEI fallback）
        - "qei"         : 強制 qEI（scalar）
        - "qucb"/"q_ucb": スカラー目的に対する qUCB

    Returns:
      best_idx, vals, is_multiobjective
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
