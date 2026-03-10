from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import torch
from botorch.acquisition.multi_objective.objective import GenericMCMultiOutputObjective
from botorch.acquisition.objective import GenericMCObjective


PerDimMode = Literal["target", "identity"]
ObjectiveKind = Literal["target_distance", "identity_multi", "linear_scalarization", "mixed_multi"]


@dataclass
class ObjectiveSpec:
    """
    生の出力 `Y_raw` を BO で最大化する目的空間に変換する仕様。
    """

    kind: ObjectiveKind
    weights: Sequence[float]
    targets: Optional[Sequence[float]] = None
    power: float = 2.0
    maximize: Optional[Sequence[bool]] = None
    modes: Optional[Sequence[PerDimMode]] = None

    def dim(self) -> int:
        return len(self.weights)

    def as_tensor(self, xs: Sequence[float] | None, like: torch.Tensor) -> Optional[torch.Tensor]:
        if xs is None:
            return None
        return torch.tensor(xs, dtype=like.dtype, device=like.device)


def to_object_space(Y_raw: torch.Tensor, spec: ObjectiveSpec) -> torch.Tensor:
    """
    生の `Y_raw (..., m)` を最大化用の目的空間に変換する。
    """
    m = Y_raw.shape[-1]
    assert m == spec.dim(), f"spec.weights (m={spec.dim()}) と Y_raw(m={m}) の次元不一致"

    if spec.kind == "target_distance":
        targets = spec.as_tensor(spec.targets, like=Y_raw)
        assert targets is not None and targets.shape[-1] == m
        diff = (Y_raw - targets).abs().pow(spec.power)
        return -diff

    if spec.kind == "identity_multi":
        maximize = spec.maximize or [True] * m
        sign = torch.tensor([1.0 if b else -1.0 for b in maximize], dtype=Y_raw.dtype, device=Y_raw.device)
        return Y_raw * sign

    if spec.kind == "linear_scalarization":
        w = spec.as_tensor(spec.weights, like=Y_raw)
        assert w is not None
        return (Y_raw * w).sum(dim=-1, keepdim=True)

    if spec.kind == "mixed_multi":
        assert spec.modes is not None, "mixed_multi では modes が必須です"
        assert len(spec.modes) == m, "modes の長さが出力次元 m と一致していません"

        maximize = spec.maximize or [True] * m
        assert len(maximize) == m, "maximize の長さが m と一致していません"
        targets = spec.as_tensor(spec.targets, like=Y_raw) if spec.targets is not None else None

        outs = []
        for j, mode_j in enumerate(spec.modes):
            yj = Y_raw[..., j]
            if mode_j == "identity":
                sign_j = 1.0 if maximize[j] else -1.0
                outs.append(sign_j * yj)
            elif mode_j == "target":
                assert targets is not None, "target モードには targets が必要です"
                diff = (yj - targets[..., j]).abs().pow(spec.power)
                outs.append(-diff)
            else:
                raise ValueError(f"Unknown per-dim mode: {mode_j}")
        return torch.stack(outs, dim=-1)

    raise ValueError(f"Unknown ObjectiveSpec.kind={spec.kind}")


def build_multiobjective(spec: ObjectiveSpec) -> GenericMCMultiOutputObjective:
    def _transform(samples: torch.Tensor, X: torch.Tensor | None = None) -> torch.Tensor:
        return to_object_space(samples, spec)

    return GenericMCMultiOutputObjective(_transform)


def build_scalar_objective(spec: ObjectiveSpec) -> GenericMCObjective:
    def _transform(samples: torch.Tensor, X: torch.Tensor | None = None) -> torch.Tensor:
        obj = to_object_space(samples, spec)
        if obj.shape[-1] == 1:
            return obj.squeeze(-1)
        return obj.sum(dim=-1)

    return GenericMCObjective(_transform)


def build_scalar_objective_for_aq(spec: ObjectiveSpec) -> GenericMCObjective:
    weights = list(spec.weights)

    def _transform(samples: torch.Tensor, X: torch.Tensor | None = None) -> torch.Tensor:
        obj = to_object_space(samples, spec)

        reduce_dims = tuple(range(obj.ndim - 1))
        mean = obj.mean(dim=reduce_dims, keepdim=True)
        std = obj.std(dim=reduce_dims, keepdim=True, unbiased=False).clamp_min(1e-12)
        obj = (obj - mean) / std

        w = torch.as_tensor(weights, dtype=obj.dtype, device=obj.device)
        if obj.shape[-1] == 1:
            return obj.squeeze(-1) * w[0]
        return (obj * w).sum(dim=-1)

    return GenericMCObjective(_transform)


def compute_ref_point(
    Y_train_raw: torch.Tensor,
    spec: ObjectiveSpec,
    eps: float | Sequence[float] = 0.1,
) -> torch.Tensor:
    """
    qEHVI の参照点（目的空間での `min - eps`）を返す。
    """
    Y_obj = to_object_space(Y_train_raw, spec)
    assert Y_obj.ndim == 2 and Y_obj.shape[-1] == spec.dim(), "qEHVI用は m次元の目的が必要"
    if isinstance(eps, float):
        eps = [eps] * spec.dim()
    eps_t = torch.tensor(eps, dtype=Y_obj.dtype, device=Y_obj.device)
    return Y_obj.min(dim=0).values - eps_t
