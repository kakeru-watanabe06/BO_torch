# src/BO_torch/objectives.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Literal

import torch
from botorch.acquisition.multi_objective.objective import GenericMCMultiOutputObjective
from botorch.acquisition.objective import GenericMCObjective


PerDimMode = Literal["target", "identity"]
ObjectiveKind = Literal["target_distance", "identity_multi", "linear_scalarization","mixed_multi"]


@dataclass
class ObjectiveSpec:
    """
    多目的の目的空間への変換仕様。
    - target_distance: 各出力 y_i を target_i からの -w_i * |y_i - target_i|^p に変換（= 目標に近いほど大きい）
    - identity_multi : そのまま（または maximize/minimize で符号反転）で多目的化（E を最大化したいなら +、最小化なら -）
    - linear_scalarization: w^T y （符号込み）で一目的に畳み込み（qEI用）
    """
    kind: ObjectiveKind
    weights: Sequence[float]                 # 長さ m
    targets: Optional[Sequence[float]] = None  # target_distance のとき必須（長さ m）
    power: float = 2.0                         # target_distance の距離の冪
    maximize: Optional[Sequence[bool]] = None  # identity_multi用（True=最大化, False=最小化）
    modes: Optional[Sequence[PerDimMode]] = None  # mixed_multi用（長さ m）

    def dim(self) -> int:
        return len(self.weights)

    def as_tensor(
        self, xs: Sequence[float] | None, like: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if xs is None:
            return None
        return torch.tensor(xs, dtype=like.dtype, device=like.device)


def to_object_space(Y_raw: torch.Tensor, spec: ObjectiveSpec) -> torch.Tensor:
    """
    生の Y_raw (…, m) を「最大化する目的空間」Y_obj(…, m or 1) に変換。
    """
    m = Y_raw.shape[-1]
    assert m == spec.dim(), f"spec.weights (m={spec.dim()}) と Y_raw(m={m}) の次元不一致"

    if spec.kind == "target_distance":
        targets = spec.as_tensor(spec.targets, like=Y_raw)
        assert targets is not None and targets.shape[-1] == m
        w = spec.as_tensor(spec.weights, like=Y_raw)
        # 目的：-w_i * |y_i - t_i|^p（最大化）
        diff = (Y_raw - targets).abs().pow(spec.power)
        # return -(w * diff)
        return -diff 

    elif spec.kind == "identity_multi":
        # maximize=True ならそのまま、False なら符号反転（最小化→最大化へ）
        maximize = spec.maximize or [True] * m
        sign = torch.tensor([1.0 if b else -1.0 for b in maximize], dtype=Y_raw.dtype, device=Y_raw.device)
        return Y_raw * sign  # (…, m)

    elif spec.kind == "linear_scalarization":
        # w^T y を最大化（長さ1へ）
        w = spec.as_tensor(spec.weights, like=Y_raw)  # (m,)
        val = (Y_raw * w).sum(dim=-1, keepdim=True)   # (…, 1)
        return val
    
    elif spec.kind == "mixed_multi":
                # 次元ごとに mode を切り替える
        assert spec.modes is not None, "mixed_multi では modes が必須です"
        assert len(spec.modes) == m, "modes の長さが出力次元 m と一致していません"

        modes = list(spec.modes)
        maximize = spec.maximize or [True] * m
        assert len(maximize) == m, "maximize の長さが m と一致していません"

        w = spec.as_tensor(spec.weights, like=Y_raw)  # (m,)
        targets = spec.as_tensor(spec.targets, like=Y_raw) if spec.targets is not None else None

        outs = []
        for j in range(m):
            yj = Y_raw[..., j]
            mode_j = modes[j]

            if mode_j == "identity":
                # そのまま or 符号反転（スケーリングしたければ w[j] かけてもOK）
                sign_j = 1.0 if maximize[j] else -1.0
                outs.append(sign_j * yj)

            elif mode_j == "target":
                assert targets is not None, "target モードには targets が必要です"
                diff = (yj - targets[..., j]).abs().pow(spec.power)
                # outs.append(-w[j] * diff)
                outs.append(-diff)

            else:
                raise ValueError(f"Unknown per-dim mode: {mode_j}")

        # (…, m)
        return torch.stack(outs, dim=-1)

    else:
        raise ValueError(f"Unknown ObjectiveSpec.kind={spec.kind}")


def build_multiobjective(spec: ObjectiveSpec) -> GenericMCMultiOutputObjective:
    """
    qEHVI 用：m次元の目的ベクトルを返す Objective。
    """
    def _transform(samples: torch.Tensor, X: torch.Tensor | None = None) -> torch.Tensor:
        # samples: (…, m)
        return to_object_space(samples, spec)
    return GenericMCMultiOutputObjective(_transform)


def build_scalar_objective(spec: ObjectiveSpec) -> GenericMCObjective:
    def _transform(samples, X=None):
        obj = to_object_space(samples, spec)
        if obj.shape[-1] == 1:
            return obj.squeeze(-1)
        return obj.sum(dim=-1)
    return GenericMCObjective(_transform)

def build_scalar_objective_for_aq(spec) -> GenericMCObjective:
    weights_list = spec.weights  # 例: [1.0, 0.2, 3.0] みたいなPython list

    def _transform(samples, X=None):
        obj = to_object_space(samples, spec)  # (..., m)

        # weights: obj と同じ dtype/device にそろえる
        w = torch.as_tensor(weights_list, dtype=obj.dtype, device=obj.device)

        if obj.shape[-1] == 1:
            # weights が [w1] でも [w1, ...] でも先頭だけ使う
            return obj.squeeze(-1) * w[0]

        # 形を合わせる: (..., m) * (m,) はブロードキャストOK
        return (obj * w).sum(dim=-1)
    return GenericMCObjective(_transform)



def compute_ref_point(Y_train_raw: torch.Tensor, spec: ObjectiveSpec, eps: float | Sequence[float] = 0.1) -> torch.Tensor:
    """
    qEHVI の参照点（最小値 - eps）を m次元で作る。
    """
    Y_obj = to_object_space(Y_train_raw, spec)   # (N, m)
    assert Y_obj.ndim == 2 and Y_obj.shape[-1] == spec.dim(), "qEHVI用は m次元の目的が必要"
    if isinstance(eps, float):
        eps = [eps] * spec.dim()
    eps_t = torch.tensor(eps, dtype=Y_obj.dtype, device=Y_obj.device)
    return Y_obj.min(dim=0).values - eps_t
