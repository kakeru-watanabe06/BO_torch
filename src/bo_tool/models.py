from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Protocol

from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from bo_tool.kernels import TanimotoKernel


class _ModelConfigLike(Protocol):
    kernel: str
    ard: bool


def _normalize_kernel_name(kernel_name: str) -> str:
    return kernel_name.lower().replace("-", "_")


def _build_matern(nu: float, ard_num_dims: int | None) -> ScaleKernel:
    return ScaleKernel(MaternKernel(nu=nu, ard_num_dims=ard_num_dims))


def _resolve_matern_nu(kernel_name: str) -> float | None:
    aliases = {
        "matern12": 0.5,
        "matern_12": 0.5,
        "matern_1_2": 0.5,
        "matern_nu_0.5": 0.5,
        "matern32": 1.5,
        "matern_32": 1.5,
        "matern_3_2": 1.5,
        "matern_nu_1.5": 1.5,
        "matern52": 2.5,
        "matern_52": 2.5,
        "matern_5_2": 2.5,
        "matern_nu_2.5": 2.5,
    }
    return aliases.get(_normalize_kernel_name(kernel_name))


@dataclass
class ModelConfig:
    kernel: str = "matern32"
    ard: bool = False


def make_covar_module(input_dim: int, cfg: _ModelConfigLike):
    """ModelConfig に応じて GPyTorch のカーネルを構成する。"""
    kernel_name = _normalize_kernel_name(cfg.kernel)
    ard_num_dims = input_dim if (cfg.ard and kernel_name != "tanimoto") else None

    nu = _resolve_matern_nu(kernel_name)
    if nu is not None:
        return _build_matern(nu, ard_num_dims)

    if kernel_name == "tanimoto":
        if cfg.ard:
            raise ValueError("Tanimoto kernel does not support ARD. Set ard=false in config.")
        return ScaleKernel(TanimotoKernel())

    raise ValueError(f"Unknown kernel type: {cfg.kernel}")


def create_kernel(kernel_name: str, ard: bool, input_dim: int):
    ard_num_dims = input_dim if ard else None
    norm_name = _normalize_kernel_name(kernel_name)

    nu = _resolve_matern_nu(norm_name)
    if nu is not None:
        return _build_matern(nu, ard_num_dims)
    if norm_name in ["rbf", "gaussian"]:
        return ScaleKernel(RBFKernel(ard_num_dims=ard_num_dims))

    raise ValueError(f"Unknown kernel type: {kernel_name}")


def build_models(X_train: torch.Tensor, Y_train: torch.Tensor, cfg: _ModelConfigLike) -> ModelListGP:
    models = []
    d = X_train.shape[1]

    # Tanimoto のときは 0/1 指紋をそのまま使うので Normalize を使わない。
    use_input_norm = _normalize_kernel_name(cfg.kernel) != "tanimoto"

    for i in range(Y_train.shape[-1]):
        input_tf = Normalize(d=d) if use_input_norm else None

        gp = SingleTaskGP(
            X_train,
            Y_train[:, i : i + 1],
            covar_module=make_covar_module(d, cfg),
            input_transform=input_tf,
            outcome_transform=Standardize(m=1),
        )

        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(X_train.device)
        gp = gp.to(X_train.device)
        fit_gpytorch_mll(mll)

        models.append(gp)

    return ModelListGP(*models)
