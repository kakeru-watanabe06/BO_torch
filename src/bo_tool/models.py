from __future__ import annotations

import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from bo_tool.kernels import TanimotoKernel


from dataclasses import dataclass

@dataclass
class ModelConfig:
    kernel: str = "matern32"
    ard: bool = False


def make_covar_module(input_dim: int, cfg: ModelConfig):
    """ModelConfig に応じて GPyTorch のカーネルを構成する。"""
    # Tanimoto では ARD 無効
    ard_num_dims = input_dim if (cfg.ard and cfg.kernel != "tanimoto") else None

    if cfg.kernel in ("matern32", "matern_3_2"):
        base = MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)
        return ScaleKernel(base)

    if cfg.kernel in ("matern52", "matern_5_2"):
        base = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
        return ScaleKernel(base)

    if cfg.kernel == "tanimoto":
        if cfg.ard:
            raise ValueError("Tanimoto kernel does not support ARD. Set ard=false in config.")
        base = TanimotoKernel()
        return ScaleKernel(base)

    raise ValueError(f"Unknown kernel type: {cfg.kernel}")
    raise ValueError(f"Unknown kernel type: {cfg.kernel}")

def create_kernel(kernel_name: str, ard: bool, input_dim: int):
    ard_num_dims = input_dim if ard else None

    if kernel_name.lower() in ["matern12", "matern-12", "matern_nu_0.5"]:
        return ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=ard_num_dims))

    if kernel_name.lower() in ["matern32", "matern-32", "matern_nu_1.5"]:
        return ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=ard_num_dims))

    if kernel_name.lower() in ["matern52", "matern-52", "matern_nu_2.5"]:
        return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=ard_num_dims))

    if kernel_name.lower() in ["rbf", "gaussian"]:
        return ScaleKernel(RBFKernel(ard_num_dims=ard_num_dims))

    raise ValueError(f"Unknown kernel type: {kernel_name}")

def build_models(X_train: torch.Tensor, Y_train: torch.Tensor, cfg: ModelConfig) -> ModelListGP:
    models = []
    d = X_train.shape[1]

    # Tanimoto のときは 0/1 指紋をそのまま使うので Normalize はオフ
    use_input_norm = (cfg.kernel != "tanimoto")

    for i in range(Y_train.shape[-1]):
        input_tf = Normalize(d=d) if use_input_norm else None

        gp = SingleTaskGP(
            X_train,
            Y_train[:, i:i+1],
            covar_module=make_covar_module(d, cfg),
            input_transform=input_tf,
            outcome_transform=Standardize(m=1),
        )

        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(X_train.device)
        gp = gp.to(X_train.device)
        fit_gpytorch_mll(mll)

        models.append(gp)

    return ModelListGP(*models)