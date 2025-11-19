from __future__ import annotations

import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel


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


def build_models(X_train, Y_train_raw, model_cfg):
    models = []
    input_dim = X_train.shape[1]

    for i in range(Y_train_raw.shape[-1]):
        gp = SingleTaskGP(
            X_train,
            Y_train_raw[:, i:i+1],
            input_transform=Normalize(d=input_dim),
            outcome_transform=Standardize(m=1),
        )

        # カーネルを JSON に合わせて設定
        gp.covar_module = create_kernel(
            kernel_name=model_cfg.kernel,
            ard=model_cfg.ard,
            input_dim=input_dim
        )

        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        models.append(gp)

    return ModelListGP(*models)

