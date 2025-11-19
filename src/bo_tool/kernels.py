# src/bo_tool/kernels.py
from __future__ import annotations
import torch
from gpytorch.kernels import Kernel


class TanimotoKernel(Kernel):
    r"""
    シンプルな Tanimoto カーネル実装。
    バイナリ or 非負の特徴量 x, x' に対して

        k(x, x') = (x · x') / (||x||^2 + ||x'||^2 - x · x')

    を返す。
    想定入力: X ~ (N, D), (M, D) の 2次元テンソル（バッチ対応はしていない簡易版）。
    """

    has_lengthscale = False  # lengthscale は持たない前提

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params) -> torch.Tensor:
        # 2D (N, D), (M, D) を想定（必要ならあとでバッチ対応拡張）
        x1 = x1.float()
        x2 = x2.float()

        # 内積 x1 @ x2^T → (N, M)
        s12 = x1 @ x2.transpose(-2, -1)

        # 自己内積 ||x||^2
        s1 = (x1.pow(2)).sum(dim=-1, keepdim=True)      # (N, 1)
        s2 = (x2.pow(2)).sum(dim=-1).unsqueeze(-2)      # (1, M)

        denom = s1 + s2 - s12                           # (N, M)
        denom = denom.clamp_min(1e-8)                   # 0 割回避

        K = s12 / denom

        if diag:
            # diag=True のときは対角だけ返す (N,) か (M,) 想定
            return K.diagonal(dim1=-2, dim2=-1)
        return K
