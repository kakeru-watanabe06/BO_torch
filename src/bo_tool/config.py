# src/BO_torch/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Literal, Any, Dict
import json
from pathlib import Path

from .objectives import ObjectiveSpec


@dataclass
class DataConfig:
    train: str
    all: str
    id_col: str
    x_cols: List[str]
    y_cols: List[str]
    # 追加：カラム範囲（0-based / iloc 用インデックス）
    x_col_start: Optional[int] = None
    x_col_end: Optional[int] = None   # Python のスライスと同じで「終端は含まない」
    smiles_col: Optional[str] = None

@dataclass
class ModelConfig:
    kernel: str = "matern32"
    ard: bool = False

@dataclass
class ObjectiveConfig:
    kind: Literal["target_distance", "identity_multi", "linear_scalarization", "mixed_multi"] = "target_distance"
    weights: Optional[List[float]] = None
    targets: Optional[List[float]] = None
    power: float = 2.0
    maximize: Optional[List[bool]] = None
    modes: Optional[List[Literal["target", "identity"]]] = None


@dataclass
class BOConfig:
    max_iters: int = 64
    mc: int = 512
    acq_type: str = "auto"      # "auto", "qei", "qehvi", "qucb"
    ucb_beta: float = 2.0       # qUCB 用 β

@dataclass
class ScalerConfig:
    """
    生の y (e.g., S1_energy_eV, Oscillator_strength) を
    固定の平均・標準偏差で標準化するための情報。
    """
    y_raw_cols: List[str]  # 例: ["S1_energy_eV", "Oscillator_strength"]
    mean: List[float]
    std: List[float]

@dataclass
class EvalConfig:
    loocv: bool = True
    min_points: int = 5

@dataclass
class OutputConfig:
    outdir: str = "results/offline_bo"
    tag: str = ""


@dataclass
class ExperimentConfig:
    data: DataConfig
    objective: ObjectiveConfig
    model: ModelConfig
    bo: BOConfig
    output: OutputConfig
    scaler: Optional[ScalerConfig] = None
    eval: Optional[EvalConfig] = None


def _as_bool_list(xs: Optional[list]) -> Optional[List[bool]]:
    if xs is None:
        return None
    return [bool(int(x)) if isinstance(x, (int, str)) else bool(x) for x in xs]


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r") as f:
        cfg = json.load(f)

    # ===== data =====
    data_cfg = cfg.get("data", {})
    dc = DataConfig(
        train=data_cfg["train"],
        all=data_cfg["all"],
        id_col=data_cfg["id_col"],
        x_cols=data_cfg["x_cols"],
        y_cols=data_cfg["y_cols"],
        x_col_start=data_cfg.get("x_col_start"),
        x_col_end=data_cfg.get("x_col_end"),
        smiles_col=data_cfg.get("smiles_col"),
    )

    # ===== objective =====
    obj_cfg = cfg.get("objective", {})
    oc = ObjectiveConfig(
        kind=obj_cfg["kind"],
        targets=obj_cfg.get("targets", []),
        weights=obj_cfg.get("weights", []),
        power=obj_cfg.get("power", 2.0),
        maximize=obj_cfg.get("maximize", []),
        modes=obj_cfg.get("modes", []),
    )

    # ===== model =====
    model_cfg = cfg.get("model", {})                    # ←★ 必須
    mc = ModelConfig(
        kernel=model_cfg.get("kernel", "matern32"),
        ard=model_cfg.get("ard", False),
    )

    # ===== bo =====
    bo_cfg = cfg.get("bo", {})
    boc = BOConfig(
        max_iters=bo_cfg.get("max_iters", 32),
        mc=bo_cfg.get("mc", 256),
        acq_type=bo_cfg.get("acq_type", "auto"),
        ucb_beta=bo_cfg.get("ucb_beta", 2.0),
    )

    # ===== eval =====
    eval_cfg_raw = cfg.get("eval", {})
    ec = EvalConfig(
        loocv=eval_cfg_raw.get("loocv", True),
        min_points=eval_cfg_raw.get("min_points", 5),
    )

    # ===== scaler =====
    scaler_cfg_raw = cfg.get("scaler")
    sc: Optional[ScalerConfig] = None
    if scaler_cfg_raw is not None:
        sc = ScalerConfig(
            y_raw_cols=scaler_cfg_raw["y_raw_cols"],
            mean=scaler_cfg_raw["mean"],
            std=scaler_cfg_raw["std"],
        )

    # ===== output =====
    out_cfg = cfg.get("output", {})
    outc = OutputConfig(
        outdir=out_cfg.get("outdir", "results"),
        tag=out_cfg.get("tag", "exp"),
    )

    return ExperimentConfig(
        data=dc,
        objective=oc,
        model=mc,       
        bo=boc,
        output=outc,
        scaler=sc,
        eval=ec,
    )



def build_objective_spec(y_cols: List[str], oc: ObjectiveConfig) -> ObjectiveSpec:
    # y次元に合わせて spec を構築
    return ObjectiveSpec(
        kind=oc.kind,
        weights=oc.weights if oc.weights is not None else [1.0] * len(y_cols),
        targets=oc.targets,
        power=oc.power,
        maximize=oc.maximize,
        modes=oc.modes,
    )
