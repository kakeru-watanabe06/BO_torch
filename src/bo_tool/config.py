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

@dataclass
class ModelConfig:
    kernel: str = "matern32"
    ard: bool = False

@dataclass
class ObjectiveConfig:
    kind: Literal["target_distance", "identity_multi", "linear_scalarization"] = "target_distance"
    weights: Optional[List[float]] = None
    targets: Optional[List[float]] = None
    power: float = 2.0
    maximize: Optional[List[bool]] = None


@dataclass
class BOConfig:
    max_iters: int = 64
    mc: int = 512


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
    )

    # ===== objective =====
    obj_cfg = cfg.get("objective", {})
    oc = ObjectiveConfig(
        kind=obj_cfg["kind"],
        targets=obj_cfg.get("targets", []),
        weights=obj_cfg.get("weights", []),
        power=obj_cfg.get("power", 2.0),
        maximize=obj_cfg.get("maximize", []),
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
    )



def build_objective_spec(y_cols: List[str], oc: ObjectiveConfig) -> ObjectiveSpec:
    # y次元に合わせて spec を構築
    return ObjectiveSpec(
        kind=oc.kind,
        weights=oc.weights if oc.weights is not None else [1.0] * len(y_cols),
        targets=oc.targets,
        power=oc.power,
        maximize=oc.maximize,
    )
