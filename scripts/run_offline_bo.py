#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from bo_tool.bo_loop_offline import offline_bo_loop
from bo_tool.data_utils import load_offline_data
from bo_tool.io_utils import ensure_dir, save_history_excel, save_metrics_excel
from bo_tool.metrics import (
    fixed_ref_point,
    history_to_appended_Ys,
    hypervolume_curve,
    hypervolume_gap_curve,
    make_metrics_dataframe,
    scalar_best_curve,
)
from bo_tool.models import ModelConfig
from bo_tool.objectives import ObjectiveSpec


def parse_args():
    parser = argparse.ArgumentParser(description="Offline BO benchmark runner")
    parser.add_argument("--train", required=True, help="path to train excel")
    parser.add_argument("--all", required=True, help="path to all excel")
    parser.add_argument("--id_col", default="Folder")
    parser.add_argument("--x_cols", nargs="+", required=True, help="feature columns")
    parser.add_argument("--y_cols", nargs="+", required=True, help="target columns (m-dim)")

    parser.add_argument(
        "--obj_kind",
        choices=["target_distance", "identity_multi", "linear_scalarization", "mixed_multi"],
        default="target_distance",
    )
    parser.add_argument("--targets", type=float, nargs="*", default=None)
    parser.add_argument("--weights", type=float, nargs="*", default=None)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--maximize", type=int, nargs="*", default=None)

    parser.add_argument("--kernel", default="matern32", choices=["matern12", "matern32", "matern52", "tanimoto"])
    parser.add_argument("--ard", action="store_true", help="enable ARD for supported kernels")

    parser.add_argument("--max_iters", type=int, default=64)
    parser.add_argument("--mc", type=int, default=512)
    parser.add_argument(
        "--acq_type",
        default="auto",
        choices=["auto", "qei", "qehvi", "qucb", "q_ucb", "ucb", "sucb", "s_ucb"],
    )
    parser.add_argument("--ucb_beta", type=float, default=2.0)

    parser.add_argument("--outdir", default="results/offline_bo")
    parser.add_argument("--tag", default="", help="extra tag for output folder name")
    return parser.parse_args()


def _read_table(path: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        return pd.read_csv(path)


def main():
    torch.set_default_dtype(torch.double)
    args = parse_args()

    used_df, pool_df, X_train, Y_train_raw, x_cols = load_offline_data(
        train_path=args.train,
        all_path=args.all,
        id_col=args.id_col,
        X_cols=args.x_cols,
        y_cols=args.y_cols,
    )

    m = len(args.y_cols)
    weights = args.weights if args.weights else [1.0] * m
    maximize = None
    if args.maximize is not None:
        if len(args.maximize) != m:
            raise ValueError("--maximize length must equal m")
        maximize = [bool(x) for x in args.maximize]

    spec = ObjectiveSpec(
        kind=args.obj_kind,
        weights=weights,
        targets=args.targets if args.targets is not None else None,
        power=args.power,
        maximize=maximize,
    )

    model_cfg = ModelConfig(kernel=args.kernel, ard=args.ard)

    history = offline_bo_loop(
        X_cols=x_cols,
        y_cols=args.y_cols,
        id_col=args.id_col,
        pool_df=pool_df.copy(),
        X_train=X_train.clone(),
        Y_train_raw=Y_train_raw.clone(),
        spec=spec,
        model_cfg=model_cfg,
        max_iters=args.max_iters,
        num_mc_samples=args.mc,
        acq_type=args.acq_type,
        ucb_beta=args.ucb_beta,
        eval_cfg=None,
        n_init=len(used_df),
    )

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = ts + (f"_{args.tag}" if args.tag else "")
    outdir = Path(args.outdir) / exp_name
    ensure_dir(outdir)

    hist_path = outdir / "history.xlsx"
    save_history_excel(history, hist_path)

    appended_Ys = history_to_appended_Ys(history, args.y_cols)
    all_df = _read_table(args.all)
    all_Y_raw = torch.tensor(all_df[args.y_cols].to_numpy(), dtype=torch.double)

    if spec.kind != "linear_scalarization" and m >= 2:
        ref_point = fixed_ref_point(all_Y_raw, spec)
        hv_vals = hypervolume_curve(Y_train_raw, appended_Ys, spec, ref_point)
        hv_gap = hypervolume_gap_curve(Y_train_raw, appended_Ys, all_Y_raw, spec, ref_point)
        best_vals = scalar_best_curve(Y_train_raw, appended_Ys, spec)
        metrics_df = make_metrics_dataframe(hv_vals, hv_gap, best_vals)
    else:
        best_vals = scalar_best_curve(Y_train_raw, appended_Ys, spec)
        metrics_df = make_metrics_dataframe(
            hv_curve_vals=None,
            hv_gap_vals=None,
            scalar_best_vals=best_vals,
        )

    metrics_path = outdir / "metrics.xlsx"
    save_metrics_excel(metrics_df, metrics_path)

    config_snapshot = {
        "id_col": args.id_col,
        "x_cols": x_cols,
        "y_cols": args.y_cols,
        "objective": {
            "kind": args.obj_kind,
            "weights": weights,
            "targets": args.targets,
            "power": args.power,
            "maximize": maximize,
        },
        "model": {
            "kernel": args.kernel,
            "ard": args.ard,
        },
        "bo": {
            "max_iters": args.max_iters,
            "mc": args.mc,
            "acq_type": args.acq_type,
            "ucb_beta": args.ucb_beta,
        },
        "paths": {
            "train": args.train,
            "all": args.all,
        },
    }

    config_path = outdir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {hist_path}")
    print(f"[OK] Saved: {metrics_path}")
    print(f"[OK] Saved: {config_path}")


if __name__ == "__main__":
    main()
