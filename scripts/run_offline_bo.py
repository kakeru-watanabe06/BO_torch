#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import pandas as pd
import torch
from datetime import datetime

# import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from bo_tool.data_utils import load_offline_data
from bo_tool.objectives import ObjectiveSpec, to_object_space
from bo_tool.optimization.bo_loop_offline import offline_bo_loop
from bo_tool.evaluation.metrics import (
    fixed_ref_point, hypervolume_curve, hypervolume_gap_curve,
    scalar_best_curve, history_to_appended_Ys, make_metrics_dataframe
)
from bo_tool.io_utils import ensure_dir, save_history_excel, save_metrics_excel


def parse_args():
    p = argparse.ArgumentParser(description="Offline BO benchmark runner")
    p.add_argument("--train", required=True, help="path to train excel")
    p.add_argument("--all",   required=True, help="path to all excel")
    p.add_argument("--id_col", default="Folder")
    p.add_argument("--x_cols",  nargs="+", required=True, help="feature columns")
    p.add_argument("--y_cols",  nargs="+", required=True, help="target columns (m-dim)")

    # ObjectiveSpec
    p.add_argument("--obj_kind", choices=["target_distance","identity_multi","linear_scalarization"],
                   default="target_distance")
    p.add_argument("--targets", type=float, nargs="*", default=None,
                   help="targets for each y (required for target_distance)")
    p.add_argument("--weights", type=float, nargs="*", default=None,
                   help="weights for each y (default=1)")
    p.add_argument("--power", type=float, default=2.0)
    p.add_argument("--maximize", type=int, nargs="*", default=None,
                   help="for identity_multi: 1=maximize,0=minimize per dim")

    # BO
    p.add_argument("--max_iters", type=int, default=64)
    p.add_argument("--mc", type=int, default=512)

    # output
    p.add_argument("--outdir", default="results/offline_bo")
    p.add_argument("--tag", default="", help="extra tag for output folder name")
    return p.parse_args()


def main():
    torch.set_default_dtype(torch.double)
    args = parse_args()

    # load data
    used_df, pool_df, X_train, Y_train_raw = load_offline_data(
        train_path=args.train, all_path=args.all,
        id_col=args.id_col, X_cols=args.x_cols, y_cols=args.y_cols
    )

    # ObjectiveSpec build
    m = len(args.y_cols)
    weights = args.weights if args.weights else [1.0] * m
    maximize = None
    if args.maximize is not None:
        assert len(args.maximize) == m, "--maximize length must equal m"
        maximize = [bool(x) for x in args.maximize]

    spec = ObjectiveSpec(
        kind=args.obj_kind,
        weights=weights,
        targets=args.targets if args.targets is not None else None,
        power=args.power,
        maximize=maximize,
    )

    # BO loop
    history = offline_bo_loop(
        X_cols=args.x_cols, y_cols=args.y_cols, id_col=args.id_col,
        pool_df=pool_df.copy(),
        X_train=X_train.clone(),
        Y_train_raw=Y_train_raw.clone(),
        spec=spec, max_iters=args.max_iters, num_mc_samples=args.mc
    )

    # outputs (folder with timestamp)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{ts}"
    if args.tag:
        exp_name += f"_{args.tag}"
    outdir = Path(args.outdir) / exp_name
    ensure_dir(outdir)

    # save history
    hist_path = outdir / "history.xlsx"
    save_history_excel(history, hist_path)

    # metrics
    appended_Ys = history_to_appended_Ys(history, args.y_cols)
    initial_Y = Y_train_raw
    all_df = pd.read_excel(args.all, engine="openpyxl")
    all_Y_raw = torch.tensor(all_df[args.y_cols].to_numpy(), dtype=torch.double)

    metrics_df = None
    if spec.kind != "linear_scalarization" and m >= 2:
        ref_point = fixed_ref_point(all_Y_raw, spec)
        hv_vals = hypervolume_curve(initial_Y, appended_Ys, spec, ref_point)
        hv_gap  = hypervolume_gap_curve(initial_Y, appended_Ys, all_Y_raw, spec, ref_point)
        best_vals = scalar_best_curve(initial_Y, appended_Ys, spec)
        metrics_df = make_metrics_dataframe(hv_vals, hv_gap, best_vals)
    else:
        best_vals = scalar_best_curve(initial_Y, appended_Ys, spec)
        metrics_df = make_metrics_dataframe(hv_curve_vals=None, hv_gap_vals=None, scalar_best_vals=best_vals)

    save_metrics_excel(metrics_df, outdir / "metrics.xlsx")

    # save config snapshot
    cfg = {
        "id_col": args.id_col,
        "x_cols": args.x_cols,
        "y_cols": args.y_cols,
        "obj": {
            "kind": args.obj_kind,
            "weights": weights,
            "targets": args.targets,
            "power": args.power,
            "maximize": maximize,
        },
        "max_iters": args.max_iters,
        "mc": args.mc,
        "paths": {"train": args.train, "all": args.all},
    }
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {hist_path}")
    print(f"[OK] Saved: {outdir / 'metrics.xlsx'}")
    print(f"[OK] Saved: {outdir / 'config.json'}")


if __name__ == "__main__":
    main()
