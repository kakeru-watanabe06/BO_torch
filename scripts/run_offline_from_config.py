#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
from datetime import datetime

import pandas as pd
import torch

# import path
ROOT = Path(__file__).resolve().parents[1]  # BO_TORCH/
sys.path.append(str(ROOT / "src"))          # BO_TORCH/src

from bo_tool.config import load_config, build_objective_spec
from bo_tool.data_utils import load_offline_data
from bo_tool.bo_loop_offline import offline_bo_loop
from bo_tool.metrics import (
    fixed_ref_point, hypervolume_curve, hypervolume_gap_curve,
    scalar_best_curve, history_to_appended_Ys, make_metrics_dataframe
)
from bo_tool.io_utils import ensure_dir, save_history_excel, save_metrics_excel


def parse_args():
    p = argparse.ArgumentParser(description="Offline BO runner (JSON config)")
    p.add_argument("--config", required=True, help="path to JSON config")
    return p.parse_args()


def main():
    torch.set_default_dtype(torch.double)
    args = parse_args()
    cfg = load_config(args.config)

    # load data
    used_df, pool_df, X_train, Y_train_raw, x_cols = load_offline_data(
        train_path=cfg.data.train,
        all_path=cfg.data.all,
        id_col=cfg.data.id_col,
        X_cols=cfg.data.x_cols,
        y_cols=cfg.data.y_cols,
        x_col_start=cfg.data.x_col_start,
        x_col_end=cfg.data.x_col_end,
    )

    # objective spec
    spec = build_objective_spec(cfg.data.y_cols, cfg.objective)
    eval_cfg = cfg.eval
    # BO loop
    n_init = len(used_df)
    history = offline_bo_loop(
        X_cols=x_cols,
        y_cols=cfg.data.y_cols,
        id_col=cfg.data.id_col,
        pool_df=pool_df,
        X_train=X_train,
        Y_train_raw=Y_train_raw,
        spec=spec,
        model_cfg=cfg.model,       
        max_iters=cfg.bo.max_iters,
        num_mc_samples=cfg.bo.mc,
        eval_cfg=eval_cfg,
        n_init=n_init,
        acq_type=cfg.bo.acq_type,
        ucb_beta=cfg.bo.ucb_beta,
    )


    # outputs
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = ts + (f"_{cfg.output.tag}" if cfg.output.tag else "")
    outdir = Path(cfg.output.outdir) / exp_name
    ensure_dir(outdir)

    # save history
    hist_path = outdir / "history.xlsx"
    save_history_excel(history, hist_path)

    # metrics
    all_df = pd.read_excel(cfg.data.all, engine="openpyxl")
    all_Y_raw = torch.tensor(all_df[cfg.data.y_cols].to_numpy(), dtype=torch.double)
    appended_Ys = history_to_appended_Ys(history, cfg.data.y_cols)
    initial_Y = Y_train_raw

    m = len(cfg.data.y_cols)
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
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(json.load(open(args.config, "r", encoding="utf-8")), f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {hist_path}")
    print(f"[OK] Saved: {outdir / 'metrics.xlsx'}")
    print(f"[OK] Saved: {outdir / 'config.json'}")


if __name__ == "__main__":
    main()
