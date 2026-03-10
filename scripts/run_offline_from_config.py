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
from bo_tool.config import build_objective_spec, load_config
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


def parse_args():
    parser = argparse.ArgumentParser(description="Offline BO runner (JSON config)")
    parser.add_argument("--config", required=True, help="path to JSON config")
    return parser.parse_args()


def _make_output_dir(base_outdir: str, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = ts + (f"_{tag}" if tag else "")
    outdir = Path(base_outdir) / exp_name
    ensure_dir(outdir)
    return outdir


def _read_table(path: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        return pd.read_csv(path)


def _save_config_snapshot(config_path: str, outpath: Path) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)


def main():
    torch.set_default_dtype(torch.double)
    args = parse_args()
    cfg = load_config(args.config)

    used_df, pool_df, X_train, Y_train_raw, x_cols = load_offline_data(
        train_path=cfg.data.train,
        all_path=cfg.data.all,
        id_col=cfg.data.id_col,
        X_cols=cfg.data.x_cols,
        y_cols=cfg.data.y_cols,
        x_col_start=cfg.data.x_col_start,
        x_col_end=cfg.data.x_col_end,
    )

    spec = build_objective_spec(cfg.data.y_cols, cfg.objective)
    eval_cfg = cfg.eval

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
        n_init=len(used_df),
        acq_type=cfg.bo.acq_type,
        ucb_beta=cfg.bo.ucb_beta,
    )

    outdir = _make_output_dir(cfg.output.outdir, cfg.output.tag)

    hist_path = outdir / "history.xlsx"
    save_history_excel(history, hist_path)

    all_df = _read_table(cfg.data.all)
    all_Y_raw = torch.tensor(all_df[cfg.data.y_cols].to_numpy(), dtype=torch.double)
    appended_Ys = history_to_appended_Ys(history, cfg.data.y_cols)
    initial_Y = Y_train_raw

    m = len(cfg.data.y_cols)
    if spec.kind != "linear_scalarization" and m >= 2:
        ref_point = fixed_ref_point(all_Y_raw, spec)
        hv_vals = hypervolume_curve(initial_Y, appended_Ys, spec, ref_point)
        hv_gap = hypervolume_gap_curve(initial_Y, appended_Ys, all_Y_raw, spec, ref_point)
        best_vals = scalar_best_curve(initial_Y, appended_Ys, spec)
        metrics_df = make_metrics_dataframe(hv_vals, hv_gap, best_vals)
    else:
        best_vals = scalar_best_curve(initial_Y, appended_Ys, spec)
        metrics_df = make_metrics_dataframe(
            hv_curve_vals=None,
            hv_gap_vals=None,
            scalar_best_vals=best_vals,
        )

    metrics_path = outdir / "metrics.xlsx"
    save_metrics_excel(metrics_df, metrics_path)

    config_path = outdir / "config.json"
    _save_config_snapshot(args.config, config_path)

    print(f"[OK] Saved: {hist_path}")
    print(f"[OK] Saved: {metrics_path}")
    print(f"[OK] Saved: {config_path}")


if __name__ == "__main__":
    main()
