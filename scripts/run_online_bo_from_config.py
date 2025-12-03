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
from bo_tool.data_utils import load_online_data
from bo_tool.bo_loop_online import online_bo_loop
from bo_tool.Conect_To_Calculation import build_observe_func
from bo_tool.metrics import (
    fixed_ref_point, hypervolume_curve, hypervolume_gap_curve,
    scalar_best_curve, history_to_appended_Ys, make_metrics_dataframe
)
from bo_tool.io_utils import ensure_dir, save_history_excel, save_metrics_excel


def parse_args():
    p = argparse.ArgumentParser(description="Online BO runner (JSON config)")
    p.add_argument("--config", required=True, help="path to JSON config")
    return p.parse_args()


def _build_fixed_scaler_from_cfg(cfg, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    cfg.scaler から y_raw → y_scaled 用の mean / std ベクトルを作る。

    期待する config 例：
      "scaler": {
        "y_raw_cols": ["S1_energy_eV", "Oscillator_strength"],
        "mean": [2.50, 0.40],
        "std":  [0.30, 0.10]
      }

    - y_cols (cfg.data.y_cols) と scaler.y_raw_cols は論理的には同じ次元数 m を持つ前提。
    - 生の戻り値は scaler.y_raw_cols の順番で並べてベクトル化し、
      (y_raw - mean) / std をとったものを BO に渡す。
    """
    raw_cols = list(cfg.scaler.y_raw_cols)
    mean = torch.tensor(list(cfg.scaler.mean), dtype=torch.double, device=device)
    std = torch.tensor(list(cfg.scaler.std), dtype=torch.double, device=device)

    if mean.numel() != len(raw_cols) or std.numel() != len(raw_cols):
        raise ValueError(
            f"scaler.mean/std の長さ ({mean.numel()}, {std.numel()}) と "
            f"scaler.y_raw_cols の長さ ({len(raw_cols)}) が一致しません"
        )
    return mean, std


def main():
    torch.set_default_dtype(torch.double)
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 1) データ読込み =====
    used_df, pool_df, X_train, Y_train_scaled_init, x_cols = load_online_data(
        train_path=cfg.data.train,
        all_path=cfg.data.all,
        id_col=cfg.data.id_col,
        smiles_col=cfg.data.smiles_col,
        X_cols=cfg.data.x_cols,
        y_cols=cfg.data.y_cols,
        x_col_start=cfg.data.x_col_start,
        x_col_end=cfg.data.x_col_end,
    )
    smiles_col = cfg.data.smiles_col
    if smiles_col not in pool_df.columns:
        raise ValueError(f"smiles_col={smiles_col} が pool_df に存在しません")
    print(f"x_cols used: {x_cols}")
    # ===== 2) 目的関数仕様 =====
    spec = build_objective_spec(cfg.data.y_cols, cfg.objective)

    # ===== 3) 固定スケーラー構築（生 → スケール済み） =====
    #   - 初期データはすでに標準化済みカラムを使っている前提なので、
    #     Y_train_scaled_init はそのまま GP に渡す。
    #   - 新規観測は「生の y_raw」から mean/std で標準化してから追加する。
    mean_raw, std_raw = _build_fixed_scaler_from_cfg(cfg, device=device)

    # ===== 4) all_df を読んでおく（オフライン検証モード / HV 用） =====
    all_df = pd.read_excel(cfg.data.all, engine="openpyxl")
    all_df_indexed = all_df.set_index(cfg.data.id_col)

    # ===== 5) 真値観測関数 =====
    observe_func = build_observe_func(
        cfg=cfg,
        device=device,
        mean_raw=mean_raw,
        std_raw=std_raw,
    )

    # ===== 6) BO ループ実行 =====
    eval_cfg = {
        "loocv": True,
        "min_points": 5,
    }

    history = online_bo_loop(
        X_cols=x_cols,
        y_cols=cfg.data.y_cols,       # 「標準化後の目的変数」の論理名
        id_col=cfg.data.id_col,
        smiles_col=cfg.data.smiles_col,
        pool_df=pool_df,
        X_train=X_train.to(device=device),
        Y_train_raw=Y_train_scaled_init.to(device=device),
        spec=spec,
        model_cfg=cfg.model,
        observe_func=observe_func,
        max_iters=cfg.bo.max_iters,
        num_mc_samples=cfg.bo.mc,
        eval_cfg=eval_cfg,
    )

    # ===== 7) 出力（offline_runner と同じスタイル） =====
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = ts + (f"_{cfg.output.tag}" if cfg.output.tag else "")
    outdir = Path(cfg.output.outdir) / exp_name
    ensure_dir(outdir)

    # history
    hist_path = outdir / "history.xlsx"
    save_history_excel(history, hist_path)

    # metrics
    all_Y_raw = torch.tensor(all_df[cfg.data.y_cols].to_numpy(), dtype=torch.double, device=device)
    appended_Ys = history_to_appended_Ys(history, cfg.data.y_cols)
    initial_Y = Y_train_scaled_init.to(device=device)

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

    save_metrics_excel(metrics_df, outdir / "metrics.xlsx")

    # config snapshot
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            json.load(open(args.config, "r", encoding="utf-8")),
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] Saved: {hist_path}")
    print(f"[OK] Saved: {outdir / 'metrics.xlsx'}")
    print(f"[OK] Saved: {outdir / 'config.json'}")


if __name__ == "__main__":
    main()
