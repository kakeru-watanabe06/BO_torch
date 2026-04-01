from pathlib import Path
import pandas as pd
import numpy as np
import re
from datetime import datetime

ROOT = Path("results")
OUTPUT_PATH = ROOT / "summary.csv"
METRICS_NAME = "metrics.xlsx"


# -----------------------------
# util
# -----------------------------

def extract_datetime_from_name(name: str):
    """
    フォルダ名から YYYYMMDD-HHMMSS / YYYYMMDD_HHMMSS を抽出
    """
    m = re.search(r"(20\d{6})[-_]?(\d{6})", name)
    if not m:
        return pd.NaT
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except Exception:
        return pd.NaT


def detect_best_iter_scalarbest(df: pd.DataFrame) -> int:
    sb = df["ScalarBest"].to_numpy(dtype=float)
    idx = sb.argmax()   # 最大化
    return int(df.loc[idx, "iter"])


# -----------------------------
# main logic
# -----------------------------

records = []

for metrics_path in ROOT.rglob(METRICS_NAME):
    try:
        df = pd.read_excel(metrics_path)
        df.columns = [str(c).strip() for c in df.columns]

        if not {"iter", "ScalarBest"} <= set(df.columns):
            raise ValueError("missing iter or ScalarBest")

        best_iter = detect_best_iter_scalarbest(df)
        best_val = float(df["ScalarBest"].max())

        # results からの相対パスをすべて列にする
        rel_parts = metrics_path.relative_to(ROOT).parts[:-1]  # metrics.xlsx を除く

        record = {
            "best_iter": best_iter,
            "best_ScalarBest": best_val,
            "total_iters": int(df["iter"].iloc[-1]),
            "metrics_path": str(metrics_path),
        }

        # 階層を level_0, level_1, ...
        for i, name in enumerate(rel_parts):
            record[f"level_{i}"] = name

        # 日付キー（run 名＝最後の階層から取る）
        record["run_datetime"] = extract_datetime_from_name(rel_parts[-1])

        records.append(record)

    except Exception as e:
        print(f"[SKIP] {metrics_path} ({e})")


if not records:
    raise RuntimeError("metrics.xlsx が見つかりません")


summary_df = pd.DataFrame(records)

# 並び順：
# 1. level_0, level_1, ...（階層順）
# 2. run_datetime（日付順）
sort_cols = [c for c in summary_df.columns if c.startswith("level_")]
sort_cols.append("run_datetime")

summary_df = summary_df.sort_values(
    sort_cols,
    ascending=True,
    na_position="last"
).reset_index(drop=True)

summary_df.to_csv(OUTPUT_PATH, index=False)

print(summary_df)
print(f"\nSaved to: {OUTPUT_PATH.resolve()}")
