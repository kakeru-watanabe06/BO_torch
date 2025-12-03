import json
import subprocess
from pathlib import Path
from typing import Tuple
import re

import torch
import pandas as pd

CALC_ROOT = Path("/home/kaker/calculation")
TEMPLATE_JSON = CALC_ROOT / "configs" / "setting.json"
RUN_SCRIPT = CALC_ROOT / "scripts" / "run_3_SingleFromConfig.py"


def _load_template_config() -> dict:
    with open(TEMPLATE_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_compound_name(picked_row: pd.Series, cfg) -> str:
    """
    ファイル名に使う安全な名前を作る。
    ここでは id_col をそのまま文字列化して、怪しい文字だけアンダースコアに。
    """
    raw = str(picked_row[cfg.data.id_col])
    # 簡易サニタイズ（英数字と_-以外を_に）
    safe = re.sub(r"[^0-9A-Za-z_\-]+", "_", raw)
    return safe or "molecule"


def _build_job_workdir_rel(template_workdir: str, compound_name: str) -> str:
    """
    "results/result_901_WorkFlowSingle/HABI" → "results/result_901_WorkFlowSingle/<compound_name>"
    のように、末尾だけ差し替える。
    """
    base, _ = template_workdir.rsplit("/", 1)
    return f"{base}/{compound_name}"


def _run_single_job(job_cfg: dict, job_cfg_path: Path) -> None:
    """JSON を書き出して run_3_SingleFromConfig.py を実行する。"""
    job_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(job_cfg_path, "w", encoding="utf-8") as f:
        json.dump(job_cfg, f, ensure_ascii=False, indent=2)

    cmd = ["python", str(RUN_SCRIPT), "--config", str(job_cfg_path)]
    # 必要なら python のフルパスに変えてもよい
    res = subprocess.run(cmd, cwd=str(CALC_ROOT), capture_output=True, text=True)

    if res.returncode != 0:
        # 失敗時はログを出して例外にする（方針次第でペナルティ値にしてもOK）
        print("=== Calculation failed ===")
        print("STDOUT:\n", res.stdout)
        print("STDERR:\n", res.stderr)
        raise RuntimeError(f"External calc failed (returncode={res.returncode})")

def _find_states_csv(workdir_rel: str) -> Path:
    """
    /home/kaker/calculation/<workdir_rel>/1/pyscf_tddft/ 以下から
    *_states.csv を探して返す。
    """
    tddft_dir = CALC_ROOT / workdir_rel / "1" / "pyscf_tddft"
    if not tddft_dir.exists():
        raise FileNotFoundError(f"pyscf_tddft directory not found: {tddft_dir}")

    matches = list(tddft_dir.glob("*_states.csv"))
    if not matches:
        raise FileNotFoundError(f"states.csv not found in {tddft_dir}")
    return matches[0]  # 基本1個のはずなので先頭

def _read_s1_energy_and_f(states_csv: Path) -> Tuple[float, float]:
    """
    states_csv から「状態 1」の E(eV) と f を読む。
    列名は実際の CSV に合わせて調整。
    """
    df = pd.read_csv(states_csv)

    # 1 行目が状態 1 という前提
    row0 = df.iloc[0]

    # ここ、実際の列名に合わせて変えてください：
    # 例: "E (eV)" や "E_eV" など
    try:
        e_ev = float(row0["E (eV)"])
    except KeyError:
        # 予備パターン
        e_ev = float(row0[[c for c in df.columns if "E" in c and "eV" in c][0]])

    try:
        f_val = float(row0["f"])
    except KeyError:
        f_val = float(row0[[c for c in df.columns if c.lower().startswith("f")][0]])

    return e_ev, f_val


def build_observe_func(cfg, device: torch.device, mean_raw: torch.Tensor, std_raw: torch.Tensor):
    """
    cfg, device, 固定スケーラー(mean/std) を束縛した observe_func を返す。
    online_bo_loop にはこの戻り値を渡す。
    """
    template_cfg = _load_template_config()
    template_workdir = template_cfg["workflow"]["workdir"]

    def observe_func(picked_row: pd.Series) -> torch.Tensor:
        """
        1. picked_row から SMILES / ID を取得
        2. テンプレ JSON に埋め込んで設定ファイルを書き出し
        3. run_3_SingleFromConfig.py で計算
        4. .out → states_csv から S1 energy / f を読み出し
        5. 固定スケールで標準化して Tensor(m,) を返す
        """
        smiles = picked_row[cfg.data.smiles_col]
        smiles = "O" # ここはデバッグ用ダミー（実際には picked_row から取る）
        print(f"Running calculation for SMILES: {smiles}")
        compound_name = _make_compound_name(picked_row, cfg)

        # --- JSON 構築 ---
        job_cfg = json.loads(json.dumps(template_cfg))  # deep copy
        job_cfg["input"]["smiles"] = smiles
        job_cfg["input"]["compound_name"] = compound_name

        workdir_rel = _build_job_workdir_rel(template_workdir, compound_name)
        job_cfg["workflow"]["workdir"] = workdir_rel

        # ジョブ用設定ファイルの置き場所（計算側の workdir は上で設定済み）
        job_cfg_path = CALC_ROOT / "bo_jobs" / compound_name / "setting.json"

        # --- 計算実行 ---
        _run_single_job(job_cfg, job_cfg_path)

        # --- 結果読み込み（ここだけシンプルに） ---
        states_csv = _find_states_csv(workdir_rel)
        df = pd.read_csv(states_csv)
        row0 = df.iloc[0]

        # 列名は実際の CSV に合わせてちょっとだけ調整してね
        e_ev = float(row0["E (eV)"])
        f_val = float(row0["f"])

        raw_vals = []
        for col in cfg.scaler.y_raw_cols:
            if col == "S1_energy_eV":
                raw_vals.append(e_ev)
            elif col == "Oscillator_strength":
                raw_vals.append(f_val)
            else:
                raise ValueError(f"Unknown y_raw_col: {col}")

        y_raw = torch.tensor(raw_vals, dtype=torch.double, device=device)
        y_scaled = (y_raw - mean_raw) / std_raw
        return y_scaled

    return observe_func
