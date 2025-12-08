import json
import subprocess
from pathlib import Path
from typing import Tuple
import re
import os

import torch
import pandas as pd

# ==============================
# 設定
# ==============================

CALC_ROOT = Path("/Users/macstudio2022/local_calculation")
RUN_SCRIPT = CALC_ROOT / "scripts" / "run_3_SingleFromConfig.py"

# 計算用 conda 環境
CALC_ENV = "/opt/miniconda3/envs/calculation"
CALC_PYTHON = f"{CALC_ENV}/bin/python"
CALC_BIN = f"{CALC_ENV}/bin"

TEMPLATE_JSON = CALC_ROOT / "configs" / "setting.json"

# 各 observe 呼び出しごとにインクリメントされるジョブカウンタ
_job_counter = 0


# ==============================
# ヘルパー
# ==============================

def _load_template_config() -> dict:
    with open(TEMPLATE_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_compound_name(picked_row: pd.Series, cfg) -> str:
    """
    ファイル名に使う安全なベース名を作る。
    ここでは id_col をそのまま文字列化して、怪しい文字だけアンダースコアに。
    """
    raw = str(picked_row[cfg.data.id_col])
    safe = re.sub(r"[^0-9A-Za-z_\-]+", "_", raw)
    return safe or "molecule"


def _build_job_workdir_rel(template_workdir: str, iter_tag: str, id_tag: str,run_tag) -> str:
    """
    テンプレートの workdir をもとに
      results/result_901_WorkFlowSingle/<iter_tag>/<id_tag>
    のような相対パスを作る。

    例:
      template_workdir = "results/result_901_WorkFlowSingle/HABI"
      iter_tag  = "iter_001"
      id_tag    = "ID52"
      -> "results/result_901_WorkFlowSingle/iter_001/ID52"
    """
    base, _ = template_workdir.rsplit("/", 1)
    return f"{base}/{run_tag}/{iter_tag}/{id_tag}"



def _run_single_job(job_cfg: dict, job_cfg_path: Path) -> None:
    job_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(job_cfg_path, "w", encoding="utf-8") as f:
        json.dump(job_cfg, f, ensure_ascii=False, indent=2)

    # calculation 環境の bin を PATH に足す
    env = os.environ.copy()
    env["PATH"] = f"{CALC_BIN}:" + env.get("PATH", "")

    cmd = [CALC_PYTHON, str(RUN_SCRIPT), "--config", str(job_cfg_path)]
    res = subprocess.run(
        cmd,
        cwd=str(CALC_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )

    if res.returncode != 0:
        print("=== Calculation failed ===")
        print("STDOUT:\n", res.stdout)
        print("STDERR:\n", res.stderr)
        raise RuntimeError(f"External calc failed (returncode={res.returncode})")


def _find_states_csv(workdir_rel: str) -> Path:
    """
    CALC_ROOT / workdir_rel 以下から *_states.csv を探して返す。
    最初は素直に
        <workdir_rel>/pyscf_tddft/*_states.csv
    を見る。もし実際は `/1/pyscf_tddft` になっているなら、
    ここを `... / "1" / "pyscf_tddft"` に直す。
    """
    # 例: CALC_ROOT / "results/result_901_WorkFlowSingle/ID52_it001/pyscf_tddft"
    tddft_dir = CALC_ROOT / workdir_rel / "pyscf_tddft"
    if not tddft_dir.exists():
        raise FileNotFoundError(f"pyscf_tddft directory not found: {tddft_dir}")

    matches = list(tddft_dir.glob("*_states.csv"))
    if not matches:
        raise FileNotFoundError(f"states.csv not found in {tddft_dir}")
    return matches[0]  # 基本1個のはずなので先頭


def _read_s1_energy_and_f(states_csv: Path) -> Tuple[float, float]:
    """
    states_csv から1番目の励起状態の E(eV) と f を読む。
    列名は実際の CSV に合わせて調整。
    """
    df = pd.read_csv(states_csv)
    row0 = df.iloc[0]

    # 例: "excitation_energy_eV", "oscillator_strength_f"
    try:
        e_ev = float(row0["excitation_energy_eV"])
    except KeyError:
        # バックアップ：E と eV を含む列を探す
        cand = [c for c in df.columns if "E" in c and "eV" in c]
        if not cand:
            raise
        e_ev = float(row0[cand[0]])

    try:
        f_val = float(row0["oscillator_strength_f"])
    except KeyError:
        cand = [c for c in df.columns if c.lower().startswith("f")]
        if not cand:
            raise
        f_val = float(row0[cand[0]])

    return e_ev, f_val


# ==============================
# メイン：observe_func ビルダー
# ==============================

def build_observe_func(cfg, device: torch.device, mean_raw: torch.Tensor, std_raw: torch.Tensor, run_tag):
    """
    cfg, device, 固定スケーラー(mean/std) を束縛した observe_func を返す。
    online_bo_loop にはこの戻り値を渡す。
    """
    template_cfg = _load_template_config()
    template_workdir = template_cfg["workflow"]["workdir"]

    def observe_func(picked_row: pd.Series) -> torch.Tensor:
        global _job_counter
        _job_counter += 1

        # --- BO が選んだ SMILES / ID ---
        smiles = str(picked_row[cfg.data.smiles_col])
        # smiles = "O"
        mol_id = picked_row[cfg.data.id_col]

        # ID をベースに安全な名前に
        id_tag = _make_compound_name(picked_row, cfg)   # 例: "52" → "52" or "ID52" は好みで
        iter_tag = f"iter_{_job_counter:03d}"           # iter_001, iter_002, ...

        print(f"[OBSERVE] iter={iter_tag} id={id_tag} smiles={smiles}")

        # --- JSON 構築 ---
        job_cfg = json.loads(json.dumps(template_cfg))  # deep copy

        job_cfg["input"]["smiles"] = smiles
        # compound_name は ID ベースにしておく（なくてもいいがわかりやすいので）
        job_cfg["input"]["compound_name"] = id_tag

        # workdir: results/result_901_WorkFlowSingle/iter_xxx/IDxx
        workdir_rel = _build_job_workdir_rel(template_workdir, iter_tag, id_tag, run_tag)
        job_cfg["workflow"]["workdir"] = workdir_rel

        # JSON の保存場所も iter/ID 構造に
        job_cfg_path = CALC_ROOT / "bo_jobs" / run_tag / iter_tag / id_tag / "setting.json"
        # --- 計算実行 ---
        _run_single_job(job_cfg, job_cfg_path)

        # --- 結果読み込み ---
        states_csv = _find_states_csv(workdir_rel)
        print(f"[OBSERVE] states_csv: {states_csv}")

        e_ev, f_val = _read_s1_energy_and_f(states_csv)

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
