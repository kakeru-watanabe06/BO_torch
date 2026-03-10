# Config Reference (`ExperimentConfig`)

`scripts/run_offline_from_config.py` / `scripts/run_online_bo_from_config.py` が読む JSON 設定仕様（現行実装ベース）です。

## Top-level

```json
{
  "data": { ... },
  "objective": { ... },
  "model": { ... },
  "bo": { ... },
  "eval": { ... },
  "output": { ... },
  "scaler": { ... }
}
```

- 必須: `data`, `objective`, `model`, `bo`, `output`
- 任意: `eval`, `scaler`

## 1. `data`

```json
{
  "data": {
    "train": "data/train/coumarine_train_31.xlsx",
    "all": "data/train/all_dataset.csv",
    "id_col": "Folder",
    "x_cols": null,
    "x_col_start": "F_3",
    "x_col_end": "R_8",
    "y_cols": ["S1_energy_eV_scaled", "Oscillator_strength_scaled"],
    "smiles_col": "SMILES"
  }
}
```

- `train`: 初期点データ
- `all`: 候補全集合
- `id_col`: 行ID
- `x_cols`: 説明変数列（`null` / `[]` 可）
- `x_col_start`, `x_col_end`: 列範囲（`int` または列名 `str`）
- `y_cols`: 目的変数列
- `smiles_col`: onlineモードで使う列

`X` 列の解決ルール:
- `x_col_start` または `x_col_end` が指定されている場合は、`all` の列から範囲抽出
- `x_col_end` を列名で指定した場合はその列を含む
- `x_col_end` を数値で指定した場合は Python スライス準拠（終端は含まない）
- 範囲指定がなければ `x_cols` をそのまま使用

## 2. `objective`

```json
{
  "objective": {
    "kind": "target_distance",
    "weights": [1.0, 1.0],
    "targets": [-3.25, 6.45],
    "power": 1.0,
    "maximize": null,
    "modes": null
  }
}
```

- `kind`: `target_distance` / `identity_multi` / `linear_scalarization` / `mixed_multi`
- `weights`: 重みベクトル
- `targets`: target系モードの目標値
- `power`: target距離の冪
- `maximize`: `identity_multi` / `mixed_multi` の符号制御
- `modes`: `mixed_multi` で次元ごとに `target` / `identity`

実装上の変換:
- `target_distance`: 各次元で `-|y-target|^power`（現状 `weights` はこの変換では使わない）
- `identity_multi`: `maximize=False` の次元だけ符号反転
- `linear_scalarization`: `sum(weights * y)` で 1 次元化
- `mixed_multi`:
- `identity` 次元は符号制御
- `target` 次元は `-|y-target|^power`（現状この変換では `weights` 不使用）

## 3. `model`

```json
{
  "model": {
    "kernel": "matern32",
    "ard": false
  }
}
```

- `kernel`: `matern12`, `matern32`, `matern52`, `tanimoto`
- `ard`: ARD使用可否（`tanimoto` では `false` 必須）

補足:
- 追加キー（例: `outcome_standardize`）があっても、現行実装では参照しない

## 4. `bo`

```json
{
  "bo": {
    "max_iters": 90,
    "mc": 512,
    "acq_type": "qucb",
    "ucb_beta": 1.0
  }
}
```

- `max_iters`: 反復回数
- `mc`: MCサンプル数
- `acq_type`: `auto`, `qei`, `qehvi`, `qucb`, `q_ucb`, `ucb`, `sucb`, `s_ucb`
- `ucb_beta`: qUCB時の β

`acq_type` の実装挙動:
- `qucb` / `q_ucb` / `ucb`: qUCB
- `sucb` / `s_ucb`: 目的次元ごとに UCB を計算し、候補間で標準化して合算
- `auto` / `qehvi` かつ `m >= 2` かつ `kind != linear_scalarization`: qEHVI
- それ以外: qEI

## 5. `eval`

```json
{
  "eval": {
    "loocv": true,
    "min_points": 5
  }
}
```

- `loocv`: 各反復で LOOCV 指標を計算するか
- `min_points`: LOOCV開始に必要な最小サンプル数

## 6. `scaler` (online)

online実行で使用します。外部計算で得た生の `y` を固定スケールに変換します。

```json
{
  "scaler": {
    "y_raw_cols": ["S1_energy_eV", "Oscillator_strength"],
    "mean": [4.239423742, 0.246500903],
    "std": [0.321735681, 0.162087901]
  }
}
```

- `run_online_bo_from_config.py` では `scaler` が必須

## 7. `output`

```json
{
  "output": {
    "outdir": "results/online_bo",
    "tag": "exp1"
  }
}
```

- `outdir`: 出力先ディレクトリ
- `tag`: 実験タグ（最終出力ディレクトリ名に付与）

## 8. サンプルJSON（ループ別）

すぐに使えるサンプルを追加しています。

- Offline ループ用: `configs/samples/offline_sample.json`
- Online ループ用: `configs/samples/online_sample.json`
- Offline + sUCB 用: `configs/samples/offline_sucb_sample.json`
- Online + sUCB 用: `configs/samples/online_sucb_sample.json`

実行例:

```bash
python scripts/run_offline_from_config.py --config configs/samples/offline_sample.json
python scripts/run_online_bo_from_config.py --config configs/samples/online_sample.json
```
