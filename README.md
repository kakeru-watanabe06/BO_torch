# BO_torch

Bayesian Optimization (BO) を `botorch` で実行するためのプロジェクトです。

- `offline`: 既知データセット上で候補を順次選ぶ検証モード
- `online`: 外部計算を呼び出して観測値を取得する実運用モード

## 1. セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. よく使う実行コマンド

### 2-1. Offline BO（JSON設定）

```bash
python scripts/run_offline_from_config.py --config configs/benchmark/s1_qehvi_target_distance.json
```

### 2-2. Online BO（JSON設定）

```bash
python scripts/run_online_bo_from_config.py --config configs/online/config.json
```

### 2-3. Offline BO（CLI直接指定）

```bash
python scripts/run_offline_bo.py \
  --train data/train/s1_summary_train_dataset.xlsx \
  --all data/train/s1_summary_scaled_FP.xlsx \
  --id_col Folder \
  --x_cols FP_0 FP_1 FP_2 \
  --y_cols S1_energy_eV_scaled Oscillator_strength_scaled
```

## 3. 出力ファイル

各実行で `results/.../<timestamp>_<tag>/` が作成され、以下を保存します。

- `history.xlsx`: 各反復の選択結果と観測値
- `metrics.xlsx`: HV / HV_gap / ScalarBest（設定に応じて）
- `config.json`: 実行時の設定スナップショット

Online BO では、途中経過として `history_iterXXX.xlsx` も同じフォルダに保存されます。

## 4. ディレクトリ構成

- `src/bo_tool/`
- `src/bo_tool/config.py`: 設定ロード
- `src/bo_tool/data_utils.py`: データ読み込み
- `src/bo_tool/models.py`: GPモデル構築
- `src/bo_tool/objectives.py`: 目的変換
- `src/bo_tool/acquisition.py`: 獲得関数構築
- `src/bo_tool/bo_loop_offline.py`: offlineループ
- `src/bo_tool/bo_loop_online.py`: onlineループ
- `src/bo_tool/metrics.py`: 指標計算（HV, LOOCVなど）
- `configs/`: 実験設定JSON
- `scripts/`: 実行エントリポイント

## 5. 設定ファイル

JSONの項目詳細は以下を参照してください。

- `configs/config_reference.md`
