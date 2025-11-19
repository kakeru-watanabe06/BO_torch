# Config Reference (ExperimentConfig)

`scripts/run_offline_from_config.py` が読む JSON 設定の仕様メモ。

トップレベル構造:

```json
{
  "data": { ... },
  "objective": { ... },
  "model": { ... },
  "bo": { ... },
  "output": { ... }
}
```

==============================

## 1. data セクション

==============================

例:

```json
{
  "data": {
    "train": "data/train/s1_summary_train_dataset.xlsx",
    "all": "data/train/s1_summary_scaled.xlsx",
    "id_col": "Folder",

    "x_cols": [],
    "x_col_start": 25,
    "x_col_end": 73,

    "y_cols": [
      "S1_energy_eV_scaled",
      "Oscillator_strength_scaled"
    ]
  }
}
```

### *学習データ関連*
- **train**: 初期 BO に使う行が入った Excel ファイルパス  
- **all** : ベンチマーク用の「既知候補空間」(train + 未使用) の Excel  
- **id_col**: 各行を一意に識別する列名 (例: Folder)

### *説明変数関連*
- **x_cols / x_col_start / x_col_end**  
  - `x_cols` が非空 → その列名リストを使用  
  - `x_cols` が空 → `all_df.columns[x_col_start : x_col_end]` を使用  
    （Python スライス：end は含まない）

### *目的変数関連*
- **y_cols**: 出力（目的値）列名のリスト  
  複数次元も対応例：  
  `["S1_energy_eV_scaled", "Oscillator_strength_scaled"]`

---

==============================

## 2. objective セクション

==============================

例:

```json
{
  "objective": {
    "kind": "target_distance",
    "targets": [-0.01, -1.02],
    "weights": [1.0, 1.0],
    "power": 2.0,
    "maximize": null
  }
}
```

### *kind（目的関数の種類）*
選べる値:
- `"target_distance"`
- `"identity_multi"`
- `"linear_scalarization"`

### *target_distance*
各次元 i について以下のスコアを最大化する：

**fᵢ(y) = − wᵢ · | yᵢ − targetᵢ |ᵖᵒʷᵉʳ**

- 目標値に近いほどスコアが大きい（最大化問題）  
- **targets**: 各次元の目標値  
- **weights**: 各次元の重み  
- **power** : 距離の冪（通常 2.0）

### *identity_multi*
- y をそのまま多目的最適化に使用  
- **maximize**: 各次元ごとの最大化/最小化設定

### *linear_scalarization*
- 線形結合で 1 次元に畳み込み  
  f(y) = Σ_i w_i * y_i  
- 単目的 (qEI) 用

※ 出力次元 m ≥ 2 かつ kind が target_distance / identity_multi のときは qEHVI。  
※ linear_scalarization または m = 1 のときは qEI。

---

==============================

## 3. model セクション

==============================

例:

```json
{
  "model": {
    "kernel": "matern32",
    "ard": true
  }
}
```

### *kernel*
- `matern32` → Matern ν=1.5  
- `matern52` → Matern ν=2.5  
- `tanimoto` → 自作 Tanimoto カーネル (0/1 fingerprint 前提, Normalize しない)

### *ard*
- Matern のとき: ARD を使うかどうか (次元ごとに別々の lengthscale)  
- tanimoto のとき: false 固定 (ARD 未対応)

---

==============================

## 4. bo セクション

==============================

例:

```json
{
  "bo": {
    "max_iters": 64,
    "mc": 512
  }
}
```

### *max_iters*
- オフライン BO の最大反復回数 (何点追加するか)

### *mc*
- qEHVI / qEI の MC サンプル数 (Sobol の sample_shape)

---

==============================

## 5. output セクション

==============================

例:

```json
{
  "output": {
    "outdir": "results/offline_bo",
    "tag": "C4_qEHVI_FP_matern32_v1"
  }
}
```

### *outdir*
- 結果を書き出すベースディレクトリ

### *tag*
- 実験のラベル (モデル・特徴量セット名など)  
  実際の出力ディレクトリは `outdir/YYYYMMDD-HHMMSS_tag/` 形式で作成される。

---

==============================

## 6. 典型例 (テンプレ)

==============================

```json
{
  "data": {
    "train": "data/train/s1_summary_train_dataset.xlsx",
    "all": "data/train/s1_summary_scaled.xlsx",
    "id_col": "Folder",

    "x_cols": [],
    "x_col_start": 25,
    "x_col_end": 73,

    "y_cols": [
      "S1_energy_eV_scaled",
      "Oscillator_strength_scaled"
    ]
  },

  "objective": {
    "kind": "target_distance",
    "targets": [-0.008576, -1.017155],
    "weights": [1.0, 1.0],
    "power": 2.0
  },

  "model": {
    "kernel": "matern32",
    "ard": true
  },

  "bo": {
    "max_iters": 64,
    "mc": 512
  },

  "output": {
    "outdir": "results/offline_bo",
    "tag": "C4_qEHVI_FP_matern32_v1"
  }
}
```
