Config Reference (ExperimentConfig)

scripts/run_offline_from_config.py が読む JSON 設定の仕様メモ。

トップレベル構造:

{
"data": { ... },
"objective": { ... },
"model": { ... },
"bo": { ... },
"output": { ... }
}

==============================

1. data セクション

==============================

例:

"data": {
"train": "data/train/s1_summary_train_dataset.xlsx",
"all": "data/train/s1_summary_scaled.xlsx",
"id_col": "Folder",

"x_cols": [],
"x_col_start": 25,
"x_col_end": 73,

"y_cols": ["S1_energy_eV_scaled", "Oscillator_strength_scaled"]
}

● train: 初期 BO に使う行が入った Excel ファイルパス
● all : ベンチマーク用の「既知候補空間」(train + 未使用) の Excel
● id_col: 各行を一意に識別する列名 (例: Folder)

● x_cols / x_col_start / x_col_end:

説明変数(特徴量) の列をどう選ぶか

x_cols が非空 → その列名リストをそのまま使用

x_cols が空 → all_df.columns[x_col_start:x_col_end] を使用 (Python のスライス, end は含まない)

● y_cols:

出力(目的値) の列名のリスト

長さ m = 目的の次元数

例: ["S1_energy_eV_scaled", "Oscillator_strength_scaled"]

==============================

2. objective セクション

==============================

例:

"objective": {
"kind": "target_distance",
"targets": [-0.01, -1.02],
"weights": [1.0, 1.0],
"power": 2.0,
"maximize": null
}

● kind (文字列):

"target_distance"

"identity_multi"

"linear_scalarization"

● target_distance:

各次元 i について
f_i(y) = - w_i * | y_i - target_i |^power

目標値に近いほどスコアが大きくなる (最大化問題)

targets: 目標値のリスト (長さ m)

weights: 各次元の重み (長さ m)

power : 距離の冪 (普通は 2.0)

● identity_multi:

y をほぼそのまま多目的最適化に使う

maximize: 各次元ごとに true/false

true → そのまま最大化

false → 符号反転して最大化に変換 (本来は最小化したい)

● linear_scalarization:

線形結合で 1 次元に畳み込む
f(y) = Σ_i w_i * y_i

一目的 (qEI) 用

※ 出力次元 m >= 2 かつ kind が target_distance / identity_multi のときは qEHVI が使われる。
※ linear_scalarization または m = 1 のときは qEI。

==============================

3. model セクション

==============================

例:

"model": {
"kernel": "matern32",
"ard": true
}

● kernel:

"matern32" → Matern ν=1.5

"matern52" → Matern ν=2.5

"tanimoto" → 自作 Tanimoto カーネル (0/1 fingerprint 前提, Normalize しない)

● ard (bool):

Matern のとき: ARD を使うかどうか (次元ごとに別々の lengthscale)

tanimoto のとき: false 固定 (ARD は未対応)

==============================

4. bo セクション

==============================

例:

"bo": {
"max_iters": 64,
"mc": 512
}

● max_iters:

オフライン BO の最大反復回数 (何点追加するか)

● mc:

qEHVI / qEI の MC サンプル数 (Sobol の sample_shape)

==============================

5. output セクション

==============================

例:

"output": {
"outdir": "results/offline_bo",
"tag": "C4_qEHVI_FP_matern32_v1"
}

● outdir:

結果を書き出すベースディレクトリ

● tag:

実験のラベル (モデル・特徴量セット名など)

実際の出力ディレクトリは
outdir/YYYYMMDD-HHMMSS_tag/
の形式で作成される

==============================

6. 典型例 (テンプレ)

==============================

{
"data": {
"train": "data/train/s1_summary_train_dataset.xlsx",
"all": "data/train/s1_summary_scaled.xlsx",
"id_col": "Folder",

"x_cols": [],
"x_col_start": 25,
"x_col_end": 73,

"y_cols": ["S1_energy_eV_scaled", "Oscillator_strength_scaled"]


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