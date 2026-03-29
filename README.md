# fig2csv-poc

折れ線グラフ画像から CSV 復元に向けた PoC です。まずは以下を対象にしています。

- y軸 と z軸 の検知
- OCR による目盛り文字の抽出
- 検知ベースのキャリブレーション点候補と OCR ベース候補のズレ比較

## 現段階の構成

### 1. 軸検知
現状は **PoC の最初の一歩**として、OpenCV ベースの直線検出で y軸 / z軸 を推定します。
これは将来的に GPU 上で動かす物体検知モデルへ差し替えやすいよう、出力スキーマを共通化しています。

### 2. OCR ベースのキャリブレーション点
EasyOCR を GPU で動かし、数値っぽい文字列を抽出します。OpenCV は headless 版を使います。
その文字の中心が、検出された y軸 / z軸 近傍にある場合にキャリブレーション点候補とみなします。

### 3. ズレ比較
- vision 側: 軸上に等間隔な仮想キャリブレーション点を配置
- OCR 側: 文字位置に基づくキャリブレーション点を作成
- 両者の最近傍対応をとり、ピクセル差を出力

## セットアップ

```bash
cd /workspace/fig2csv-poc
uv sync
```

### GPU 版 PyTorch について
この PoC では OCR に EasyOCR を使っているため、内部で PyTorch が必要です。
環境によっては `uv sync` だけでは GPU 版 torch が入らないことがあります。
その場合は CUDA 対応 torch を別途追加してください。

## 実行例

```bash
cd /workspace/fig2csv-poc
uv run python scripts/make_sample_chart.py
uv run python scripts/run_pipeline.py data/samples/sample_chart.png --output-dir outputs/sample1
```

## 出力

- `outputs/.../axes_overlay.png` : 検出された軸の重ね描き画像
- `outputs/.../summary_overlay.png` : 軸・OCR bbox・calibration point・対応ズレをまとめて可視化した画像
- `outputs/.../result.json` : OCR, 軸, キャリブレーション点, 比較結果

## 現在の成果

- GPU 上で EasyOCR が動作することを確認
- y軸 / z軸 の検知コードを実装
- OCR による calibration point 候補抽出を実装
- 両者のピクセル差比較を JSON で保存
- サンプル画像 `data/samples/sample_chart.png` に対する実行結果を `docs/results_sample1.md` に整理

## 制約

- 現状の軸検知は古典的手法であり、学習済み GPU 検知モデルではない
- vision 側の calibration point はまだ等間隔の仮想点
- z軸 OCR 抽出はまだ弱い

## 次の拡張候補

1. y軸 / z軸 の教師データを作り、YOLO 系で GPU 学習
2. 軸交点・目盛り線・凡例の検出追加
3. OCR 文字列の値整合性チェック
4. 折れ線自体の抽出と pixel-to-value 変換
