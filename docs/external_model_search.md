# 既存モデル探索メモ

## 目的
キャリブレーションポイントを直接検出する既存の深層学習モデル、またはそれに近い既存モデルを探して実際に試す。

## 探索キーワード例
- Hugging Face chart calibration point detection model
- chart parsing keypoint detection axis tick detection
- plot digitization deep learning tick detection
- ChartOCR line chart keypoint

## 見つかった候補
### 1. luke-harriman/chart_object_detection (Hugging Face)
- 種別: DETR ベースの object detection
- URL: https://huggingface.co/luke-harriman/chart_object_detection
- 状況: 実際に推論実行
- 結果: サンプル画像に対して画像全体に近い単一 bbox のみを検出
- 所見: calibration point 直検出には使えない可能性が高い

### 2. zmykevin/ChartOCR (GitHub)
- URL: https://github.com/zmykevin/ChartOCR
- 特徴: line chart を含む chart data extraction の deep hybrid framework
- 備考: keypoint ベースの説明あり。ただし導入が重く、すぐ使える公開済み pretrained が見つけにくい

### 3. yanjigao49/Chart-Recognition (GitHub)
- URL: https://github.com/yanjigao49/Chart-Recognition
- 特徴: axis detector, tick label detector, OCR, RCNN ベース統合パイプライン
- 備考: tick label / axis 検出には近いが calibration point 直検出そのものではない

## 現時点の結論
- Hugging Face 上で公開されていて即推論できる public モデルとしては `luke-harriman/chart_object_detection` が見つかった
- これを実際にサンプル画像へ適用したが、calibration point 検出には十分ではなかった
- ChartOCR 系は論文・実装としては有望だが、即試せる軽量 pretrained 取得経路をさらに調査する必要がある
