# sample1 実行結果

## 入力
- `data/samples/sample_chart.png`

## 軸検知
- y軸: x=87 付近の縦線を検出
- z軸: y=663 付近の横線を検出

## OCR
- y軸目盛り `24, 22, 20, 18, 16, 14, 12, 10` を取得
- z軸では `10` のみ誤って拾われた

## 可視化成果物
- `outputs/sample1/axes_overlay.png`: 軸検知のみの重ね描き
- `outputs/sample1/summary_overlay.png`: 軸・OCR bbox・OCR calibration point・vision calibration point・対応ズレをまとめて表示

## 所見
- 軸そのものの位置推定は PoC として十分うまくいった
- OCR からの calibration point は y軸でかなり良好
- ただし vision 側は現在「等間隔サンプリング」なので、OCR の実際の目盛りとは完全一致しない
- z軸は OCR ベース抽出ロジックがまだ弱く、文字の並び方向と近傍判定を強化する必要がある

## 外部既存モデルの試行
- Hugging Face の `luke-harriman/chart_object_detection` をサンプル画像に適用した
- 結果は `outputs/external_hf_chart_detector/overlay.png` と `outputs/external_hf_chart_detector/results.json` に保存
- 今回のサンプルでは画像全体に近い 1 bbox のみが出力され、calibration point 直検出には使えなかった

## 次の改善
1. y軸/z軸の tick mark 自体の視覚検出
2. OCR 数字列の単調性チェック
3. x軸（今回 z軸として扱っている側）の OCR grouping 強化
4. より目的に近い既存 keypoint / chart parsing モデルの継続調査
