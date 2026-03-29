# ChartDete サンプル結果

## モデル
- Hugging Face: `tdsone/chartdete`
- config: `cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py`
- checkpoint: `checkpoint.pth`

## 入力
- `data/samples/sample_chart.png`

## 結論
GPU 推論は成功。
このサンプルでは、`y_axis_area` と `x_axis_area` は高信頼で検出できた。
一方で `x_tick`, `y_tick`, `tick_grouping` は低信頼だった。

## 主な検出
- `y_axis_area`: score `0.9475`
- `x_axis_area`: score `0.9228`
- `tick_grouping`: 最高 `0.0671`
- `y_tick`: 最高 `0.0114`
- `x_tick`: 最高 `0.0080`

## 解釈
- 軸領域の深層学習検出には有効
- 今回のサンプル1枚に対しては calibration point を安定に取るには tick スコアが足りない
- ただしモデル自体が `x_tick` / `y_tick` クラスを持つことは確認できたため、画像条件や閾値調整、別サンプルでは有望

## 出力ファイル
- `outputs/chartdete_sample/overlay.png`
- `outputs/chartdete_sample/raw_detections.json`
- `outputs/chartdete_sample/summary.json`
- `outputs/chartdete_sample/README.md`
