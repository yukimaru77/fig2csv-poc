# ChartDete 推論メモ

## 結論
Hugging Face の `tdsone/chartdete` を `/workspace/fig2csv-poc/data/samples/sample_chart.png` に対して実行できました。GPU 推論は成功です。

ただし、このサンプルでは **高信頼で取れたのは `y_axis_area` と `x_axis_area` の2件** でした。`x_tick` / `y_tick` / `tick_grouping` も候補は出ましたが、かなり低信頼です。

## 主要結果
- `y_axis_area`: score `0.9475`
- `x_axis_area`: score `0.9228`
- `tick_grouping`: 最高 score `0.0671`（低い）
- `y_tick`: 最高 score `0.0114`（かなり低い）
- `x_tick`: 最高 score `0.0080`（かなり低い）

`score >= 0.2` で残る interested class は以下のみ:
1. `y_axis_area` `[16.19, 100.46, 94.04, 647.02]`
2. `x_axis_area` `[122.06, 660.11, 1041.02, 730.15]`

## 出力ファイル
- `raw_detections.json`: 全検出と interested class 抜粋
- `summary.json`: 要約
- `overlay.png`: 可視化画像

## 実行手順
1. 作業ディレクトリ作成
   - `/workspace/chartdete-run`
2. Hugging Face モデル取得
   - `snapshot_download("tdsone/chartdete", local_dir="hf-model")`
3. ChartDete 本体取得
   - `git clone https://github.com/tdsone/ChartDete.git /workspace/chartdete-run/ChartDete`
4. 既存 venv に old MMDetection 系を追加
   - `uv pip install --python /workspace/fig2csv-poc/.venv/bin/python --no-build-isolation "mmcv==1.7.2"`
   - `uv pip install --python /workspace/fig2csv-poc/.venv/bin/python --no-build-isolation -e /workspace/chartdete-run/ChartDete`
5. `mmcv-full` はホストでは CUDA toolkit 不足でビルド不可だったため、CUDA 13.0 devel Docker 内でビルドして同じ venv に導入
   - `docker run --rm --gpus all -v /workspace:/workspace -w /workspace/chartdete-run nvidia/cuda:13.0.0-devel-ubuntu24.04 ... /workspace/fig2csv-poc/.venv/bin/python -m pip install --no-build-isolation mmcv-full==1.7.2`
6. 推論スクリプト実行
   - `/workspace/fig2csv-poc/.venv/bin/python /workspace/chartdete-run/run_chartdete_inference.py`

## 補足
- モデル config は HF の `cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py` をそのまま使用
- checkpoint は HF の `checkpoint.pth` を使用
- 追加の config patch は不要でした
- この1枚では tick 系の信頼度が低く、軸領域検出の方が安定していました
