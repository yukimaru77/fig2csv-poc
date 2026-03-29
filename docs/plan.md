# 実装計画メモ

## 目的
折れ線グラフ画像から CSV に戻す PoC の前段として、以下を行う。

1. y軸 / z軸 の検知
2. OCR からキャリブレーションポイント抽出
3. 両者の位置ズレ比較

## 方針
- まずは軸検知を古典的手法で実装し、出力スキーマを固定
- OCR は GPU 利用可能な EasyOCR を使用
- 後から物体検知モデルへ差し替える

## モデル化の見取り図
- detector: y_axis / z_axis を bbox or line segment で返す
- ocr branch: tick text + bbox を返す
- fusion: axis geometry と OCR text を対応付ける
- compare: pixel-level discrepancy を出す
