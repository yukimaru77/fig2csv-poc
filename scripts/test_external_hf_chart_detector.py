from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, DetrForObjectDetection

MODEL_ID = "luke-harriman/chart_object_detection"


def main() -> None:
    image_path = Path("data/samples/sample_chart.png")
    out_dir = Path("outputs/external_hf_chart_detector")
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = DetrForObjectDetection.from_pretrained(MODEL_ID)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    inputs = processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[image.height, image.width]], device=outputs.logits.device)
    results = processor.post_process_object_detection(outputs, threshold=0.2, target_sizes=target_sizes)[0]

    id2label = getattr(model.config, "id2label", None) or {}
    detections = []
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(x, 2) for x in box.tolist()]
        label_id = int(label.item())
        label_name = id2label.get(label_id, f"LABEL_{label_id}")
        score_val = float(score.item())
        detections.append({"label_id": label_id, "label": label_name, "score": score_val, "box": box})
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 4, box[1] + 4), f"{label_name}:{score_val:.2f}", fill="red")

    (out_dir / "results.json").write_text(json.dumps(detections, ensure_ascii=False, indent=2), encoding="utf-8")
    canvas.save(out_dir / "overlay.png")
    print(json.dumps({"detections": detections, "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
