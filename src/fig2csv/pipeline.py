from __future__ import annotations

import json
from pathlib import Path

import cv2
import easyocr
import numpy as np

from .compare import compare_points
from .detectors import detect_axes_classical, render_axis_overlay
from .ocr_points import extract_calibration_points_from_ocr, synthesize_vision_calibration_points, to_ocr_ticks
from .visualize import draw_summary_overlay


def run_pipeline(image_path: str, output_dir: str = "outputs") -> dict:
    image_path = str(image_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    axes = detect_axes_classical(image)
    overlay = render_axis_overlay(image, axes)
    overlay_path = output_root / "axes_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    reader = easyocr.Reader(["en"], gpu=True)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ocr_raw = reader.readtext(rgb)
    ocr_ticks = to_ocr_ticks(ocr_raw)
    ocr_points = extract_calibration_points_from_ocr(ocr_ticks, axes)
    vision_points = synthesize_vision_calibration_points(axes)
    comparisons = compare_points(vision_points, ocr_points)

    summary_overlay = draw_summary_overlay(overlay, ocr_ticks, ocr_points, vision_points, comparisons)
    summary_overlay_path = output_root / "summary_overlay.png"
    cv2.imwrite(str(summary_overlay_path), summary_overlay)

    result = {
        "image_path": image_path,
        "axes": [a.model_dump() for a in axes],
        "ocr_ticks": [t.model_dump() for t in ocr_ticks],
        "ocr_calibration_points": [p.model_dump() for p in ocr_points],
        "vision_calibration_points": [p.model_dump() for p in vision_points],
        "comparisons": [c.model_dump() for c in comparisons],
        "artifacts": {
            "axes_overlay": str(overlay_path),
            "summary_overlay": str(summary_overlay_path),
        },
    }

    result_path = output_root / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
