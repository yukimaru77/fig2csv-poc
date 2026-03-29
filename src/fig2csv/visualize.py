from __future__ import annotations

import cv2
import numpy as np

from .schemas import CalibrationPoint, ComparisonRecord, OCRTick


def _draw_cross(img: np.ndarray, x: int, y: int, color: tuple[int, int, int], size: int = 6, thickness: int = 2) -> None:
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)


def draw_summary_overlay(
    image: np.ndarray,
    ocr_ticks: list[OCRTick],
    ocr_points: list[CalibrationPoint],
    vision_points: list[CalibrationPoint],
    comparisons: list[ComparisonRecord],
) -> np.ndarray:
    canvas = image.copy()

    for tick in ocr_ticks:
        pts = np.array([[(int(x), int(y)) for x, y in tick.bbox]], dtype=np.int32)
        cv2.polylines(canvas, pts, isClosed=True, color=(0, 200, 255), thickness=2)
        cv2.putText(
            canvas,
            tick.text,
            (int(tick.center.x) + 4, int(tick.center.y) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

    for point in vision_points:
        x, y = int(point.pixel.x), int(point.pixel.y)
        _draw_cross(canvas, x, y, (255, 0, 255), size=7, thickness=2)
        cv2.putText(canvas, f"V:{point.axis_name}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

    for point in ocr_points:
        x, y = int(point.pixel.x), int(point.pixel.y)
        cv2.circle(canvas, (x, y), 6, (0, 255, 255), -1)
        label = f"O:{point.value}" if point.value is not None else f"O:{point.axis_name}"
        cv2.putText(canvas, label, (x + 6, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    for comp in comparisons:
        if comp.vision_pixel is not None and comp.ocr_pixel is not None:
            p1 = (int(comp.vision_pixel.x), int(comp.vision_pixel.y))
            p2 = (int(comp.ocr_pixel.x), int(comp.ocr_pixel.y))
            cv2.line(canvas, p1, p2, (0, 128, 255), 1)
            mx, my = int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)
            if comp.euclidean is not None:
                cv2.putText(canvas, f"{comp.euclidean:.1f}px", (mx + 3, my - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 255), 1)

    legend_lines = [
        "Legend:",
        "Green/Blue line: detected axes",
        "Orange boxes: OCR text boxes",
        "Magenta cross: vision calibration points",
        "Yellow dot: OCR calibration points",
        "Orange line: matched displacement",
    ]
    x0, y0 = 20, 25
    for i, text in enumerate(legend_lines):
        cv2.putText(canvas, text, (x0, y0 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, text, (x0, y0 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas
