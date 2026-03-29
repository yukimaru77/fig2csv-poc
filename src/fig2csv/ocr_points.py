from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np

from .schemas import AxisDetection, CalibrationPoint, OCRTick, Point


NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")


def is_numeric_text(text: str) -> bool:
    return bool(NUMERIC_RE.match(text.strip().replace(",", "")))


def bbox_center(bbox: list[list[float]]) -> Point:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return Point(x=float(sum(xs) / len(xs)), y=float(sum(ys) / len(ys)))


def to_ocr_ticks(results: Iterable) -> list[OCRTick]:
    ticks: list[OCRTick] = []
    for bbox, text, confidence in results:
        center = bbox_center(bbox)
        ticks.append(
            OCRTick(
                text=text,
                confidence=float(confidence),
                center=center,
                bbox=[[float(x), float(y)] for x, y in bbox],
            )
        )
    return ticks


def extract_calibration_points_from_ocr(
    ocr_ticks: list[OCRTick], axes: list[AxisDetection], distance_threshold: float = 32.0
) -> list[CalibrationPoint]:
    out: list[CalibrationPoint] = []
    axis_map = {a.axis_name: a for a in axes}

    for tick in ocr_ticks:
        if tick.confidence < 0.2 or not is_numeric_text(tick.text):
            continue

        if "y_axis" in axis_map:
            y_axis = axis_map["y_axis"]
            axis_x = (y_axis.p1.x + y_axis.p2.x) / 2
            dx = abs(tick.center.x - axis_x)
            if dx <= distance_threshold:
                out.append(
                    CalibrationPoint(
                        axis_name="y_axis",
                        pixel=Point(x=axis_x, y=tick.center.y),
                        value=tick.text,
                        source="ocr",
                    )
                )

        if "z_axis" in axis_map:
            z_axis = axis_map["z_axis"]
            axis_y = (z_axis.p1.y + z_axis.p2.y) / 2
            dy = abs(tick.center.y - axis_y)
            if dy <= distance_threshold:
                out.append(
                    CalibrationPoint(
                        axis_name="z_axis",
                        pixel=Point(x=tick.center.x, y=axis_y),
                        value=tick.text,
                        source="ocr",
                    )
                )

    return out


def synthesize_vision_calibration_points(axes: list[AxisDetection], tick_count: int = 6) -> list[CalibrationPoint]:
    out: list[CalibrationPoint] = []
    for axis in axes:
        if axis.axis_name == "y_axis":
            x = (axis.p1.x + axis.p2.x) / 2
            y1, y2 = sorted([axis.p1.y, axis.p2.y])
            ys = np.linspace(y1, y2, tick_count)
            for y in ys:
                out.append(
                    CalibrationPoint(
                        axis_name="y_axis",
                        pixel=Point(x=x, y=float(y)),
                        value=None,
                        source="vision_model",
                    )
                )
        elif axis.axis_name == "z_axis":
            y = (axis.p1.y + axis.p2.y) / 2
            x1, x2 = sorted([axis.p1.x, axis.p2.x])
            xs = np.linspace(x1, x2, tick_count)
            for x in xs:
                out.append(
                    CalibrationPoint(
                        axis_name="z_axis",
                        pixel=Point(x=float(x), y=y),
                        value=None,
                        source="vision_model",
                    )
                )
    return out


def euclidean(p1: Point, p2: Point) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)
