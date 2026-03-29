from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .schemas import AxisDetection, Point


@dataclass
class LineCandidate:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1))


def _preprocess(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges


def _hough_lines(edges: np.ndarray) -> list[LineCandidate]:
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=12)
    candidates: list[LineCandidate] = []
    if raw is None:
        return candidates
    for item in raw[:, 0, :]:
        x1, y1, x2, y2 = map(int, item)
        length = math.hypot(x2 - x1, y2 - y1)
        candidates.append(LineCandidate(x1, y1, x2, y2, score=float(length)))
    return candidates


def _pick_vertical(candidates: Iterable[LineCandidate], width: int, height: int) -> LineCandidate | None:
    verticals = []
    for c in candidates:
        angle = abs(c.angle_deg)
        if 80 <= angle <= 100 and c.length > height * 0.35:
            x_mean = (c.x1 + c.x2) / 2
            left_bias = 1.0 - min(x_mean / max(width, 1), 1.0)
            verticals.append((c.score + left_bias * 50.0, c))
    if not verticals:
        return None
    verticals.sort(key=lambda t: t[0], reverse=True)
    return verticals[0][1]


def _pick_horizontal(candidates: Iterable[LineCandidate], width: int, height: int) -> LineCandidate | None:
    horizontals = []
    for c in candidates:
        angle = abs(c.angle_deg)
        if angle <= 10 and c.length > width * 0.35:
            y_mean = (c.y1 + c.y2) / 2
            lower_bias = min(y_mean / max(height, 1), 1.0)
            horizontals.append((c.score + lower_bias * 50.0, c))
    if not horizontals:
        return None
    horizontals.sort(key=lambda t: t[0], reverse=True)
    return horizontals[0][1]


def detect_axes_classical(image: np.ndarray) -> list[AxisDetection]:
    h, w = image.shape[:2]
    edges = _preprocess(image)
    lines = _hough_lines(edges)
    out: list[AxisDetection] = []

    vertical = _pick_vertical(lines, w, h)
    if vertical is not None:
        out.append(
            AxisDetection(
                axis_name="y_axis",
                p1=Point(x=vertical.x1, y=vertical.y1),
                p2=Point(x=vertical.x2, y=vertical.y2),
                score=vertical.score,
                source="classical",
            )
        )

    horizontal = _pick_horizontal(lines, w, h)
    if horizontal is not None:
        out.append(
            AxisDetection(
                axis_name="z_axis",
                p1=Point(x=horizontal.x1, y=horizontal.y1),
                p2=Point(x=horizontal.x2, y=horizontal.y2),
                score=horizontal.score,
                source="classical",
            )
        )

    return out


def render_axis_overlay(image: np.ndarray, detections: list[AxisDetection]) -> np.ndarray:
    canvas = image.copy()
    color_map = {"y_axis": (0, 255, 0), "z_axis": (255, 0, 0)}
    for det in detections:
        color = color_map[det.axis_name]
        cv2.line(
            canvas,
            (int(det.p1.x), int(det.p1.y)),
            (int(det.p2.x), int(det.p2.y)),
            color,
            3,
        )
        cv2.putText(
            canvas,
            det.axis_name,
            (int(det.p1.x) + 5, int(det.p1.y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas
