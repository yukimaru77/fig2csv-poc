from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float
    y: float


class AxisDetection(BaseModel):
    axis_name: Literal["y_axis", "z_axis"]
    p1: Point
    p2: Point
    score: float = Field(default=1.0)
    source: Literal["vision_model", "classical", "ocr"] = "classical"


class OCRTick(BaseModel):
    text: str
    confidence: float
    center: Point
    bbox: list[list[float]]


class CalibrationPoint(BaseModel):
    axis_name: Literal["y_axis", "z_axis"]
    pixel: Point
    value: str | None = None
    source: Literal["vision_model", "ocr"]


class ComparisonRecord(BaseModel):
    axis_name: Literal["y_axis", "z_axis"]
    vision_pixel: Point | None = None
    ocr_pixel: Point | None = None
    dx: float | None = None
    dy: float | None = None
    euclidean: float | None = None
