from __future__ import annotations

from collections import defaultdict

from .ocr_points import euclidean
from .schemas import CalibrationPoint, ComparisonRecord


def compare_points(
    vision_points: list[CalibrationPoint],
    ocr_points: list[CalibrationPoint],
    match_threshold: float = 40.0,
) -> list[ComparisonRecord]:
    grouped_vision = defaultdict(list)
    grouped_ocr = defaultdict(list)

    for p in vision_points:
        grouped_vision[p.axis_name].append(p)
    for p in ocr_points:
        grouped_ocr[p.axis_name].append(p)

    records: list[ComparisonRecord] = []
    for axis_name, v_points in grouped_vision.items():
        remaining = grouped_ocr.get(axis_name, []).copy()
        for v in v_points:
            best = None
            best_d = float("inf")
            for o in remaining:
                d = euclidean(v.pixel, o.pixel)
                if d < best_d:
                    best_d = d
                    best = o
            if best is not None and best_d <= match_threshold:
                remaining.remove(best)
                records.append(
                    ComparisonRecord(
                        axis_name=axis_name,
                        vision_pixel=v.pixel,
                        ocr_pixel=best.pixel,
                        dx=best.pixel.x - v.pixel.x,
                        dy=best.pixel.y - v.pixel.y,
                        euclidean=best_d,
                    )
                )
            else:
                records.append(ComparisonRecord(axis_name=axis_name, vision_pixel=v.pixel))
        for o in remaining:
            records.append(ComparisonRecord(axis_name=axis_name, ocr_pixel=o.pixel))
    return records
