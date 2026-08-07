"""Debug overlays. A pipeline with tuned thresholds is only trustworthy when
each stage can be eyeballed, so these images are part of the algorithm, not
an afterthought — ``parse(..., debug_dir=...)`` writes one per stage.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from roomify.walls import WallExtraction

_PALETTE = [  # BGR, dark enough to read on light plans
    (180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214),
    (189, 103, 148), (75, 86, 140), (194, 119, 227), (127, 127, 127),
    (34, 189, 188), (207, 190, 23),
]


def save(debug_dir: str | Path, name: str, image: np.ndarray) -> None:
    directory = Path(debug_dir)
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / name), image)


def walls_overlay(bgr: np.ndarray, extraction: WallExtraction) -> np.ndarray:
    out = bgr.copy()
    tint = out.copy()
    tint[extraction.solid > 0] = (0, 200, 0)
    tint[extraction.lines > 0] = (255, 80, 0)
    out = cv2.addWeighted(tint, 0.55, out, 0.45, 0)
    x0, y0, x1, y1 = extraction.footprint
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 255), 1)
    lo, hi = extraction.band
    label = f"band {lo}-{hi}{' (fallback)' if extraction.band_fallback else ''}"
    label += f"  thickness {extraction.thickness_px:.1f}px"
    cv2.putText(out, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out


def polygons_overlay(
    bgr: np.ndarray, polygons: list[np.ndarray], labels: list[str] | None = None
) -> np.ndarray:
    out = bgr.copy()
    fill = out.copy()
    for i, poly in enumerate(polygons):
        pts = np.round(poly).astype(np.int32)
        cv2.fillPoly(fill, [pts], _PALETTE[i % len(_PALETTE)])
    out = cv2.addWeighted(fill, 0.4, out, 0.6, 0)
    for i, poly in enumerate(polygons):
        pts = np.round(poly).astype(np.int32)
        color = _PALETTE[i % len(_PALETTE)]
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        text = labels[i] if labels else str(i + 1)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(out, text, (cx - 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out
