"""Wall extraction.

Walls in residential floor plans are the only content that is simultaneously
LOW-CHROMA (grey/black, unlike wood/tile fills), TONALLY COHERENT (one grey
band, unlike mixed annotations), and THICK (7px+, unlike 1px dimension lines
and text strokes). Each property alone misfires; the conjunction is robust.

Two complementary masks are produced:

- ``solid``: in-band ∧ thickness — clean poché walls at their TRUE width.
  Reliable for the building footprint (and thus scale calibration) but blind
  to thin strokes such as door sills and window lines.
- ``lines``: adaptive-threshold ∧ long horizontal/vertical runs, gated to the
  footprint — carries exactly those thin strokes, which are what seal door
  and window gaps when rooms are extracted from ``union = solid | lines``.

The grey band is auto-estimated per image. Hard-coding it is the documented
failure mode of prior art (band 80–140 vs real-world walls at grey ≈154).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MAX_CHROMA = 20  # max(B,G,R) - min(B,G,R) above this = colored fill, not wall
BAND_BELOW_PEAK = 25
BAND_ABOVE_PEAK = 30
FALLBACK_BAND = (60, 200)
MIN_CORE_HALF_WIDTH = 2.5  # px; strokes thinner than ~5px are not walls


@dataclass(frozen=True)
class WallExtraction:
    solid: np.ndarray  # uint8 {0,255}: thick in-band walls at true width
    lines: np.ndarray  # uint8 {0,255}: thin H/V structure inside the footprint
    union: np.ndarray  # solid | lines — the room-enclosure mask
    band: tuple[int, int]  # grey band actually used
    band_fallback: bool  # True when no band peak was found
    thickness_px: float  # estimated wall thickness (working px)
    footprint: tuple[int, int, int, int]  # x0, y0, x1, y1 (inclusive)


def estimate_wall_band(bgr: np.ndarray) -> tuple[int, int] | None:
    """Find the wall-grey band from the histogram of low-chroma pixels.

    The dominant low-chroma peak is the paper/background; the strongest
    remaining peak is the wall tone (grey #999 in listing plans, near-black
    in line drawings). Returns None when no secondary peak carries enough
    mass — e.g. plans whose walls are colored.
    """
    chroma = bgr.max(axis=2).astype(np.int16) - bgr.min(axis=2).astype(np.int16)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    low_chroma = gray[chroma < MAX_CHROMA]
    if low_chroma.size == 0:
        return None

    hist = np.bincount(low_chroma.ravel().astype(np.int64), minlength=256).astype(np.float64)
    hist = np.convolve(hist, np.ones(7) / 7, mode="same")
    background_peak = int(np.argmax(hist))
    hist[max(0, background_peak - 35) :] = 0  # mask background and brighter
    peak = int(np.argmax(hist))
    if hist[peak] < 0.001 * low_chroma.size:
        return None
    return max(0, peak - BAND_BELOW_PEAK), min(255, peak + BAND_ABOVE_PEAK)


def extract_walls(bgr: np.ndarray) -> WallExtraction:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    band = estimate_wall_band(bgr)
    band_fallback = band is None
    if band is None:
        band = FALLBACK_BAND

    chroma = bgr.max(axis=2).astype(np.int16) - bgr.min(axis=2).astype(np.int16)
    in_band = (
        (gray >= band[0]) & (gray <= band[1]) & (chroma < MAX_CHROMA)
    ).astype(np.uint8) * 255

    solid, thickness = _solid_walls(in_band, MIN_CORE_HALF_WIDTH)
    # Self-calibrate once: on high-res inputs walls are much thicker than the
    # 2.5px floor, and a proportional core threshold rejects proportionally
    # thicker non-wall strokes (text at 300 DPI easily exceeds 5px).
    if thickness > 12:
        solid, thickness = _solid_walls(in_band, max(MIN_CORE_HALF_WIDTH, 0.25 * thickness))

    footprint = _bbox(solid)
    lines = _thin_lines(gray, footprint)
    if footprint is None:
        footprint = _bbox(lines) or (0, 0, w - 1, h - 1)
        lines = _thin_lines(gray, footprint)
        if lines.any():
            # Thin-line/CAD plans have no solid core, but downstream wall
            # geometry still requires a positive raster stroke width.
            thickness = 1.0

    union = cv2.bitwise_or(solid, lines)
    return WallExtraction(
        solid=solid,
        lines=lines,
        union=union,
        band=band,
        band_fallback=band_fallback,
        thickness_px=thickness,
        footprint=footprint,
    )


def _solid_walls(in_band: np.ndarray, core_half_width: float) -> tuple[np.ndarray, float]:
    """Keep in-band components that contain a thick core.

    ``distanceTransform >= r`` is an exact "stroke at least 2r wide" test;
    keeping whole components that contain such cores then restores the true
    wall width (a reconstruction-by-component: measured areas would otherwise
    inherit a systematic bias from approximating width with a fixed dilate).
    """
    dist = cv2.distanceTransform(in_band, cv2.DIST_L2, 5)
    cores = dist >= core_half_width
    if not cores.any():
        return np.zeros_like(in_band), 0.0

    n_labels, labels = cv2.connectedComponents(in_band, connectivity=8)
    keep = np.zeros(n_labels, dtype=bool)
    keep[np.unique(labels[cores])] = True
    keep[0] = False
    solid = np.where(keep[labels], np.uint8(255), np.uint8(0))

    # Component-keep restores true wall width but also swallows thin in-band
    # strokes CONNECTED to walls (door sills, window frames) — which would
    # make every door read as "wall present". A small opening sheds those
    # appendages while leaving wall runs untouched; the thin strokes remain
    # available in the lines mask where they belong. The kernel is tied to
    # the minimum core width, NOT the measured thickness — a thickness
    # inflated by a filled blob (elevator core, column) must never erase
    # real walls.
    k = 2 * int(MIN_CORE_HALF_WIDTH) + 1
    solid = cv2.morphologyEx(
        solid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    )
    if not solid.any():
        return solid, 0.0

    # Thickness = 2 × the modal ridge value. Stripe centerlines dominate the
    # distance-transform ridge, so filled blobs (whose interiors would wreck
    # a percentile over all pixels) barely register.
    ridge = (dist >= cv2.dilate(dist, np.ones((3, 3))) - 0.01) & (solid > 0)
    ridge_vals = np.round(dist[ridge]).astype(np.int64)
    ridge_vals = ridge_vals[ridge_vals > 0]
    if ridge_vals.size:
        thickness = 2.0 * float(np.bincount(ridge_vals).argmax())
    else:
        thickness = 2.0 * float(np.percentile(dist[solid > 0], 90))
    return solid, min(max(thickness, 3.0), 60.0)


def _thin_lines(gray: np.ndarray, footprint: tuple[int, int, int, int] | None) -> np.ndarray:
    """Long horizontal/vertical strokes (door sills, window lines, thin walls).

    Adaptive threshold captures anything locally darker than its surround;
    the directional openings keep only runs longer than ~1/30 of the image,
    which drops text and short texture marks. Gating to the footprint drops
    the dimension lines living in the margins.

    No chroma guard here, deliberately: JPEG chroma ringing contaminates thin
    grey strokes wherever they border colored fills, so a chroma cut deletes
    real sills (measured: it costs rooms on real listing plans). Long
    high-contrast texture stripes may therefore leak in — the fallback ladder
    in rooms.py is the designed mitigation for that case.
    """
    h, w = gray.shape
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10
    )
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 30), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 30)))
    lines = cv2.bitwise_or(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel),
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel),
    )
    if footprint is not None:
        x0, y0, x1, y1 = footprint
        gated = np.zeros_like(lines)
        pad = 2
        gated[max(0, y0 - pad) : y1 + 1 + pad, max(0, x0 - pad) : x1 + 1 + pad] = lines[
            max(0, y0 - pad) : y1 + 1 + pad, max(0, x0 - pad) : x1 + 1 + pad
        ]
        lines = gated
    return lines


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
