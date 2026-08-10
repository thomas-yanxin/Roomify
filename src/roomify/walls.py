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
# Walls are ink, printed to contrast with the paper; room fills, balcony
# floors, and glazing tints sit near the paper tone (measured bg-15..bg-55
# across the corpus). Texture/JPEG noise fragments those light fills into
# stroke-sized chunks that defeat the per-component stroke gate, and a fill
# admitted into the mask erases the room it covers — so no band may come
# closer to the background tone than this. Thin light strokes that matter
# (glazing, sills) are sealed by the ``lines`` mask, which needs no band.
INK_CLEARANCE = 60
FALLBACK_BAND = (60, 200)
MIN_CORE_HALF_WIDTH = 2.5  # px; strokes thinner than ~5px are not walls


@dataclass(frozen=True)
class WallExtraction:
    solid: np.ndarray  # uint8 {0,255}: thick in-band walls at true width
    lines: np.ndarray  # uint8 {0,255}: thin H/V structure inside the footprint
    union: np.ndarray  # solid | lines — the room-enclosure mask
    band: tuple[int, int]  # primary wall band (strongest stroke score)
    bands: list[tuple[int, int]]  # every band used (multi-tone walls)
    band_fallback: bool  # True when no band peak was found
    thickness_px: float  # estimated wall thickness (working px)
    footprint: tuple[int, int, int, int]  # x0, y0, x1, y1 (inclusive)
    # Non-rectilinear wall directions carrying real mass (diamond wings).
    # Room sealing repeats its directional closes at these angles.
    angles: tuple[float, ...] = ()
    # Dashed zone dividers (玄关/走廊 sub-zone boundaries): drawn evidence
    # for functional splits, sealed like walls but tagged on the rooms they
    # bound. None when the plan draws no dashes.
    zones: np.ndarray | None = None


def estimate_wall_bands(bgr: np.ndarray) -> list[tuple[int, int]]:
    """Find the wall-tone band(s) from the histogram of low-chroma pixels.

    The dominant low-chroma peak is the paper/background. Every remaining
    peak is a candidate wall tone — plans routinely draw load-bearing walls
    black AND partition walls grey — but fills (balconies, annotations) also
    peak, so histogram mass alone picks the wrong tone. Walls are STROKES:
    each candidate band is scored by its stroke-core mass (pixels whose
    distance transform sits in the wall half-width range); big fills score
    only along their rims and drop out. All bands scoring at least 8% of the
    best are returned, best first. Empty when no candidate carries stroke
    mass — e.g. plans whose walls are colored.
    """
    chroma = bgr.max(axis=2).astype(np.int16) - bgr.min(axis=2).astype(np.int16)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    low_chroma_mask = chroma < MAX_CHROMA
    low_chroma = gray[low_chroma_mask]
    if low_chroma.size == 0:
        return []

    hist = np.bincount(low_chroma.ravel().astype(np.int64), minlength=256).astype(np.float64)
    hist = np.convolve(hist, np.ones(7) / 7, mode="same")
    background_peak = int(np.argmax(hist))
    ink_ceiling = max(0, background_peak - INK_CLEARANCE)
    hist[ink_ceiling:] = 0  # mask the background and everything fill-toned

    scored: list[tuple[float, tuple[int, int]]] = []
    while len(scored) < 4:
        peak = int(np.argmax(hist))
        # A loose noise floor only: the stroke-core score below is what
        # actually separates wall tones from junk, and thin partition walls
        # can carry little histogram mass.
        if hist[peak] < 0.0002 * low_chroma.size:
            break
        band = (
            max(0, peak - BAND_BELOW_PEAK),
            min(255, peak + BAND_ABOVE_PEAK, ink_ceiling - 1),
        )
        hist[band[0] : band[1] + 1] = 0  # consume the peak
        in_band = (
            (gray >= band[0]) & (gray <= band[1]) & low_chroma_mask
        ).astype(np.uint8) * 255
        stroke_core = _stroke_core_mass(in_band)
        if stroke_core > 0:
            scored.append((stroke_core, band))
    if not scored:
        return []
    # Junk bands (fills, text, anti-aliasing halos) score ~zero strokes, so
    # this relative cut only guards against marginal noise — thin partition
    # walls legitimately carry a small fraction of the main walls' mass.
    best = max(score for score, _ in scored)
    return [band for score, band in sorted(scored, reverse=True) if score >= 0.08 * best]


def _stroke_core_mass(in_band: np.ndarray) -> float:
    """Length-weighted medial-axis mass of wall-width strokes.

    Long walls retain a ridge along their run even when they are thick or
    touch a column. A filled blob has only a few very deep ridge samples;
    weighting ridge pixels by inverse half-width prevents that blob from
    outvoting a wall without rejecting the wall's whole connected component.
    """
    dist = cv2.distanceTransform(in_band, cv2.DIST_L2, 5)
    ridge = dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)) - 0.01
    core_ridge = ridge & (dist >= MIN_CORE_HALF_WIDTH)
    if not core_ridge.any():
        return 0.0
    return float(np.sum(1.0 / dist[core_ridge]))


def _segment_angle_mass(binary: np.ndarray) -> np.ndarray:
    """Length-weighted histogram of long-stroke directions in a binary mask,
    folded to (-45, 45] degrees (wall families repeat every 90°). 1° bins,
    index i ↔ angle i - 44."""
    h, w = binary.shape
    segs = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 360,
        threshold=60,
        minLineLength=max(40, min(h, w) // 12),
        maxLineGap=4,
    )
    hist = np.zeros(90)
    if segs is None:
        return hist
    for x1, y1, x2, y2 in segs.reshape(-1, 4):
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        fold = (angle + 45.0) % 90.0 - 45.0  # (-45, 45]
        length = float(np.hypot(x2 - x1, y2 - y1))
        hist[int(round(fold)) + 44] += length
    return hist


def estimate_plan_rotation(bgr: np.ndarray) -> float:
    """Global plan rotation in degrees, 0.0 for rectilinear plans.

    Whole-sheet rotations (decorative listing exports) put nearly ALL
    structural mass off-axis; mixed-wing plans (a 45° diamond wing on a
    rectilinear body) keep a rectilinear majority and must NOT derotate —
    their diagonals are handled per-angle during room sealing instead.
    The returned angle is the strongest single wall family, not a blend:
    irregular units carry several off-axis families and the residual ones
    are re-detected as secondary angles after derotation.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10
    )
    hist = _segment_angle_mass(binary)
    total = hist.sum()
    if total <= 0:
        return 0.0
    off_axis = total - hist[40:49].sum()  # |angle| ≤ 4° counts as axis-aligned
    if off_axis < 0.55 * total:
        return 0.0
    smoothed = np.convolve(hist, np.ones(3), mode="same")
    smoothed[40:49] = 0.0  # the strongest OFF-AXIS family, axis bins can't win
    peak = int(np.argmax(smoothed))
    if smoothed[peak] <= 0:
        return 0.0
    lo, hi = max(0, peak - 1), min(90, peak + 2)
    angles = np.arange(90) - 44.0
    return float(np.average(angles[lo:hi], weights=np.maximum(hist[lo:hi], 1e-9)))


def _secondary_angles(union: np.ndarray) -> tuple[float, ...]:
    """Off-axis directions worth sealing along (e.g. a 45° diamond wing):
    at least one locally long run and at least 8° off both axes. These are
    candidates only; ``extract_walls`` still requires substantial wall-tone
    overlap before adding their recovered strokes to the wall mask."""
    hist = _segment_angle_mass(union)
    if hist.sum() <= 0:
        return ()
    found: list[float] = []
    smoothed = np.convolve(hist, np.ones(3), mode="same")
    min_mass = float(max(40, min(union.shape) // 12))
    for idx in np.argsort(smoothed)[::-1]:
        angle = int(idx) - 44
        # Below ~18° a rotated 30px line kernel still matches ordinary
        # 10px-thick AXIS walls (drift 30·tanθ < thickness), so near-axis
        # candidates self-confirm off real rectilinear structure and then
        # merge rooms via rotated closes. Genuine wings run 30-45°.
        if abs(angle) < 15:
            continue
        if smoothed[idx] < min_mass:
            break
        if all(abs(angle - a) > 6 for a in found):
            found.append(float(angle))
        if len(found) >= 2:
            break
    return tuple(found)


def extract_walls(bgr: np.ndarray) -> WallExtraction:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    bands = estimate_wall_bands(bgr)
    band_fallback = not bands
    if not bands:
        bands = [FALLBACK_BAND]

    chroma = bgr.max(axis=2).astype(np.int16) - bgr.min(axis=2).astype(np.int16)
    # Keep exactly the independently qualified wall-tone bands. Filling the
    # gaps between them would admit an arbitrary room fill whose tone happens
    # to sit between black load-bearing walls and grey partitions.
    low_chroma = chroma < MAX_CHROMA
    in_band = np.zeros_like(gray)
    for lo, hi in bands:
        in_band[(gray >= lo) & (gray <= hi) & low_chroma] = 255

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

    # Diamond wings: raw strokes may suggest an angle, but a single long
    # annotation must not become a wall just because Hough sees it. Accept a
    # recovered family only when it substantially overlaps an independently
    # selected wall tone.
    candidate_angles = _secondary_angles(_stroke_binary(gray, footprint))
    wall_tone_neighborhood = cv2.dilate(in_band, np.ones((3, 3), np.uint8))
    accepted_angles: list[float] = []
    for angle in candidate_angles:
        recovered = _thin_lines(gray, footprint, (angle,))
        added = cv2.bitwise_and(recovered, cv2.bitwise_not(lines))
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(added)
        long_runs = [
            label
            for label in range(1, n_labels)
            if stats[label, cv2.CC_STAT_AREA] >= 40
        ]
        if not long_runs:
            continue
        keep = np.where(np.isin(labels, long_runs), np.uint8(255), np.uint8(0))
        tone_overlap = cv2.bitwise_and(keep, wall_tone_neighborhood)
        keep_mass = cv2.countNonZero(keep)
        tone_mass = cv2.countNonZero(tone_overlap)
        tone_qualified = tone_mass >= 40 and tone_mass >= 0.3 * keep_mass
        if not tone_qualified:
            continue
        lines = cv2.bitwise_or(lines, keep)
        accepted_angles.append(angle)
    angles = tuple(accepted_angles)

    union = cv2.bitwise_or(solid, lines)
    return WallExtraction(
        solid=solid,
        lines=lines,
        union=union,
        band=bands[0],
        bands=bands,
        band_fallback=band_fallback,
        thickness_px=thickness,
        footprint=footprint,
        angles=angles,
        zones=_zone_lines(_stroke_binary(gray, footprint), union, footprint, angles),
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


def _zone_lines(
    binary: np.ndarray,
    structure: np.ndarray,
    footprint: tuple[int, int, int, int] | None,
    angles: tuple[float, ...] = (),
) -> np.ndarray | None:
    """Dashed zone dividers at any wall angle (see ``_zone_lines_axis``)."""
    zones = _zone_lines_axis(binary, structure)
    h, w = binary.shape
    for angle in angles:
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
        m[0, 2] += nw / 2.0 - w / 2.0
        m[1, 2] += nh / 2.0 - h / 2.0
        rot_b = cv2.warpAffine(binary, m, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
        rot_s = cv2.warpAffine(
            structure, m, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0
        )
        z = _zone_lines_axis(rot_b, rot_s)
        if z is None:
            continue
        z_back = cv2.warpAffine(
            z, cv2.invertAffineTransform(m), (w, h), flags=cv2.INTER_NEAREST, borderValue=0
        )
        zones = z_back if zones is None else cv2.bitwise_or(zones, z_back)
    return zones


def _zone_lines_axis(binary: np.ndarray, structure: np.ndarray) -> np.ndarray | None:
    """Dashed zone dividers: thin dash runs fused into straight lines.

    Listing plans split open spaces (玄关/走廊/餐厅) with dashed lines and
    print a per-zone area — the dashes are drawn evidence of a boundary, so
    sealing along them aligns measured geometry with the printed truth.
    Discriminating them from label text (which also fuses into short rows):
    a zone divider spans structure-to-structure, so both endpoints must
    anchor on the wall mask; a floating text row anchors nowhere.
    """
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    thin = np.where(dist <= 2.0, binary, np.uint8(0))
    h, w = binary.shape
    length = max(40, min(h, w) // 16)
    anchor = cv2.dilate(structure, np.ones((13, 13), np.uint8))
    zones = np.zeros_like(binary)
    found = False
    # 3px-tall/wide join: rotated-frame dashes carry ±1px resampling
    # stair-steps that a single-row kernel cannot bridge
    for join, run in (((9, 3), (length, 1)), ((3, 9), (1, length))):
        fused = cv2.morphologyEx(
            thin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, join)
        )
        runs = cv2.morphologyEx(
            fused, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, run)
        )
        # only dash-BUILT runs: continuous strokes already live in ``lines``
        runs = cv2.bitwise_and(runs, cv2.bitwise_not(structure))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(runs, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if max(bw, bh) < length:
                continue
            component = labels[y : y + bh, x : x + bw] == i
            comp_px = int(component.sum())
            # dash-ness: a divider is roughly half gaps before the join
            # close; continuous strokes (fused wood grain) score ~1.0
            pre = thin[y : y + bh, x : x + bw][component]
            dashness = float(np.count_nonzero(pre)) / max(1, comp_px)
            if not 0.25 <= dashness <= 0.85:
                continue
            # isolation: grain comes in parallel packs — a real divider has
            # no thin siblings in a band around it
            pad = 8
            yl, yh = max(0, y - (pad if bw >= bh else 0)), y + bh + (pad if bw >= bh else 0)
            xl, xh = max(0, x - (0 if bw >= bh else pad)), x + bw + (0 if bw >= bh else pad)
            band_px = int(np.count_nonzero(thin[yl:yh, xl:xh]))
            if band_px > 1.8 * max(1, int(np.count_nonzero(pre))):
                continue
            ys, xs = np.nonzero(component)
            if bw >= bh:  # horizontal run: anchor both x-extremes
                p1 = (x + int(xs.min()), y + int(ys[np.argmin(xs)]))
                p2 = (x + int(xs.max()), y + int(ys[np.argmax(xs)]))
            else:
                p1 = (x + int(xs[np.argmin(ys)]), y + int(ys.min()))
                p2 = (x + int(xs[np.argmax(ys)]), y + int(ys.max()))
            if anchor[p1[1], p1[0]] and anchor[p2[1], p2[0]]:
                zones[labels == i] = 255
                found = True
    return zones if found else None


def _stroke_binary(
    gray: np.ndarray, footprint: tuple[int, int, int, int] | None
) -> np.ndarray:
    """Adaptive-threshold stroke binary, gated to the wall footprint (drops
    the dimension chains and margin annotations living outside it)."""
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10
    )
    if footprint is not None:
        x0, y0, x1, y1 = footprint
        gated = np.zeros_like(binary)
        pad = 2
        gated[max(0, y0 - pad) : y1 + 1 + pad, max(0, x0 - pad) : x1 + 1 + pad] = binary[
            max(0, y0 - pad) : y1 + 1 + pad, max(0, x0 - pad) : x1 + 1 + pad
        ]
        binary = gated
    return binary


def _thin_lines(
    gray: np.ndarray,
    footprint: tuple[int, int, int, int] | None,
    angles: tuple[float, ...] = (),
) -> np.ndarray:
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
    binary = _stroke_binary(gray, footprint)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 30), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 30)))
    lines = cv2.bitwise_or(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel),
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel),
    )
    # Diagonal wall families: run the same long-run opening in each rotated
    # frame. Thin diagonal walls anti-alias too slim for the solid mask's
    # thickness test, so this is their only route into the enclosure union —
    # and an exact-45° line kernel would miss a 43° family entirely.
    for angle in angles:
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
        m[0, 2] += nw / 2.0 - w / 2.0
        m[1, 2] += nh / 2.0 - h / 2.0
        rot = cv2.warpAffine(binary, m, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
        rk_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, nw // 30), 1))
        rk_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, nh // 30)))
        # Pre-close along the run direction: rotation resampling plus glazing
        # mullions chop thin diagonal strokes into sub-length pieces; joining
        # ≤9px breaks lets the length gate judge the actual run. Axis passes
        # need no such help (no resampling) and must not get it — closing
        # would fuse label glyphs into fake runs there.
        join_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        join_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        rot_lines = cv2.bitwise_or(
            cv2.morphologyEx(
                cv2.morphologyEx(rot, cv2.MORPH_CLOSE, join_h), cv2.MORPH_OPEN, rk_h
            ),
            cv2.morphologyEx(
                cv2.morphologyEx(rot, cv2.MORPH_CLOSE, join_v), cv2.MORPH_OPEN, rk_v
            ),
        )
        lines = cv2.bitwise_or(
            lines,
            cv2.warpAffine(
                rot_lines,
                cv2.invertAffineTransform(m),
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            ),
        )
    return lines


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
