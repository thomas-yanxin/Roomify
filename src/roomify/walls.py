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
    # Door furniture (swing arcs + door leaves): thin strokes hanging off
    # the wall network. Sill-less arc-style plans seal their doorways with
    # exactly these strokes — the leaf+arc chain connects jamb to jamb.
    door_hints: np.ndarray | None = None
    # Straight jamb-to-jamb chords derived from ``door_hints``. Room
    # extraction seals with these so a swing arc never becomes a room wall.
    door_seals: np.ndarray | None = None
    # Functional cuts introduced by a VLM-labelled room split. Keep these
    # separate from drawn zone dividers: only inferred cuts can divide an
    # otherwise continuous facade opening.
    inferred_zones: np.ndarray | None = None


def _arc_stroke_coverage(
    binary: np.ndarray,
    valid: np.ndarray,
    hinge: tuple[float, float],
    along: np.ndarray,
    normal: np.ndarray,
    radius: float,
) -> float:
    """Share of a quarter-circle covered by stroke pixels."""
    angles = np.linspace(0.08, np.pi / 2 - 0.08, 32)
    height, width = binary.shape
    hits = usable = 0
    for angle in angles:
        direction = along * np.cos(angle) + normal * np.sin(angle)
        x = int(round(hinge[0] + direction[0] * radius))
        y = int(round(hinge[1] + direction[1] * radius))
        if not (0 <= x < width and 0 <= y < height) or not valid[y, x]:
            continue
        usable += 1
        hits += int(binary[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2].any())
    return hits / usable if usable >= len(angles) * 0.5 else 0.0


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
        if hist[peak] < 0.00005 * low_chroma.size:
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

    # A facade can use two short black load-bearing caps beside a light
    # window while the rest of the plan uses grey walls. Their tone is too
    # sparse to qualify as a whole-image band; keep only dark thick pieces
    # that touch structure and extend its footprint.
    base_footprint = _bbox(solid)
    if base_footprint is not None and all(lo > 33 for lo, _hi in bands):
        dark = np.where((gray <= 33) & low_chroma, np.uint8(255), np.uint8(0))
        dark_solid, _ = _solid_walls(dark, MIN_CORE_HALF_WIDTH)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(dark_solid)
        near_solid = cv2.dilate(solid, np.ones((5, 5), np.uint8))
        x0, y0, x1, y1 = base_footprint
        keep = np.zeros(count, dtype=bool)
        for label in range(1, count):
            x, y, width, height = stats[label, :4]
            outside = x < x0 or y < y0 or x + width - 1 > x1 or y + height - 1 > y1
            keep[label] = outside and bool(np.any(near_solid[labels == label]))
        caps = np.where(keep[labels], np.uint8(255), np.uint8(0))
        solid = cv2.bitwise_or(solid, caps)
        in_band = cv2.bitwise_or(in_band, caps)

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
    stroke_bin = _stroke_binary(gray, footprint)
    zones = _zone_lines(stroke_bin, union, footprint, angles)
    door_hints, door_seals = _door_hints(stroke_bin, union, zones, thickness)
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
        zones=zones,
        door_hints=door_hints,
        door_seals=door_seals,
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
    h, w = binary.shape
    # length gate from the SOURCE frame: rotated canvases are larger and
    # would silently raise the bar on exactly the frames that need it
    length = max(40, min(h, w) // 16)
    zones = _zone_lines_axis(binary, structure, length)
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
        z = _zone_lines_axis(rot_b, rot_s, length)
        if z is None:
            continue
        z_back = cv2.warpAffine(
            z, cv2.invertAffineTransform(m), (w, h), flags=cv2.INTER_NEAREST, borderValue=0
        )
        zones = z_back if zones is None else cv2.bitwise_or(zones, z_back)
    return zones


def _zone_lines_axis(
    binary: np.ndarray, structure: np.ndarray, length: int | None = None
) -> np.ndarray | None:
    """Dashed zone dividers: thin dash runs fused into straight lines.

    Listing plans split open spaces (玄关/走廊/餐厅) with dashed lines and
    print a per-zone area — the dashes are drawn evidence of a boundary, so
    sealing along them aligns measured geometry with the printed truth.
    Against the look-alikes: label text fuses into short unanchored rows;
    wood grain fuses into CONTINUOUS lines (dashness ~1); hatch/tile texture
    is 2D-dense around any run it produces. Dividers legitimately cross each
    other and end on door arcs, so runs may anchor on OTHER long runs.
    """
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    thin = np.where(dist <= 2.0, binary, np.uint8(0))
    h, w = binary.shape
    if length is None:
        length = max(40, min(h, w) // 16)
    anchor_structure = cv2.dilate(structure, np.ones((13, 13), np.uint8))

    candidates: list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]] = []
    long_runs = np.zeros_like(binary)
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
            # Texture veto by one-sided OTHERS-density: hatch/tile fields are
            # 2D-dense on BOTH sides of any run they spawn, while a real
            # divider keeps at least one side mostly empty — a bathroom's
            # dry/wet dashed boundary legitimately borders its own dot-hatch
            # fill on one side. Own pixels never count (a 4px dashed line
            # already fills ~10% of its own band).
            pad = 8
            if bw >= bh:  # horizontal run: bands above and below
                sides = (thin[max(0, y - pad) : y, x : x + bw],
                         thin[y + bh : min(h, y + bh + pad), x : x + bw])
            else:
                sides = (thin[y : y + bh, max(0, x - pad) : x],
                         thin[y : y + bh, x + bw : min(w, x + bw + pad)])
            densities = [
                np.count_nonzero(side) / side.size for side in sides if side.size
            ]
            if not densities or min(densities) > 0.06:
                continue
            ys, xs = np.nonzero(component)
            if bw >= bh:  # horizontal run: anchor both x-extremes
                p1 = (x + int(xs.min()), y + int(ys[np.argmin(xs)]))
                p2 = (x + int(xs.max()), y + int(ys[np.argmax(xs)]))
            else:
                p1 = (x + int(xs[np.argmin(ys)]), y + int(ys.min()))
                p2 = (x + int(xs[np.argmax(ys)]), y + int(ys.max()))
            mask = np.zeros_like(binary)
            mask[y : y + bh, x : x + bw][component] = 255
            candidates.append((mask, p1, p2))
            long_runs = cv2.bitwise_or(long_runs, mask)

    if not candidates:
        return None
    # Dividers may meet each other at zone corners (玄关's two dashed
    # sides), but a run must never satisfy its own free endpoint.
    zones = np.zeros_like(binary)
    found = False
    for mask, p1, p2 in candidates:
        other_runs = cv2.bitwise_and(long_runs, cv2.bitwise_not(mask))
        anchor_any = cv2.bitwise_or(
            anchor_structure, cv2.dilate(other_runs, np.ones((13, 13), np.uint8))
        )
        first, second = anchor_any[p1[1], p1[0]], anchor_any[p2[1], p2[0]]
        if (anchor_structure[p1[1], p1[0]] or anchor_structure[p2[1], p2[0]]) and (
            first and second
        ):
            zones = cv2.bitwise_or(zones, mask)
            found = True
    return zones if found else None


def _door_hints(
    binary: np.ndarray,
    union: np.ndarray,
    zones: np.ndarray | None,
    wall_thickness: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Thin strokes hanging off the wall network: door leaves and swing arcs.

    Arc-style plans draw no sill across the doorway — the only strokes
    spanning the gap are the leaf (a short straight stroke from the hinge
    jamb) and the quarter-circle arc (leaf tip to the far jamb). Both start
    ON the wall network, so admitting wall-adjacent thin components lets the
    drawing seal its own doorways. Size caps keep beds and sofas out; a
    stray bedside table against a wall costs a sub-room-size nook, not a
    fake divider.
    """
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    thin = np.where(dist <= 2.0, binary, np.uint8(0))
    if zones is not None:
        thin = cv2.bitwise_and(thin, cv2.bitwise_not(zones))
    # Arcs and leaves are drawn TOUCHING their jamb — connected to the wall
    # network they'd merge into one giant component and fail the size gate.
    # Cut the wall out first so each piece of door furniture stands alone,
    # then re-join hairline breaks (some plans draw the swing arc itself as
    # a fine dash-dot curve).
    thin = cv2.bitwise_and(
        thin, cv2.bitwise_not(cv2.dilate(union, np.ones((3, 3), np.uint8)))
    )
    thin = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    near_radius = max(4, int(round(0.5 * wall_thickness)))
    near_wall = cv2.dilate(
        union, np.ones((2 * near_radius + 1, 2 * near_radius + 1), np.uint8)
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
    hints = np.zeros_like(binary)
    seals = np.zeros_like(binary)
    found = False
    min_span = max(6.0, 0.75 * wall_thickness)
    max_span = 20.0 * wall_thickness
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        long_side = max(bw, bh)
        if not min_span <= long_side <= max_span:
            continue
        component = labels[y : y + bh, x : x + bw] == i
        touching = component & (near_wall[y : y + bh, x : x + bw] > 0)
        if not touching.any():
            continue
        # Closed curves (leaf+arc sector boundaries, tinted-sector edges)
        # are allowed: the swing-sector pocket they excise is suppressed in
        # room extraction, and on plans where hints only cost coverage the
        # plain candidate variant wins the selection anyway.
        hints[y : y + bh, x : x + bw][component] = 255
        # When the leaf and arc form one component, its two wall contacts
        # are the doorway jambs. Add their chord: room contours then follow
        # the structural threshold instead of tracing the swing sector.
        count, _labels, _stats, centroids = cv2.connectedComponentsWithStats(
            touching.astype(np.uint8), connectivity=8
        )
        anchors = centroids[1:count]
        sealed = False
        if len(anchors) >= 2 and long_side > 8.0 * wall_thickness:
            left, right = max(
                (
                    (a, b)
                    for i, a in enumerate(anchors)
                    for b in anchors[i + 1 :]
                ),
                key=lambda pair: float(np.sum((pair[0] - pair[1]) ** 2)),
            )
            left = left + (x, y)
            right = right + (x, y)
            if min(abs(left[0] - right[0]), abs(left[1] - right[1])) > near_radius:
                # One radius (the leaf) may already be in ``union`` as a
                # long thin line, leaving only the arc in this component.
                # Its orthogonal corner nearest the wall is the hinge; the
                # less-inked radius from there is the actual threshold.
                corners = (np.array((left[0], right[1])), np.array((right[0], left[1])))
                distance = cv2.distanceTransform(
                    cv2.bitwise_not(union), cv2.DIST_L2, 5
                )
                hinge = min(
                    corners,
                    key=lambda point: distance[
                        int(np.clip(round(point[1]), 0, union.shape[0] - 1)),
                        int(np.clip(round(point[0]), 0, union.shape[1] - 1)),
                    ],
                )
                structural = cv2.dilate(
                    union, np.ones((2 * near_radius + 1, 2 * near_radius + 1), np.uint8)
                )

                def ink_share(
                    end: np.ndarray,
                    hinge: np.ndarray = hinge,
                    structural: np.ndarray = structural,
                ) -> float:
                    samples = np.linspace(hinge, end, max(2, int(np.linalg.norm(end - hinge))))
                    xs = np.clip(np.round(samples[:, 0]).astype(int), 0, union.shape[1] - 1)
                    ys = np.clip(np.round(samples[:, 1]).astype(int), 0, union.shape[0] - 1)
                    return float((structural[ys, xs] > 0).mean())

                right = min((left, right), key=ink_share)
                left = hinge
            cv2.line(
                seals,
                (int(round(left[0])), int(round(left[1]))),
                (int(round(right[0])), int(round(right[1]))),
                255,
                max(1, int(round(0.2 * wall_thickness))),
            )
            sealed = True
        if not sealed:
            seals[y : y + bh, x : x + bw][component] = 255
        found = True
    return (hints, seals) if found else (None, None)


def _stroke_binary(
    gray: np.ndarray, footprint: tuple[int, int, int, int] | None
) -> np.ndarray:
    """Adaptive-threshold stroke binary, gated to the wall footprint (drops
    the dimension chains and margin annotations living outside it) and
    cleared of dot-hatch FIELDS (bathroom/tile floor texture).

    Dot hatch is floor decoration, never structure — the same principle
    that keeps light fills out of the wall bands. A dot is a small roundish
    blob; a FIELD is many of them packed together. Isolated small marks
    (dimension ticks) and dashed zone dividers survive: a dashed LINE
    through the density window covers ~3% of it, a hatch field 8%+.
    Derotation resampling can smear dots to 5-8px, hence the 8px cap.
    """
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

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n > 1:
        long_side = np.maximum(stats[:, 2], stats[:, 3])
        short_side = np.maximum(np.minimum(stats[:, 2], stats[:, 3]), 1)
        # dots are ROUND (hatch marks, resampling smears); dash SEGMENTS are
        # elongated and must survive even inside grain-speckled areas — a
        # zone divider running between two wood floors is exactly that case
        roundish = long_side < 2 * short_side
        tiny = (long_side <= 4) & roundish  # hatch marks, 2x2 divider dots
        smear = (long_side > 4) & (long_side <= 8) & roundish  # rotation blur
        tiny[0] = smear[0] = False
        dots = np.where((tiny | smear)[labels], np.float32(1.0), np.float32(0.0))
        density = cv2.boxFilter(dots, -1, (31, 31), normalize=True)
        # Size-tiered density floors: true hatch fields run 0.15-0.25, while
        # grain speckle WITH a dashed divider embedded runs ~0.08 and its
        # tiny marks must survive. 5-7px round smears exist only as hatch
        # blurred by derotation resampling (~0.08-0.10) — no legitimate
        # drawing uses them, so their floor is low.
        field = ((density >= 0.12) & tiny[labels] & (dots > 0)) | (
            (density >= 0.05) & smear[labels] & (dots > 0)
        )
        binary = np.where(field, np.uint8(0), binary)
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
    if angles:
        # Dot-hatch immunity: the rotated pass pre-closes small breaks (see
        # below), and a bathroom's dot field lines up diagonally at almost
        # any angle — pure dots must never reach that join. Mullion-broken
        # glazing fragments are elongated (≥5px) and pass.
        n_comp, comp_labels, comp_stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        elongated = np.maximum(comp_stats[:, 2], comp_stats[:, 3]) > 4
        elongated[0] = False
        binary = np.where(elongated[comp_labels], binary, np.uint8(0))
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
