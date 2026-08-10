"""Fusing CV geometry with VLM semantics, and estimating the px→mm scale.

Room recall is max(CV, VLM): CV polygons get VLM names by marker id, the
VLM can veto a non-room, and rooms only the VLM saw enter as bounding boxes
marked ``source="vlm"``. Matching is strictly id-keyed — position-zipping
mislabels every room the moment the model reorders its answer.

Scale estimation combines two independent, individually flawed sources:

- Dimension chains give per-axis scales (which anisotropically resized
  exports need) but a single OCR slip corrupts a whole chain, and chains
  measure axis-to-axis — the footprint extent minus one wall thickness.
- Printed room areas give a robust magnitude — the MEDIAN of area ratios
  over many rooms shrugs off individual misreads — but only the geometric
  mean sqrt(sx·sy), never the axis split.

So: chains propose per-axis candidates, printed areas elect the candidate
pair whose product matches the median ratio. Each source alone still
degrades gracefully (with the appropriate confidence and warnings).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from roomify.rooms import CVRoom
from roomify.schema import OUTDOOR_ROOM_TYPES, ParseWarning, Scale, Unresolved
from roomify.vlm import ChainRead, OpeningsRead, RoomRead
from roomify.walls import WallExtraction

if TYPE_CHECKING:  # openings imports merge; annotate without the cycle
    from roomify.openings import OpeningCandidate, WallSegment

MIN_ROOM_SQM = 0.5  # below this a nameless room is segmentation junk
DEVIATION_FLAG_THRESHOLD = 0.10
MIN_ROOMS_FOR_AREA_SCALE = 3
AGREE_HIGH = 0.05  # relative disagreement (on the area scale) between sources
AGREE_MEDIUM = 0.15


@dataclass
class RoomDraft:
    """A room mid-pipeline, in WORKING pixels (converted at assembly)."""

    polygon: np.ndarray
    area_px: float
    perimeter_px: float
    edge_lengths_px: list[float]
    seed: tuple[float, float]
    source: str  # "cv" | "vlm" | "cv+vlm"
    confidence: float
    name: str | None = None
    room_type: str = "unknown_space"
    printed_area_sqm: float | None = None
    marker: str | None = None  # overlay marker id, for warnings/debug
    zone_bounded: bool = False  # boundary includes a dashed zone divider
    recovered: bool = False  # reclaimed from uncovered floor space


@dataclass
class ScaleDraft:
    """Scale in WORKING px per mm; converted alongside geometry at assembly."""

    px_per_mm_x: float
    px_per_mm_y: float
    method: str
    confidence: str
    px_per_mm_from_areas: float | None
    n_rooms_used: int
    n_chain_values_used: int

    def to_schema(self, to_original: float) -> Scale:
        # working px × to_original = original px, so px-per-mm grows by the
        # same factor when expressed in original-image pixels.
        return Scale(
            px_per_mm_x=self.px_per_mm_x * to_original,
            px_per_mm_y=self.px_per_mm_y * to_original,
            method=self.method,  # type: ignore[arg-type]
            confidence=self.confidence,  # type: ignore[arg-type]
            px_per_mm_from_areas=(
                self.px_per_mm_from_areas * to_original if self.px_per_mm_from_areas else None
            ),
            anisotropy=self.px_per_mm_y / self.px_per_mm_x - 1,
            n_rooms_used=self.n_rooms_used,
            n_chain_values_used=self.n_chain_values_used,
        )


@dataclass
class MergeOutcome:
    rooms: list[RoomDraft]
    warnings: list[ParseWarning] = field(default_factory=list)
    unresolved: list[Unresolved] = field(default_factory=list)


MIN_FREE_FLOOR = 0.30  # of an extra room's box; see merge_rooms


def merge_rooms(
    cv_rooms: list[CVRoom],
    read: RoomRead | None,
    image_shape: tuple[int, int],
    free_floor: np.ndarray | None = None,
) -> MergeOutcome:
    """CV polygons + VLM semantics table → room drafts.

    Marker ids are 1-based strings matching render_room_overlay's numbering.

    ``free_floor`` is the floor inside the building that no CV polygon
    claims (``rooms.uncovered_floor``). An extra room needs somewhere to
    exist: the model reports every printed label it sees, so the zones of an
    open-plan space (走廊/玄关 inside a 客餐厅) come back as "unmarkered
    rooms" and used to be emitted as boxes lying ON the measured room —
    double-counting its area and overlapping its polygon. Measured on the
    corpus, the one genuinely missed room stood on 65% free floor while
    every redundant box stood on 0-7%.
    """
    out = MergeOutcome(rooms=[])
    entries = read.rooms if read is not None else {}

    for i, cv_room in enumerate(cv_rooms):
        marker = str(i + 1)
        entry = entries.get(marker)
        draft = RoomDraft(
            polygon=cv_room.polygon,
            area_px=cv_room.area_px,
            perimeter_px=cv_room.perimeter_px,
            edge_lengths_px=cv_room.edge_lengths_px,
            seed=cv_room.seed,
            source="cv",
            confidence=0.3,
            marker=marker,
            zone_bounded=cv_room.zone_bounded,
            recovered=cv_room.recovered,
        )
        if entry is None:
            if read is not None:
                out.warnings.append(
                    ParseWarning(
                        code="room_semantics_missing",
                        message=f"VLM returned no entry for marker {marker}; kept CV geometry",
                        ref=marker,
                    )
                )
            out.unresolved.append(
                Unresolved(path=f"rooms/{marker}/name", reason="no VLM reading for this room")
            )
            out.rooms.append(draft)
            continue
        if entry.not_a_room:
            out.warnings.append(
                ParseWarning(
                    code="room_vetoed",
                    message=f"VLM judged marker {marker} not to be a room; polygon dropped",
                    ref=marker,
                )
            )
            continue
        out.rooms.append(
            replace(
                draft,
                source="cv+vlm",
                confidence=entry.confidence,
                name=entry.name,
                room_type=entry.room_type,
                printed_area_sqm=entry.printed_area_sqm,
            )
        )

    h, w = image_shape
    for j, extra in enumerate(read.extra_rooms if read is not None else []):
        y0, x0, y1, x1 = (v / 1000.0 for v in extra.box_2d)
        polygon = np.array(
            [[x0 * w, y0 * h], [x1 * w, y0 * h], [x1 * w, y1 * h], [x0 * w, y1 * h]],
            dtype=np.float64,
        )
        width, height = (x1 - x0) * w, (y1 - y0) * h
        marker = f"vlm_{j + 1}"
        if free_floor is not None:
            box = free_floor[
                max(0, int(y0 * h)) : int(y1 * h) + 1,
                max(0, int(x0 * w)) : int(x1 * w) + 1,
            ]
            if box.size == 0 or float((box > 0).mean()) < MIN_FREE_FLOOR:
                out.warnings.append(
                    ParseWarning(
                        code="room_already_measured",
                        message=(
                            f"VLM reported {extra.name or 'an extra room'} where no "
                            "unclaimed floor remains; it is a zone of an "
                            "already-measured room, not a room of its own"
                        ),
                        ref=marker,
                    )
                )
                continue
        out.rooms.append(
            RoomDraft(
                polygon=polygon,
                area_px=float(width * height),
                perimeter_px=float(2 * (width + height)),
                edge_lengths_px=[float(width), float(height), float(width), float(height)],
                seed=(float((x0 + x1) / 2 * w), float((y0 + y1) / 2 * h)),
                source="vlm",
                confidence=min(extra.confidence, 0.5),
                name=extra.name,
                room_type=extra.room_type,
                printed_area_sqm=extra.printed_area_sqm,
                marker=marker,
            )
        )
        out.warnings.append(
            ParseWarning(
                code="room_geometry_is_bbox",
                message=(
                    f"room {extra.name or marker} was found only by the VLM; its geometry is "
                    "an approximate bounding box"
                ),
                ref=marker,
            )
        )
    return out


RECONCILE_RATIO = 1.4  # a pairing is a misfit beyond ±40% of expectation
RECONCILE_FIT = 1.25  # a repair must land within ±25% to be adopted
MERGE_SUM_TOL = 0.20  # fragment sums must match the label within ±20%


def reconcile_rooms(
    rooms: list[RoomDraft],
    scale: ScaleDraft | None,
    wall_thickness_px: float,
) -> tuple[list[RoomDraft], list[ParseWarning]]:
    """Repair label↔polygon pairing using the printed areas as ink truth.

    The VLM reads printed labels reliably but attaches them to overlay
    markers noisily once fragments multiply. Measured pixel areas are exact,
    so a pairing whose measured/printed ratio strays far from the plan-wide
    median is a wrong ATTACHMENT, not a wrong measurement. Three repairs,
    all confined to the misfit subset (well-fitting rooms are never touched)
    and all logged:

    1. reassign labels among misfit rooms (labels move with their printed
       value — name and area are one piece of ink);
    2. merge ADJACENT misfit/unnamed fragments whose joint area matches an
       otherwise-unplaceable label (a room chopped by a texture accident);
    3. restore a label in place when no better geometry fits it.
    """
    import itertools

    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    warnings: list[ParseWarning] = []
    if scale is None or len(rooms) < 2:
        return rooms, warnings
    px_per_sqm = scale.px_per_mm_x * scale.px_per_mm_y * 1e6

    def misfit_of(area_px: float, printed: float) -> float:
        return abs(math.log(area_px / max(printed * px_per_sqm, 1e-9)))

    log_bad = math.log(RECONCILE_RATIO)
    log_fit = math.log(RECONCILE_FIT)

    # cv-measured rooms only: bbox geometry cannot anchor a repair
    misfits = [
        i
        for i, r in enumerate(rooms)
        if r.source != "vlm"
        and r.printed_area_sqm
        and misfit_of(r.area_px, r.printed_area_sqm) > log_bad
    ]
    unnamed = [
        i
        for i, r in enumerate(rooms)
        if r.source != "vlm" and not r.printed_area_sqm and not r.name
    ]
    if not misfits:
        return rooms, warnings

    labels: list[tuple[str | None, str, float, int]] = []
    for i in misfits:
        printed_area = rooms[i].printed_area_sqm
        assert printed_area is not None
        labels.append((rooms[i].name, rooms[i].room_type, printed_area, i))
    slots = sorted(set(misfits) | set(unnamed))

    polys: dict[int, ShapelyPolygon] = {}
    for i in slots:
        try:
            p = ShapelyPolygon(rooms[i].polygon)
            polys[i] = p if p.is_valid else p.buffer(0)
        except Exception:
            polys[i] = ShapelyPolygon()

    out = list(rooms)
    for i in misfits:  # detach the misfit labels; slots start clean
        out[i] = replace(out[i], name=None, printed_area_sqm=None,
                         room_type="unknown_space", confidence=0.3)

    gap = max(8.0, 1.5 * wall_thickness_px)
    used: set[int] = set()
    for name, room_type, printed, origin in labels:
        expected = printed * px_per_sqm
        free = [i for i in slots if i not in used]
        # single-room fit first
        best_i, best_m = None, log_fit
        for i in free:
            m = misfit_of(out[i].area_px, printed)
            if m < best_m:
                best_i, best_m = i, m
        # fragment merge: pairs/triples of mutually adjacent free rooms
        best_group = None
        best_group_m = math.inf
        if best_i is None:
            for size in (2, 3):
                for combo in itertools.combinations(free, size):
                    total = sum(out[i].area_px for i in combo)
                    group_m = abs(math.log(max(total, 1e-9) / expected))
                    if group_m > math.log(1 + MERGE_SUM_TOL):
                        continue
                    if all(
                        polys[a].distance(polys[b]) <= gap
                        for a, b in itertools.combinations(combo, 2)
                    ) and group_m < best_group_m:
                        best_group = combo
                        best_group_m = group_m
        if best_i is not None:
            out[best_i] = replace(
                out[best_i],
                name=name,
                room_type=room_type,
                printed_area_sqm=printed,
                source="cv+vlm",
                confidence=0.6,
            )
            used.add(best_i)
            warnings.append(
                ParseWarning(
                    code="label_reassigned",
                    message=f"printed label {name or '?'} ({printed}㎡) re-attached "
                    "to the room whose measured area matches it",
                    ref=out[best_i].marker,
                )
            )
        elif best_group is not None:
            merged_poly = unary_union(
                [polys[i] for i in best_group]
            ).buffer(gap / 2).buffer(-gap / 2)
            if merged_poly.geom_type == "MultiPolygon":
                merged_poly = max(merged_poly.geoms, key=lambda g: g.area)
            ring = np.asarray(merged_poly.exterior.coords[:-1], dtype=np.float64)
            closed = np.vstack([ring, ring[:1]])
            edges = np.linalg.norm(np.diff(closed, axis=0), axis=1)
            seed = merged_poly.representative_point()
            keeper = min(best_group)
            out[keeper] = replace(
                out[keeper],
                polygon=ring,
                area_px=float(merged_poly.area),
                perimeter_px=float(edges.sum()),
                edge_lengths_px=[float(e) for e in edges],
                seed=(float(seed.x), float(seed.y)),
                name=name,
                room_type=room_type,
                printed_area_sqm=printed,
                source="cv+vlm",
                confidence=0.5,
                zone_bounded=any(out[i].zone_bounded for i in best_group),
            )
            used.update(best_group)
            for i in best_group:
                if i != keeper:
                    out[i] = None  # type: ignore[call-overload]
            warnings.append(
                ParseWarning(
                    code="rooms_merged_for_label",
                    message=(
                        f"{len(best_group)} adjacent fragments merged: their joint "
                        f"area matches the printed label {name or '?'} ({printed}㎡)"
                    ),
                    ref=out[keeper].marker,
                )
            )
        elif origin not in used and out[origin] is not None:
            # No better home anywhere: the label was PRINTED inside this
            # room, and that spatial certainty outranks the area mismatch
            # (0.01㎡ duct labels, glazing-eaten balconies). Restore the
            # original pairing untouched — the deviation flag still tells
            # the truth about the mismatch.
            out[origin] = replace(
                out[origin],
                name=name,
                room_type=room_type,
                printed_area_sqm=printed,
                source="cv+vlm",
                confidence=rooms[origin].confidence,
            )
            used.add(origin)
        else:
            warnings.append(
                ParseWarning(
                    code="label_unplaced",
                    message=(
                        f"printed label {name or '?'} ({printed}㎡) matches no "
                        "measured room or adjacent fragment group; left unassigned"
                    ),
                )
            )

    return [r for r in out if r is not None], warnings


def estimate_scale(
    chains: list[ChainRead],
    walls: WallExtraction,
    rooms: list[RoomDraft],
) -> tuple[ScaleDraft | None, list[ParseWarning]]:
    warnings: list[ParseWarning] = []
    x0, y0, x1, y1 = walls.footprint
    # Chains measure between wall AXES: the outer-face footprint extent is
    # one wall thickness (half at each end) larger than the chain span.
    extent_x = (x1 - x0) - walls.thickness_px
    extent_y = (y1 - y0) - walls.thickness_px

    x_candidates = _axis_candidates(chains, ("top", "bottom"), extent_x)
    y_candidates = _axis_candidates(chains, ("left", "right"), extent_y)

    # Median of per-room area ratios: px² per mm². Only CV-measured polygons
    # participate — VLM bbox geometry is approximate by construction, and
    # zone-bounded rooms (dashed functional splits) measure against printed
    # values too loosely to calibrate a scale.
    ratios = [
        r.area_px / (r.printed_area_sqm * 1e6)
        for r in rooms
        if r.printed_area_sqm and r.source != "vlm" and not r.zone_bounded
    ]
    area_scale_sq = statistics.median(ratios) if len(ratios) >= MIN_ROOMS_FOR_AREA_SCALE else None
    from_areas = area_scale_sq**0.5 if area_scale_sq else None

    if x_candidates and y_candidates and area_scale_sq:
        best = min(
            ((sx, nx, sy, ny) for sx, nx in x_candidates for sy, ny in y_candidates),
            key=lambda c: abs(c[0] * c[2] - area_scale_sq),
        )
        sx, nx, sy, ny = best
        disagreement = abs(sx * sy / area_scale_sq - 1)
        # A degenerate pairing can nail the area PRODUCT with an absurd
        # per-axis split (seen: 11:1 when a lone chain matched the wrong
        # extent on a derotated plan); real listing exports stay ≤ ~1.15.
        anisotropy = max(sx, sy) / max(min(sx, sy), 1e-12)
        if disagreement > AGREE_MEDIUM or anisotropy > 1.35:
            # No credible chain pair — typical of diagonal units, where
            # dimension chains only span the rectilinear wing while the
            # footprint bbox covers the whole plan. The area median is
            # self-consistent with the polygons being measured; a wrong
            # per-axis split is worse than an isotropic assumption.
            warnings.append(
                ParseWarning(
                    code="scale_disagreement",
                    message=(
                        f"dimension chains disagree with printed areas "
                        f"(area {disagreement:.0%}, axis ratio "
                        f"{anisotropy:.2f}); using the printed-area scale "
                        "(equal axes assumed)"
                    ),
                )
            )
            assert from_areas is not None
            return (
                ScaleDraft(
                    px_per_mm_x=from_areas,
                    px_per_mm_y=from_areas,
                    method="printed_areas",
                    confidence="medium" if len(ratios) >= 5 else "low",
                    px_per_mm_from_areas=from_areas,
                    n_rooms_used=len(ratios),
                    n_chain_values_used=0,
                ),
                warnings,
            )
        confidence = "high" if disagreement <= AGREE_HIGH else "medium"
        return (
            ScaleDraft(
                px_per_mm_x=sx,
                px_per_mm_y=sy,
                method="dimension_chains+printed_areas",
                confidence=confidence,
                px_per_mm_from_areas=from_areas,
                n_rooms_used=len(ratios),
                n_chain_values_used=nx + ny,
            ),
            warnings,
        )

    if x_candidates and y_candidates:
        # No printed areas to elect a pair: the chain with the largest mm sum
        # (the smallest px/mm candidate) is likeliest to span the full extent.
        sx, nx = min(x_candidates, key=lambda c: c[0])
        sy, ny = min(y_candidates, key=lambda c: c[0])
        return (
            ScaleDraft(
                px_per_mm_x=sx,
                px_per_mm_y=sy,
                method="dimension_chains",
                confidence="medium",
                px_per_mm_from_areas=None,
                n_rooms_used=0,
                n_chain_values_used=nx + ny,
            ),
            warnings,
        )

    if area_scale_sq and (x_candidates or y_candidates):
        # One measured axis plus the area-scale product determines the
        # missing axis; do not throw away anisotropy evidence.
        if x_candidates:
            sx, nx = min(x_candidates, key=lambda c: c[0])
            sy, ny = area_scale_sq / sx, 0
        else:
            sy, ny = min(y_candidates, key=lambda c: c[0])
            sx, nx = area_scale_sq / sy, 0
        # Same degenerate-pairing guard as above: a lone chain matched to
        # the wrong extent forces the derived axis absurdly far away (seen:
        # 11:1 on a derotated plan). Real exports stay ≤ ~1.15.
        if max(sx, sy) / max(min(sx, sy), 1e-12) <= 1.35:
            return (
                ScaleDraft(
                    px_per_mm_x=sx,
                    px_per_mm_y=sy,
                    method="dimension_chains+printed_areas",
                    confidence="medium" if len(ratios) >= 5 else "low",
                    px_per_mm_from_areas=from_areas,
                    n_rooms_used=len(ratios),
                    n_chain_values_used=nx + ny,
                ),
                warnings,
            )
        warnings.append(
            ParseWarning(
                code="scale_disagreement",
                message=(
                    "single dimension chain contradicts the printed-area scale "
                    f"(axis ratio {max(sx, sy) / max(min(sx, sy), 1e-12):.1f}); "
                    "using the printed-area scale (equal axes assumed)"
                ),
            )
        )

    if from_areas:
        warnings.append(
            ParseWarning(
                code="isotropy_assumed",
                message=(
                    "scale derived from printed areas only; equal horizontal/vertical "
                    "scale assumed, per-axis lengths may be off on anisotropically "
                    "resized images"
                ),
            )
        )
        return (
            ScaleDraft(
                px_per_mm_x=from_areas,
                px_per_mm_y=from_areas,
                method="printed_areas",
                confidence="medium" if len(ratios) >= 5 else "low",
                px_per_mm_from_areas=from_areas,
                n_rooms_used=len(ratios),
                n_chain_values_used=0,
            ),
            warnings,
        )

    warnings.append(
        ParseWarning(
            code="no_scale",
            message="no readable dimension chains or printed areas; output is pixel-only",
        )
    )
    return None, warnings


def _axis_candidates(
    chains: list[ChainRead], sides: tuple[str, str], extent_px: float
) -> list[tuple[float, int]]:
    """(px_per_mm, n_values) per usable chain on this axis."""
    if extent_px <= 0:
        return []
    candidates = []
    for chain in chains:
        if chain.side in sides and chain.values_mm:
            total_mm = sum(chain.values_mm)
            if total_mm > 0:
                candidates.append((extent_px / total_mm, len(chain.values_mm)))
    return candidates


@dataclass
class OpeningDraft:
    """An opening mid-pipeline, in WORKING pixels."""

    marker: str
    element_type: str
    raw_text: str | None
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    axis: str
    width_px: float
    wall_index: int
    connects: tuple[int | str, int | str]
    swing: str | None
    hinge: tuple[float, float] | None
    source: str
    confidence: float


@dataclass
class ElementDraft:
    """A VLM-only element with box_2d geometry, in WORKING pixels."""

    element_type: str
    raw_text: str | None
    bbox: tuple[float, float, float, float]
    source: str
    confidence: float


# Types whose leaves swing on a hinge — the only ones arc evidence applies to.
_SWINGING = {"single_door", "double_door", "folding_door"}
# Legend elements that legitimately stand free of walls; anything else the
# VLM reports as an "extra" is an opening claim without usable geometry.
_FREE_STANDING = {"stair", "railing", "elevator", "escalator",
                  "equipment_platform", "column", "chimney", "unknown_symbol"}


def _both_sides_indoor(
    connects: tuple[int | str, int | str], rooms: list[RoomDraft]
) -> bool:
    return all(
        isinstance(side, int)
        and side < len(rooms)
        and rooms[side].room_type not in OUTDOOR_ROOM_TYPES
        for side in connects
    )


# Beyond this an interior break is not credible as one opening: the corpus's
# genuine room-to-room mouths top out around 2m, while every wider one sits
# on a wall the ``solid`` mask lost. Openings are scanned against ``solid``,
# which by construction drops partitions thinner than ~5px, so on plans that
# draw thin partitions a whole wall reads as absent (measured on fp6: a 3.7m
# "passage" between a living-dining space and a bedroom). Pixels cannot
# settle it — at that stroke width a partition and a glazing band are
# identical — so the span is reported with its class unresolved rather than
# asserted. ponytail: a width rule; the real fix is a solid mask that keeps
# thin partitions, which needs a corpus that draws them at more than 4px.
MAX_INTERIOR_SPAN_MM = 2500.0

# A door is something you walk through, so its width is bounded by what a
# leaf can be built as — four sliding leaves at ~900mm is the widest thing
# in the domain. Past that the drawing is a glazed facade or a whole wall
# run the scan merged, not a door: the corpus's real balcony doors stop at
# 2957mm and the next one is 4085mm, so the bound sits in an empty band.
# Single-leaf width is NOT bounded here — the measurement is the break in
# the wall, frame and reveal included, and single_door widths run
# continuously from 1112 to 1375mm with no gap to cut at.
MAX_DOOR_SPAN_MM = 3500.0
_WALK_THROUGH = {"single_door", "double_door", "sliding_door", "folding_door"}


def merge_openings(
    candidates: list[OpeningCandidate],
    read: OpeningsRead | None,
    image_shape: tuple[int, int],
    rooms: list[RoomDraft] | None = None,
    scale: ScaleDraft | None = None,
    segments: list[WallSegment] | None = None,
) -> tuple[list[OpeningDraft], list[ElementDraft], list[ParseWarning], list[Unresolved]]:
    drafts: list[OpeningDraft] = []
    elements: list[ElementDraft] = []
    warnings: list[ParseWarning] = []
    unresolved: list[Unresolved] = []
    entries = read.candidates if read is not None else {}

    for cand in candidates:
        entry = entries.get(cand.marker)
        if entry is None:
            element_type = "window" if cand.kind_hint == "window" else "unknown_symbol"
            source, confidence, raw_text = "cv", 0.35, None
            if cand.arc is None:  # an arc classifies it below; nothing unresolved
                unresolved.append(
                    Unresolved(
                        path=f"openings/{cand.marker}/element_type",
                        reason="no VLM classification for this opening",
                    )
                )
        elif not entry.is_real:
            warnings.append(
                ParseWarning(
                    code="opening_vetoed",
                    message=f"VLM judged candidate {cand.marker} not to be a real opening",
                    ref=cand.marker,
                )
            )
            continue
        else:
            element_type = entry.element_type
            source, confidence, raw_text = "cv+vlm", entry.confidence, entry.raw_text

        swing = hinge = None
        if cand.arc is not None:
            # A drawn quarter-disc at a jamb is a door leaf sweeping the
            # floor: windows, sliding leaves and plain passages do not have
            # one, and the arc is measured from pixels at leaf scale while
            # the class is a reading of a small crop. The drawing wins.
            if element_type not in _SWINGING:
                if source != "cv":  # a disagreement, not merely a CV-only read
                    warnings.append(
                        ParseWarning(
                            code="opening_reclassified_by_arc",
                            message=(
                                f"candidate {cand.marker} was read as {element_type} but "
                                "the plan draws a door-leaf swing sector at its jamb; "
                                "reclassified single_door"
                            ),
                            ref=cand.marker,
                        )
                    )
                element_type = "single_door"
                confidence = min(confidence, 0.7) if source != "cv" else 0.5
            swing, hinge = cand.arc.swing, cand.arc.hinge
        # A sliding door and a sliding window are the SAME drawn symbol —
        # parallel overlapping leaves — so only adjacency separates them, and
        # between two indoor rooms there is no exterior for a window to face.
        # The prompt says so; this makes it hold (a 1.5m 客厅↔门厅 sliding
        # door came back as sliding_window at 0.9 confidence).
        if element_type == "sliding_window" and _both_sides_indoor(
            cand.connects, rooms or []
        ):
            warnings.append(
                ParseWarning(
                    code="opening_reclassified_by_adjacency",
                    message=(
                        f"candidate {cand.marker} was read as sliding_window but both "
                        "sides are indoor rooms; reclassified sliding_door"
                    ),
                    ref=cand.marker,
                )
            )
            element_type = "sliding_door"
            confidence = min(confidence, 0.7)
        if scale is not None and element_type != "unknown_symbol":
            per_mm = scale.px_per_mm_x if cand.axis == "h" else scale.px_per_mm_y
            span_mm = cand.width_px / per_mm if per_mm > 0 else 0.0
            reason = None
            if element_type in _WALK_THROUGH and span_mm > MAX_DOOR_SPAN_MM:
                reason = (
                    f"spans {span_mm:.0f}mm — wider than any door leaf can be built, "
                    "so the drawing is a glazed facade or a merged wall run"
                )
            elif span_mm > MAX_INTERIOR_SPAN_MM and _both_sides_indoor(
                cand.connects, rooms or []
            ):
                reason = (
                    f"spans {span_mm:.0f}mm between two indoor rooms — too wide for "
                    "one opening, and a partition the wall mask missed reads identically"
                )
            if reason is not None:
                warnings.append(
                    ParseWarning(
                        code="opening_span_implausible",
                        message=(
                            f"candidate {cand.marker} was read as {element_type} but it "
                            f"{reason}; class left unresolved"
                        ),
                        ref=cand.marker,
                    )
                )
                element_type = "unknown_symbol"
                confidence = min(confidence, 0.3)
                unresolved.append(
                    Unresolved(
                        path=f"openings/{cand.marker}/element_type",
                        reason=f"measured span contradicts the reading: {reason}",
                    )
                )
        if element_type in _SWINGING and cand.arc is None:
            unresolved.append(
                Unresolved(
                    path=f"openings/{cand.marker}/swing",
                    reason="no swing-arc evidence in the image",
                )
            )

        drafts.append(
            OpeningDraft(
                marker=cand.marker,
                element_type=element_type,
                raw_text=raw_text,
                bbox=cand.bbox,
                center=cand.center,
                axis=cand.axis,
                width_px=cand.width_px,
                wall_index=cand.wall_index,
                connects=cand.connects,
                swing=swing,
                hinge=hinge,
                source=source,
                confidence=confidence,
            )
        )

    h, w = image_shape
    for extra in read.extra_elements if read is not None else []:
        y0, x0, y1, x1 = (v / 1000.0 for v in extra.box_2d)
        bbox = (x0 * w, y0 * h, x1 * w, y1 * h)
        approximate_opening = extra.element_type not in _FREE_STANDING
        if approximate_opening and any(
            _boxes_overlap(bbox, c.bbox, slack=8.0) for c in candidates
        ):
            continue
        if approximate_opening and segments:
            snapped = _snap_to_wall(bbox, segments)
            if snapped is None:
                warnings.append(
                    ParseWarning(
                        code="opening_without_a_wall",
                        message=(
                            f"a {extra.element_type} was reported where no wall runs; "
                            "an opening is a hole in a wall, so it is not emitted"
                        ),
                    )
                )
                continue
            bbox = snapped
        elements.append(
            ElementDraft(
                element_type=extra.element_type,
                raw_text=extra.raw_text,
                bbox=bbox,
                source="vlm",
                confidence=min(extra.confidence, 0.5),
            )
        )
        if approximate_opening:
            warnings.append(
                ParseWarning(
                    code="opening_geometry_is_bbox",
                    message=(
                        f"a {extra.element_type} was found only by the VLM; it is reported "
                        "under elements with approximate bounding-box geometry"
                    ),
                )
            )
    return drafts, elements, warnings, unresolved


MAX_SNAP_THICKNESSES = 3.0  # of the matched wall; beyond it there is no wall to be in


def _snap_to_wall(
    bbox: tuple[float, float, float, float], segments: list[WallSegment]
) -> tuple[float, float, float, float] | None:
    """Move a VLM-only opening onto the wall it belongs in, or reject it.

    ``box_2d`` coordinates drift — measured on the corpus, 18 of 49
    opening-shaped extras sat beside a wall rather than on one and 9 had no
    wall within three thicknesses, up to 1.4m adrift, floating in the middle
    of a room. A door or window is a hole in a wall: a nearby wall says
    exactly where it is, and no nearby wall means we do not know, so nothing
    is emitted rather than geometry that cannot be true.
    """
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    best: tuple[float, float, float] | None = None
    for seg in segments:
        (ax, ay), (bx, by) = seg.start, seg.end
        dx, dy = bx - ax, by - ay
        span = dx * dx + dy * dy
        u = 0.0 if span == 0 else max(0.0, min(1.0, ((cx - ax) * dx + (cy - ay) * dy) / span))
        px, py = ax + u * dx, ay + u * dy
        gap = math.hypot(cx - px, cy - py) - MAX_SNAP_THICKNESSES * seg.thickness_px
        if best is None or gap < best[0]:
            best = (gap, px, py)
    if best is None or best[0] > 0:
        return None
    ox, oy = best[1] - cx, best[2] - cy
    return (bbox[0] + ox, bbox[1] + oy, bbox[2] + ox, bbox[3] + oy)


def _boxes_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    slack: float,
) -> bool:
    """True when grown boxes share at least 30% of the smaller box."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = (b[0] - slack, b[1] - slack, b[2] + slack, b[3] + slack)
    ix = min(ax1, bx1) - max(ax0, bx0)
    iy = min(ay1, by1) - max(ay0, by0)
    if ix <= 0 or iy <= 0:
        return False
    smaller = min((ax1 - ax0) * (ay1 - ay0), (b[2] - b[0]) * (b[3] - b[1]))
    return smaller > 0 and ix * iy >= 0.3 * smaller


def apply_area_checks(rooms: list[RoomDraft], scale: ScaleDraft | None) -> MergeOutcome:
    """Drop segmentation junk and compute deviation-vs-printed per room.

    Deviations are returned via RoomDraft-adjacent metadata at assembly time;
    here the tiny-room filter needs the scale, so it runs after estimation.
    """
    out = MergeOutcome(rooms=[])
    for draft in rooms:
        if scale is not None and draft.name is None:
            area_sqm = draft.area_px / (scale.px_per_mm_x * scale.px_per_mm_y) / 1e6
            if area_sqm < MIN_ROOM_SQM:
                out.warnings.append(
                    ParseWarning(
                        code="tiny_room_dropped",
                        message=(
                            f"unnamed region of {area_sqm:.2f}㎡ dropped as segmentation junk"
                        ),
                        ref=draft.marker,
                    )
                )
                continue
        out.rooms.append(draft)
    return out
