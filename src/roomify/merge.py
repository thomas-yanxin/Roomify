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

import statistics
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from roomify.rooms import CVRoom
from roomify.schema import ParseWarning, Scale, Unresolved
from roomify.vlm import ChainRead, OpeningsRead, RoomRead
from roomify.walls import WallExtraction

if TYPE_CHECKING:  # openings imports merge; annotate without the cycle
    from roomify.openings import OpeningCandidate

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


def merge_rooms(
    cv_rooms: list[CVRoom], read: RoomRead | None, image_shape: tuple[int, int]
) -> MergeOutcome:
    """CV polygons + VLM semantics table → room drafts.

    Marker ids are 1-based strings matching render_room_overlay's numbering.
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


def merge_openings(
    candidates: list[OpeningCandidate],
    read: OpeningsRead | None,
    image_shape: tuple[int, int],
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
            if element_type in _SWINGING or element_type == "unknown_symbol":
                swing, hinge = cand.arc.swing, cand.arc.hinge
            else:
                warnings.append(
                    ParseWarning(
                        code="arc_evidence_conflict",
                        message=(
                            f"swing-arc pixels found at {cand.marker} but it was classified "
                            f"{element_type}; swing not reported"
                        ),
                        ref=cand.marker,
                    )
                )
        elif element_type in _SWINGING:
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
