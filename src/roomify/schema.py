"""Roomify's public JSON contract.

Conventions:

- All ``*_px`` coordinates live in the ORIGINAL input image's pixel frame:
  x grows right, y grows down, origin at the top-left corner.
- All ``*_mm`` values are derived from pixels via ``scale``; the millimetre
  origin is the top-left corner of the detected building footprint.
- Polygon rings are OPEN: the first vertex is not repeated at the end.
  Edge ``i`` runs from vertex ``i`` to vertex ``(i + 1) % n`` — the closing
  edge is implicit.
- ``scale is None`` exactly when no calibration source (dimension chains or
  printed room areas) was readable; image-derived mm/sqm fields are then None,
  while the documented vertical defaults remain usable.
- Facts that are not observable in the drawing are ``None`` and listed under
  ``unresolved`` — except the documented vertical defaults needed to lift a
  2D plan into a usable 3D scene.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.2"
DEFAULT_LEVEL_HEIGHT_MM = 2800.0
DEFAULT_DOOR_HEIGHT_MM = 2100.0
DEFAULT_WINDOW_SILL_HEIGHT_MM = 900.0
DEFAULT_WINDOW_HEIGHT_MM = 1500.0

# Element vocabulary follows the tecton_v1_mapping naming of common Chinese
# residential floor-plan legend symbols (doors B20-B24, windows B28-B35,
# circulation B36-B42, structure, and an explicit unknown fallback).
ElementType = Literal[
    "passage",
    "single_door",
    "double_door",
    "sliding_door",
    "folding_door",
    "window",
    "casement_window",
    "sliding_window",
    "fixed_window",
    "bay_window",
    "floor_to_ceiling_window",
    "blind_window",
    "stair",
    "railing",
    "elevator",
    "escalator",
    "equipment_platform",
    "column",
    "chimney",
    "unknown_symbol",
]

RoomType = Literal[
    "living_room",
    "dining_room",
    "living_dining",
    "bedroom",
    "kitchen",
    "bathroom",
    "balcony",
    "hallway",
    "entrance",
    "closet",
    "storage",
    "study",
    "multipurpose",
    "equipment_platform",
    "unknown_space",
]

Source = Literal["cv", "vlm", "cv+vlm"]

EXTERIOR = "exterior"  # beyond the building envelope
UNKNOWN = "unknown"  # inside the building but not covered by any room polygon


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Point(StrictModel):
    x: float
    y: float


class BBox(StrictModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def _positive_extent(self) -> Self:
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError("bbox must have positive width and height")
        return self


def _validate_open_ring(polygon: list[Point], where: str) -> None:
    if len(polygon) < 3:
        raise ValueError(f"{where}: polygon needs at least 3 vertices")
    first, last = polygon[0], polygon[-1]
    if math.isclose(first.x, last.x) and math.isclose(first.y, last.y):
        raise ValueError(f"{where}: polygon ring must be open (first vertex not repeated)")


class Scale(StrictModel):
    """Pixel→millimetre calibration.

    ``px_per_mm_x`` and ``px_per_mm_y`` are separate on purpose: exported
    listing floor plans are frequently resized anisotropically (the test
    corpus measures ~10% difference between axes), so a single px/m factor
    cannot make horizontal and vertical wall lengths correct at once.
    """

    px_per_mm_x: float = Field(gt=0)
    px_per_mm_y: float = Field(gt=0)
    method: Literal["dimension_chains", "printed_areas", "dimension_chains+printed_areas"]
    confidence: Literal["high", "medium", "low"]
    px_per_mm_from_areas: float | None = None  # sqrt(median(area_px / printed_mm2))
    anisotropy: float  # px_per_mm_y / px_per_mm_x - 1
    n_rooms_used: int = Field(ge=0)
    n_chain_values_used: int = Field(ge=0)


class Room(StrictModel):
    id: str
    name: str | None  # label text exactly as printed on the plan (e.g. "卧室")
    room_type: RoomType
    polygon_px: list[Point]  # open ring, wall-centerline convention
    area_px: float = Field(gt=0)
    perimeter_px: float = Field(gt=0)
    edge_lengths_px: list[float]  # edge i = vertex i -> i+1 (wrapping)
    polygon_mm: list[Point] | None = None
    area_sqm: float | None = None
    perimeter_mm: float | None = None
    edge_lengths_mm: list[float] | None = None
    printed_area_sqm: float | None = None  # read off the plan by the VLM
    area_deviation: float | None = None  # (area_sqm - printed) / printed
    area_deviation_flag: bool = False  # |area_deviation| > 0.10; False when unknown
    source: Source
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        _validate_open_ring(self.polygon_px, f"room {self.id}")
        if len(self.edge_lengths_px) != len(self.polygon_px):
            raise ValueError(f"room {self.id}: edge_lengths_px must match polygon edge count")
        return self


class Wall(StrictModel):
    """Centerline wall segment derived from adjacent room boundaries."""

    id: str
    start_px: Point
    end_px: Point
    thickness_px: float = Field(gt=0)
    rooms: tuple[str, str]  # room ids, "exterior", or uncovered interior "unknown"
    start_mm: Point | None = None
    end_mm: Point | None = None
    thickness_mm: float | None = None
    source: Source
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _endpoints_differ(self) -> Self:
        if math.isclose(self.start_px.x, self.end_px.x) and math.isclose(
            self.start_px.y, self.end_px.y
        ):
            raise ValueError(f"wall {self.id}: endpoints must differ")
        return self


class Opening(StrictModel):
    """A wall-hosted element: door, window, passage, or bay window."""

    id: str
    element_type: ElementType
    raw_text: str | None = None  # nearby symbol code (e.g. "C1"), if printed
    bbox_px: BBox  # evidence region in the image
    center_px: Point
    width_px: float = Field(gt=0)  # opening length along the wall
    width_mm: float | None = None
    sill_height_mm: float | None = Field(default=None, ge=0)
    height_mm: float | None = Field(default=None, gt=0)
    wall_id: str | None = None
    connects: tuple[str, str] | None = None  # room ids, "exterior", or "unknown"
    # Swing facts come exclusively from swing-arc pixel evidence. Plans that
    # draw no arc get None here plus an entry in FloorPlan.unresolved.
    swing: Literal["clockwise", "counterclockwise"] | None = None
    hinge_px: Point | None = None
    protrusion_polygon_px: list[Point] | None = None  # bay windows
    protrusion_polygon_mm: list[Point] | None = None
    source: Source
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _ring_open(self) -> Self:
        if self.protrusion_polygon_px is not None:
            _validate_open_ring(self.protrusion_polygon_px, f"opening {self.id}")
        if self.protrusion_polygon_mm is not None:
            _validate_open_ring(self.protrusion_polygon_mm, f"opening {self.id}")

        is_door = self.element_type == "passage" or self.element_type.endswith("_door")
        is_window = self.element_type == "window" or self.element_type.endswith("_window")
        if is_door:
            self.sill_height_mm = 0.0 if self.sill_height_mm is None else self.sill_height_mm
            self.height_mm = DEFAULT_DOOR_HEIGHT_MM if self.height_mm is None else self.height_mm
        elif is_window:
            self.sill_height_mm = (
                (
                    0.0
                    if self.element_type == "floor_to_ceiling_window"
                    else DEFAULT_WINDOW_SILL_HEIGHT_MM
                )
                if self.sill_height_mm is None
                else self.sill_height_mm
            )
            self.height_mm = (
                (
                    DEFAULT_LEVEL_HEIGHT_MM
                    if self.element_type == "floor_to_ceiling_window"
                    else DEFAULT_WINDOW_HEIGHT_MM
                )
                if self.height_mm is None
                else self.height_mm
            )
        return self


class Element(StrictModel):
    """A free-standing legend element: stair, elevator, column, unknown symbol…"""

    id: str
    element_type: ElementType
    raw_text: str | None = None
    bbox_px: BBox
    room_id: str | None = None  # containing room, if inside one
    source: Source
    confidence: float = Field(ge=0, le=1)


class ParseWarning(StrictModel):
    code: str  # stable identifier, e.g. "wall_band_fallback"
    message: str
    ref: str | None = None  # id of the affected object, if any


class Unresolved(StrictModel):
    path: str  # e.g. "openings/op_3/swing"
    reason: str


class FloorPlan(StrictModel):
    schema_version: Literal["1.0", "1.1", "1.2"] = SCHEMA_VERSION  # type: ignore[assignment]
    source_file: str
    source_sha256: str
    page: int | None = None  # PDF page index; None for images
    image_width_px: int = Field(gt=0)
    image_height_px: int = Field(gt=0)
    units: Literal["mm"] = "mm"
    coordinate_system: Literal["x_right_y_down_origin_top_left"] = "x_right_y_down_origin_top_left"
    north_angle_deg: float | None = None  # 0 = up, clockwise; None if no north arrow
    level_height_mm: float = Field(default=DEFAULT_LEVEL_HEIGHT_MM, gt=0)
    scale: Scale | None
    rooms: list[Room]
    walls: list[Wall]
    openings: list[Opening]
    elements: list[Element]
    warnings: list[ParseWarning]
    unresolved: list[Unresolved]
    # The metric node graph stays empty when no scale is known.
    nodes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    rootNodeIds: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _cross_references(self) -> Self:
        ids: list[str] = [r.id for r in self.rooms]
        ids += [w.id for w in self.walls]
        ids += [o.id for o in self.openings]
        ids += [e.id for e in self.elements]
        if len(ids) != len(set(ids)):
            raise ValueError("object ids must be unique")

        room_ids = {r.id for r in self.rooms}
        wall_ids = {w.id for w in self.walls}
        for wall in self.walls:
            unknown = set(wall.rooms) - room_ids - {EXTERIOR, UNKNOWN}
            if unknown:
                raise ValueError(f"wall {wall.id} references unknown rooms: {sorted(unknown)}")
        for opening in self.openings:
            if opening.wall_id is not None and opening.wall_id not in wall_ids:
                raise ValueError(f"opening {opening.id} references unknown wall {opening.wall_id}")
            if opening.connects is not None:
                bad = set(opening.connects) - room_ids - {EXTERIOR, UNKNOWN}
                if bad:
                    raise ValueError(f"opening {opening.id} connects unknown rooms: {sorted(bad)}")
        for element in self.elements:
            if element.room_id is not None and element.room_id not in room_ids:
                raise ValueError(f"element {element.id} references unknown room {element.room_id}")

        self._check_mm_presence()
        return self

    def _check_mm_presence(self) -> None:
        """mm fields are all-or-nothing, tied to the presence of scale."""
        want = self.scale is not None
        for room in self.rooms:
            mm_fields = (room.polygon_mm, room.area_sqm, room.perimeter_mm, room.edge_lengths_mm)
            if want and any(v is None for v in mm_fields):
                raise ValueError(f"room {room.id}: scale is set but mm fields are missing")
            if not want and any(v is not None for v in mm_fields):
                raise ValueError(f"room {room.id}: mm fields present without scale")
        for wall in self.walls:
            wall_mm = (wall.start_mm, wall.end_mm, wall.thickness_mm)
            if want and any(v is None for v in wall_mm):
                raise ValueError(f"wall {wall.id}: scale is set but mm fields are missing")
            if not want and any(v is not None for v in wall_mm):
                raise ValueError(f"wall {wall.id}: mm fields present without scale")
        for opening in self.openings:
            if want and opening.width_mm is None:
                raise ValueError(f"opening {opening.id}: scale is set but width_mm is missing")
            if not want and opening.width_mm is not None:
                raise ValueError(f"opening {opening.id}: width_mm present without scale")
            if (
                want
                and opening.protrusion_polygon_px is not None
                and opening.protrusion_polygon_mm is None
            ):
                raise ValueError(
                    f"opening {opening.id}: scale is set but protrusion_polygon_mm is missing"
                )
            if not want and opening.protrusion_polygon_mm is not None:
                raise ValueError(
                    f"opening {opening.id}: protrusion_polygon_mm present without scale"
                )
