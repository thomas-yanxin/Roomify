import pytest
from pydantic import ValidationError

from roomify.schema import (
    BBox,
    FloorPlan,
    Opening,
    ParseWarning,
    Point,
    Room,
    Scale,
    Wall,
)


def _room(id="room_1", **overrides):
    base = dict(
        id=id,
        name="卧室",
        room_type="bedroom",
        polygon_px=[Point(x=0, y=0), Point(x=100, y=0), Point(x=100, y=80), Point(x=0, y=80)],
        area_px=8000.0,
        perimeter_px=360.0,
        edge_lengths_px=[100.0, 80.0, 100.0, 80.0],
        source="cv+vlm",
        confidence=0.9,
    )
    base.update(overrides)
    return Room(**base)


def _plan(**overrides):
    base = dict(
        source_file="plan.png",
        source_sha256="0" * 64,
        image_width_px=800,
        image_height_px=600,
        scale=None,
        rooms=[_room()],
        walls=[],
        openings=[],
        elements=[],
        warnings=[],
        unresolved=[],
    )
    base.update(overrides)
    return FloorPlan(**base)


def test_roundtrip():
    plan = _plan()
    again = FloorPlan.model_validate_json(plan.model_dump_json())
    assert again == plan


def test_closed_ring_rejected():
    with pytest.raises(ValidationError, match="open"):
        _room(
            polygon_px=[Point(x=0, y=0), Point(x=1, y=0), Point(x=1, y=1), Point(x=0, y=0)],
            edge_lengths_px=[1.0, 1.0, 1.0, 1.0],
        )


def test_edge_lengths_must_match_vertex_count():
    with pytest.raises(ValidationError, match="edge_lengths"):
        _room(edge_lengths_px=[100.0, 80.0])


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError, match="unique"):
        _plan(rooms=[_room(), _room()])


def test_unknown_references_rejected():
    wall = Wall(
        id="wall_1",
        start_px=Point(x=0, y=0),
        end_px=Point(x=100, y=0),
        thickness_px=8.0,
        rooms=("room_1", "nope"),
        source="cv",
        confidence=0.8,
    )
    with pytest.raises(ValidationError, match="unknown rooms"):
        _plan(walls=[wall])

    opening = Opening(
        id="op_1",
        element_type="single_door",
        bbox_px=BBox(x0=10, y0=0, x1=50, y1=10),
        center_px=Point(x=30, y=5),
        width_px=40.0,
        wall_id="missing_wall",
        source="cv+vlm",
        confidence=0.7,
    )
    with pytest.raises(ValidationError, match="unknown wall"):
        _plan(openings=[opening])


def test_unknown_connects_is_valid():
    opening = Opening(
        id="op_1",
        element_type="single_door",
        bbox_px=BBox(x0=10, y0=0, x1=50, y1=10),
        center_px=Point(x=30, y=5),
        width_px=40.0,
        connects=("room_1", "unknown"),
        source="cv+vlm",
        confidence=0.7,
    )
    assert _plan(openings=[opening]).openings[0].connects == ("room_1", "unknown")


def test_exterior_is_a_valid_reference():
    wall = Wall(
        id="wall_1",
        start_px=Point(x=0, y=0),
        end_px=Point(x=100, y=0),
        thickness_px=8.0,
        rooms=("room_1", "exterior"),
        source="cv",
        confidence=0.8,
    )
    assert _plan(walls=[wall]).walls[0].rooms == ("room_1", "exterior")


def test_mm_fields_require_scale():
    with pytest.raises(ValidationError, match="without scale"):
        _plan(rooms=[_room(area_sqm=9.6)])


def test_scale_requires_mm_fields():
    scale = Scale(
        px_per_mm_x=0.044,
        px_per_mm_y=0.049,
        method="dimension_chains",
        confidence="high",
        anisotropy=0.11,
        n_rooms_used=0,
        n_chain_values_used=13,
    )
    with pytest.raises(ValidationError, match="mm fields are missing"):
        _plan(scale=scale)

    mm_room = _room(
        polygon_mm=[Point(x=0, y=0), Point(x=2262, y=0), Point(x=2262, y=1632), Point(x=0, y=1632)],
        area_sqm=3.69,
        perimeter_mm=7788.0,
        edge_lengths_mm=[2262.0, 1632.0, 2262.0, 1632.0],
    )
    plan = _plan(scale=scale, rooms=[mm_room])
    assert plan.scale is not None


def test_bbox_extent_validated():
    with pytest.raises(ValidationError, match="positive"):
        BBox(x0=10, y0=10, x1=10, y1=20)


def test_swing_defaults_absent_not_invented():
    opening = Opening(
        id="op_1",
        element_type="single_door",
        bbox_px=BBox(x0=10, y0=0, x1=50, y1=10),
        center_px=Point(x=30, y=5),
        width_px=40.0,
        source="cv+vlm",
        confidence=0.7,
    )
    plan = _plan(openings=[opening])
    assert plan.openings[0].swing is None
    assert plan.openings[0].hinge_px is None


def test_warning_shape():
    plan = _plan(warnings=[ParseWarning(code="wall_band_fallback", message="no grey peak found")])
    assert plan.warnings[0].ref is None
