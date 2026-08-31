import cv2
import numpy as np
import pytest

from conftest import WALL_GREY, blank, draw_gap, draw_sill, draw_wall_rect
from roomify import parse
from roomify.merge import RoomDraft, ScaleDraft
from roomify.pipeline import (
    _drop_isolated_fixture_regions,
    _drop_unlabelled_fixture_regions,
    _inferred_zone_mask,
    _upgrade_openings_from_support,
)
from roomify.walls import WallExtraction


def test_vertical_defaults_must_fit_level():
    with pytest.raises(ValueError, match="fit within"):
        parse("unused.png", use_vlm=False, level_height_mm=2000)


def test_parse_thin_line_plan(tmp_path):
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=1, grey=20)
    path = tmp_path / "thin-line.png"
    assert cv2.imwrite(str(path), img)

    plan = parse(path, use_vlm=False)

    assert plan.rooms
    assert plan.walls
    assert all(wall.thickness_px > 0 for wall in plan.walls)


def test_inferred_zone_excludes_a_drawn_wall():
    shape = (80, 80)
    solid = np.zeros(shape, np.uint8)
    cv2.line(solid, (20, 10), (20, 60), 255, 3)
    walls = WallExtraction(
        solid=solid,
        lines=np.zeros(shape, np.uint8),
        union=solid,
        band=(100, 150),
        bands=[(100, 150)],
        band_fallback=False,
        thickness_px=8.0,
        footprint=(0, 0, 79, 79),
    )

    inferred = _inferred_zone_mask(
        walls, [((20.0, 10.0), (20.0, 60.0)), ((60.0, 10.0), (60.0, 60.0))]
    )

    assert inferred is not None
    assert inferred[35, 20] == 0
    assert inferred[35, 60] == 255


def test_unlabelled_functional_zone_is_not_a_room():
    shape = (100, 100)
    empty = np.zeros(shape, np.uint8)
    walls = WallExtraction(
        solid=empty,
        lines=empty,
        union=empty,
        band=(100, 150),
        bands=[(100, 150)],
        band_fallback=False,
        thickness_px=8,
        footprint=(0, 0, 99, 99),
    )
    room = RoomDraft(
        polygon=np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=float),
        area_px=6400,
        perimeter_px=320,
        edge_lengths_px=[80] * 4,
        seed=(50, 50),
        source="cv",
        confidence=0.3,
        zone_bounded=True,
    )

    kept, dropped = _drop_unlabelled_fixture_regions(
        [room], walls, np.full((100, 100, 3), 255, np.uint8)
    )

    assert kept == []
    assert dropped == [room]


def test_supported_unlabelled_zone_needs_an_explicit_vlm_veto_to_drop():
    shape = (100, 100)
    solid = np.zeros(shape, np.uint8)
    cv2.rectangle(solid, (10, 10), (90, 90), 255, 8)
    walls = WallExtraction(
        solid=solid,
        lines=solid,
        union=solid,
        band=(100, 150),
        bands=[(100, 150)],
        band_fallback=False,
        thickness_px=8,
        footprint=(0, 0, 99, 99),
    )
    room = RoomDraft(
        polygon=np.array([[14, 14], [86, 14], [86, 86], [14, 86]], dtype=float),
        area_px=5184,
        perimeter_px=288,
        edge_lengths_px=[72] * 4,
        seed=(50, 50),
        source="cv",
        confidence=0.3,
        zone_bounded=True,
    )
    floor = np.full((100, 100, 3), (80, 140, 190), np.uint8)

    kept, dropped = _drop_unlabelled_fixture_regions([room], walls, floor)
    assert kept == [room]
    assert dropped == []

    room.vlm_vetoed = True
    kept, dropped = _drop_unlabelled_fixture_regions([room], walls, floor)
    assert kept == []
    assert dropped == [room]


def test_unlabelled_door_pocket_is_not_a_room():
    shape = (250, 250)
    solid = np.zeros(shape, np.uint8)
    hints = np.zeros(shape, np.uint8)
    cv2.rectangle(solid, (20, 60), (220, 240), 255, 8)
    cv2.rectangle(solid, (20, 20), (50, 48), 255, 1)
    cv2.line(hints, (20, 20), (50, 20), 255, 1)
    cv2.line(hints, (50, 20), (50, 48), 255, 1)
    walls = WallExtraction(
        solid=solid,
        lines=solid,
        union=solid,
        band=(100, 150),
        bands=[(100, 150)],
        band_fallback=False,
        thickness_px=10,
        footprint=(0, 0, 249, 249),
        door_hints=hints,
    )
    pocket = RoomDraft(
        polygon=np.array([[20, 20], [50, 20], [50, 48], [20, 48]], dtype=float),
        area_px=840,
        perimeter_px=116,
        edge_lengths_px=[30, 28, 30, 28],
        seed=(35, 34),
        source="cv",
        confidence=0.3,
    )
    room = RoomDraft(
        polygon=np.array([[20, 60], [220, 60], [220, 240], [20, 240]], dtype=float),
        area_px=36_000,
        perimeter_px=760,
        edge_lengths_px=[200, 180, 200, 180],
        seed=(120, 150),
        source="cv+vlm",
        confidence=0.9,
        name="客厅",
    )

    kept, dropped = _drop_unlabelled_fixture_regions(
        [room, pocket], walls, np.full((250, 250, 3), 255, np.uint8)
    )

    assert kept == [room]
    assert dropped == [pocket]


def test_vlm_hallway_guess_does_not_protect_an_unlabelled_door_pocket():
    shape = (250, 250)
    solid = np.zeros(shape, np.uint8)
    hints = np.zeros(shape, np.uint8)
    cv2.rectangle(solid, (20, 60), (220, 240), 255, 8)
    cv2.rectangle(solid, (20, 20), (50, 48), 255, 1)
    cv2.line(hints, (20, 20), (50, 20), 255, 1)
    cv2.line(hints, (50, 20), (50, 48), 255, 1)
    walls = WallExtraction(
        solid=solid,
        lines=solid,
        union=solid,
        band=(100, 150),
        bands=[(100, 150)],
        band_fallback=False,
        thickness_px=10,
        footprint=(0, 0, 249, 249),
        door_hints=hints,
    )
    pocket = RoomDraft(
        polygon=np.array([[20, 20], [50, 20], [50, 48], [20, 48]], dtype=float),
        area_px=840,
        perimeter_px=116,
        edge_lengths_px=[30, 28, 30, 28],
        seed=(35, 34),
        source="cv+vlm",
        confidence=0.7,
        room_type="hallway",
    )
    room = RoomDraft(
        polygon=np.array([[20, 60], [220, 60], [220, 240], [20, 240]], dtype=float),
        area_px=36_000,
        perimeter_px=760,
        edge_lengths_px=[200, 180, 200, 180],
        seed=(120, 150),
        source="cv+vlm",
        confidence=0.9,
        name="客厅",
    )

    kept, dropped = _drop_unlabelled_fixture_regions(
        [room, pocket], walls, np.full((250, 250, 3), 255, np.uint8)
    )

    assert kept == [room]
    assert dropped == [pocket]


def test_vlm_name_does_not_protect_an_isolated_cabinet_void():
    walls = WallExtraction(
        solid=np.zeros((200, 200), np.uint8),
        lines=np.zeros((200, 200), np.uint8),
        union=np.zeros((200, 200), np.uint8),
        band=(100, 150),
        bands=[(100, 150)],
        band_fallback=False,
        thickness_px=8,
        footprint=(0, 0, 199, 199),
    )
    room = RoomDraft(
        polygon=np.array([[0, 0], [190, 0], [190, 190], [0, 190]], dtype=float),
        area_px=36_100,
        perimeter_px=760,
        edge_lengths_px=[190] * 4,
        seed=(95, 95),
        source="cv+vlm",
        confidence=0.9,
        name="客厅",
    )
    cabinet = RoomDraft(
        polygon=np.array([[20, 20], [40, 20], [40, 30], [20, 30]], dtype=float),
        area_px=200,
        perimeter_px=60,
        edge_lengths_px=[20, 10, 20, 10],
        seed=(30, 25),
        source="cv+vlm",
        confidence=0.7,
        name="客厅",
        room_type="living_room",
    )

    kept, dropped = _drop_isolated_fixture_regions([room, cabinet], [], walls)

    assert kept == [room]
    assert dropped == [cabinet]


def test_vlm_bbox_room_requires_a_walkable_connection():
    walls = WallExtraction(
        solid=np.zeros((200, 200), np.uint8),
        lines=np.zeros((200, 200), np.uint8),
        union=np.zeros((200, 200), np.uint8),
        band=(100, 150),
        bands=[(100, 150)],
        band_fallback=False,
        thickness_px=8,
        footprint=(0, 0, 199, 199),
    )
    room = RoomDraft(
        polygon=np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=float),
        area_px=3600,
        perimeter_px=240,
        edge_lengths_px=[60] * 4,
        seed=(50, 50),
        source="vlm",
        confidence=0.5,
        name="卫生间",
        room_type="bathroom",
    )

    kept, dropped = _drop_isolated_fixture_regions([room], [], walls)

    assert kept == []
    assert dropped == [room]


def test_dropped_void_can_complete_but_not_create_a_door():
    from roomify.openings import OpeningCandidate

    def candidate(width, kind="doorlike"):
        return OpeningCandidate(
            marker="A",
            bbox=(10, 20, 20, 20 + width),
            center=(15, 20 + width / 2),
            axis="v",
            width_px=width,
            kind_hint=kind,
            connects=(0, 1),
            wall_index=0,
            arc=None,
        )

    partial = candidate(30)
    complete = candidate(70, "double_door")

    upgraded = _upgrade_openings_from_support([partial], [complete], [0, 1], 10)

    assert len(upgraded) == 1
    assert upgraded[0].kind_hint == "double_door"
    assert upgraded[0].width_px == 70
    assert _upgrade_openings_from_support([], [complete], [0, 1], 10) == []


def test_parse_embeds_node_graph_when_scale_is_available(
    tmp_path, simple_plan, monkeypatch
):
    path = tmp_path / "calibrated.png"
    assert cv2.imwrite(str(path), simple_plan)
    scale = ScaleDraft(
        px_per_mm_x=0.1,
        px_per_mm_y=0.1,
        method="dimension_chains",
        confidence="medium",
        px_per_mm_from_areas=None,
        n_rooms_used=0,
        n_chain_values_used=2,
    )
    monkeypatch.setattr("roomify.pipeline.estimate_scale", lambda *_args: (scale, []))

    plan = parse(path, use_vlm=False)

    assert plan.rootNodeIds == ["site_roomify"]
    assert plan.nodes["site_roomify"]["type"] == "site"
    assert any(node["type"] == "wall" for node in plan.nodes.values())


def test_unresolved_paths_use_final_object_ids(tmp_path, monkeypatch):
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    cv2.line(img, (300, 100), (300, 400), (WALL_GREY,) * 3, 8)
    draw_gap(img, 296, 200, 304, 245)
    draw_sill(img, 296, 200, 296, 245)
    draw_sill(img, 304, 200, 304, 245)
    path = tmp_path / "two-rooms.png"
    assert cv2.imwrite(str(path), img)
    monkeypatch.setattr(
        "roomify.pipeline.reconcile_rooms",
        lambda rooms, *_args: (rooms[:1], []),
    )

    plan = parse(path, use_vlm=False)

    assert len(plan.rooms) == 1
    ids = {
        "rooms": {room.id for room in plan.rooms},
        "openings": {opening.id for opening in plan.openings},
    }
    assert plan.openings
    for item in plan.unresolved:
        collection, object_id, _ = item.path.split("/", 2)
        assert object_id in ids[collection]


def test_habitability_audit_reports_only_unresolved_topology():
    from types import SimpleNamespace

    from roomify.pipeline import _habitability_warnings

    rooms = [
        SimpleNamespace(id="living", name="客厅", room_type="living_room"),
        SimpleNamespace(id="bedroom", name="卧室", room_type="bedroom"),
        SimpleNamespace(id="balcony", name="阳台", room_type="balcony"),
    ]
    warnings = _habitability_warnings(rooms, [])
    assert [warning.code for warning in warnings] == [
        "dwelling_circulation_disconnected",
        "balcony_access_unresolved",
    ]

    openings = [
        SimpleNamespace(
            id="op_1", element_type="passage", connects=("living", "bedroom")
        ),
        SimpleNamespace(
            id="op_2", element_type="sliding_door", connects=("bedroom", "balcony")
        ),
    ]
    assert _habitability_warnings(rooms, openings) == []

    openings.append(
        SimpleNamespace(id="op_3", element_type="window", connects=("living", "unknown"))
    )
    warnings = _habitability_warnings(rooms, openings)
    assert [warning.code for warning in warnings] == ["opening_adjacency_unresolved"]
