import cv2
import numpy as np
import pytest

from conftest import WALL_GREY, blank, draw_gap, draw_wall_rect
from roomify.rooms import detect_rooms
from roomify.walls import extract_walls


def test_two_rooms_sealed_by_sills(simple_plan):
    detection = detect_rooms(extract_walls(simple_plan))
    assert detection.strategy == "close5"
    assert len(detection.rooms) == 2
    # centerline convention: each half is ~200x300 px
    for room in detection.rooms:
        assert room.area_px == pytest.approx(200 * 300, rel=0.12)
        assert 4 <= len(room.polygon) <= 8


def test_fallback_ladder_bridges_bare_door_gaps():
    # No sill strokes at all: close5 sees one merged void, the directional
    # sweep must recover both rooms.
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    cv2.line(img, (300, 100), (300, 400), (WALL_GREY,) * 3, 8)
    draw_gap(img, 294, 200, 306, 240)  # 40px bare doorway
    detection = detect_rooms(extract_walls(img))
    assert detection.strategy.startswith("directional_close")
    assert len(detection.rooms) == 2


def test_l_shaped_room_vertices():
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    # carve an L: block the top-right quadrant with wall fill
    cv2.rectangle(img, (300, 100), (500, 250), (WALL_GREY,) * 3, -1)
    detection = detect_rooms(extract_walls(img))
    assert len(detection.rooms) == 1
    room = detection.rooms[0]
    assert 6 <= len(room.polygon) <= 10  # an L needs 6; noise may add a couple
    assert room.edge_lengths_px == pytest.approx(
        [float(np.linalg.norm(a - b)) for a, b in zip(
            room.polygon, np.roll(room.polygon, -1, axis=0), strict=True
        )]
    )


def test_harsh_texture_degrades_without_crashing():
    # Full-room stripes above the adaptive threshold's contrast are outside
    # the CV domain (real listing plans stay below it): the striped room may
    # be lost here — that is what the VLM extra_rooms rescue is for — but the
    # untextured room must survive and nothing may crash.
    from conftest import add_texture, draw_sill

    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    cv2.line(img, (300, 100), (300, 400), (WALL_GREY,) * 3, 8)
    draw_gap(img, 296, 200, 304, 245)
    draw_sill(img, 296, 200, 296, 245)
    draw_sill(img, 304, 200, 304, 245)
    add_texture(img, 110, 110, 290, 390, harsh=True)
    detection = detect_rooms(extract_walls(img))
    assert any(r.area_px > 20000 and r.seed[0] > 300 for r in detection.rooms)


def test_column_island_is_not_a_room():
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    cv2.rectangle(img, (280, 220), (330, 270), (WALL_GREY,) * 3, -1)  # column
    detection = detect_rooms(extract_walls(img))
    assert len(detection.rooms) == 1  # the room around it, not the column top


def test_seed_lies_inside_polygon(simple_plan):
    from shapely.geometry import Point, Polygon

    for room in detect_rooms(extract_walls(simple_plan)).rooms:
        assert Polygon(room.polygon).contains(Point(room.seed))
