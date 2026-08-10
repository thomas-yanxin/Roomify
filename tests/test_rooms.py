import cv2
import numpy as np
import pytest

from conftest import WALL_GREY, blank, draw_gap, draw_wall_rect
from roomify.rooms import detect_rooms
from roomify.walls import WallExtraction, extract_walls


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
    # bare gaps resolve only via fallback closes → composite reports that
    assert detection.strategy == "composite"
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


def test_square_corners_removes_diagonal_jogs():
    from roomify.rooms import _square_corners

    # a rectangle whose one corner is cut by a short 45° jog
    pts = np.array(
        [[0.0, 0.0], [90.0, 0.0], [100.0, 10.0], [100.0, 100.0], [0.0, 100.0]]
    )
    squared = _square_corners(pts, max_cut=20.0)
    assert len(squared) == 4
    assert [100.0, 0.0] in squared.tolist()

    # a genuine long chamfer survives
    pts_big = np.array(
        [[0.0, 0.0], [60.0, 0.0], [100.0, 40.0], [100.0, 100.0], [0.0, 100.0]]
    )
    assert len(_square_corners(pts_big, max_cut=20.0)) == 5


def test_spike_spurs_removed():
    from roomify.rooms import _remove_spikes

    # a rectangle with a needle poking inward from its bottom edge
    pts = np.array(
        [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [60.0, 100.0],
         [59.0, 80.0], [58.0, 100.0], [0.0, 100.0]]
    )
    cleaned = _remove_spikes(pts, max_len=25.0)
    assert len(cleaned) <= 6
    assert not any(abs(p[1] - 80.0) < 1 for p in cleaned)  # the tip is gone


def test_open_courtyard_is_not_closed_into_a_room():
    union = np.zeros((400, 400), np.uint8)
    cv2.line(union, (40, 40), (40, 360), 255, 10)
    cv2.line(union, (40, 40), (360, 40), 255, 10)
    cv2.line(union, (360, 40), (360, 170), 255, 10)
    cv2.line(union, (360, 230), (360, 360), 255, 10)
    cv2.line(union, (40, 360), (360, 360), 255, 10)
    walls = WallExtraction(
        solid=union,
        lines=np.zeros_like(union),
        union=union,
        band=(130, 185),
        bands=[(130, 185)],
        band_fallback=False,
        thickness_px=10.0,
        footprint=(35, 35, 365, 365),
    )

    assert detect_rooms(walls).rooms == []


def test_dashed_zone_divider_splits_and_tags():
    # 玄关/走廊-style functional split: a dashed line spanning wall to wall
    # is drawn evidence of a zone boundary — seal it, measure both zones,
    # and tag them as zone-bounded (weaker evidence than a physical wall).
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    for y in range(108, 396, 12):
        cv2.line(img, (300, y), (300, y + 6), (120, 120, 120), 2)
    det = detect_rooms(extract_walls(img))
    assert len(det.rooms) == 2, [r.area_px for r in det.rooms]
    assert all(r.zone_bounded for r in det.rooms)


def test_label_text_row_is_not_a_zone_divider():
    # A printed label floats mid-room: fused glyphs must not become a
    # boundary (they anchor on no wall).
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    cv2.putText(img, "23.5", (260, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
    det = detect_rooms(extract_walls(img))
    assert len(det.rooms) == 1
    assert not det.rooms[0].zone_bounded


def test_dot_hatch_room_survives_composition():
    # A dot-hatched bathroom only exists at the 5px close (bigger closes
    # solidify its texture into mask); the neighbouring room needs nothing.
    # Composition must keep both instead of electing one close size.
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    cv2.line(img, (300, 100), (300, 400), (WALL_GREY,) * 3, 8)
    draw_gap(img, 296, 200, 304, 240)
    for x in (296, 304):
        cv2.line(img, (x, 200), (x, 240), (120, 120, 120), 1)
    for y in range(115, 390, 7):
        for x in range(115, 285, 7):
            img[y : y + 2, x : x + 2] = 60
    det = detect_rooms(extract_walls(img))
    assert len(det.rooms) == 2, [r.area_px for r in det.rooms]


def test_composite_keeps_stable_parent_and_local_room():
    from roomify.rooms import CVRoom, RoomDetection, _composite

    def rect(x0, y0, x1, y1):
        polygon = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
        width, height = x1 - x0, y1 - y0
        return CVRoom(
            polygon=polygon,
            area_px=width * height,
            perimeter_px=2 * (width + height),
            edge_lengths_px=[width, height, width, height],
            seed=((x0 + x1) / 2, (y0 + y1) / 2),
        )

    parent = rect(0, 0, 350, 200)
    local = rect(400, 0, 500, 100)
    left, right = rect(0, 0, 175, 200), rect(175, 0, 350, 200)
    candidates = [RoomDetection([parent, local], "close5")]
    candidates += [RoomDetection([parent], f"close{i}") for i in range(5)]
    candidates.append(RoomDetection([left, right], "one-off-split"))

    result = _composite(candidates, building_area=100_000)

    assert [room.area_px for room in result.rooms] == [70_000, 10_000]


def test_one_ended_dashed_line_is_not_a_zone_boundary():
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    for x in range(108, 300, 12):
        cv2.line(img, (x, 250), (x + 6, 250), (120, 120, 120), 2)

    walls = extract_walls(img)
    detection = detect_rooms(walls)

    assert walls.zones is None
    assert len(detection.rooms) == 1
    assert not detection.rooms[0].zone_bounded
