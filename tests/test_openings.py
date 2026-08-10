import cv2
import numpy as np
import pytest

from conftest import WALL_GREY, blank, draw_gap, draw_sill, draw_wall_rect
from roomify.merge import RoomDraft, merge_rooms
from roomify.openings import (
    WallSegment,
    derive_wall_segments,
    find_openings,
    measure_bay_protrusion,
)
from roomify.rooms import detect_rooms
from roomify.walls import extract_walls


def _drafts(img):
    wx = extract_walls(img)
    return wx, merge_rooms(detect_rooms(wx).rooms, None, img.shape[:2]).rooms


def _two_room_plan(with_sills=True):
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    cv2.line(img, (300, 100), (300, 400), (WALL_GREY,) * 3, 8)
    draw_gap(img, 294, 200, 306, 240)
    if with_sills:
        draw_sill(img, 296, 200, 296, 240)
        draw_sill(img, 304, 200, 304, 240)
    return img


def test_door_gap_found_with_connects_and_wall():
    img = _two_room_plan()
    wx, drafts = _drafts(img)
    cands, segments = find_openings(wx, drafts, img, door_px=40)
    doors = [c for c in cands if set(c.connects) == {0, 1}]
    assert len(doors) == 1
    door = doors[0]
    assert door.axis == "v"
    assert door.width_px == pytest.approx(40, abs=8)
    assert segments[door.wall_index].rooms in ((0, 1), (1, 0))
    assert door.arc is None  # no arc drawn -> no swing invented


def test_window_hint_from_parallel_strokes():
    img = _two_room_plan()
    draw_gap(img, 150, 96, 240, 104)
    for off in (-3, 0, 3):
        draw_sill(img, 150, 100 + off, 240, 100 + off, grey=130)
    wx, drafts = _drafts(img)
    cands, _ = find_openings(wx, drafts, img, door_px=40)
    windows = [c for c in cands if c.kind_hint == "window" and c.center[1] < 110]
    assert len(windows) == 1
    assert "exterior" in windows[0].connects


def test_tinted_sector_yields_swing():
    img = _two_room_plan()
    # left room floor: flat warm fill; swing sector: clearly lighter tint
    cv2.rectangle(img, (110, 110), (290, 390), (168, 196, 216), -1)
    center = (296, 240)  # hinge at the lower jamb, sweeping into left room
    axes = (40, 40)
    cv2.ellipse(img, center, axes, 0, 180, 270, (215, 235, 245), -1)
    wx, drafts = _drafts(img)
    cands, _ = find_openings(wx, drafts, img, door_px=40)
    doors = [c for c in cands if set(c.connects) == {0, 1}]
    assert len(doors) == 1
    arc = doors[0].arc
    assert arc is not None
    assert arc.hinge[1] == pytest.approx(240, abs=8)
    left_room_idx = next(
        i for i, d in enumerate(drafts) if d.seed[0] < 300 and d.source != "vlm"
    )
    assert arc.opens_into == left_room_idx


def test_flat_room_has_no_arc():
    img = _two_room_plan()
    wx, drafts = _drafts(img)
    cands, _ = find_openings(wx, drafts, img, door_px=40)
    assert all(c.arc is None for c in cands)


def test_wide_passage_detected():
    # A wide open mouth (balcony/kitchen style): the floor-transition strokes
    # real plans draw across it keep the two rooms separable, and the whole
    # mouth must come back as ONE wide opening, not be silently dropped.
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    cv2.line(img, (300, 100), (300, 400), (WALL_GREY,) * 3, 8)
    draw_gap(img, 294, 150, 306, 280)  # 130px-wide passage, no door
    draw_sill(img, 296, 150, 296, 280)
    draw_sill(img, 304, 150, 304, 280)
    wx, drafts = _drafts(img)
    cands, _ = find_openings(wx, drafts, img, door_px=40)
    passages = [c for c in cands if set(c.connects) == {0, 1}]
    assert len(passages) == 1
    assert passages[0].width_px == pytest.approx(130, abs=10)
    assert passages[0].arc is None


def test_wall_segments_shared_and_exterior():
    img = _two_room_plan()
    wx, drafts = _drafts(img)
    _, segments = find_openings(wx, drafts, img, door_px=40)
    shared = [s for s in segments if isinstance(s.rooms[0], int) and isinstance(s.rooms[1], int)]
    assert len(shared) >= 1  # the divider
    assert any("exterior" in s.rooms for s in segments)
    divider = shared[0]
    assert abs(divider.start[0] - 300) < 8 and abs(divider.end[0] - 300) < 8


def test_exterior_segments_use_wall_center_not_inner_face():
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    wx, drafts = _drafts(img)

    segments = derive_wall_segments(drafts, wx)

    exterior = [segment for segment in segments if "exterior" in segment.rooms]
    fixed = [
        (segment.start[1] + segment.end[1]) / 2
        for segment in exterior
        if abs(segment.end[0] - segment.start[0]) > 300
    ] + [
        (segment.start[0] + segment.end[0]) / 2
        for segment in exterior
        if abs(segment.end[1] - segment.start[1]) > 200
    ]
    assert sorted(fixed) == pytest.approx([100, 100, 400, 500], abs=2)


def test_bay_protrusion_measured_from_outer_and_side_strokes():
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    cv2.line(img, (180, 100), (190, 60), (80,) * 3, 1)
    cv2.line(img, (190, 60), (250, 60), (80,) * 3, 1)
    cv2.line(img, (250, 60), (260, 100), (80,) * 3, 1)
    walls = extract_walls(img)
    ring = np.array([[106.0, 106.0], [494.0, 106.0], [494.0, 394.0], [106.0, 394.0]])
    room = RoomDraft(
        polygon=ring,
        area_px=388 * 288,
        perimeter_px=2 * (388 + 288),
        edge_lengths_px=[388, 288, 388, 288],
        seed=(300, 250),
        source="cv",
        confidence=0.8,
    )
    segment = WallSegment((180, 100), (260, 100), walls.thickness_px, (0, "exterior"))

    polygon = measure_bay_protrusion(segment, (180, 94, 260, 106), walls, [room], img)

    assert polygon is not None
    assert [point[1] for point in polygon[:2]] == pytest.approx([100, 100], abs=1)
    assert [point[1] for point in polygon[2:]] == pytest.approx([60, 60], abs=4)
    assert [point[0] for point in polygon[2:]] == pytest.approx([250, 190], abs=4)


def test_unmatched_internal_wall_is_unknown():
    img = _two_room_plan()
    wx, drafts = _drafts(img)
    left_room = [draft for draft in drafts if draft.seed[0] < 300]
    segments = derive_wall_segments(left_room, wx)
    divider = [
        segment for segment in segments if abs((segment.start[0] + segment.end[0]) / 2 - 300) < 10
    ]
    assert divider
    assert all("unknown" in segment.rooms for segment in divider)


def test_tee_junction_ends_meet_perpendicular_walls():
    from roomify.openings import WallSegment, _close_tee_junctions

    horizontal = WallSegment(start=(104.0, 200.0), end=(300.0, 200.0),
                             thickness_px=8.0, rooms=(0, 1))
    vertical = WallSegment(start=(100.0, 100.0), end=(100.0, 300.0),
                           thickness_px=8.0, rooms=(0, "exterior"))
    closed = _close_tee_junctions([horizontal, vertical], t=8.0)
    assert closed[0].start == (100.0, 200.0)  # extended 4px to the crossing wall
    assert closed[1] == vertical  # the crossed wall itself is untouched


def test_short_diagonal_stubs_are_not_walls(simple_plan):
    import numpy as np

    from roomify.merge import merge_rooms
    from roomify.openings import derive_wall_segments
    from roomify.rooms import detect_rooms
    from roomify.walls import extract_walls

    wx = extract_walls(simple_plan)
    drafts = merge_rooms(detect_rooms(wx).rooms, None, simple_plan.shape[:2]).rooms
    for seg in derive_wall_segments(drafts, wx):
        dx = abs(seg.end[0] - seg.start[0])
        dy = abs(seg.end[1] - seg.start[1])
        if min(dx, dy) > 0.09 * max(dx, dy):  # diagonal
            assert float(np.hypot(dx, dy)) >= 2.5 * wx.thickness_px


def test_no_swing_sector_is_read_at_more_than_a_leaf_width():
    """A quarter-disc's radius IS the leaf: 2m+ mouths have no swing.

    Wide balcony and corridor mouths sit on a floor-tint boundary, which
    reads as a tinted sector and used to promote them to swinging doors.
    """
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    cv2.line(img, (300, 100), (300, 400), (WALL_GREY,) * 3, 8)
    draw_gap(img, 294, 150, 306, 300)  # 150px mouth = 3.75m at door_px=40
    draw_sill(img, 296, 150, 296, 300)
    draw_sill(img, 304, 150, 304, 300)
    cv2.rectangle(img, (110, 110), (290, 390), (168, 196, 216), -1)
    cv2.ellipse(img, (296, 300), (150, 150), 0, 180, 270, (215, 235, 245), -1)
    wx, drafts = _drafts(img)
    cands, _ = find_openings(wx, drafts, img, door_px=40)
    mouth = [c for c in cands if set(c.connects) == {0, 1}]
    assert len(mouth) == 1
    assert mouth[0].width_px > 100
    assert mouth[0].arc is None
    # (the same tinted sector at leaf scale still yields a swing —
    # test_tinted_sector_yields_swing)
