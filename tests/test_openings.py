import cv2
import pytest

from conftest import WALL_GREY, blank, draw_gap, draw_sill, draw_wall_rect
from roomify.merge import merge_rooms
from roomify.openings import derive_wall_segments, find_openings
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


def test_unmatched_internal_wall_is_unknown():
    img = _two_room_plan()
    wx, drafts = _drafts(img)
    left_room = [draft for draft in drafts if draft.seed[0] < 300]
    segments = derive_wall_segments(left_room, wx)
    divider = [
        segment
        for segment in segments
        if abs((segment.start[0] + segment.end[0]) / 2 - 300) < 10
    ]
    assert divider
    assert all("unknown" in segment.rooms for segment in divider)
