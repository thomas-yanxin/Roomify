import cv2
import numpy as np
import pytest

from conftest import WALL_GREY, blank, draw_gap, draw_sill, draw_wall_rect
from roomify.merge import RoomDraft, merge_rooms
from roomify.openings import (
    ArcEvidence,
    OpeningCandidate,
    WallSegment,
    _dedupe_corner_swings,
    _merge_zone_facade_panels,
    _recover_corner_swing,
    _resolve_unknown_adjacencies,
    _scan_segment,
    derive_wall_segments,
    find_openings,
    find_zone_passages,
    measure_bay_protrusion,
)
from roomify.rooms import detect_rooms
from roomify.walls import WallExtraction, extract_walls


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


def test_plain_internal_gap_without_leaf_ink_is_a_passage():
    img = _two_room_plan(with_sills=False)
    wx, drafts = _drafts(img)

    openings, _ = find_openings(wx, drafts, img, door_px=40)
    internal = [opening for opening in openings if set(opening.connects) == {0, 1}]

    assert len(internal) == 1
    assert internal[0].kind_hint == "passage"


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
    left_room_idx = next(i for i, d in enumerate(drafts) if d.seed[0] < 300 and d.source != "vlm")
    assert arc.opens_into == left_room_idx


def test_stroked_arc_tolerates_frame_width():
    img = _two_room_plan(with_sills=False)
    # The wall break includes its reveal (41px), while the drawn leaf is 36px.
    cv2.ellipse(img, (300, 240), (36, 36), 0, 180, 270, (40, 40, 40), 2)
    wx, drafts = _drafts(img)

    doors = [
        candidate
        for candidate in find_openings(wx, drafts, img, door_px=40)[0]
        if set(candidate.connects) == {0, 1}
    ]

    assert len(doors) == 1
    assert doors[0].arc is not None


def test_open_door_leaf_is_not_a_second_corner_opening():
    threshold = OpeningCandidate(
        marker="A",
        bbox=(104.0, 128.0, 140.0, 138.0),
        center=(122.0, 133.0),
        axis="h",
        width_px=36.0,
        kind_hint="doorlike",
        connects=(0, 1),
        wall_index=0,
        arc=ArcEvidence((104.0, 133.0), "clockwise", 1),
    )
    leaf = OpeningCandidate(
        marker="B",
        bbox=(94.0, 100.0, 104.0, 130.0),
        center=(99.0, 115.0),
        axis="v",
        width_px=30.0,
        kind_hint="doorlike",
        connects=(0, 1),
        wall_index=1,
        arc=ArcEvidence((99.0, 100.0), "clockwise", 1),
    )

    assert _dedupe_corner_swings([threshold, leaf], 10.0) == [threshold]


def test_weak_corner_arc_does_not_merge_different_doors():
    threshold = OpeningCandidate(
        marker="A",
        bbox=(100.0, 70.0, 108.0, 106.0),
        center=(104.0, 88.0),
        axis="v",
        width_px=36.0,
        kind_hint="doorlike",
        connects=(0, 1),
        wall_index=0,
        arc=None,
    )
    neighbour = OpeningCandidate(
        marker="B",
        bbox=(106.0, 102.0, 136.0, 110.0),
        center=(121.0, 106.0),
        axis="h",
        width_px=31.0,
        kind_hint="doorlike",
        connects=(1, "unknown"),
        wall_index=1,
        arc=ArcEvidence((108.0, 106.0), "clockwise", 1),
    )
    shape = (180, 180)
    blank_mask = np.zeros(shape, np.uint8)
    walls = WallExtraction(
        solid=blank_mask,
        lines=blank_mask,
        union=blank_mask,
        band=(100, 150),
        bands=[(100, 150)],
        band_fallback=False,
        thickness_px=10.0,
        footprint=(0, 0, 179, 179),
    )

    recovered = _recover_corner_swing(
        [threshold, neighbour], walls, [], np.zeros((*shape, 3), np.uint8), blank_mask
    )

    assert recovered == [threshold, neighbour]


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


def test_only_spatially_grounded_vlm_room_contributes_wall_segments():
    from dataclasses import replace

    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    wx, drafts = _drafts(img)
    approximate = replace(drafts[0], source="vlm")

    assert derive_wall_segments([approximate], wx) == []
    assert derive_wall_segments(
        [replace(approximate, spatially_grounded=True)], wx
    )

    divided = _two_room_plan()
    wx, drafts = _drafts(divided)
    grounded = [
        replace(room, source="vlm", spatially_grounded=True)
        if room.seed[0] > 300
        else room
        for room in drafts
    ]
    candidates, _ = find_openings(wx, grounded, divided, door_px=40)
    assert any(set(candidate.connects) == {0, 1} for candidate in candidates)


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

    # Door notches can leave the traced face almost two wall thicknesses short.
    horizontal = WallSegment(
        start=(115.0, 200.0), end=(300.0, 200.0), thickness_px=8.0, rooms=(0, 1)
    )
    vertical = WallSegment(
        start=(100.0, 100.0), end=(100.0, 300.0), thickness_px=8.0, rooms=(0, "exterior")
    )
    closed = _close_tee_junctions([horizontal, vertical], t=8.0)
    assert closed[0].start == (100.0, 200.0)
    assert closed[1] == vertical  # the crossed wall itself is untouched


def test_tee_junction_trims_short_overrun():
    from roomify.openings import WallSegment, _close_tee_junctions

    horizontal = WallSegment(
        start=(92.0, 200.0), end=(300.0, 200.0), thickness_px=8.0, rooms=(0, 1)
    )
    vertical = WallSegment(
        start=(100.0, 100.0), end=(100.0, 300.0), thickness_px=8.0, rooms=(0, "exterior")
    )

    closed = _close_tee_junctions([horizontal, vertical], t=8.0)

    assert closed[0].start == (100.0, 200.0)


def test_door_sized_gap_keeps_one_host_wall():
    from roomify.openings import _fuse_collinear_segments

    pieces = [
        WallSegment((100.0, 100.0), (100.0, 180.0), 10.0, (0, 1)),
        WallSegment((100.0, 210.0), (100.0, 300.0), 10.0, (0, 1)),
    ]
    fused = _fuse_collinear_segments(pieces, 10.0)
    assert len(fused) == 1
    assert fused[0].start == (100.0, 100.0)
    assert fused[0].end == (100.0, 300.0)


def test_diagonal_parallel_strokes_are_scanned_as_an_opening():
    from roomify.walls import WallExtraction

    shape = (220, 220)
    lines = np.zeros(shape, np.uint8)
    start, end = (40, 180), (180, 40)
    cv2.line(lines, start, end, 255, 2)
    # A second parallel leaf only over the opening interval.
    cv2.line(lines, (89, 139), (139, 89), 255, 2)
    walls = WallExtraction(
        solid=np.zeros(shape, np.uint8),
        lines=lines,
        union=lines,
        band=(130, 185),
        bands=[(130, 185)],
        band_fallback=False,
        thickness_px=6.0,
        footprint=(20, 20, 200, 200),
    )
    segment = WallSegment(start, end, 6.0, (0, 1))

    found = _scan_segment(
        0,
        segment,
        walls,
        [None, None],
        np.full(shape, 255, np.uint8),
        min_len=20,
        max_len=100,
    )

    assert len(found) == 1
    assert found[0].axis == "d"
    assert found[0].kind_hint == "window"
    assert found[0].connects == (0, 1)


def test_thin_wall_is_refined_before_its_gap_becomes_an_opening():
    from roomify.walls import WallExtraction

    shape = (160, 260)
    lines = np.zeros(shape, np.uint8)
    cv2.line(lines, (30, 80), (104, 80), 255, 2)
    cv2.line(lines, (145, 80), (230, 80), 255, 2)
    walls = WallExtraction(
        solid=np.zeros(shape, np.uint8),
        lines=lines,
        union=lines,
        band=(130, 185),
        bands=[(130, 185)],
        band_fallback=False,
        thickness_px=6.0,
        footprint=(20, 20, 240, 140),
    )
    segment = WallSegment((30, 80), (230, 80), 6.0, (0, 1))

    found = _scan_segment(
        0,
        segment,
        walls,
        [None, None],
        np.full(shape, 255, np.uint8),
        min_len=20,
        max_len=220,
    )

    assert len(found) == 1
    assert found[0].center == pytest.approx((124.5, 80.0), abs=1)
    assert found[0].width_px == pytest.approx(40, abs=3)


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


def test_diagonal_room_edge_needs_a_detected_wall_family(simple_plan):
    from dataclasses import replace

    wx = extract_walls(simple_plan)
    polygon = np.array(
        [[100.0, 100.0], [300.0, 100.0], [400.0, 200.0], [400.0, 400.0], [100.0, 400.0]]
    )
    draft = RoomDraft(
        polygon=polygon,
        area_px=85000.0,
        perimeter_px=1000.0,
        edge_lengths_px=[200.0, 141.4, 200.0, 300.0, 300.0],
        seed=(250.0, 250.0),
        source="cv",
        confidence=0.8,
    )

    without_family = derive_wall_segments([draft], wx)
    supported = wx.union.copy()
    cv2.line(supported, (300, 100), (400, 200), 255, round(wx.thickness_px))
    with_family = derive_wall_segments(
        [draft], replace(wx, angles=(45.0,), union=supported)
    )

    assert not any(
        min(abs(s.end[0] - s.start[0]), abs(s.end[1] - s.start[1])) > 1 for s in without_family
    )
    assert any(min(abs(s.end[0] - s.start[0]), abs(s.end[1] - s.start[1])) > 1 for s in with_family)


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


def test_facade_beyond_a_notch_is_exterior_not_unknown():
    """Outside-ness is the building's silhouette, not its bounding box.

    An L-shaped or diagonal footprint puts most of its facade INSIDE the
    footprint's bounding box, so a box test calls the open air behind those
    walls "interior circulation space" — which then reaches the classifier
    prompt as the wrong adjacency prior.
    """
    img = blank()
    # L-shaped outline; the top-right quadrant is open air, not a room
    corners = [(100, 100), (300, 100), (300, 250), (500, 250), (500, 400), (100, 400)]
    for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
        cv2.line(img, a, b, (WALL_GREY,) * 3, 10)
    # a window in the notch-facing wall, well inside the bounding box
    draw_gap(img, 296, 150, 304, 210)
    for off in (-3, 0, 3):
        draw_sill(img, 296 + off, 150, 296 + off, 210, grey=130)

    wx, drafts = _drafts(img)
    cands, segments = find_openings(wx, drafts, img, door_px=40)
    notch = [c for c in cands if abs(c.center[0] - 300) < 12 and 140 < c.center[1] < 220]
    assert notch, [c.center for c in cands]
    assert "exterior" in notch[0].connects, notch[0].connects


def test_wall_peninsula_yields_one_centerline_not_two_faces():
    """A wall the same room wraps around must not come back as two lines.

    Facing edges of two DIFFERENT rooms already merge onto the wall
    centerline. A stub or peninsula has the same room on both faces, so the
    pairing skipped it and emitted each face as its own segment — a double
    line down every stub (20 such pairs on the corpus).
    """
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    # a stub poking into the room from the left wall, same room both sides
    cv2.rectangle(img, (105, 240), (260, 250), (WALL_GREY,) * 3, -1)
    wx, drafts = _drafts(img)
    segments = derive_wall_segments(drafts, wx)

    stub = [
        s
        for s in segments
        if abs(s.start[1] - s.end[1]) < 3
        and 230 < (s.start[1] + s.end[1]) / 2 < 260
        and max(s.start[0], s.end[0]) < 300
    ]
    assert len(stub) == 1, [(s.start, s.end, s.rooms) for s in stub]
    assert (stub[0].start[1] + stub[0].end[1]) / 2 == pytest.approx(245, abs=3)


def test_one_wall_stretch_gets_one_segment():
    """Three edges within a thickness must not draw the wall twice.

    A recess, or a stub whose face is also another room's boundary, puts
    three parallel edges close together. Pairing every qualifying partner
    emitted a segment per pair; the nearest-first claim makes the pairing a
    partition, so each stretch of wall is drawn once.
    """
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    cv2.line(img, (300, 100), (300, 400), (WALL_GREY,) * 3, 10)
    draw_gap(img, 294, 220, 306, 260)
    draw_sill(img, 296, 220, 296, 260)
    draw_sill(img, 304, 220, 304, 260)
    # a shallow recess in the left room's face of the divider
    cv2.rectangle(img, (286, 300), (296, 380), (255, 255, 255), -1)
    wx, drafts = _drafts(img)
    segments = derive_wall_segments(drafts, wx)

    divider = [
        s
        for s in segments
        if abs(s.start[0] - s.end[0]) < 3 and 275 < (s.start[0] + s.end[0]) / 2 < 320
    ]
    # every point of the divider is covered by at most one segment
    for y in range(110, 390, 10):
        covering = [
            s
            for s in divider
            if min(s.start[1], s.end[1]) - 1 <= y <= max(s.start[1], s.end[1]) + 1
        ]
        assert len(covering) <= 1, (y, [(s.start, s.end, s.rooms) for s in covering])


def test_stub_wall_hosts_no_opening():
    """A wall with the same room on both faces cannot connect anything."""
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    cv2.rectangle(img, (105, 240), (260, 250), (WALL_GREY,) * 3, -1)
    wx, drafts = _drafts(img)
    cands, segments = find_openings(wx, drafts, img, door_px=40)
    assert any(s.rooms[0] == s.rooms[1] for s in segments), "expected a stub segment"
    assert not [c for c in cands if c.connects[0] == c.connects[1]]


def test_far_side_of_a_wall_is_the_room_behind_it():
    """A wall's far side is whatever floor is there, room included.

    Leftover edges resolve their far side by probing beyond the wall, but
    the probe only asked the building silhouette — inside/outside — so a
    room right behind the wall came back as "unknown" whenever the two
    polygons' edges had not paired. The opening then linked nothing, and
    the plan came apart into rooms you cannot walk between (12 such groups
    on the corpus).
    """
    from roomify.openings import WallSegment, _resolve_connects
    from roomify.walls import WallExtraction

    shape = (200, 300)
    left = np.zeros(shape, np.uint8)
    left[40:160, 40:140] = 255
    right = np.zeros(shape, np.uint8)
    right[40:160, 160:260] = 255  # wall body spans x 140..160
    silhouette = np.zeros(shape, np.uint8)
    silhouette[30:170, 30:270] = 255
    masks = [left, right]
    walls = WallExtraction(
        solid=np.zeros(shape, np.uint8),
        lines=np.zeros(shape, np.uint8),
        union=np.zeros(shape, np.uint8),
        band=(130, 185),
        bands=[(130, 185)],
        band_fallback=False,
        thickness_px=10.0,
        footprint=(30, 30, 269, 169),
    )
    seg = WallSegment((140.0, 50.0), (140.0, 150.0), 10.0, (0, "exterior"))

    assert _resolve_connects(seg, walls, masks, silhouette) == (0, 1)

    # …and a wall with nothing but building behind it is still "unknown"
    lonely = WallSegment((140.0, 50.0), (140.0, 150.0), 10.0, (0, "exterior"))
    assert _resolve_connects(lonely, walls, [left, None], silhouette) == (0, "unknown")


def test_unknown_opening_uses_only_a_unique_nearby_room():
    def room(x0, x1, y0, y1):
        polygon = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
        return RoomDraft(
            polygon=polygon,
            area_px=float((x1 - x0) * (y1 - y0)),
            perimeter_px=float(2 * (x1 - x0 + y1 - y0)),
            edge_lengths_px=[float(x1 - x0), float(y1 - y0)] * 2,
            seed=((x0 + x1) / 2, (y0 + y1) / 2),
            source="cv",
            confidence=0.8,
        )

    candidate = OpeningCandidate(
        marker="A",
        bbox=(45.0, 95.0, 55.0, 105.0),
        center=(50.0, 100.0),
        axis="h",
        width_px=10.0,
        kind_hint="doorlike",
        connects=(0, "unknown"),
        wall_index=0,
        arc=None,
    )
    near = _resolve_unknown_adjacencies(
        [candidate], [room(0, 100, 0, 90), room(0, 100, 110, 200)], 10.0
    )
    assert near[0].connects == (0, 1)

    ambiguous = _resolve_unknown_adjacencies(
        [candidate],
        [room(0, 100, 0, 90), room(0, 45, 110, 200), room(55, 100, 110, 200)],
        10.0,
    )
    assert ambiguous[0].connects == (0, "unknown")

    circulation_rooms = [
        room(0, 100, 0, 90),
        room(0, 45, 110, 200),
        room(55, 100, 110, 200),
    ]
    circulation_rooms[1].room_type = "living_room"
    circulation_rooms[2].room_type = "hallway"
    resolved = _resolve_unknown_adjacencies([candidate], circulation_rooms, 10.0)
    assert resolved[0].connects == (0, 2)

    recovered_landing = _resolve_unknown_adjacencies(
        [candidate],
        [room(0, 100, 0, 90), room(0, 48, 108, 200), room(60, 100, 110, 200)],
        10.0,
        exterior_rooms={1},
    )
    assert recovered_landing[0].connects == (0, 2)


def test_dashed_zone_divider_is_not_reported_as_a_wall():
    """You cannot lean on a dashed line.

    Listing plans split open space (玄关/走廊/餐厅) with dashed dividers, and
    room extraction seals along them — so the room polygon has an edge there
    and the pairing turns it into a wall segment. It is a functional split,
    not structure: 15 such segments on the corpus.
    """
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    for y in range(108, 396, 12):
        cv2.line(img, (300, y), (300, y + 6), (120, 120, 120), 2)

    wx, drafts = _drafts(img)
    assert wx.zones is not None, "fixture must produce a zone divider"
    assert len(drafts) == 2, "the divider must still split the space"

    segments = derive_wall_segments(drafts, wx)
    on_divider = [
        s
        for s in segments
        if abs((s.start[0] + s.end[0]) / 2 - 300) < 10 and abs(s.end[1] - s.start[1]) > 60
    ]
    assert on_divider == [], [(s.start, s.end, s.rooms) for s in on_divider]

    # the real walls around the space are untouched
    assert len(segments) >= 4

    left, right = sorted(drafts, key=lambda room: room.seed[0])
    left.room_type = "living_room"
    right.room_type = "dining_room"
    passages = find_zone_passages(wx, drafts, segments)
    assert len(passages) == 1
    assert set(passages[0].connects) == {0, 1}
    assert passages[0].width_px > 200

    # The pixels are physically open even when the semantics are unusual;
    # the pipeline reports the privacy conflict separately.
    right.room_type = "bedroom"
    assert len(find_zone_passages(wx, drafts, segments)) == 1


def test_zone_divider_that_leaks_into_lines_is_not_a_wall():
    """A texture-bridged dashed line is still a zone, not structure."""
    from dataclasses import replace

    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    for y in range(108, 396, 12):
        cv2.line(img, (300, y), (300, y + 6), (120, 120, 120), 2)

    wx, drafts = _drafts(img)
    assert wx.zones is not None
    leaked_lines = cv2.bitwise_or(wx.lines, wx.zones)
    leaked = replace(wx, lines=leaked_lines, union=cv2.bitwise_or(wx.solid, leaked_lines))

    segments = derive_wall_segments(drafts, leaked)

    assert not any(
        abs((segment.start[0] + segment.end[0]) / 2 - 300) < 10
        and abs(segment.end[1] - segment.start[1]) > 60
        for segment in segments
    )


def test_zone_cut_does_not_split_one_facade_opening():
    empty = np.zeros((100, 100), np.uint8)
    zones = empty.copy()
    cv2.line(zones, (30, 22), (30, 70), 255, 1)
    walls = WallExtraction(
        solid=empty,
        lines=empty,
        union=empty,
        band=(100, 150),
        bands=[(100, 150)],
        band_fallback=False,
        thickness_px=4,
        footprint=(0, 0, 99, 99),
        inferred_zones=zones,
    )
    rooms = [
        RoomDraft(
            polygon=np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float),
            area_px=area,
            perimeter_px=80,
            edge_lengths_px=[20] * 4,
            seed=(10, 10),
            source="cv",
            confidence=0.5,
            zone_bounded=True,
        )
        for area in (400, 100)
    ]
    candidates = [
        OpeningCandidate(
            "A", (10, 20, 30, 24), (20, 22), "h", 21, "window", (0, "exterior"), 0, None
        ),
        OpeningCandidate(
            "B",
            (30, 20, 45, 24),
            (37.5, 22),
            "h",
            16,
            "window",
            (1, "exterior"),
            1,
            None,
        ),
    ]
    segments = [
        WallSegment((10, 22), (30, 22), 4, (0, "exterior")),
        WallSegment((30, 22), (60, 22), 4, (1, "exterior")),
    ]

    merged, hosts = _merge_zone_facade_panels(
        candidates,
        segments,
        rooms,
        walls,
        [None, None],
        np.full((100, 100, 3), 255, np.uint8),
        empty,
    )

    assert len(merged) == 1
    assert merged[0].bbox == (10.0, 20.0, 45.0, 24.0)
    assert merged[0].connects == (0, "exterior")
    assert hosts[merged[0].wall_index].start == (10, 22.0)
    assert hosts[merged[0].wall_index].end == (45, 22.0)
