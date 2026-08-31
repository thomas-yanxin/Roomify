"""Frozen examples regression; exact pixels here are not release evidence."""

import math
from pathlib import Path

import pytest
from shapely.geometry import LineString, Polygon

from roomify import parse
from roomify.io import load
from roomify.walls import estimate_plan_rotation, extract_walls

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
pytestmark = pytest.mark.dev_corpus


def test_all_example_plans_keep_physical_geometry():
    paths = sorted(EXAMPLES.glob("floorplan-*.png"))
    assert paths

    for path in paths:
        source = load(path)
        rotation = estimate_plan_rotation(source.bgr)
        extraction = extract_walls(source.bgr)
        plan = parse(path, use_vlm=False)
        wall_by_id = {wall.id: wall for wall in plan.walls}

        assert plan.rooms, path.name
        assert plan.walls, path.name
        polygons = [
            Polygon([(point.x, point.y) for point in room.polygon_px])
            for room in plan.rooms
        ]
        assert all(polygon.is_valid and polygon.area > 0 for polygon in polygons), path.name
        assert all(
            left.intersection(right).area <= 1
            for index, left in enumerate(polygons)
            for right in polygons[index + 1 :]
        ), path.name
        # CV-only must find a wall break for every room; class may honestly
        # stay unresolved until VLM review.
        walk_through = {
            "single_door",
            "double_door",
            "sliding_door",
            "passage",
            "unknown_symbol",
        }
        assert all(
            any(
                room.id in opening.connects and opening.element_type in walk_through
                for opening in plan.openings
            )
            for room in plan.rooms
        ), path.name
        assert all(
            0 <= point.x <= plan.image_width_px and 0 <= point.y <= plan.image_height_px
            for wall in plan.walls
            for point in (wall.start_px, wall.end_px)
        ), path.name
        wall_lines = [
            LineString(
                ((wall.start_px.x, wall.start_px.y), (wall.end_px.x, wall.end_px.y))
            )
            for wall in plan.walls
        ]
        assert all(
            left.intersection(right).length <= 2
            for index, left in enumerate(wall_lines)
            for right in wall_lines[index + 1 :]
        ), path.name

        # A rectilinear drawing cannot acquire diagonal walls from door leaves
        # or contour notches. Genuine angled plans advertise a wall family (or
        # a whole-sheet rotation) before diagonal segments are allowed.
        if abs(rotation) < 3 and not extraction.angles:
            assert all(
                math.isclose(wall.start_px.x, wall.end_px.x, abs_tol=2)
                or math.isclose(wall.start_px.y, wall.end_px.y, abs_tol=2)
                for wall in plan.walls
            ), path.name
            assert all(
                math.isclose(a.x, b.x, abs_tol=1) or math.isclose(a.y, b.y, abs_tol=1)
                for room in plan.rooms
                for a, b in zip(
                    room.polygon_px,
                    room.polygon_px[1:] + room.polygon_px[:1],
                    strict=True,
                )
            ), path.name

        for opening in plan.openings:
            if opening.wall_id is None:
                assert opening.element_type == "passage", (path.name, opening.id)
                continue
            wall = wall_by_id[opening.wall_id]
            wall_length = math.hypot(
                wall.end_px.x - wall.start_px.x,
                wall.end_px.y - wall.start_px.y,
            )
            assert opening.width_px <= wall_length + 3, (path.name, opening.id)


def test_fp2_recognizes_common_opening_symbols_without_vlm():
    plan = parse(EXAMPLES / "floorplan-2.png", use_vlm=False)
    types = [opening.element_type for opening in plan.openings]

    assert types.count("single_door") == 6
    assert types.count("sliding_door") == 2
    assert types.count("window") == 6
    assert types.count("floor_to_ceiling_window") == 1
    assert types.count("railing") == 1
    assert types.count("unknown_symbol") == 0

    windows = [opening for opening in plan.openings if opening.element_type == "window"]
    top_left = min(windows, key=lambda opening: opening.bbox_px.x0)
    bottom = max(windows, key=lambda opening: opening.bbox_px.y0)
    assert (top_left.bbox_px.x0, top_left.bbox_px.x1, top_left.width_px) == (238, 324, 87)
    assert (bottom.bbox_px.x0, bottom.bbox_px.x1, bottom.width_px) == (519, 630, 112)

    entry = next(
        opening
        for opening in plan.openings
        if opening.element_type == "single_door" and "exterior" in opening.connects
    )
    ensuite = max(
        (opening for opening in plan.openings if opening.element_type == "single_door"),
        key=lambda opening: opening.bbox_px.x0,
    )
    assert (
        entry.bbox_px.x0,
        entry.bbox_px.y0,
        entry.bbox_px.x1,
        entry.bbox_px.y1,
        entry.width_px,
        entry.swing,
    ) == (
        126,
        275,
        169,
        285,
        44,
        "clockwise",
    )
    assert entry.hinge_px is not None
    assert (entry.hinge_px.x, entry.hinge_px.y) == (
        126,
        280,
    )
    assert (ensuite.bbox_px.y0, ensuite.bbox_px.y1, ensuite.width_px) == (343, 383, 41)

    glazing = next(
        opening for opening in plan.openings if opening.element_type == "floor_to_ceiling_window"
    )
    assert (glazing.bbox_px.x0, glazing.bbox_px.x1, glazing.width_px) == (251, 470, 220)


def test_dense_furnished_plans_ignore_fixture_contours():
    forbidden = {
        11: {
            (196, 113, 230, 121),
            (348, 399, 366, 407),
            (348, 416, 377, 424),
        },
        12: {(492, 195, 502, 217), (266, 384, 276, 409)},
        16: {
            (421, 460, 484, 470),
            (417, 510, 427, 558),
            (244, 687, 254, 727),
            (350, 687, 360, 727),
            (672, 687, 682, 727),
        },
        17: {(460, 510, 472, 570)},
    }
    for number, false_boxes in forbidden.items():
        plan = parse(EXAMPLES / f"floorplan-{number}.png", use_vlm=False)
        boxes = {
            (
                round(opening.bbox_px.x0),
                round(opening.bbox_px.y0),
                round(opening.bbox_px.x1),
                round(opening.bbox_px.y1),
            )
            for opening in plan.openings
        }
        assert boxes.isdisjoint(false_boxes), number

    false_walls = {
        12: {(412, 294, 433, 294)},
        16: {
            (421, 465, 470, 465),
            (421, 453, 484, 453),
            (422, 509, 422, 559),
            (246, 541, 268, 541),
            (732, 562, 753, 562),
            (300, 231, 330, 231),
        },
        17: {(290, 474, 318, 474), (351, 196, 351, 283)},
    }
    for number, forbidden_walls in false_walls.items():
        plan = parse(EXAMPLES / f"floorplan-{number}.png", use_vlm=False)
        walls = {
            tuple(
                round(value)
                for value in (
                    wall.start_px.x,
                    wall.start_px.y,
                    wall.end_px.x,
                    wall.end_px.y,
                )
            )
            for wall in plan.walls
        }
        walls |= {(x1, y1, x0, y0) for x0, y0, x1, y1 in walls}
        assert walls.isdisjoint(forbidden_walls), number


def test_split_tracks_and_open_door_leaves_are_not_duplicate_doors():
    fp6 = parse(EXAMPLES / "floorplan-6.png", use_vlm=False)
    wall_by_id = {wall.id: wall for wall in fp6.walls}
    diagonal = [
        opening
        for opening in fp6.openings
        if opening.wall_id is not None
        and "exterior" not in opening.connects
        and abs(
            wall_by_id[opening.wall_id].end_px.x
            - wall_by_id[opening.wall_id].start_px.x
        )
        > 10
        and abs(
            wall_by_id[opening.wall_id].end_px.y
            - wall_by_id[opening.wall_id].start_px.y
        )
        > 10
    ]
    assert len(diagonal) == 1
    assert diagonal[0].element_type == "sliding_door"
    assert diagonal[0].width_px > 60
    bathroom_track = next(
        opening
        for opening in fp6.openings
        if tuple(
            round(value)
            for value in (
                opening.bbox_px.x0,
                opening.bbox_px.y0,
                opening.bbox_px.x1,
                opening.bbox_px.y1,
            )
        )
        == (295, 193, 383, 199)
    )
    assert bathroom_track.element_type == "sliding_door"
    assert bathroom_track.wall_id is not None
    assert all(
        "unknown" not in wall.rooms
        for wall in fp6.walls
        if abs(wall.end_px.x - wall.start_px.x) > 3
        and abs(wall.end_px.y - wall.start_px.y) > 3
    )

    fp7 = parse(EXAMPLES / "floorplan-7.png", use_vlm=False)
    fp7_types = {
        tuple(
            round(value)
            for value in (
                opening.bbox_px.x0,
                opening.bbox_px.y0,
                opening.bbox_px.x1,
                opening.bbox_px.y1,
            )
        ): opening.element_type
        for opening in fp7.openings
    }
    assert fp7_types[(81, 213, 170, 216)] == "sliding_door"

    fp9 = parse(EXAMPLES / "floorplan-9.png", use_vlm=False)
    balcony_links = [
        opening
        for opening in fp9.openings
        if opening.element_type == "sliding_door"
        and opening.center_px.y > 0.75 * fp9.image_height_px
        and "exterior" not in opening.connects
    ]
    assert len(balcony_links) == 1
    assert balcony_links[0].wall_id is not None

    fp10 = parse(EXAMPLES / "floorplan-10.png", use_vlm=False)
    assert all(
        tuple(
            round(value)
            for value in (
                opening.bbox_px.x0,
                opening.bbox_px.y0,
                opening.bbox_px.x1,
                opening.bbox_px.y1,
            )
        )
        != (237, 421, 238, 558)
        for opening in fp10.openings
    )

    fp11 = parse(EXAMPLES / "floorplan-11.png", use_vlm=False)
    types_by_box = {
        tuple(
            round(value)
            for value in (
                opening.bbox_px.x0,
                opening.bbox_px.y0,
                opening.bbox_px.x1,
                opening.bbox_px.y1,
            )
        ): opening.element_type
        for opening in fp11.openings
    }
    assert (308, 287, 316, 333) not in types_by_box
    assert types_by_box[(312, 326, 342, 334)] == "single_door"
    assert types_by_box[(450, 140, 486, 148)] == "sliding_door"
    assert types_by_box[(265, 144, 302, 152)] == "sliding_door"
    assert types_by_box[(295, 535, 341, 543)] == "sliding_door"

    fp15 = parse(EXAMPLES / "floorplan-15.png", use_vlm=False)
    types_by_box = {
        tuple(
            round(value)
            for value in (
                opening.bbox_px.x0,
                opening.bbox_px.y0,
                opening.bbox_px.x1,
                opening.bbox_px.y1,
            )
        ): opening.element_type
        for opening in fp15.openings
    }
    assert {
        types_by_box[(252, 178, 339, 194)],
        types_by_box[(194, 381, 210, 478)],
    } == {"sliding_door"}
    assert types_by_box[(105, 310, 146, 326)] == "window"
    assert types_by_box[(146, 310, 201, 326)] == "single_door"
    assert types_by_box[(421, 317, 467, 333)] == "single_door"
    assert (406, 276, 422, 325) not in types_by_box


def test_isolated_rooms_recover_their_drawn_doors_and_fixture_voids_are_dropped():
    expected = {
        6: {("single_door", (259, 193, 283, 199))},
        7: {("single_door", (276, 417, 282, 459))},
        8: {("single_door", (553, 266, 598, 272))},
        12: {("double_door", (266, 379, 276, 451))},
        16: {
            ("single_door", (411, 263, 440, 273)),
            ("single_door", (690, 478, 700, 508)),
        },
        19: {("single_door", (546, 424, 577, 434))},
    }
    for number, required in expected.items():
        plan = parse(EXAMPLES / f"floorplan-{number}.png", use_vlm=False)
        actual = {
            (
                opening.element_type,
                tuple(
                    round(value)
                    for value in (
                        opening.bbox_px.x0,
                        opening.bbox_px.y0,
                        opening.bbox_px.x1,
                        opening.bbox_px.y1,
                    )
                ),
            )
            for opening in plan.openings
        }
        assert required <= actual, number

    assert len(parse(EXAMPLES / "floorplan-7.png", use_vlm=False).rooms) == 10
    assert len(parse(EXAMPLES / "floorplan-10.png", use_vlm=False).rooms) == 10
    assert len(parse(EXAMPLES / "floorplan-13.png", use_vlm=False).rooms) == 8
    assert len(parse(EXAMPLES / "floorplan-16.png", use_vlm=False).rooms) >= 9


def test_short_openings_and_corner_frames_keep_their_physical_class():
    fp4 = parse(EXAMPLES / "floorplan-4.png", use_vlm=False)
    assert len(fp4.rooms) == 5
    assert min(room.area_px for room in fp4.rooms) > 1_000
    fp4_types = {
        (
            opening.element_type,
            tuple(
                round(value)
                for value in (
                    opening.bbox_px.x0,
                    opening.bbox_px.y0,
                    opening.bbox_px.x1,
                    opening.bbox_px.y1,
                )
            ),
        )
        for opening in fp4.openings
    }
    assert {
        ("window", (367, 112, 391, 122)),
        ("passage", (392, 264, 402, 274)),
        ("passage", (390, 169, 403, 179)),
    } <= fp4_types

    fp5 = parse(EXAMPLES / "floorplan-5.png", use_vlm=False)
    assert len(fp5.rooms) == 6
    double_doors = [opening for opening in fp5.openings if opening.element_type == "double_door"]
    assert len(double_doors) == 1
    double_door = double_doors[0]
    assert "exterior" not in double_door.connects
    height = double_door.bbox_px.y1 - double_door.bbox_px.y0
    width = double_door.bbox_px.x1 - double_door.bbox_px.x0
    assert 40 <= height <= 70
    assert width <= height / 3

    fp13 = parse(EXAMPLES / "floorplan-13.png", use_vlm=False)
    horizontal_passages = sorted(
        (opening.bbox_px.x0, opening.bbox_px.x1)
        for opening in fp13.openings
        if opening.element_type == "passage" and round(opening.bbox_px.y0) == 477
    )
    assert horizontal_passages == [(316.0, 447.5), (447.5, 503.0)]
    assert any(
        opening.element_type == "passage"
        and tuple(
            round(value)
            for value in (
                opening.bbox_px.x0,
                opening.bbox_px.y0,
                opening.bbox_px.x1,
                opening.bbox_px.y1,
            )
        )
        == (448, 573, 501, 581)
        for opening in fp13.openings
    )

    fp8 = parse(EXAMPLES / "floorplan-8.png", use_vlm=False)
    fp19 = parse(EXAMPLES / "floorplan-19.png", use_vlm=False)
    assert any(
        opening.element_type == "double_door"
        and round(opening.bbox_px.y0) == 95
        for opening in fp8.openings
    )
    assert any(
        opening.element_type == "double_door"
        and tuple(
            round(value)
            for value in (
                opening.bbox_px.x0,
                opening.bbox_px.y0,
                opening.bbox_px.x1,
                opening.bbox_px.y1,
            )
        )
        == (484, 260, 537, 270)
        for opening in fp19.openings
    )
    upper_internal = [
        opening
        for opening in fp19.openings
        if 170 <= opening.center_px.y <= 190 and "exterior" not in opening.connects
    ]
    assert any(opening.element_type == "window" for opening in upper_internal)
    assert any(opening.element_type == "sliding_door" for opening in upper_internal)

    fp11 = parse(EXAMPLES / "floorplan-11.png", use_vlm=False)
    exterior_doors = {
        tuple(
            round(value)
            for value in (
                opening.bbox_px.x0,
                opening.bbox_px.y0,
                opening.bbox_px.x1,
                opening.bbox_px.y1,
            )
        )
        for opening in fp11.openings
        if "exterior" in opening.connects and opening.element_type.endswith("_door")
    }
    assert exterior_doors == {(507, 295, 515, 331)}


def test_exterior_swing_arcs_are_doors_even_when_the_threshold_has_window_tracks():
    expected = {
        1: set(),
        6: {("single_door", (67, 227, 114, 273))},
        7: {("single_door", (470, 124, 508, 130))},
        10: {("single_door", (526, 301, 536, 341))},
        12: {("single_door", (638, 138, 648, 180))},
        15: {("single_door", (114, 636, 159, 652))},
        17: {("single_door", (673, 283, 685, 326))},
        18: {("single_door", (519, 119, 562, 129))},
        19: {("double_door", (484, 260, 537, 270))},
    }
    for number, wanted in expected.items():
        plan = parse(EXAMPLES / f"floorplan-{number}.png", use_vlm=False)
        actual = {
            (
                opening.element_type,
                tuple(
                    round(value)
                    for value in (
                        opening.bbox_px.x0,
                        opening.bbox_px.y0,
                        opening.bbox_px.x1,
                        opening.bbox_px.y1,
                    )
                ),
            )
            for opening in plan.openings
            if "exterior" in opening.connects
            and opening.element_type.endswith("_door")
        }
        assert actual == wanted, number
