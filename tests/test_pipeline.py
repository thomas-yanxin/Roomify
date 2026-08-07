import cv2

from conftest import WALL_GREY, blank, draw_gap, draw_sill, draw_wall_rect
from roomify import parse


def test_parse_thin_line_plan(tmp_path):
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=1, grey=20)
    path = tmp_path / "thin-line.png"
    assert cv2.imwrite(str(path), img)

    plan = parse(path, use_vlm=False)

    assert plan.rooms
    assert plan.walls
    assert all(wall.thickness_px > 0 for wall in plan.walls)


def test_unresolved_paths_use_final_object_ids(tmp_path):
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    cv2.line(img, (300, 100), (300, 400), (WALL_GREY,) * 3, 8)
    draw_gap(img, 296, 200, 304, 245)
    draw_sill(img, 296, 200, 296, 245)
    draw_sill(img, 304, 200, 304, 245)
    path = tmp_path / "two-rooms.png"
    assert cv2.imwrite(str(path), img)

    plan = parse(path, use_vlm=False)

    ids = {
        "rooms": {room.id for room in plan.rooms},
        "openings": {opening.id for opening in plan.openings},
    }
    assert plan.openings
    for item in plan.unresolved:
        collection, object_id, _ = item.path.split("/", 2)
        assert object_id in ids[collection]
