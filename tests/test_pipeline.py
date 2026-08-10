import cv2
import pytest

from conftest import WALL_GREY, blank, draw_gap, draw_sill, draw_wall_rect
from roomify import parse
from roomify.merge import ScaleDraft


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
