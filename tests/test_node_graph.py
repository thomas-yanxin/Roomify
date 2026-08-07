import pytest

from roomify.node_graph import embed_node_graph
from roomify.schema import BBox, FloorPlan, Opening, Point, Room, Scale, Wall


def _metric_plan() -> FloorPlan:
    scale = Scale(
        px_per_mm_x=0.1,
        px_per_mm_y=0.1,
        method="dimension_chains",
        confidence="high",
        anisotropy=0,
        n_rooms_used=0,
        n_chain_values_used=2,
    )
    room = Room(
        id="room_1",
        name="卧室",
        room_type="bedroom",
        polygon_px=[Point(x=0, y=0), Point(x=400, y=0), Point(x=400, y=300), Point(x=0, y=300)],
        area_px=120_000,
        perimeter_px=1400,
        edge_lengths_px=[400, 300, 400, 300],
        polygon_mm=[
            Point(x=0, y=0),
            Point(x=4000, y=0),
            Point(x=4000, y=3000),
            Point(x=0, y=3000),
        ],
        area_sqm=12,
        perimeter_mm=14_000,
        edge_lengths_mm=[4000, 3000, 4000, 3000],
        source="cv+vlm",
        confidence=0.9,
    )
    wall = Wall(
        id="wall_1",
        start_px=Point(x=0, y=0),
        end_px=Point(x=400, y=0),
        thickness_px=20,
        rooms=("room_1", "exterior"),
        start_mm=Point(x=0, y=0),
        end_mm=Point(x=4000, y=0),
        thickness_mm=200,
        source="cv",
        confidence=0.8,
    )
    door = Opening(
        id="op_1",
        element_type="single_door",
        bbox_px=BBox(x0=55, y0=-10, x1=145, y1=10),
        center_px=Point(x=100, y=0),
        width_px=90,
        width_mm=900,
        wall_id="wall_1",
        connects=("room_1", "exterior"),
        source="cv+vlm",
        confidence=0.9,
    )
    window = Opening(
        id="op_2",
        element_type="window",
        bbox_px=BBox(x0=225, y0=-10, x1=375, y1=10),
        center_px=Point(x=300, y=0),
        width_px=150,
        width_mm=1500,
        wall_id="wall_1",
        connects=("room_1", "exterior"),
        source="cv+vlm",
        confidence=0.9,
    )
    return FloorPlan(
        source_file="plan.png",
        source_sha256="0" * 64,
        image_width_px=400,
        image_height_px=300,
        scale=scale,
        rooms=[room],
        walls=[wall],
        openings=[door, window],
        elements=[],
        warnings=[],
        unresolved=[],
    )


def test_node_graph_has_valid_hierarchy_units_and_vertical_defaults():
    merged = embed_node_graph(_metric_plan())
    nodes = merged.nodes

    assert merged.rootNodeIds == ["site_roomify"]
    assert nodes["wall_1"]["children"] == ["door_op_1", "window_op_2"]
    assert nodes["wall_1"]["start"] == [0, 0]
    assert nodes["wall_1"]["end"] == [4, 0]
    assert nodes["wall_1"]["thickness"] == 0.2
    assert nodes["door_op_1"]["position"] == [1, 1.05, 0]
    assert nodes["door_op_1"]["height"] == 2.1
    assert nodes["window_op_2"]["position"] == [3, 1.65, 0]
    assert nodes["window_op_2"]["height"] == 1.5
    assert nodes["zone_room_1"]["polygon"][2] == [4, 3]
    assert nodes["site_roomify"]["metadata"]["roomify"]["schema_version"] == "1.2"
    assert all(node["parentId"] is None or node["parentId"] in nodes for node in nodes.values())


def test_node_graph_can_be_embedded_without_recursion():
    merged = embed_node_graph(_metric_plan()).model_dump(mode="json")

    assert merged["rooms"][0]["id"] == "room_1"
    assert merged["rootNodeIds"] == ["site_roomify"]
    assert merged["nodes"]["wall_1"]["type"] == "wall"
    retained = merged["nodes"]["site_roomify"]["metadata"]["roomify"]
    assert "nodes" not in retained
    assert "rootNodeIds" not in retained


def test_node_graph_requires_metric_scale():
    plan = FloorPlan(
        source_file="plan.png",
        source_sha256="0" * 64,
        image_width_px=400,
        image_height_px=300,
        scale=None,
        rooms=[],
        walls=[],
        openings=[],
        elements=[],
        warnings=[],
        unresolved=[],
    )

    with pytest.raises(ValueError, match="scale is null"):
        embed_node_graph(plan)
