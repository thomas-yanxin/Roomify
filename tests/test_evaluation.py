from roomify.evaluation import (
    HoldoutTruth,
    ReleaseGates,
    apply_release_gates,
    build_report,
    evaluate_case,
)
from roomify.schema import BBox, FloorPlan, Opening, Point, Room, Wall


def test_perfect_cross_source_holdout_passes_every_metric():
    def room(room_id: str, x0: float, x1: float) -> Room:
        points = [Point(x=x0, y=0), Point(x=x1, y=0), Point(x=x1, y=100), Point(x=x0, y=100)]
        width = x1 - x0
        return Room(
            id=room_id,
            name=None,
            room_type="unknown_space",
            polygon_px=points,
            area_px=width * 100,
            perimeter_px=2 * (width + 100),
            edge_lengths_px=[width, 100, width, 100],
            source="cv",
            confidence=1,
        )

    rooms = [room("room_1", 0, 50), room("room_2", 50, 100)]
    wall = Wall(
        id="wall_1",
        start_px=Point(x=50, y=0),
        end_px=Point(x=50, y=100),
        thickness_px=4,
        rooms=("room_1", "room_2"),
        source="cv",
        confidence=1,
    )
    opening = Opening(
        id="opening_1",
        element_type="single_door",
        bbox_px=BBox(x0=48, y0=40, x1=52, y1=60),
        center_px=Point(x=50, y=50),
        width_px=101,
        wall_id=wall.id,
        connects=("room_1", "room_2"),
        source="cv",
        confidence=1,
    )
    passage = Opening(
        id="opening_2",
        element_type="passage",
        bbox_px=BBox(x0=48, y0=70, x1=52, y1=80),
        center_px=Point(x=50, y=75),
        width_px=11,
        wall_id=None,
        connects=("room_1", "room_2"),
        source="cv",
        confidence=1,
    )
    plan = FloorPlan(
        source_file="private.png",
        source_sha256="0" * 64,
        image_width_px=100,
        image_height_px=100,
        scale=None,
        rooms=rooms,
        walls=[wall],
        openings=[opening, passage],
        elements=[],
        warnings=[],
        unresolved=[],
    )
    truth = HoldoutTruth(
        source_group="source-a",
        image_sha256="0" * 64,
        image_width_px=100,
        image_height_px=100,
        rooms=[
            {"id": item.id, "polygon_px": item.polygon_px}
            for item in rooms
        ],
        walls=[{"start_px": wall.start_px, "end_px": wall.end_px}],
        openings=[
            {
                "element_type": opening.element_type,
                "bbox_px": opening.bbox_px,
                "connects": opening.connects,
            },
            {
                "element_type": passage.element_type,
                "bbox_px": passage.bbox_px,
                "connects": passage.connects,
            },
        ],
    )
    cases = [
        evaluate_case(plan, truth, "a"),
        evaluate_case(plan, truth.model_copy(update={"source_group": "source-b"}), "b"),
    ]
    report = build_report(cases)
    gates = ReleaseGates(
        minimum_cases=2,
        minimum_source_groups=2,
        room_iou=1,
        wall_f1=1,
        opening_f1=1,
        adjacency_f1=1,
        physical_violation_rate=0,
    )

    assert all(report["aggregate"][metric] == 1 for metric in (
        "room_iou", "wall_f1", "opening_f1", "adjacency_f1"
    ))
    assert report["aggregate"]["physical_violation_rate"] == 0
    assert apply_release_gates(report, gates) == []
