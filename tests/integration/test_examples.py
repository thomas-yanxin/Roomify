"""End-to-end acceptance on real listing floor plans.

The first plan is bundled under examples/. A second plan can be supplied via
ROOMIFY_EXAMPLES_DIR. Live VLM credentials are always required; these tests
spend real calls and minutes of wall time.
"""

import json
import os
from pathlib import Path

import pytest

DEFAULT_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
EXAMPLES = Path(os.environ.get("ROOMIFY_EXAMPLES_DIR", DEFAULT_EXAMPLES))
FIXTURE = Path(__file__).parent.parent / "fixtures" / "fp1_expected.json"

pytestmark = [
    pytest.mark.skipif(
        not (EXAMPLES / "floorplan-1.png").exists(),
        reason="example floor plans not available (set ROOMIFY_EXAMPLES_DIR)",
    ),
    pytest.mark.skipif(
        not os.environ.get("ROOMIFY_VLM_API_KEY"),
        reason="VLM credentials not configured",
    ),
]


@pytest.fixture(scope="module")
def fp1_plan():
    from roomify import parse

    return parse(EXAMPLES / "floorplan-1.png")


def test_fp1_rooms_names_and_areas(fp1_plan):
    expected = json.loads(FIXTURE.read_text())
    assert len(fp1_plan.rooms) == len(expected["rooms"]) == 9

    got = sorted(
        ((r.name, r.printed_area_sqm) for r in fp1_plan.rooms),
        key=lambda x: (x[0] or "", x[1] or 0),
    )
    want = sorted(
        ((r["name"], r["printed_area_sqm"]) for r in expected["rooms"]),
        key=lambda x: (x[0] or "", x[1] or 0),
    )
    # allow a single VLM misread of a printed area; names must all match
    mismatches = sum(1 for g, w in zip(got, want, strict=True) if g != w)
    assert [g[0] for g in got] == [w[0] for w in want]
    assert mismatches <= 1

    within = sum(
        1
        for r in fp1_plan.rooms
        if r.area_deviation is not None and abs(r.area_deviation) <= 0.10
    )
    assert within >= 8


def test_fp1_scale(fp1_plan):
    scale = fp1_plan.scale
    assert scale is not None
    assert scale.confidence in ("high", "medium")
    assert 0.040 <= scale.px_per_mm_x <= 0.047
    assert 0.045 <= scale.px_per_mm_y <= 0.052
    assert 0.05 <= scale.anisotropy <= 0.18  # the export really is anisotropic


def test_fp1_no_invented_swings(fp1_plan):
    # floorplan-1 draws no swing arcs: every swing must be None and doors
    # must carry unresolved entries instead of fabricated facts.
    assert all(o.swing is None and o.hinge_px is None for o in fp1_plan.openings)


def test_fp1_openings_coverage(fp1_plan):
    expected = json.loads(FIXTURE.read_text())
    n = len(fp1_plan.openings)
    assert 0.75 * expected["openings_total_reference"] <= n <= 1.5 * expected[
        "openings_total_reference"
    ]
    assert sum(1 for o in fp1_plan.openings if "window" in o.element_type) >= 6


def test_fp1_document_valid_json(fp1_plan):
    from roomify.schema import FloorPlan

    round_tripped = FloorPlan.model_validate_json(fp1_plan.model_dump_json())
    assert round_tripped == fp1_plan
    assert fp1_plan.north_angle_deg == 0
    assert fp1_plan.rootNodeIds == ["site_roomify"]
    assert sum(node["type"] == "wall" for node in fp1_plan.nodes.values()) == len(
        fp1_plan.walls
    )
    assert all(
        node["parentId"] is None or node["parentId"] in fp1_plan.nodes
        for node in fp1_plan.nodes.values()
    )


@pytest.mark.skipif(
    not (EXAMPLES / "floorplan-2.png").exists(),
    reason="second example floor plan not available",
)
def test_fp2_full_parse():
    from roomify import parse

    plan = parse(EXAMPLES / "floorplan-2.png")
    assert len(plan.rooms) == 10
    names = {r.name for r in plan.rooms if r.name}
    assert {"客餐厅", "主卧", "厨房", "步入式衣柜", "卫生间"} <= names
    flagged = [r for r in plan.rooms if r.area_deviation_flag]
    assert len(flagged) <= 1
    doors = [o for o in plan.openings if o.element_type.endswith("_door")]
    assert len(doors) >= 4
    assert any(o.swing is not None for o in doors), "fp2 draws swing arcs"
    # The wide kitchen mouth is drawn with a thin double-line track band:
    # "passage" and "sliding_door" are both defensible reads of it.
    assert any(
        o.element_type in ("passage", "sliding_door") and (o.width_mm or 0) > 1800
        for o in plan.openings
    )
