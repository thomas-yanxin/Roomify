import numpy as np
import pytest

from roomify.merge import (
    MIN_ROOM_SQM,
    RoomDraft,
    ScaleDraft,
    apply_area_checks,
    estimate_scale,
    merge_rooms,
)
from roomify.rooms import CVRoom
from roomify.vlm import ChainRead, RoomRead
from roomify.walls import WallExtraction


def _cv_room(area=10000.0, seed=(100.0, 100.0)):
    side = area**0.5
    return CVRoom(
        polygon=np.array([[0.0, 0.0], [side, 0.0], [side, side], [0.0, side]]),
        area_px=area,
        perimeter_px=4 * side,
        edge_lengths_px=[side] * 4,
        seed=seed,
    )


def _walls(footprint=(0, 0, 427, 569), thickness=9.0):
    empty = np.zeros((10, 10), np.uint8)
    return WallExtraction(
        solid=empty,
        lines=empty,
        union=empty,
        band=(130, 185),
        band_fallback=False,
        thickness_px=thickness,
        footprint=footprint,
    )


def _draft(area_px, printed, source="cv+vlm", name="房间"):
    side = area_px**0.5
    return RoomDraft(
        polygon=np.array([[0.0, 0.0], [side, 0.0], [side, side], [0.0, side]]),
        area_px=area_px,
        perimeter_px=4 * side,
        edge_lengths_px=[side] * 4,
        seed=(side / 2, side / 2),
        source=source,
        confidence=0.9,
        name=name,
        printed_area_sqm=printed,
    )


# ---------------------------------------------------------------- merge_rooms


def test_merge_by_marker_id_not_position():
    read = RoomRead.model_validate(
        {
            "rooms": {
                "2": {"name": "客厅", "room_type": "living_room", "confidence": 0.9},
                "1": {"name": "卧室", "room_type": "bedroom", "confidence": 0.9},
            }
        }
    )
    outcome = merge_rooms([_cv_room(), _cv_room()], read, (700, 700))
    assert outcome.rooms[0].name == "卧室"
    assert outcome.rooms[1].name == "客厅"
    assert all(r.source == "cv+vlm" for r in outcome.rooms)


def test_not_a_room_veto_drops_polygon():
    read = RoomRead.model_validate(
        {"rooms": {"1": {"not_a_room": True}, "2": {"name": "厨房", "room_type": "kitchen"}}}
    )
    outcome = merge_rooms([_cv_room(), _cv_room()], read, (700, 700))
    assert len(outcome.rooms) == 1
    assert outcome.rooms[0].name == "厨房"
    assert any(w.code == "room_vetoed" for w in outcome.warnings)


def test_missing_entry_keeps_cv_geometry_and_records_unresolved():
    read = RoomRead.model_validate({"rooms": {"1": {"name": "卧室", "room_type": "bedroom"}}})
    outcome = merge_rooms([_cv_room(), _cv_room()], read, (700, 700))
    assert len(outcome.rooms) == 2
    assert outcome.rooms[1].source == "cv"
    assert outcome.rooms[1].name is None
    assert any(u.path == "rooms/2/name" for u in outcome.unresolved)
    assert any(w.code == "room_semantics_missing" for w in outcome.warnings)


def test_vlm_call_failure_degrades_all_rooms():
    outcome = merge_rooms([_cv_room()], None, (700, 700))
    assert outcome.rooms[0].source == "cv"
    assert not outcome.warnings  # a failed call is reported once by the pipeline
    assert outcome.unresolved


def test_extra_room_becomes_bbox_with_warning():
    read = RoomRead.model_validate(
        {
            "rooms": {},
            "extra_rooms": [
                {"box_2d": [100, 200, 300, 600], "name": "阳台", "room_type": "balcony"}
            ],
        }
    )
    outcome = merge_rooms([], read, (1000, 1000))
    room = outcome.rooms[0]
    assert room.source == "vlm"
    assert room.name == "阳台"
    assert room.area_px == pytest.approx(400 * 200)  # (x: 200-600, y: 100-300 in px)
    assert any(w.code == "room_geometry_is_bbox" for w in outcome.warnings)


# ------------------------------------------------------------- estimate_scale


def _fp1_chains(right_sum_misread=True):
    right = [3909, 1954, 1158, 1954 if right_sum_misread else 1594, 3071]
    return [
        ChainRead(side="top", values_mm=[2632, 3709, 3364]),
        ChainRead(side="bottom", values_mm=[1869, 350, 3040, 3364]),  # spans less than top
        ChainRead(side="left", values_mm=[4068, 1197, 2194, 2194, 1684, 350]),
        ChainRead(side="right", values_mm=right),
    ]


def _fp1_rooms(sx=0.0431, sy=0.0479):
    printed = [37.52, 14.28, 9.83, 9.65, 5.95, 3.59, 3.36, 2.72, 1.37]
    return [_draft(p * 1e6 * sx * sy, p) for p in printed]


def test_scale_elects_pair_matching_printed_areas():
    # The misread right chain (sum too large) and the short bottom chain are
    # both on offer; the printed-area median must elect top×left.
    scale, warnings = estimate_scale(_fp1_chains(), _walls(), _fp1_rooms())
    assert scale is not None
    assert scale.method == "dimension_chains+printed_areas"
    assert scale.px_per_mm_x == pytest.approx((427 - 9) / 9705, rel=1e-6)
    assert scale.px_per_mm_y == pytest.approx((569 - 9) / 11687, rel=1e-6)
    assert scale.confidence == "high"
    assert scale.n_rooms_used == 9
    assert not warnings


def test_scale_chains_only_prefers_longest_span():
    scale, _ = estimate_scale(_fp1_chains(right_sum_misread=False), _walls(), [])
    assert scale is not None
    assert scale.method == "dimension_chains"
    assert scale.confidence == "medium"
    assert scale.px_per_mm_x == pytest.approx((427 - 9) / 9705, rel=1e-6)  # top > bottom
    assert scale.n_rooms_used == 0


def test_scale_combines_one_axis_chain_with_printed_areas():
    walls = _walls(footprint=(0, 0, 409, 509), thickness=9.0)
    scale, warnings = estimate_scale(
        [ChainRead(side="top", values_mm=[10_000])],
        walls,
        _fp1_rooms(sx=0.04, sy=0.05),
    )
    assert scale is not None
    assert scale.method == "dimension_chains+printed_areas"
    assert scale.px_per_mm_x == pytest.approx(0.04)
    assert scale.px_per_mm_y == pytest.approx(0.05)
    assert scale.n_chain_values_used == 1
    assert not warnings


def test_scale_areas_only_assumes_isotropy():
    scale, warnings = estimate_scale([], _walls(), _fp1_rooms(sx=0.045, sy=0.045))
    assert scale is not None
    assert scale.method == "printed_areas"
    assert scale.px_per_mm_x == scale.px_per_mm_y == pytest.approx(0.045, rel=0.01)
    assert any(w.code == "isotropy_assumed" for w in warnings)


def test_scale_ignores_vlm_bbox_rooms():
    rooms = [_draft(1e6 * 0.002, 1.0, source="vlm") for _ in range(5)]
    scale, warnings = estimate_scale([], _walls(), rooms)
    assert scale is None
    assert any(w.code == "no_scale" for w in warnings)


def test_scale_none_without_sources():
    scale, warnings = estimate_scale([], _walls(), [])
    assert scale is None
    assert any(w.code == "no_scale" for w in warnings)


# ---------------------------------------------------------- apply_area_checks


def test_tiny_nameless_room_dropped():
    scale = ScaleDraft(
        px_per_mm_x=0.044,
        px_per_mm_y=0.048,
        method="dimension_chains",
        confidence="high",
        px_per_mm_from_areas=None,
        n_rooms_used=0,
        n_chain_values_used=4,
    )
    tiny_px = MIN_ROOM_SQM * 0.5 * 1e6 * scale.px_per_mm_x * scale.px_per_mm_y
    tiny = _draft(tiny_px, printed=None, name=None)
    named = _draft(tiny_px, printed=None, name="储物间")
    big = _draft(9e6 * scale.px_per_mm_x * scale.px_per_mm_y, printed=None, name=None)
    outcome = apply_area_checks([tiny, named, big], scale)
    assert [r.name for r in outcome.rooms] == ["储物间", None]
    assert any(w.code == "tiny_room_dropped" for w in outcome.warnings)


def test_no_scale_keeps_everything():
    tiny = _draft(100.0, printed=None, name=None)
    outcome = apply_area_checks([tiny], None)
    assert len(outcome.rooms) == 1


# ------------------------------------------------------------ merge_openings


def test_extra_elements_deduped_against_marked_openings():
    from dataclasses import dataclass

    from roomify.merge import merge_openings
    from roomify.vlm import OpeningsRead

    @dataclass
    class _Cand:  # minimal OpeningCandidate stand-in
        marker: str = "A"
        bbox: tuple = (100.0, 200.0, 160.0, 210.0)
        center: tuple = (130.0, 205.0)
        axis: str = "h"
        width_px: float = 60.0
        kind_hint: str = "window"
        connects: tuple = (0, "exterior")
        wall_index: int = 0
        arc: object = None

    read = OpeningsRead.model_validate(
        {
            "candidates": [
                {"marker": "A", "element_type": "window", "raw_text": None,
                 "confidence": 0.9, "is_real": True}
            ],
            "extra_elements": [
                # duplicate of A, a few px off -> dropped
                {"box_2d": [195, 98, 215, 158], "element_type": "window",
                 "raw_text": None, "confidence": 0.8, "is_real": True},
                # genuinely elsewhere -> kept
                {"box_2d": [700, 700, 800, 800], "element_type": "stair",
                 "raw_text": None, "confidence": 0.8, "is_real": True},
            ],
        }
    )
    drafts, elements, warnings, unresolved = merge_openings([_Cand()], read, (1000, 1000))
    assert len(drafts) == 1
    assert [e.element_type for e in elements] == ["stair"]
