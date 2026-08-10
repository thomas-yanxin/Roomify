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
        bands=[(130, 185)],
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


def test_unlabelled_extra_room_keeps_known_type_and_bbox():
    read = RoomRead.model_validate(
        {
            "rooms": {},
            "extra_rooms": [
                {"box_2d": [100, 200, 300, 600], "room_type": "balcony"}
            ],
        }
    )

    outcome = merge_rooms([], read, (1000, 1000))

    assert len(outcome.rooms) == 1
    assert outcome.rooms[0].room_type == "balcony"
    assert outcome.rooms[0].area_px == pytest.approx(400 * 200)


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


def test_extra_elements_only_dedupe_opening_duplicates():
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
                # same projection as A but a separate free-standing element -> kept
                {"box_2d": [250, 100, 300, 160], "element_type": "stair",
                 "raw_text": None, "confidence": 0.8, "is_real": True},
                # same projection on a nearby parallel wall -> distinct opening
                {"box_2d": [250, 100, 300, 160], "element_type": "window",
                 "raw_text": None, "confidence": 0.8, "is_real": True},
                # a VLM-only opening that CV missed -> kept with approximate geometry
                {"box_2d": [700, 700, 750, 760], "element_type": "window",
                 "raw_text": None, "confidence": 0.8, "is_real": True},
            ],
        }
    )
    drafts, elements, warnings, unresolved = merge_openings([_Cand()], read, (1000, 1000))
    assert len(drafts) == 1
    assert [e.element_type for e in elements] == ["stair", "window", "window"]
    assert [warning.code for warning in warnings] == [
        "opening_geometry_is_bbox",
        "opening_geometry_is_bbox",
    ]


def test_scale_falls_back_to_areas_when_chains_disagree():
    # Chains that cannot explain the printed areas (diagonal units) must not
    # ship a bogus per-axis split; the isotropic area median wins.
    walls = _walls(footprint=(0, 0, 581, 569), thickness=6.0)
    chains = [ChainRead(side="top", values_mm=[12218]), ChainRead(side="left", values_mm=[10292])]
    rooms = [_draft(p * 1e6 * 0.032 * 0.032, p) for p in (19.46, 17.94, 21.86, 5.94, 7.07)]
    scale, warnings = estimate_scale(chains, walls, rooms)
    assert scale is not None
    assert scale.method == "printed_areas"
    assert scale.px_per_mm_x == scale.px_per_mm_y == pytest.approx(0.032, rel=0.01)
    assert any(w.code == "scale_disagreement" for w in warnings)


def _sq_draft(x0, y0, side, printed=None, name=None, marker="m"):
    ring = np.array(
        [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]],
        dtype=np.float64,
    )
    return RoomDraft(
        polygon=ring,
        area_px=float(side * side),
        perimeter_px=4.0 * side,
        edge_lengths_px=[float(side)] * 4,
        seed=(x0 + side / 2, y0 + side / 2),
        source="cv+vlm" if printed or name else "cv",
        confidence=0.8,
        name=name,
        printed_area_sqm=printed,
        marker=marker,
    )


def _scale_1600():  # 40 px/m → 1600 px² per m²
    return ScaleDraft(
        px_per_mm_x=0.04, px_per_mm_y=0.04, method="printed_areas",
        confidence="medium", px_per_mm_from_areas=0.04,
        n_rooms_used=5, n_chain_values_used=0,
    )


def test_reconcile_repairs_swapped_labels():
    from roomify.merge import reconcile_rooms

    # 10㎡ and 20㎡ rooms with their printed labels SWAPPED by the VLM
    rooms = [
        _sq_draft(0, 0, 126.5, printed=20.0, name="客厅", marker="1"),    # ~10㎡
        _sq_draft(300, 0, 179, printed=10.0, name="次卧", marker="2"),    # ~20㎡
        _sq_draft(600, 0, 126.5, printed=10.1, name="主卧", marker="3"),  # good pair
    ]
    fixed, warns = reconcile_rooms(rooms, _scale_1600(), 10.0)
    by_marker = {r.marker: r for r in fixed}
    assert by_marker["1"].printed_area_sqm == 10.0 and by_marker["1"].name == "次卧"
    assert by_marker["2"].printed_area_sqm == 20.0 and by_marker["2"].name == "客厅"
    assert by_marker["3"].printed_area_sqm == 10.1  # untouched
    assert sum(1 for w in warns if w.code == "label_reassigned") == 2


def test_reconcile_merges_adjacent_fragments():
    from roomify.merge import reconcile_rooms

    # a 4.5㎡ label sits on a 1.5㎡ fragment; the other 3㎡ fragment is
    # unnamed and 6px away — merged they match the label
    rooms = [
        _sq_draft(0, 0, 49, printed=4.5, name="卫生间", marker="1"),   # 1.5㎡
        _sq_draft(55, 0, 69.3, marker="2"),                            # 3.0㎡ unnamed
        _sq_draft(600, 0, 126.5, printed=10.0, name="主卧", marker="3"),
    ]
    fixed, warns = reconcile_rooms(rooms, _scale_1600(), 8.0)
    assert len(fixed) == 2
    merged = next(r for r in fixed if r.name == "卫生间")
    assert merged.printed_area_sqm == 4.5
    assert merged.area_px == pytest.approx(4.5 * 1600, rel=0.25)
    assert any(w.code == "rooms_merged_for_label" for w in warns)


def test_reconcile_restores_hopeless_labels_in_place():
    from roomify.merge import reconcile_rooms

    # 0.01㎡ duct labels and glazing-eaten balconies: the label fits nothing
    # better, but it was PRINTED inside this room — keep the original
    # pairing (the deviation flag downstream tells the mismatch truth).
    rooms = [
        _sq_draft(0, 0, 126.5, printed=10.0, name="主卧", marker="1"),  # good
        _sq_draft(300, 0, 60, printed=15.0, name="阳台", marker="2"),   # 2.25㎡ vs 15
    ]
    fixed, warns = reconcile_rooms(rooms, _scale_1600(), 8.0)
    by_marker = {r.marker: r for r in fixed}
    assert by_marker["1"].printed_area_sqm == 10.0  # untouched
    assert by_marker["2"].printed_area_sqm == 15.0 and by_marker["2"].name == "阳台"
    assert not any(w.code == "label_unplaced" for w in warns)


def test_reconcile_chooses_closest_fragment_sum():
    from roomify.merge import reconcile_rooms

    rooms = [
        _sq_draft(-300, 0, (20 * 1600) ** 0.5, printed=10.0, name="目标", marker="0"),
        _sq_draft(0, 0, 6000**0.5, marker="1"),
        _sq_draft(82, 0, 7600**0.5, marker="2"),
        _sq_draft(0, 82, 10000**0.5, marker="3"),
    ]

    fixed, _ = reconcile_rooms(rooms, _scale_1600(), 8.0)
    markers = {room.marker for room in fixed}

    assert "2" in markers  # 85% pair left alone
    assert "3" not in markers  # exact 100% pair merged with marker 1


def test_drawn_swing_sector_overrules_a_window_reading():
    """Pixel evidence owns swing — and therefore owns "is this a door".

    The reference endpoint reads door-width breaks with a plainly drawn
    quarter-disc as "window" with 0.95 confidence (7 cases on the corpus).
    A window has no leaf sweeping the floor, so the drawing wins.
    """
    from dataclasses import dataclass

    from roomify.merge import merge_openings
    from roomify.openings import ArcEvidence
    from roomify.vlm import OpeningsRead

    @dataclass
    class _Cand:
        marker: str = "A"
        bbox: tuple = (100.0, 200.0, 160.0, 210.0)
        center: tuple = (130.0, 205.0)
        axis: str = "h"
        width_px: float = 60.0
        kind_hint: str = "doorlike"
        connects: tuple = (0, 1)
        wall_index: int = 0
        arc: object = ArcEvidence(hinge=(100.0, 205.0), swing="clockwise", opens_into=1)

    def classified(element_type: str):
        read = OpeningsRead.model_validate(
            {
                "candidates": [
                    {"marker": "A", "element_type": element_type, "raw_text": None,
                     "confidence": 0.95, "is_real": True}
                ],
                "extra_elements": [],
            }
        )
        drafts, _, warnings, _ = merge_openings([_Cand()], read, (1000, 1000))
        return drafts[0], warnings

    for wrong in ("window", "sliding_door", "passage"):
        draft, warnings = classified(wrong)
        assert draft.element_type == "single_door", wrong
        assert draft.swing == "clockwise" and draft.hinge == (100.0, 205.0)
        assert draft.confidence <= 0.7
        assert [w.code for w in warnings] == ["opening_reclassified_by_arc"]

    # a reading that already agrees keeps its own (finer) type and confidence
    draft, warnings = classified("double_door")
    assert draft.element_type == "double_door"
    assert draft.swing == "clockwise"
    assert draft.confidence == pytest.approx(0.95)
    assert warnings == []


def test_sliding_window_between_indoor_rooms_is_a_sliding_door():
    """Same symbol, different name: only adjacency separates the two.

    A sliding door and a sliding window are both drawn as parallel
    overlapping leaves, so the reading flips on the plan's least legible
    detail. Two indoor rooms have no exterior between them.
    """
    from dataclasses import dataclass

    from roomify.merge import merge_openings
    from roomify.vlm import OpeningsRead

    @dataclass
    class _Cand:
        marker: str = "A"
        bbox: tuple = (100.0, 200.0, 160.0, 210.0)
        center: tuple = (130.0, 205.0)
        axis: str = "h"
        width_px: float = 60.0
        kind_hint: str = "window"
        connects: tuple = (0, 1)
        wall_index: int = 0
        arc: object = None

    read = OpeningsRead.model_validate(
        {
            "candidates": [
                {"marker": "A", "element_type": "sliding_window", "raw_text": None,
                 "confidence": 0.9, "is_real": True}
            ],
            "extra_elements": [],
        }
    )

    indoor = [_draft(1000.0, None, name="客厅"), _draft(1000.0, None, name="门厅")]
    for room in indoor:
        room.room_type = "living_room"
    drafts, _, warnings, _ = merge_openings([_Cand()], read, (1000, 1000), indoor)
    assert drafts[0].element_type == "sliding_door"
    assert [w.code for w in warnings] == ["opening_reclassified_by_adjacency"]

    # onto a balcony, or onto the exterior, the reading stands
    outdoor = [indoor[0], _draft(1000.0, None, name="阳台")]
    outdoor[1].room_type = "balcony"
    drafts, _, warnings, _ = merge_openings([_Cand()], read, (1000, 1000), outdoor)
    assert drafts[0].element_type == "sliding_window"
    assert warnings == []

    drafts, _, warnings, _ = merge_openings(
        [_Cand(connects=(0, "exterior"))], read, (1000, 1000), indoor
    )
    assert drafts[0].element_type == "sliding_window"
    assert warnings == []


def test_extra_room_with_no_free_floor_left_is_not_a_room():
    """Open-plan zones come back as "unmarkered rooms" and must not be emitted.

    The model reports every printed label it sees, so 走廊/玄关 inside a
    measured 客餐厅 arrive as extra_rooms. Emitting their boxes double-counts
    the floor and overlaps the polygon that already measured it.
    """
    import numpy as np

    from roomify.vlm import RoomRead

    def read(box):
        return RoomRead.model_validate(
            {
                "rooms": [],
                "extra_rooms": [
                    {"box_2d": box, "name": "走廊", "room_type": "hallway",
                     "printed_area_sqm": 9.91, "confidence": 0.9, "not_a_room": False}
                ],
            }
        )

    free = np.zeros((100, 100), np.uint8)
    free[10:40, 10:40] = 255  # the only unclaimed floor, top-left

    # a box over claimed floor -> rejected, and the reason is recorded
    outcome = merge_rooms([], read([500, 500, 900, 900]), (100, 100), free)
    assert outcome.rooms == []
    assert [w.code for w in outcome.warnings] == ["room_already_measured"]

    # a box over unclaimed floor -> kept as an approximate bbox room
    outcome = merge_rooms([], read([120, 120, 380, 380]), (100, 100), free)
    assert [r.name for r in outcome.rooms] == ["走廊"]
    assert [w.code for w in outcome.warnings] == ["room_geometry_is_bbox"]

    # without the mask the veto cannot run and nothing is dropped
    outcome = merge_rooms([], read([500, 500, 900, 900]), (100, 100))
    assert [r.name for r in outcome.rooms] == ["走廊"]


def test_wide_interior_span_is_left_unresolved_not_asserted():
    """A 3m break between two indoor rooms is not one opening.

    Openings are scanned against the ``solid`` mask, which drops partitions
    thinner than ~5px; on plans that draw them thin a whole wall reads as
    absent and comes back as a confident "passage". Pixels cannot settle it,
    so the class is reported unresolved rather than invented.
    """
    from dataclasses import dataclass

    from roomify.merge import ScaleDraft, merge_openings
    from roomify.vlm import OpeningsRead

    @dataclass
    class _Cand:
        marker: str = "A"
        bbox: tuple = (100.0, 200.0, 250.0, 210.0)
        center: tuple = (175.0, 205.0)
        axis: str = "h"
        width_px: float = 150.0  # 3000mm at 0.05 px/mm
        kind_hint: str = "doorlike"
        connects: tuple = (0, 1)
        wall_index: int = 0
        arc: object = None

    scale = ScaleDraft(0.05, 0.05, "printed_areas", "medium", 0.05, 4, 0)
    read = OpeningsRead.model_validate(
        {
            "candidates": [
                {"marker": "A", "element_type": "passage", "raw_text": None,
                 "confidence": 0.9, "is_real": True}
            ],
            "extra_elements": [],
        }
    )
    indoor = [_draft(1000.0, None, name="客餐厅"), _draft(1000.0, None, name="次卧")]
    for room in indoor:
        room.room_type = "bedroom"

    drafts, _, warnings, unresolved = merge_openings(
        [_Cand()], read, (1000, 1000), indoor, scale
    )
    assert drafts[0].element_type == "unknown_symbol"
    assert drafts[0].width_px == 150.0  # the measured span is still reported
    assert drafts[0].confidence <= 0.3
    assert [w.code for w in warnings] == ["opening_span_implausible"]
    assert [u.path for u in unresolved] == ["openings/A/element_type"]

    # a door-width break on the same wall is untouched
    drafts, _, warnings, _ = merge_openings(
        [_Cand(width_px=45.0)], read, (1000, 1000), indoor, scale
    )
    assert drafts[0].element_type == "passage"
    assert warnings == []

    # so is a wide mouth onto the exterior — only indoor↔indoor is implausible
    drafts, _, warnings, _ = merge_openings(
        [_Cand(connects=(0, "exterior"))], read, (1000, 1000), indoor, scale
    )
    assert drafts[0].element_type == "passage"
    assert warnings == []


def test_door_wider_than_a_leaf_can_be_is_not_asserted_as_a_door():
    """A 6.5m "sliding door" is a glazed facade, whatever the crop looks like."""
    from dataclasses import dataclass

    from roomify.merge import ScaleDraft, merge_openings
    from roomify.vlm import OpeningsRead

    @dataclass
    class _Cand:
        marker: str = "A"
        bbox: tuple = (100.0, 200.0, 400.0, 210.0)
        center: tuple = (250.0, 205.0)
        axis: str = "h"
        width_px: float = 300.0  # 6000mm at 0.05 px/mm
        kind_hint: str = "doorlike"
        connects: tuple = (0, "exterior")
        wall_index: int = 0
        arc: object = None

    scale = ScaleDraft(0.05, 0.05, "printed_areas", "medium", 0.05, 4, 0)

    def parse(element_type, **kw):
        read = OpeningsRead.model_validate(
            {
                "candidates": [
                    {"marker": "A", "element_type": element_type, "raw_text": None,
                     "confidence": 0.9, "is_real": True}
                ],
                "extra_elements": [],
            }
        )
        rooms = [_draft(1000.0, None, name="阳台")]
        rooms[0].room_type = "balcony"
        return merge_openings([_Cand(**kw)], read, (1000, 1000), rooms, scale)

    drafts, _, warnings, unresolved = parse("sliding_door")
    assert drafts[0].element_type == "unknown_symbol"
    assert drafts[0].width_px == 300.0  # the span is still measured and reported
    assert [w.code for w in warnings] == ["opening_span_implausible"]
    assert [u.path for u in unresolved] == ["openings/A/element_type"]

    # the same span read as glazing is entirely plausible and stands
    drafts, _, warnings, _ = parse("floor_to_ceiling_window")
    assert drafts[0].element_type == "floor_to_ceiling_window"
    assert warnings == []

    # and a normal balcony door is untouched
    drafts, _, warnings, _ = parse("sliding_door", width_px=140.0)  # 2800mm
    assert drafts[0].element_type == "sliding_door"
    assert warnings == []


def test_vlm_only_opening_is_snapped_to_its_wall_or_dropped():
    """A door or window is a hole in a wall; it cannot float in a room.

    VLM box_2d coordinates drift: on the corpus 18 of 49 opening-shaped
    extras sat beside a wall and 9 had none within three thicknesses, up to
    1.4m adrift. Near ones are moved onto the wall they belong in; ones with
    no wall to be in have no geometry worth reporting.
    """
    from roomify.merge import merge_openings
    from roomify.openings import WallSegment
    from roomify.vlm import OpeningsRead

    wall = WallSegment((100.0, 500.0), (900.0, 500.0), 10.0, (0, "exterior"))

    def extras(*boxes):
        return OpeningsRead.model_validate({
            "candidates": [],
            "extra_elements": [
                {"box_2d": list(b), "element_type": t, "raw_text": None,
                 "confidence": 0.8, "is_real": True}
                for b, t in boxes
            ],
        })

    # box_2d is [ymin, xmin, ymax, xmax] scaled 0-1000 over a 1000x1000 image
    near = ([515, 300, 535, 400], "window")      # 15px below the wall
    far = ([800, 300, 820, 400], "window")       # 300px away: nothing there
    free = ([800, 600, 830, 640], "column")      # free-standing, no wall needed

    _, elements, warnings, _ = merge_openings(
        [], extras(near, far, free), (1000, 1000), None, None, [wall])

    kinds = [e.element_type for e in elements]
    assert kinds == ["window", "column"], kinds
    window = elements[0]
    assert (window.bbox[1] + window.bbox[3]) / 2 == pytest.approx(500.0, abs=0.5)
    assert window.bbox[2] - window.bbox[0] == pytest.approx(100.0)  # size kept
    assert elements[1].bbox[1] == 800.0  # the column is left exactly where it was
    assert "opening_without_a_wall" in [w.code for w in warnings]
