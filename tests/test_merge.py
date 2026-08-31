import numpy as np
import pytest

from roomify.merge import (
    MIN_ROOM_SQM,
    RoomDraft,
    ScaleDraft,
    apply_area_checks,
    estimate_scale,
    ground_room_inventory,
    merge_rooms,
    pixels_per_mm_in_direction,
)
from roomify.rooms import CVRoom
from roomify.vlm import ChainRead, RoomRead, SpatialRoomRead
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


def test_directional_scale_respects_anisotropic_resize():
    scale = ScaleDraft(0.1, 0.2, "dimension_chains", "medium", None, 0, 2)

    assert pixels_per_mm_in_direction(1, 0, scale) == pytest.approx(0.1)
    assert pixels_per_mm_in_direction(0, 1, scale) == pytest.approx(0.2)
    assert pixels_per_mm_in_direction(1, 1, scale) == pytest.approx(0.1264911)


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


def test_not_a_room_veto_keeps_cv_geometry_for_physical_validation():
    read = RoomRead.model_validate(
        {"rooms": {"1": {"not_a_room": True}, "2": {"name": "厨房", "room_type": "kitchen"}}}
    )
    outcome = merge_rooms([_cv_room(), _cv_room()], read, (700, 700))
    assert len(outcome.rooms) == 2
    assert outcome.rooms[0].source == "cv"
    assert outcome.rooms[0].name is None
    assert outcome.rooms[0].vlm_vetoed
    assert outcome.rooms[1].name == "厨房"
    assert any(w.code == "room_vetoed" for w in outcome.warnings)


def test_missing_entry_keeps_cv_geometry_and_records_unresolved():
    read = RoomRead.model_validate({"rooms": {"1": {"name": "卧室", "room_type": "bedroom"}}})
    outcome = merge_rooms([_cv_room(), _cv_room()], read, (700, 700))
    assert len(outcome.rooms) == 2
    assert outcome.rooms[1].source == "cv"
    assert outcome.rooms[1].name is None
    assert any(u.path == "rooms/2/name" for u in outcome.unresolved)
    assert any(w.code == "room_semantics_missing" for w in outcome.warnings)


def test_spatial_inventory_grounds_labels_and_repairs_an_ocr_area():
    def room(x0, width, height):
        polygon = np.array(
            [[x0, 0.0], [x0 + width, 0.0], [x0 + width, height], [x0, height]]
        )
        return CVRoom(
            polygon=polygon,
            area_px=width * height,
            perimeter_px=2 * (width + height),
            edge_lengths_px=[width, height, width, height],
            seed=(x0 + width / 2, height / 2),
        )

    cv_rooms = [room(0, 90, 100), room(200, 125, 100), room(400, 100, 60), room(600, 80, 50)]
    markers = RoomRead.model_validate(
        {
            "rooms": {
                "1": {"name": "阳台", "printed_area_sqm": 9},
                "2": {"name": "次卧", "printed_area_sqm": 12.48},
                "3": {"name": "露台", "printed_area_sqm": 1.96},
                "4": {"name": "卫生间", "printed_area_sqm": 4},
            }
        }
    )
    def item(name, area, label_box, room_box):
        return {
            "name": name,
            "printed_area_sqm": area,
            "label_box_2d": label_box,
            "room_box_2d": room_box,
        }

    spatial = SpatialRoomRead.model_validate(
        {
            "rooms": [
                item("阳台", 9, [40, 40, 60, 60], [0, 0, 100, 90]),
                item("次卧", 22.48, [40, 240, 60, 260], [0, 200, 100, 325]),
                item("厨房", 6, [700, 700, 720, 720], [680, 680, 740, 740]),
                item("卫生间", 4, [15, 630, 35, 650], [0, 600, 50, 680]),
                item("储物间", 2, [790, 790, 810, 810], [750, 750, 850, 850]),
            ]
        }
    )

    read = ground_room_inventory(cv_rooms, markers, spatial, (1000, 1000))

    assert read is not None
    assert [read.rooms[str(i)].name for i in range(1, 5)] == ["阳台", "次卧", "厨房", "卫生间"]
    assert read.rooms["2"].printed_area_sqm == 12.48
    assert [(room.name, room.printed_area_sqm) for room in read.extra_rooms] == [("储物间", 2)]


def test_missing_spatial_room_does_not_steal_an_overlapping_neighbour():
    spatial = SpatialRoomRead.model_validate(
        {
            "rooms": [
                {
                    "name": "卧室",
                    "room_type": "bedroom",
                    "printed_area_sqm": 12,
                    "label_box_2d": [700, 700, 720, 720],
                    "room_box_2d": [0, 0, 500, 500],
                }
            ]
        }
    )

    read = ground_room_inventory([_cv_room()], RoomRead(), spatial, (1000, 1000))

    assert read is not None
    assert read.rooms == {}
    assert [room.name for room in read.extra_rooms] == ["卧室"]


def test_unmatched_spatial_duplicate_keeps_marker_assignment():
    markers = RoomRead.model_validate(
        {"rooms": {"1": {"name": "客厅", "printed_area_sqm": 10}}}
    )
    spatial = SpatialRoomRead.model_validate(
        {
            "rooms": [
                {
                    "name": "客厅",
                    "printed_area_sqm": 10,
                    "label_box_2d": [700, 700, 720, 720],
                    "room_box_2d": [0, 0, 100, 100],
                }
            ]
        }
    )

    read = ground_room_inventory([_cv_room()], markers, spatial, (1000, 1000))

    assert read is not None
    assert read.rooms["1"].name == "客厅"


def test_spatial_inventory_keeps_semantics_for_an_unlabelled_marker():
    markers = RoomRead.model_validate(
        {
            "rooms": {
                "1": {"name": "客厅", "room_type": "living_room"},
                "2": {"name": None, "room_type": "hallway"},
            }
        }
    )
    spatial = SpatialRoomRead.model_validate(
        {
            "rooms": [
                {
                    "name": "客厅",
                    "room_type": "living_room",
                    "label_box_2d": [40, 40, 60, 60],
                    "room_box_2d": [0, 0, 100, 100],
                }
            ]
        }
    )

    read = ground_room_inventory(
        [_cv_room(), _cv_room(seed=(200, 200))], markers, spatial, (1000, 1000)
    )

    assert read is not None
    assert read.rooms["2"].name is None
    assert read.rooms["2"].room_type == "hallway"


def test_vlm_call_failure_degrades_all_rooms():
    outcome = merge_rooms([_cv_room()], None, (700, 700))
    assert outcome.rooms[0].source == "cv"
    assert not outcome.warnings  # a failed call is reported once by the pipeline
    assert outcome.unresolved


def test_unmeasured_extra_room_cannot_invent_bbox_geometry():
    read = RoomRead.model_validate(
        {
            "rooms": {},
            "extra_rooms": [
                {"box_2d": [100, 200, 300, 600], "name": "阳台", "room_type": "balcony"}
            ],
        }
    )
    outcome = merge_rooms([], read, (1000, 1000))
    assert outcome.rooms == []
    assert [warning.code for warning in outcome.warnings] == [
        "room_without_physical_evidence"
    ]


def test_grounded_label_recovers_a_textured_room_missing_from_free_floor():
    read = RoomRead.model_validate(
        {
            "rooms": {"1": {"name": "客厅"}},
            "extra_rooms": [
                {
                    "box_2d": [600, 600, 800, 800],
                    "label_box_2d": [660, 660, 700, 700],
                    "expected_area_px": 400,
                    "spatially_grounded": True,
                    "name": "卫生间",
                    "printed_area_sqm": 3.62,
                }
            ],
        }
    )

    outcome = merge_rooms(
        [_cv_room(area=400)], read, (100, 100), np.zeros((100, 100), np.uint8)
    )

    assert [room.name for room in outcome.rooms] == ["客厅", "卫生间"]
    assert outcome.rooms[1].source == "vlm"


def test_grounded_duplicate_room_is_not_emitted_over_existing_geometry():
    read = RoomRead.model_validate(
        {
            "rooms": {"1": {"name": "客厅"}},
            "extra_rooms": [
                {
                    "box_2d": [0, 0, 1000, 1000],
                    "label_box_2d": [900, 900, 920, 920],
                    "expected_area_px": 10_000,
                    "spatially_grounded": True,
                    "name": "客厅",
                    "printed_area_sqm": 10,
                }
            ],
        }
    )

    outcome = merge_rooms(
        [_cv_room()], read, (100, 100), np.zeros((100, 100), np.uint8)
    )

    assert [room.name for room in outcome.rooms] == ["客厅"]
    assert [warning.code for warning in outcome.warnings] == ["room_already_measured"]


def test_vetoed_extra_region_is_not_emitted():
    read = RoomRead.model_validate(
        {"extra_rooms": [{"box_2d": [100, 100, 200, 200], "name": "未命名"}]}
    )

    outcome = merge_rooms([], read, (100, 100))

    assert outcome.rooms == []
    assert [warning.code for warning in outcome.warnings] == ["room_vetoed"]


def test_unlabelled_extra_room_cannot_invent_bbox_geometry():
    read = RoomRead.model_validate(
        {
            "rooms": {},
            "extra_rooms": [{"box_2d": [100, 200, 300, 600], "room_type": "balcony"}],
        }
    )

    outcome = merge_rooms([], read, (1000, 1000))

    assert outcome.rooms == []
    assert [warning.code for warning in outcome.warnings] == [
        "room_without_physical_evidence"
    ]


def test_labelled_open_alcove_splits_from_larger_room():
    ring = np.array(
        [[0, 0], [100, 0], [100, 40], [140, 40], [140, 80], [100, 80], [100, 120], [0, 120]],
        dtype=np.float64,
    )
    parent = CVRoom(
        polygon=ring,
        area_px=13_600,
        perimeter_px=520,
        edge_lengths_px=[100, 40, 40, 40, 40, 40, 100, 120],
        seed=(50, 60),
    )
    read = RoomRead.model_validate(
        {
            "rooms": {"1": {"name": "客厅", "room_type": "living_room"}},
            "extra_rooms": [
                {
                    "box_2d": [1, 1, 999, 999],
                    "label_box_2d": [225, 525, 375, 675],
                    "expected_area_px": 1600,
                    "name": "储物间",
                    "room_type": "storage",
                    "printed_area_sqm": 2.0,
                }
            ],
        }
    )

    outcome = merge_rooms([parent], read, (200, 200), np.zeros((200, 200), np.uint8))

    assert [room.name for room in outcome.rooms] == ["客厅", "储物间"]
    assert outcome.rooms[0].area_px == pytest.approx(12_000)
    assert outcome.rooms[1].area_px == pytest.approx(1_600)
    assert outcome.zone_boundaries == [((100.0, 40.0), (100.0, 80.0))]


@pytest.mark.parametrize(
    ("extra_name", "extra_type"),
    [("客厅", "living_room"), ("卧室", "bedroom")],
)
def test_unmeasured_vlm_zone_does_not_split_an_existing_room(extra_name, extra_type):
    read = RoomRead.model_validate(
        {
            "rooms": {"1": {"name": "客餐厅", "room_type": "living_dining"}},
            "extra_rooms": [
                {
                    "box_2d": [100, 100, 500, 500],
                    "name": extra_name,
                    "room_type": extra_type,
                    "spatially_grounded": True,
                }
            ],
        }
    )

    outcome = merge_rooms(
        [_cv_room(area=10_000)], read, (100, 100), np.zeros((100, 100), np.uint8)
    )

    assert [room.name for room in outcome.rooms] == ["客餐厅"]
    assert outcome.zone_boundaries == []
    assert [warning.code for warning in outcome.warnings] == ["room_already_measured"]


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
    chain_x = (427 - 9) / 9705
    chain_y = (569 - 9) / 11687
    assert scale.px_per_mm_x / scale.px_per_mm_y == pytest.approx(
        chain_x / chain_y, rel=1e-6
    )
    assert scale.px_per_mm_x * scale.px_per_mm_y == pytest.approx(
        0.0431 * 0.0479, rel=1e-6
    )
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


def test_scale_uses_labelled_zone_rooms_when_all_boundaries_are_dashed():
    rooms = _fp1_rooms(sx=0.045, sy=0.045)
    for room in rooms:
        room.zone_bounded = True
    scale, _ = estimate_scale([], _walls(), rooms)
    assert scale is not None
    assert scale.px_per_mm_x == scale.px_per_mm_y == pytest.approx(0.045)
    assert scale.n_rooms_used == len(rooms)


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


def test_cv_classifies_common_door_window_and_sliding_symbols():
    from roomify.merge import merge_openings
    from roomify.openings import OpeningCandidate, WallSegment
    from roomify.vlm import OpeningsRead

    rooms = [_draft(1000.0, None), _draft(1000.0, None)]
    segment = WallSegment((0.0, 5.0), (200.0, 5.0), 10.0, (0, 1))

    def candidate(marker, width, kind, connects=(0, 1)):
        return OpeningCandidate(
            marker=marker,
            bbox=(0.0, 0.0, float(width), 10.0),
            center=(width / 2, 5.0),
            axis="h",
            width_px=float(width),
            kind_hint=kind,
            connects=connects,
            wall_index=0,
            arc=None,
        )

    candidates = [
        candidate("A", 50, "window"),
        candidate("B", 100, "window"),
        candidate("C", 40, "doorlike", (0, "exterior")),
        candidate("D", 100, "doorlike"),
        candidate("E", 50, "window", (0, "exterior")),
        candidate("F", 20, "doorlike"),
    ]
    drafts, _, _, unresolved = merge_openings(
        candidates, None, (100, 200), rooms, segments=[segment]
    )

    assert [draft.element_type for draft in drafts] == [
        "single_door",
        "sliding_door",
        "window",
        "passage",
        "window",
        "unknown_symbol",
    ]
    assert [item.path for item in unresolved if item.path.endswith("element_type")] == [
        "openings/F/element_type"
    ]

    wrong = OpeningsRead.model_validate(
        {
            "candidates": [
                {
                    "marker": "B",
                    "element_type": "passage",
                    "raw_text": None,
                    "confidence": 0.9,
                    "is_real": True,
                }
            ],
            "extra_elements": [],
        }
    )
    drafts, _, warnings, _ = merge_openings(
        [candidates[1]], wrong, (100, 200), rooms, segments=[segment]
    )
    assert drafts[0].element_type == "sliding_door"
    assert [warning.code for warning in warnings] == ["opening_reclassified_by_geometry"]

    plain_gap = candidate("G", 40, "passage")
    wrong = OpeningsRead.model_validate(
        {
            "candidates": [
                {
                    "marker": "G",
                    "element_type": "single_door",
                    "confidence": 0.9,
                    "is_real": True,
                }
            ]
        }
    )
    drafts, _, warnings, _ = merge_openings(
        [plain_gap], wrong, (100, 200), rooms, segments=[segment]
    )
    assert drafts[0].element_type == "passage"
    assert [warning.code for warning in warnings] == ["opening_reclassified_by_geometry"]


def test_shallow_balcony_edge_is_a_railing_not_a_window():
    from roomify.merge import merge_openings
    from roomify.openings import OpeningCandidate, WallSegment
    from roomify.vlm import OpeningsRead

    balcony = _draft(32000.0, None)
    balcony.polygon = np.array([[0.0, 0.0], [400.0, 0.0], [400.0, 80.0], [0.0, 80.0]])
    balcony.area_px = 32000.0
    segment = WallSegment((0.0, 80.0), (400.0, 80.0), 10.0, (0, "exterior"))
    candidate = OpeningCandidate(
        marker="A",
        bbox=(40.0, 75.0, 360.0, 85.0),
        center=(200.0, 80.0),
        axis="h",
        width_px=320.0,
        kind_hint="window",
        connects=(0, "exterior"),
        wall_index=0,
        arc=None,
    )

    drafts, _, _, unresolved = merge_openings(
        [candidate], None, (100, 400), [balcony], segments=[segment]
    )
    assert drafts[0].element_type == "railing"
    assert unresolved == []

    for wrong_type in ("window", "sliding_door"):
        wrong = OpeningsRead.model_validate(
            {
                "candidates": [
                    {
                        "marker": "A",
                        "element_type": wrong_type,
                        "confidence": 0.9,
                        "is_real": True,
                    }
                ]
            }
        )
        drafts, _, warnings, _ = merge_openings(
            [candidate], wrong, (100, 400), [balcony], segments=[segment]
        )
        assert drafts[0].element_type == "railing"
        assert [warning.code for warning in warnings] == ["opening_reclassified_by_geometry"]

    # A deep balcony still ends at a railing; depth only distinguishes
    # shallow facade strips when room semantics are unavailable.
    balcony.room_type = "balcony"
    balcony.polygon[:, 1] = [0.0, 0.0, 300.0, 300.0]
    drafts, _, warnings, _ = merge_openings(
        [candidate], wrong, (320, 400), [balcony], segments=[segment]
    )
    assert drafts[0].element_type == "railing"
    assert [warning.code for warning in warnings] == ["opening_reclassified_by_geometry"]


def test_wide_glazing_needs_a_shallow_balcony_side():
    from roomify.merge import merge_openings
    from roomify.openings import OpeningCandidate, WallSegment

    living = _draft(160000.0, None)
    living.polygon = np.array([[0.0, 0.0], [400.0, 0.0], [400.0, 400.0], [0.0, 400.0]])
    balcony = _draft(32000.0, None)
    balcony.polygon = np.array([[0.0, 410.0], [400.0, 410.0], [400.0, 490.0], [0.0, 490.0]])
    segment = WallSegment((0.0, 405.0), (400.0, 405.0), 10.0, (0, 1))
    candidate = OpeningCandidate(
        marker="A",
        bbox=(90.0, 400.0, 310.0, 410.0),
        center=(200.0, 405.0),
        axis="h",
        width_px=220.0,
        kind_hint="window",
        connects=(0, 1),
        wall_index=0,
        arc=None,
    )

    drafts, _, _, unresolved = merge_openings(
        [candidate], None, (500, 400), [living, balcony], segments=[segment]
    )
    assert drafts[0].element_type == "floor_to_ceiling_window"
    assert unresolved == []

    balcony.polygon[:, 1] = [410.0, 410.0, 710.0, 710.0]
    drafts, _, _, unresolved = merge_openings(
        [candidate], None, (720, 400), [living, balcony], segments=[segment]
    )
    assert drafts[0].element_type == "unknown_symbol"
    assert [item.path for item in unresolved] == ["openings/A/element_type"]


def test_extra_elements_never_replace_measured_opening_geometry():
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
                {
                    "marker": "A",
                    "element_type": "window",
                    "raw_text": None,
                    "confidence": 0.9,
                    "is_real": True,
                }
            ],
            "extra_elements": [
                # duplicate of A, a few px off -> dropped
                {
                    "box_2d": [195, 98, 215, 158],
                    "element_type": "window",
                    "raw_text": None,
                    "confidence": 0.8,
                    "is_real": True,
                },
                # same projection as A but a separate free-standing element -> kept
                {
                    "box_2d": [250, 100, 300, 160],
                    "element_type": "stair",
                    "raw_text": None,
                    "confidence": 0.8,
                    "is_real": True,
                },
                # same projection on a nearby parallel wall -> distinct opening
                {
                    "box_2d": [250, 100, 300, 160],
                    "element_type": "window",
                    "raw_text": None,
                    "confidence": 0.8,
                    "is_real": True,
                },
                # a VLM-only opening with no host wall -> rejected
                {
                    "box_2d": [700, 700, 750, 760],
                    "element_type": "window",
                    "raw_text": None,
                    "confidence": 0.8,
                    "is_real": True,
                },
            ],
        }
    )
    drafts, elements, warnings, unresolved = merge_openings([_Cand()], read, (1000, 1000))
    assert len(drafts) == 1
    assert [e.element_type for e in elements] == ["stair"]
    assert [warning.code for warning in warnings] == [
        "opening_without_physical_evidence"
    ] * 3


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
        px_per_mm_x=0.04,
        px_per_mm_y=0.04,
        method="printed_areas",
        confidence="medium",
        px_per_mm_from_areas=0.04,
        n_rooms_used=5,
        n_chain_values_used=0,
    )


def test_reconcile_repairs_swapped_labels():
    from roomify.merge import reconcile_rooms

    # 10㎡ and 20㎡ rooms with their printed labels SWAPPED by the VLM
    rooms = [
        _sq_draft(0, 0, 126.5, printed=20.0, name="客厅", marker="1"),  # ~10㎡
        _sq_draft(300, 0, 179, printed=10.0, name="次卧", marker="2"),  # ~20㎡
        _sq_draft(600, 0, 126.5, printed=10.1, name="主卧", marker="3"),  # good pair
    ]
    fixed, warns = reconcile_rooms(rooms, _scale_1600(), 10.0)
    by_marker = {r.marker: r for r in fixed}
    assert by_marker["1"].printed_area_sqm == 10.0 and by_marker["1"].name == "次卧"
    assert by_marker["2"].printed_area_sqm == 20.0 and by_marker["2"].name == "客厅"
    assert by_marker["3"].printed_area_sqm == 10.1  # untouched
    assert sum(1 for w in warns if w.code == "label_reassigned") == 2


def test_reconcile_does_not_move_spatially_grounded_labels():
    from roomify.merge import reconcile_rooms

    rooms = [
        _sq_draft(0, 0, 126.5, printed=1, name="玄关", marker="1"),
        _sq_draft(300, 0, 40, printed=10, name="露台", marker="2"),
    ]
    for room in rooms:
        room.spatially_grounded = True

    fixed, warnings = reconcile_rooms(rooms, _scale_1600(), 8.0)

    assert [room.name for room in fixed] == ["玄关", "露台"]
    assert warnings == []


def test_reconcile_merges_adjacent_fragments():
    from roomify.merge import reconcile_rooms

    # a 4.5㎡ label sits on a 1.5㎡ fragment; the other 3㎡ fragment is
    # unnamed and 6px away — merged they match the label
    rooms = [
        _sq_draft(0, 0, 49, printed=4.5, name="卫生间", marker="1"),  # 1.5㎡
        _sq_draft(55, 0, 69.3, marker="2"),  # 3.0㎡ unnamed
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
        _sq_draft(300, 0, 60, printed=15.0, name="阳台", marker="2"),  # 2.25㎡ vs 15
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

    def classified(element_type: str, kind_hint: str = "doorlike"):
        read = OpeningsRead.model_validate(
            {
                "candidates": [
                    {
                        "marker": "A",
                        "element_type": element_type,
                        "raw_text": None,
                        "confidence": 0.95,
                        "is_real": True,
                    }
                ],
                "extra_elements": [],
            }
        )
        drafts, _, warnings, _ = merge_openings(
            [_Cand(kind_hint=kind_hint)], read, (1000, 1000)
        )
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

    for wrong in ("single_door", "sliding_door"):
        draft, warnings = classified(wrong, "double_door")
        assert draft.element_type == "double_door"
        assert [warning.code for warning in warnings] == ["opening_reclassified_by_arc"]


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
                {
                    "marker": "A",
                    "element_type": "sliding_window",
                    "raw_text": None,
                    "confidence": 0.9,
                    "is_real": True,
                }
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

    impossible = OpeningsRead.model_validate(
        {
            "candidates": [
                {
                    "marker": "A",
                    "element_type": "column",
                    "raw_text": None,
                    "confidence": 0.9,
                    "is_real": True,
                }
            ],
            "extra_elements": [],
        }
    )
    drafts, _, warnings, unresolved = merge_openings([_Cand()], impossible, (1000, 1000), indoor)
    assert drafts[0].element_type == "unknown_symbol"
    assert [warning.code for warning in warnings] == ["opening_legend_conflict"]
    assert [item.path for item in unresolved] == ["openings/A/element_type"]

    drafts, _, warnings, _ = merge_openings(
        [_Cand(connects=(0, "exterior"))], read, (1000, 1000), indoor
    )
    assert drafts[0].element_type == "sliding_window"
    assert warnings == []


def test_window_into_a_privacy_room_is_left_unresolved():
    """Do not assert a bedroom/bathroom window as an indoor opening."""
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

    def classify(element_type):
        return OpeningsRead.model_validate(
            {
                "candidates": [
                    {
                        "marker": "A",
                        "element_type": element_type,
                        "raw_text": None,
                        "confidence": 0.9,
                        "is_real": True,
                    }
                ],
                "extra_elements": [],
            }
        )

    rooms = [_draft(1000.0, None, name="客厅"), _draft(1000.0, None, name="卧室")]
    rooms[0].room_type = "living_room"
    rooms[1].room_type = "bedroom"
    drafts, _, warnings, unresolved = merge_openings(
        [_Cand()], classify("fixed_window"), (1000, 1000), rooms
    )
    assert drafts[0].element_type == "unknown_symbol"
    assert [warning.code for warning in warnings] == ["opening_habitability_conflict"]
    assert [item.path for item in unresolved] == ["openings/A/element_type"]

    # Sliding symbols have a deterministic indoor interpretation: a door.
    drafts, _, _, _ = merge_openings([_Cand()], classify("sliding_window"), (1000, 1000), rooms)
    assert drafts[0].element_type == "sliding_door"

    # A serving window between kitchen and living space remains possible.
    rooms[1].room_type = "kitchen"
    drafts, _, warnings, _ = merge_openings(
        [_Cand()], classify("fixed_window"), (1000, 1000), rooms
    )
    assert drafts[0].element_type == "fixed_window"
    assert warnings == []

    # A home cannot have a permanently open passage through its envelope.
    drafts, _, warnings, unresolved = merge_openings(
        [_Cand(connects=(0, "exterior"))], classify("passage"), (1000, 1000), rooms
    )
    assert drafts[0].element_type == "unknown_symbol"
    assert [warning.code for warning in warnings] == ["opening_habitability_conflict"]
    assert [item.path for item in unresolved] == ["openings/A/element_type"]


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
                    {
                        "box_2d": box,
                        "name": "走廊",
                        "room_type": "hallway",
                        "printed_area_sqm": 9.91,
                        "confidence": 0.9,
                        "not_a_room": False,
                    }
                ],
            }
        )

    free = np.zeros((100, 100), np.uint8)
    free[10:40, 10:40] = 255  # the only unclaimed floor, top-left

    # a box over claimed floor -> rejected, and the reason is recorded
    outcome = merge_rooms([], read([500, 500, 900, 900]), (100, 100), free)
    assert outcome.rooms == []
    assert [w.code for w in outcome.warnings] == ["room_already_measured"]

    # unclaimed pixels alone do not validate an ungrounded VLM box
    outcome = merge_rooms([], read([120, 120, 380, 380]), (100, 100), free)
    assert outcome.rooms == []
    assert [warning.code for warning in outcome.warnings] == [
        "room_without_physical_evidence"
    ]

    # absence of a mask is not permission to invent geometry either
    outcome = merge_rooms([], read([500, 500, 900, 900]), (100, 100))
    assert outcome.rooms == []
    assert [warning.code for warning in outcome.warnings] == [
        "room_without_physical_evidence"
    ]


def test_wide_interior_span_is_left_unresolved_not_asserted():
    """A 3m break between two indoor rooms is not one opening.

    Openings are scanned against the ``solid`` mask, which drops partitions
    thinner than ~5px; on plans that draw them thin a whole wall reads as
    absent and comes back as a confident "passage". Pixels cannot settle it,
    so the class is reported unresolved rather than invented.
    """
    from dataclasses import dataclass
    from types import SimpleNamespace

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
                {
                    "marker": "A",
                    "element_type": "passage",
                    "raw_text": None,
                    "confidence": 0.9,
                    "is_real": True,
                }
            ],
            "extra_elements": [],
        }
    )
    indoor = [_draft(1000.0, None, name="客餐厅"), _draft(1000.0, None, name="次卧")]
    for room in indoor:
        room.room_type = "bedroom"

    drafts, _, warnings, unresolved = merge_openings([_Cand()], read, (1000, 1000), indoor, scale)
    assert drafts[0].element_type == "unknown_symbol"
    assert drafts[0].width_px == 150.0  # the measured span is still reported
    assert drafts[0].confidence <= 0.3
    assert [w.code for w in warnings] == ["opening_span_implausible"]
    assert [u.path for u in unresolved] == ["openings/A/element_type"]

    # A wide opening is normal between open-plan public zones.
    indoor[0].room_type = "living_room"
    indoor[1].room_type = "dining_room"
    drafts, _, warnings, unresolved = merge_openings([_Cand()], read, (1000, 1000), indoor, scale)
    assert drafts[0].element_type == "passage"
    assert warnings == []
    assert unresolved == []

    # a door-width break on the same wall is untouched
    drafts, _, warnings, _ = merge_openings(
        [_Cand(width_px=45.0)], read, (1000, 1000), indoor, scale
    )
    assert drafts[0].element_type == "passage"
    assert warnings == []

    drafts, _, warnings, unresolved = merge_openings(
        [_Cand(width_px=20.0)], read, (1000, 1000), indoor, scale
    )
    assert drafts[0].element_type == "unknown_symbol"
    assert [warning.code for warning in warnings] == ["opening_span_implausible"]
    assert [item.path for item in unresolved] == ["openings/A/element_type"]

    # A narrow wet-zone leaf is not a general dwelling circulation route;
    # its drawn swing arc is stronger evidence than the habitable-door bound.
    for room in indoor:
        room.room_type = "bathroom"
    arc = SimpleNamespace(radius_px=20.0, swing="clockwise", hinge=(100.0, 205.0))
    drafts, _, warnings, unresolved = merge_openings(
        [_Cand(width_px=20.0, arc=arc)], read, (1000, 1000), indoor, scale
    )
    assert drafts[0].element_type == "single_door"
    assert [warning.code for warning in warnings] == ["opening_reclassified_by_arc"]
    assert unresolved == []

    # An unsealed exterior mouth is not a habitable residential opening.
    drafts, _, warnings, unresolved = merge_openings(
        [_Cand(connects=(0, "exterior"))], read, (1000, 1000), indoor, scale
    )
    assert drafts[0].element_type == "unknown_symbol"
    assert [warning.code for warning in warnings] == ["opening_habitability_conflict"]
    assert [item.path for item in unresolved] == ["openings/A/element_type"]


def test_wide_outdoor_edge_is_not_asserted_as_a_door():
    """A 6m outdoor edge is a railing or glazing system, never one door."""
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

    def parse(element_type, rooms=None, **kw):
        read = OpeningsRead.model_validate(
            {
                "candidates": [
                    {
                        "marker": "A",
                        "element_type": element_type,
                        "raw_text": None,
                        "confidence": 0.9,
                        "is_real": True,
                    }
                ],
                "extra_elements": [],
            }
        )
        if rooms is None:
            rooms = [_draft(1000.0, None, name="阳台")]
            rooms[0].room_type = "balcony"
        return merge_openings([_Cand(**kw)], read, (1000, 1000), rooms, scale)

    drafts, _, warnings, unresolved = parse("sliding_door")
    assert drafts[0].element_type == "railing"
    assert drafts[0].width_px == 300.0  # the span is still measured and reported
    assert [w.code for w in warnings] == ["opening_reclassified_by_geometry"]
    assert unresolved == []

    # the same span read as glazing is entirely plausible and stands
    drafts, _, warnings, _ = parse("floor_to_ceiling_window")
    assert drafts[0].element_type == "floor_to_ceiling_window"
    assert warnings == []

    # and a normal balcony door is untouched
    drafts, _, warnings, _ = parse("sliding_door", width_px=140.0)  # 2800mm
    assert drafts[0].element_type == "sliding_door"
    assert warnings == []

    living = _draft(1000.0, None, name="客厅")
    living.room_type = "living_room"
    balcony = _draft(1000.0, None, name="阳台")
    balcony.room_type = "balcony"
    drafts, _, warnings, _ = parse(
        "sliding_door", rooms=[living, balcony], connects=(0, 1)
    )
    assert drafts[0].element_type == "sliding_door"
    assert warnings == []


def test_wall_less_zone_divider_cannot_become_a_door_or_window():
    from dataclasses import dataclass

    from roomify.merge import merge_openings
    from roomify.vlm import OpeningsRead

    @dataclass
    class _Cand:
        marker: str = "A"
        bbox: tuple = (100.0, 200.0, 250.0, 210.0)
        center: tuple = (175.0, 205.0)
        axis: str = "h"
        width_px: float = 150.0
        kind_hint: str = "passage"
        connects: tuple = (0, 1)
        wall_index: int = -1
        arc: object = None

    read = OpeningsRead.model_validate(
        {
            "candidates": [
                {"marker": "A", "element_type": "window", "confidence": 0.9}
            ]
        }
    )

    drafts, _, warnings, _ = merge_openings([_Cand()], read, (1000, 1000))

    assert drafts[0].element_type == "passage"
    assert [warning.code for warning in warnings] == [
        "opening_reclassified_by_geometry"
    ]


def test_vlm_only_opening_is_not_invented_without_cv_geometry():
    """A semantic box alone cannot prove a hole in a wall."""
    from roomify.merge import merge_openings
    from roomify.openings import WallSegment
    from roomify.vlm import OpeningsRead

    wall = WallSegment((100.0, 500.0), (900.0, 500.0), 10.0, (0, "exterior"))

    def extras(*boxes):
        return OpeningsRead.model_validate(
            {
                "candidates": [],
                "extra_elements": [
                    {
                        "box_2d": list(b),
                        "element_type": t,
                        "raw_text": None,
                        "confidence": 0.8,
                        "is_real": True,
                    }
                    for b, t in boxes
                ],
            }
        )

    # box_2d is [ymin, xmin, ymax, xmax] scaled 0-1000 over a 1000x1000 image
    near = ([515, 300, 535, 400], "window")  # 15px below the wall
    far = ([800, 300, 820, 400], "window")  # 300px away: nothing there
    free = ([800, 600, 830, 640], "column")  # free-standing, no wall needed

    drafts, elements, warnings, _ = merge_openings(
        [], extras(near, far, free), (1000, 1000), None, None, [wall]
    )

    assert drafts == []
    assert [element.element_type for element in elements] == ["column"]
    assert elements[0].bbox[1] == 800.0  # the column is left exactly where it was
    assert [w.code for w in warnings] == [
        "opening_without_physical_evidence",
        "opening_without_physical_evidence",
    ]


def test_vlm_only_door_does_not_become_a_connected_opening():
    from roomify.merge import merge_openings
    from roomify.openings import WallSegment
    from roomify.vlm import OpeningsRead

    wall = WallSegment((100.0, 500.0), (900.0, 500.0), 10.0, (0, 1))
    read = OpeningsRead.model_validate(
        {
            "candidates": [],
            "extra_elements": [
                {
                    "box_2d": [490, 300, 510, 400],
                    "element_type": "single_door",
                    "raw_text": None,
                    "confidence": 0.9,
                    "is_real": True,
                }
            ],
        }
    )

    drafts, elements, warnings, _ = merge_openings(
        [], read, (1000, 1000), None, None, [wall]
    )

    assert elements == []
    assert drafts == []
    assert [warning.code for warning in warnings] == [
        "opening_without_physical_evidence"
    ]
