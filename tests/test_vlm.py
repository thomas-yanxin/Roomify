import json
from types import SimpleNamespace

import numpy as np
import pytest

from roomify.rooms import CVRoom
from roomify.vlm import (
    ChainRead,
    PlanRead,
    RoomRead,
    VLMClient,
    VLMUnavailable,
    extra_elements_prompt,
    marker_ids,
    remap_room_sheet_extras,
    render_candidate_sheet,
    render_room_overlay,
    render_room_sheet,
    room_semantics_prompt,
)


def test_client_requires_env(monkeypatch):
    for var in ("ROOMIFY_VLM_API_KEY", "ROOMIFY_VLM_BASE_URL", "ROOMIFY_VLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(VLMUnavailable, match="ROOMIFY_VLM_API_KEY"):
        VLMClient()


def test_room_entry_leniency():
    read = RoomRead.model_validate(
        {
            "rooms": {
                "1": {"name": "卧室", "room_type": "bedroom", "printed_area_sqm": "9.65㎡"},
                "2": {"name": None, "room_type": "master bedroom!!", "confidence": 3},
            },
            "extra_rooms": [
                {"box_2d": [100, 50, 400, 300], "name": "阳台", "room_type": "balcony"},
                {"box_2d": [999, 0, 5], "name": "bad box is dropped"},
                "not even a dict",
            ],
        }
    )
    assert read.rooms["1"].printed_area_sqm == 9.65
    assert read.rooms["2"].room_type == "unknown_space"
    assert read.rooms["2"].confidence == 1.0
    assert len(read.extra_rooms) == 1
    assert read.extra_rooms[0].box_2d == [100, 50, 400, 300]


def test_room_sheet_extra_box_maps_from_overlay_panel():
    read = RoomRead.model_validate(
        {"extra_rooms": [{"box_2d": [100, 870, 200, 910], "room_type": "storage"}]}
    )

    mapped = remap_room_sheet_extras(read, panel_width=120)

    assert mapped.extra_rooms[0].box_2d == pytest.approx([100, 731.333, 200, 814])


def test_room_prompt_keeps_secondary_labelled_spaces():
    prompt = room_semantics_prompt(2, review_sheet=True)

    assert "Do not force an inventory label onto a marker" in prompt
    assert "even when the overlay polygon covers it" in prompt
    assert "with a toilet is a bathroom" in prompt
    assert "never box a bed, sofa, dining table" in prompt


def test_vetoed_room_drops_service_shaft_semantics():
    from roomify.vlm import RoomEntry

    entry = RoomRead.model_validate(
        {
            "rooms": {
                "1": {
                    "name": "管道",
                    "room_type": "storage",
                    "printed_area_sqm": 0.25,
                    "not_a_room": False,
                }
            }
        }
    ).rooms["1"]

    assert entry.name is None
    assert entry.printed_area_sqm is None
    assert entry.not_a_room
    assert RoomEntry(name="未命名", printed_area_sqm=0.01).not_a_room
    assert RoomEntry(name="入户花园", room_type="multipurpose").room_type == "entrance"


def test_plan_read_leniency():
    read = PlanRead.model_validate(
        {
            "footprint_box_2d": [200, 100, 90, 950],  # ymax < ymin -> invalid
            "dimension_chains": [
                {"side": "top", "values_mm": [2632, "3709", None, "abc", -5]},
                {"side": "diagonal", "values_mm": [1]},  # bad side -> whole call would fail
                "garbage",
            ],
        }
    )
    assert read.footprint_box_2d is None
    # the bad-side chain raises at ChainRead level; ensure top survived
    assert any(c.side == "top" and c.values_mm == [2632.0, 3709.0] for c in read.dimension_chains)
    assert PlanRead.model_validate([{"north_angle_deg": 0}]).north_angle_deg == 0
    assert PlanRead.model_validate([]).dimension_chains == []


def test_chain_read_rejects_unknown_side():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChainRead.model_validate({"side": "diagonal", "values_mm": [1]})


class _FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)  # str reply, or an Exception to raise
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        content = self.replies.pop(0)
        if isinstance(content, Exception):
            raise content
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _client_with(monkeypatch, replies):
    monkeypatch.setenv("ROOMIFY_VLM_API_KEY", "k")
    monkeypatch.setenv("ROOMIFY_VLM_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("ROOMIFY_VLM_MODEL", "m")
    client = VLMClient()
    fake = _FakeCompletions(replies)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return client, fake


def test_call_retries_once_then_succeeds(monkeypatch):
    good = json.dumps({"footprint_box_2d": [10, 10, 900, 900], "dimension_chains": []})
    client, fake = _client_with(monkeypatch, ["{not json", good])
    result = client.call("prompt", [np.zeros((4, 4, 3), np.uint8)], PlanRead)
    assert result is not None and result.footprint_box_2d == [10, 10, 900, 900]
    assert len(fake.requests) == 2
    # the retry must carry the error feedback
    assert "corrected JSON" in fake.requests[1]["messages"][-1]["content"]


def test_call_degrades_to_none(monkeypatch):
    client, _ = _client_with(monkeypatch, ["{not json", "still not json"])
    assert client.call("prompt", [np.zeros((4, 4, 3), np.uint8)], PlanRead) is None


def test_call_accepts_fenced_json(monkeypatch):
    good = '```json\n{"rooms": {}, "extra_rooms": []}\n```'
    client, _ = _client_with(monkeypatch, [good])
    assert client.call("p", [np.zeros((4, 4, 3), np.uint8)], RoomRead) is not None


def test_call_retries_a_schema_echo(monkeypatch):
    from roomify.vlm import OpeningsRead

    echo = json.dumps({"type": "object", "properties": {"candidates": {"type": "array"}}})
    good = json.dumps({"candidates": [], "extra_elements": []})
    client, fake = _client_with(monkeypatch, [echo, good])

    assert client.call("p", [np.zeros((4, 4, 3), np.uint8)], OpeningsRead) is not None
    assert len(fake.requests) == 2


def _bad_request(message):
    import httpx
    from openai import BadRequestError

    response = httpx.Response(
        400, request=httpx.Request("POST", "http://localhost:1/v1/chat/completions")
    )
    return BadRequestError(message, response=response, body={"message": message})


def test_marker_array_contract_normalizes_to_dict():
    read = RoomRead.model_validate(
        {
            "rooms": [
                {"marker": "2", "name": "客厅", "room_type": "living_room",
                 "printed_area_sqm": 37.52, "confidence": 0.9, "not_a_room": False},
                {"marker": "1", "name": "卧室", "room_type": "bedroom",
                 "printed_area_sqm": 9.65, "confidence": 0.9, "not_a_room": False},
                {"name": "no marker — dropped"},
            ],
            "extra_rooms": [],
        }
    )
    assert set(read.rooms) == {"1", "2"}
    assert read.rooms["2"].name == "客厅"

    root = RoomRead.model_validate(
        [{"marker": "1", "name": "厨房", "room_type": "kitchen"}]
    )
    assert root.rooms["1"].name == "厨房"


def test_single_object_array_keeps_all_vlm_tables():
    from roomify.vlm import OpeningsRead, SpatialRoomRead

    room = RoomRead.model_validate(
        [{"rooms": [{"marker": "1", "name": "卧室"}], "extra_rooms": []}]
    )
    spatial = SpatialRoomRead.model_validate(
        [{"rooms": [{"name": "厨房", "label_box_2d": [1, 2, 3, 4],
                      "room_box_2d": [0, 0, 100, 100]}]}]
    )
    openings = OpeningsRead.model_validate(
        [{"candidates": [], "extra_elements": [
            {"element_type": "single_door", "box_2d": [1, 2, 3, 4]}
        ]}]
    )

    assert room.rooms["1"].name == "卧室"
    assert spatial.rooms[0].name == "厨房"
    assert openings.extra_elements[0].element_type == "single_door"

    split = RoomRead.model_validate(
        [
            {"marker": "2", "name": "客厅"},
            {"extra_rooms": [{"name": "阳台", "box_2d": [1, 2, 3, 4]}]},
        ]
    )
    assert split.rooms["2"].name == "客厅"
    assert split.extra_rooms[0].name == "阳台"

    bad_room_box = SpatialRoomRead.model_validate(
        {"rooms": [{"name": "卫生间", "label_box_2d": [10, 20, 30, 40],
                    "room_box_2d": [10, 20, 10, 80]}]}
    )
    assert bad_room_box.rooms[0].room_box_2d == [10, 20, 30, 40]


def test_fast_path_sends_schema_and_disables_thinking(monkeypatch):
    from roomify.vlm import ROOMS_WIRE

    client, fake = _client_with(monkeypatch, ['{"rooms": [], "extra_rooms": []}'])
    assert client.call("p", [np.zeros((4, 4, 3), np.uint8)], RoomRead,
                       wire_schema=ROOMS_WIRE) is not None
    request = fake.requests[0]
    assert request["response_format"]["type"] == "json_schema"
    assert request["extra_body"] == {"enable_thinking": False}


def test_schema_rejection_falls_back_to_json_object(monkeypatch):
    from roomify.vlm import ROOMS_WIRE

    client, fake = _client_with(
        monkeypatch,
        [_bad_request("response_format json_schema is not supported"),
         '{"rooms": {}, "extra_rooms": []}'],
    )
    assert client.call("p", [np.zeros((4, 4, 3), np.uint8)], RoomRead,
                       wire_schema=ROOMS_WIRE) is not None
    assert fake.requests[0]["response_format"]["type"] == "json_schema"
    assert fake.requests[1]["response_format"] == {"type": "json_object"}
    assert client._schema_ok is False  # remembered for the rest of the session


def test_thinking_rejection_drops_extra_body(monkeypatch):
    client, fake = _client_with(
        monkeypatch,
        [_bad_request("unknown parameter enable_thinking"),
         '{"rooms": {}, "extra_rooms": []}'],
    )
    assert client.call("p", [np.zeros((4, 4, 3), np.uint8)], RoomRead) is not None
    assert "extra_body" in fake.requests[0]
    assert "extra_body" not in fake.requests[1]
    assert client._nothink_ok is False


def test_validation_retry_escalates_to_thinking(monkeypatch):
    client, fake = _client_with(
        monkeypatch, ["{not json", '{"rooms": {}, "extra_rooms": []}']
    )
    assert client.call("p", [np.zeros((4, 4, 3), np.uint8)], RoomRead) is not None
    assert "extra_body" in fake.requests[0]  # fast path
    assert "extra_body" not in fake.requests[1]  # careful retry thinks


def test_marker_ids_extend_past_z():
    ids = marker_ids(30)
    assert ids[:3] == ["A", "B", "C"]
    assert ids[26] == "AA" and ids[29] == "AD"
    assert len(set(ids)) == 30


def test_room_overlay_draws_markers():
    bgr = np.full((200, 200, 3), 245, np.uint8)
    room = CVRoom(
        polygon=np.array([[50.0, 50.0], [150.0, 50.0], [150.0, 150.0], [50.0, 150.0]]),
        area_px=10000.0,
        perimeter_px=400.0,
        edge_lengths_px=[100.0] * 4,
        seed=(100.0, 100.0),
    )
    out = render_room_overlay(bgr, [room])
    assert (out != bgr).any()
    ys, xs = np.where((out[:, :, 2] == 255) & (out[:, :, 1] == 255) & (out[:, :, 0] == 0))
    assert ys.size > 0 and ys.mean() < 100  # marker sits above the seed


def test_room_overlay_keeps_marker_inside_narrow_room():
    bgr = np.full((700, 700, 3), 245, np.uint8)
    room = CVRoom(
        polygon=np.array([[100.0, 100.0], [500.0, 100.0], [500.0, 120.0], [100.0, 120.0]]),
        area_px=8000.0,
        perimeter_px=840.0,
        edge_lengths_px=[400.0, 20.0, 400.0, 20.0],
        seed=(300.0, 110.0),
    )
    out = render_room_overlay(bgr, [room])
    yellow = (out[:, :, 2] == 255) & (out[:, :, 1] == 255) & (out[:, :, 0] == 0)
    ys = np.where(yellow)[0]
    assert ys.size > 0
    assert 100 <= ys.mean() <= 120


def test_room_sheet_keeps_an_unmodified_text_panel():
    bgr = np.full((100, 120, 3), 245, np.uint8)
    room = CVRoom(
        polygon=np.array([[10.0, 10.0], [110.0, 10.0], [110.0, 90.0], [10.0, 90.0]]),
        area_px=8000.0,
        perimeter_px=360.0,
        edge_lengths_px=[100.0, 80.0, 100.0, 80.0],
        seed=(60.0, 50.0),
    )

    sheet = render_room_sheet(bgr, [room])

    assert sheet.shape == (100, 248, 3)
    assert np.array_equal(sheet[:, :120], bgr)


def test_openings_prompt_carries_structural_context():
    from roomify.vlm import openings_prompt

    prompt = openings_prompt(
        ["A", "B"],
        include_extras=False,
        context={"A": "connects 客厅 (interior room) ↔ 卧室 (interior room), width ≈ 880mm"},
    )
    assert "connects 客厅" in prompt
    assert "doors or passages, not windows" in prompt
    assert "genuinely unmarked doors/windows" in openings_prompt(["A"])


def test_candidate_sheet_combines_context_and_crops():
    from roomify.vlm import openings_prompt

    bgr = np.full((200, 300, 3), 245, np.uint8)
    candidates = [
        SimpleNamespace(marker="A", bbox=(40, 95, 100, 105)),
        SimpleNamespace(marker="B", bbox=(180, 95, 250, 105)),
    ]

    sheet = render_candidate_sheet(bgr, candidates)

    assert sheet.shape[0] > bgr.shape[0]
    assert sheet.shape[1] == 640
    assert (sheet != 255).any()
    assert "single image is a review sheet" in openings_prompt(
        ["A", "B"], contact_sheet=True
    )


def test_extra_elements_prompt_excludes_already_boxed_symbols():
    prompt = extra_elements_prompt()

    assert "NO magenta box covers" in prompt
    assert "Never repeat a boxed object" in prompt
    assert "empty candidates list" in prompt
