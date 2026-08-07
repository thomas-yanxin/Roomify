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
    marker_ids,
    render_room_overlay,
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
