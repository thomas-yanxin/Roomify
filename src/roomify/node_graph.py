"""Embed a metric node graph in Roomify JSON."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from roomify.schema import FloorPlan, Opening, Point, Wall

_NodeGraph = dict[str, Any]

_DOOR_TYPES = {
    "passage": ("hinged", 1, "opening"),
    "single_door": ("hinged", 1, "door"),
    "double_door": ("double", 2, "door"),
    "sliding_door": ("sliding", 1, "door"),
    "folding_door": ("folding", 2, "door"),
}
_WINDOW_TYPES = {
    "window": "fixed",
    "casement_window": "casement",
    "sliding_window": "sliding",
    "fixed_window": "fixed",
    "bay_window": "bay",
    "floor_to_ceiling_window": "fixed",
    "blind_window": "louvered",
}


def _build_graph(plan: FloorPlan) -> _NodeGraph:
    """Build the metric node graph embedded in a Roomify document.

    Node dimensions use metres. The complete evidence document is retained
    under the root node's ``metadata.roomify`` field.
    """
    if plan.scale is None:
        raise ValueError("metric node fields require calibration (scale is null)")

    site_id, building_id, level_id = "site_roomify", "building_roomify", "level_roomify"
    wall_by_id = {wall.id: wall for wall in plan.walls}
    opening_ids: dict[str, str] = {}
    wall_children: dict[str, list[str]] = {wall.id: [] for wall in plan.walls}
    for opening in plan.openings:
        prefix = "door" if opening.element_type in _DOOR_TYPES else "window"
        if opening.element_type not in _DOOR_TYPES and opening.element_type not in _WINDOW_TYPES:
            continue
        if opening.wall_id not in wall_by_id:
            continue
        opening_node_id = f"{prefix}_{opening.id}"
        opening_ids[opening.id] = opening_node_id
        wall_children[opening.wall_id].append(opening_node_id)

    nodes: dict[str, dict[str, Any]] = {}
    wall_ids = [wall.id for wall in plan.walls]
    zone_ids = [f"zone_{room.id}" for room in plan.rooms]
    slab_ids = [f"slab_{room.id}" for room in plan.rooms]
    level_children = [*wall_ids, *zone_ids, *slab_ids]

    site_polygon = _site_polygon(plan.walls)
    nodes[site_id] = _node(
        site_id,
        "site",
        None,
        name=Path(plan.source_file).stem or "Roomify import",
        children=[building_id],
        polygon={"type": "polygon", "points": site_polygon},
        metadata={
            "roomify": plan.model_dump(mode="json", exclude={"nodes", "rootNodeIds"})
        },
    )
    nodes[building_id] = _node(
        building_id,
        "building",
        site_id,
        name="Roomify Building",
        children=[level_id],
        position=[0, 0, 0],
        rotation=[0, 0, 0],
    )
    nodes[level_id] = _node(
        level_id,
        "level",
        building_id,
        name="Level 0",
        children=level_children,
        level=0,
        baseElevation=0,
        height=plan.level_height_mm / 1000,
    )

    for wall in plan.walls:
        assert wall.start_mm is not None and wall.end_mm is not None
        assert wall.thickness_mm is not None
        nodes[wall.id] = _node(
            wall.id,
            "wall",
            level_id,
            children=wall_children[wall.id],
            start=_point_m(wall.start_mm),
            end=_point_m(wall.end_mm),
            thickness=wall.thickness_mm / 1000,
            height=plan.level_height_mm / 1000,
            frontSide="unknown",
            backSide="unknown",
            metadata={"roomifyId": wall.id, "rooms": list(wall.rooms)},
        )

    for opening in plan.openings:
        mapped_id = opening_ids.get(opening.id)
        host_wall = wall_by_id.get(opening.wall_id or "")
        if mapped_id is None or host_wall is None:
            continue
        node = _opening_node(opening, host_wall, mapped_id)
        nodes[mapped_id] = node

    for room, zone_id, slab_id in zip(plan.rooms, zone_ids, slab_ids, strict=True):
        assert room.polygon_mm is not None
        polygon = [_point_m(point) for point in room.polygon_mm]
        boundary_walls = [wall.id for wall in plan.walls if room.id in wall.rooms]
        nodes[zone_id] = _node(
            zone_id,
            "zone",
            level_id,
            name=room.name or room.id,
            polygon=polygon,
            autoFromWalls=False,
            boundaryWallIds=boundary_walls,
            spaceRole="room",
            roomNumber=room.id.removeprefix("room_"),
            enclosureStatus="enclosed",
            ceilingHeight=plan.level_height_mm / 1000,
            metadata={"roomifyId": room.id, "roomType": room.room_type},
        )
        nodes[slab_id] = _node(
            slab_id,
            "slab",
            level_id,
            name=f"Floor - {room.name or room.id}",
            polygon=polygon,
            holes=[],
            holeMetadata=[],
            elevation=0,
            thickness=0.1,
            autoFromWalls=False,
            metadata={"roomifyId": room.id},
        )

    return {"nodes": nodes, "rootNodeIds": [site_id]}


def embed_node_graph(plan: FloorPlan) -> FloorPlan:
    """Return a Roomify document with its metric node graph populated."""
    graph = _build_graph(plan)
    return plan.model_copy(
        update={"nodes": graph["nodes"], "rootNodeIds": graph["rootNodeIds"]}
    )


def _node(node_id: str, node_type: str, parent_id: str | None, **fields: Any) -> dict[str, Any]:
    return {
        "object": "node",
        "id": node_id,
        "type": node_type,
        "parentId": parent_id,
        "visible": True,
        "metadata": {},
        **fields,
    }


def _opening_node(opening: Opening, wall: Wall, node_id: str) -> dict[str, Any]:
    if opening.width_mm is None or opening.height_mm is None or opening.sill_height_mm is None:
        raise ValueError(f"opening {opening.id} has no metric 3D dimensions")
    position = [
        _distance_along_wall_m(opening.center_px, wall),
        (opening.sill_height_mm + opening.height_mm / 2) / 1000,
        0,
    ]
    common = dict(
        position=position,
        rotation=[0, 0, 0],
        wallId=wall.id,
        width=opening.width_mm / 1000,
        height=opening.height_mm / 1000,
        metadata={"roomifyId": opening.id},
    )
    if opening.element_type in _DOOR_TYPES:
        door_type, leaves, opening_kind = _DOOR_TYPES[opening.element_type]
        hinges_side = "left"
        if opening.hinge_px is not None:
            hinges_side = (
                "left" if _distance_along_wall_m(opening.hinge_px, wall) < position[0] else "right"
            )
        return _node(
            node_id,
            "door",
            wall.id,
            doorType=door_type,
            leafCount=leaves,
            openingKind=opening_kind,
            hingesSide=hinges_side,
            threshold=opening_kind == "door",
            handle=opening_kind == "door",
            **common,
        )
    return _node(
        node_id,
        "window",
        wall.id,
        windowType=_WINDOW_TYPES[opening.element_type],
        sill=opening.sill_height_mm > 0,
        **common,
    )


def _distance_along_wall_m(point: Point, wall: Wall) -> float:
    px = point.x - wall.start_px.x
    py = point.y - wall.start_px.y
    dx = wall.end_px.x - wall.start_px.x
    dy = wall.end_px.y - wall.start_px.y
    denominator = dx * dx + dy * dy
    fraction = 0.0 if denominator == 0 else (px * dx + py * dy) / denominator
    assert wall.start_mm is not None and wall.end_mm is not None
    length_mm = math.hypot(wall.end_mm.x - wall.start_mm.x, wall.end_mm.y - wall.start_mm.y)
    return min(max(fraction, 0.0), 1.0) * length_mm / 1000


def _point_m(point: Point) -> list[float]:
    return [point.x / 1000, point.y / 1000]


def _site_polygon(walls: list[Wall]) -> list[list[float]]:
    points = [
        point for wall in walls for point in (wall.start_mm, wall.end_mm) if point is not None
    ]
    if not points:
        return [[-15, -15], [15, -15], [15, 15], [-15, 15]]
    xs = [point.x / 1000 for point in points]
    ys = [point.y / 1000 for point in points]
    margin = 2.0
    return [
        [min(xs) - margin, min(ys) - margin],
        [max(xs) + margin, min(ys) - margin],
        [max(xs) + margin, max(ys) + margin],
        [min(xs) - margin, max(ys) + margin],
    ]
