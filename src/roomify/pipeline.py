"""The parse() orchestrator: load → CV geometry → VLM semantics → FloorPlan.

Geometry never blocks on the VLM: every VLM call that fails (or is disabled
via use_vlm=False) routes to a degradation path that still yields a valid —
if less semantic — document, with the degradation recorded in ``warnings``
and ``unresolved``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon

from roomify import debug as debug_mod
from roomify.io import SourceImage, load
from roomify.merge import (
    DEVIATION_FLAG_THRESHOLD,
    ElementDraft,
    OpeningDraft,
    RoomDraft,
    ScaleDraft,
    apply_area_checks,
    estimate_scale,
    merge_openings,
    merge_rooms,
)
from roomify.openings import WallSegment, find_openings
from roomify.rooms import detect_rooms
from roomify.schema import (
    BBox,
    Element,
    FloorPlan,
    Opening,
    ParseWarning,
    Room,
    Unresolved,
    Wall,
)
from roomify.schema import (
    Point as SchemaPoint,
)
from roomify.walls import WallExtraction, extract_walls

logger = logging.getLogger("roomify.pipeline")


def parse(
    path: str | Path,
    *,
    page: int | None = None,
    use_vlm: bool = True,
    debug_dir: str | Path | None = None,
    max_dim: int = 2000,
) -> FloorPlan:
    """Parse a floor-plan image or PDF into a structured FloorPlan.

    use_vlm=False produces a pixel-only document (no names, no mm) from CV
    alone. debug_dir writes per-stage overlay PNGs — with this many tuned
    thresholds they are part of the algorithm, not a nicety.
    """
    src = load(path, page=page, max_dim=max_dim)
    warnings: list[ParseWarning] = []
    unresolved: list[Unresolved] = []

    client = None
    if use_vlm:
        from roomify.vlm import VLMClient, VLMUnavailable

        try:
            client = VLMClient()
        except VLMUnavailable as exc:
            warnings.append(ParseWarning(code="vlm_unavailable", message=str(exc)))

    # The plan-read call runs concurrently with the CV stages and the
    # room-semantics call: none of them depend on it, and this endpoint's
    # reasoning latency (~45s/call) would otherwise serialize.
    plan_future = None
    executor = None
    if client is not None:
        from concurrent.futures import ThreadPoolExecutor

        from roomify.vlm import PlanRead, plan_read_prompt

        # 2 workers, measured: this endpoint's multi-image requests start
        # timing out under 3-way concurrency.
        executor = ThreadPoolExecutor(max_workers=2)
        plan_future = executor.submit(client.call, plan_read_prompt(), [src.bgr], PlanRead)

    walls = extract_walls(src.bgr)
    if walls.band_fallback:
        warnings.append(
            ParseWarning(
                code="wall_band_fallback",
                message="no wall-grey histogram peak found; wide fallback band used",
            )
        )

    detection = detect_rooms(walls)
    if detection.strategy != "close5":
        warnings.append(
            ParseWarning(
                code="room_gap_fallback",
                message=f"rooms sealed with {detection.strategy} (thin seal strokes absent)",
            )
        )

    room_read = None
    if client is not None:
        from roomify.vlm import RoomRead, render_room_overlay, room_semantics_prompt

        overlay = render_room_overlay(src.bgr, detection.rooms)
        if debug_dir:
            debug_mod.save(debug_dir, "vlm_room_overlay.png", overlay)
        room_read = client.call(
            room_semantics_prompt(len(detection.rooms)), [overlay], RoomRead
        )
        if room_read is None:
            warnings.append(
                ParseWarning(code="vlm_call_failed", message="room-semantics call failed; "
                             "rooms keep CV geometry with unknown names")
            )

    outcome = merge_rooms(detection.rooms, room_read, src.bgr.shape[:2])
    warnings += outcome.warnings
    unresolved += outcome.unresolved

    plan_read = plan_future.result() if plan_future is not None else None
    if plan_future is not None and plan_read is None:
        warnings.append(
            ParseWarning(code="vlm_call_failed", message="plan-read call failed; "
                         "scale falls back to printed areas, footprint to CV")
        )

    chains = plan_read.dimension_chains if plan_read is not None else []
    scale_draft, scale_warnings = estimate_scale(chains, walls, outcome.rooms)
    warnings += scale_warnings

    checked = apply_area_checks(outcome.rooms, scale_draft)
    warnings += checked.warnings
    rooms = checked.rooms

    door_px = 1000.0 * scale_draft.px_per_mm_x if scale_draft else None
    candidates, segments = find_openings(walls, rooms, src.bgr, door_px)

    openings_read = None
    if client is not None and candidates:
        from roomify.vlm import (
            OpeningsRead,
            openings_prompt,
            render_candidate_crop,
            render_openings_overlay,
        )

        overlay = render_openings_overlay(src.bgr, candidates)
        if debug_dir:
            debug_mod.save(debug_dir, "vlm_openings_overlay.png", overlay)
        # Small chunks, in parallel: this endpoint 502s beyond ~6 images per
        # request, and at ~45s per call serial chunks would dominate runtime.
        chunk_size = 5
        chunks = [candidates[s : s + chunk_size] for s in range(0, len(candidates), chunk_size)]
        futures = []
        assert executor is not None
        for idx, chunk in enumerate(chunks):
            crops = [render_candidate_crop(src.bgr, c) for c in chunk]
            futures.append(
                executor.submit(
                    client.call,
                    openings_prompt([c.marker for c in chunk], include_extras=idx == 0),
                    [overlay, *crops],
                    OpeningsRead,
                )
            )
        merged = OpeningsRead()
        got_any = False
        failed: list[list] = []
        for chunk, future in zip(chunks, futures, strict=True):
            part = future.result()
            if part is None:
                failed.append(chunk)
                continue
            got_any = True
            merged.candidates.update(part.candidates)
            merged.extra_elements.extend(part.extra_elements)
        # Reasoning latency is content-driven: ambiguous crops can stall the
        # model past the upstream's own ~240s kill switch, so retrying a
        # failed chunk at the same size can never succeed. Retry as
        # single-crop requests instead — the smallest possible reasoning
        # load — and accept the degradation only per candidate.
        retry = [c for chunk in failed for c in chunk][:12]  # bound worst-case time
        skipped = [c for chunk in failed for c in chunk][12:]
        for cand in retry:
            part = client.call(
                openings_prompt([cand.marker], include_extras=False),
                [overlay, render_candidate_crop(src.bgr, cand)],
                OpeningsRead,
            )
            if part is None:
                warnings.append(
                    ParseWarning(
                        code="vlm_call_failed",
                        message=f"openings call failed for marker {cand.marker}; "
                        "CV kind hint kept",
                        ref=cand.marker,
                    )
                )
                continue
            got_any = True
            merged.candidates.update(part.candidates)
        for cand in skipped:
            warnings.append(
                ParseWarning(
                    code="vlm_call_failed",
                    message=f"openings retry budget exhausted for marker {cand.marker}; "
                    "CV kind hint kept",
                    ref=cand.marker,
                )
            )
        openings_read = merged if got_any else None

    if executor is not None:
        executor.shutdown(wait=False)

    opening_drafts, element_drafts, op_warnings, op_unresolved = merge_openings(
        candidates, openings_read, src.bgr.shape[:2]
    )
    warnings += op_warnings
    unresolved += op_unresolved

    if debug_dir:
        debug_mod.save(debug_dir, "01_walls.png", debug_mod.walls_overlay(src.bgr, walls))
        debug_mod.save(
            debug_dir,
            "02_rooms.png",
            debug_mod.polygons_overlay(
                src.bgr,
                [r.polygon for r in rooms],
                [f"{i + 1}:{r.name or '?'}" for i, r in enumerate(rooms)],
            ),
        )
        from roomify.vlm import render_openings_overlay as _roo

        debug_mod.save(debug_dir, "03_openings.png", _roo(src.bgr, candidates))

    plan = _assemble(
        src,
        walls,
        rooms,
        segments,
        opening_drafts,
        element_drafts,
        scale_draft,
        plan_read,
        warnings,
        unresolved,
    )
    return plan


# ------------------------------------------------------------------ assembly


def _assemble(
    src: SourceImage,
    walls: WallExtraction,
    rooms: list[RoomDraft],
    segments: list[WallSegment],
    opening_drafts: list[OpeningDraft],
    element_drafts: list[ElementDraft],
    scale_draft: ScaleDraft | None,
    plan_read: object | None,
    warnings: list[ParseWarning],
    unresolved: list[Unresolved],
) -> FloorPlan:
    f = src.scale_to_original  # working px -> original px
    scale = scale_draft.to_schema(f) if scale_draft else None
    origin = (walls.footprint[0] * f, walls.footprint[1] * f)  # mm origin

    def pt(x: float, y: float) -> SchemaPoint:
        return SchemaPoint(x=x * f, y=y * f)

    def pt_mm(x: float, y: float) -> SchemaPoint:
        assert scale is not None
        return SchemaPoint(
            x=(x * f - origin[0]) / scale.px_per_mm_x,
            y=(y * f - origin[1]) / scale.px_per_mm_y,
        )

    room_ids: list[str] = [f"room_{i + 1}" for i in range(len(rooms))]
    opening_ids: list[str] = [f"op_{i + 1}" for i in range(len(opening_drafts))]
    ids_by_collection: dict[str, dict[str | None, str]] = {
        "rooms": {draft.marker: room_ids[i] for i, draft in enumerate(rooms)},
        "openings": {draft.marker: opening_ids[i] for i, draft in enumerate(opening_drafts)},
    }
    marker_to_id = ids_by_collection["rooms"] | ids_by_collection["openings"]
    warnings = [
        warning.model_copy(update={"ref": marker_to_id.get(warning.ref, warning.ref)})
        for warning in warnings
    ]
    remapped_unresolved: list[Unresolved] = []
    for item in unresolved:
        parts = item.path.split("/", 2)
        if len(parts) == 3 and parts[0] in ids_by_collection:
            parts[1] = ids_by_collection[parts[0]].get(parts[1], parts[1])
        remapped_unresolved.append(item.model_copy(update={"path": "/".join(parts)}))
    unresolved = remapped_unresolved

    schema_rooms: list[Room] = []
    for i, draft in enumerate(rooms):
        ring = draft.polygon
        polygon_px = [pt(x, y) for x, y in ring]
        closed = np.vstack([ring, ring[:1]]) * f
        edge_lengths_px = [float(np.linalg.norm(d)) for d in np.diff(closed, axis=0)]

        polygon_mm = area_sqm = perimeter_mm = edge_lengths_mm = None
        deviation = None
        flag = False
        if scale is not None:
            polygon_mm = [pt_mm(x, y) for x, y in ring]
            mm = np.array([[p.x, p.y] for p in polygon_mm])
            closed_mm = np.vstack([mm, mm[:1]])
            edge_lengths_mm = [float(np.linalg.norm(d)) for d in np.diff(closed_mm, axis=0)]
            perimeter_mm = float(sum(edge_lengths_mm))
            area_sqm = float(
                abs(
                    np.sum(
                        mm[:, 0] * np.roll(mm[:, 1], -1) - np.roll(mm[:, 0], -1) * mm[:, 1]
                    )
                )
                / 2
                / 1e6
            )
            if draft.printed_area_sqm:
                deviation = (area_sqm - draft.printed_area_sqm) / draft.printed_area_sqm
                flag = abs(deviation) > DEVIATION_FLAG_THRESHOLD

        schema_rooms.append(
            Room(
                id=room_ids[i],
                name=draft.name,
                room_type=draft.room_type,  # type: ignore[arg-type]
                polygon_px=polygon_px,
                area_px=draft.area_px * f * f,
                perimeter_px=draft.perimeter_px * f,
                edge_lengths_px=edge_lengths_px,
                polygon_mm=polygon_mm,
                area_sqm=area_sqm,
                perimeter_mm=perimeter_mm,
                edge_lengths_mm=edge_lengths_mm,
                printed_area_sqm=draft.printed_area_sqm,
                area_deviation=deviation,
                area_deviation_flag=flag,
                source=draft.source,  # type: ignore[arg-type]
                confidence=draft.confidence,
            )
        )

    def side_name(side: int | str) -> str:
        if isinstance(side, int):
            return room_ids[side]
        return side  # "exterior" | "unknown"

    schema_walls: list[Wall] = []
    wall_id_of_segment: dict[int, str] = {}
    for si, seg in enumerate(segments):
        start, end = pt(*seg.start), pt(*seg.end)
        if math.isclose(start.x, end.x) and math.isclose(start.y, end.y):
            continue
        wall_id = f"wall_{len(schema_walls) + 1}"
        wall_id_of_segment[si] = wall_id
        horizontal = abs(end.x - start.x) >= abs(end.y - start.y)
        thickness_mm = None
        start_mm = end_mm = None
        if scale is not None:
            start_mm, end_mm = pt_mm(*seg.start), pt_mm(*seg.end)
            thickness_mm = seg.thickness_px * f / (
                scale.px_per_mm_y if horizontal else scale.px_per_mm_x
            )
        schema_walls.append(
            Wall(
                id=wall_id,
                start_px=start,
                end_px=end,
                thickness_px=seg.thickness_px * f,
                rooms=(side_name(seg.rooms[0]), side_name(seg.rooms[1])),
                start_mm=start_mm,
                end_mm=end_mm,
                thickness_mm=thickness_mm,
                source="cv",
                confidence=0.6,
            )
        )

    schema_openings: list[Opening] = []
    for i, op in enumerate(opening_drafts):
        x0, y0, x1, y1 = op.bbox
        width_mm = None
        if scale is not None:
            width_mm = op.width_px * f / (
                scale.px_per_mm_x if op.axis == "h" else scale.px_per_mm_y
            )
        schema_openings.append(
            Opening(
                id=opening_ids[i],
                element_type=op.element_type,  # type: ignore[arg-type]
                raw_text=op.raw_text,
                bbox_px=BBox(x0=x0 * f, y0=y0 * f, x1=x1 * f, y1=y1 * f),
                center_px=pt(*op.center),
                width_px=op.width_px * f,
                width_mm=width_mm,
                wall_id=wall_id_of_segment.get(op.wall_index),
                connects=(side_name(op.connects[0]), side_name(op.connects[1])),
                swing=op.swing,  # type: ignore[arg-type]
                hinge_px=pt(*op.hinge) if op.hinge else None,
                protrusion_polygon_px=None,
                source=op.source,  # type: ignore[arg-type]
                confidence=op.confidence,
            )
        )

    room_polys = [
        (room_ids[i], Polygon(r.polygon)) for i, r in enumerate(rooms) if len(r.polygon) >= 3
    ]
    schema_elements: list[Element] = []
    for i, el in enumerate(element_drafts):
        x0, y0, x1, y1 = el.bbox
        center = Point((x0 + x1) / 2, (y0 + y1) / 2)
        room_id = next((rid for rid, poly in room_polys if poly.contains(center)), None)
        schema_elements.append(
            Element(
                id=f"el_{i + 1}",
                element_type=el.element_type,  # type: ignore[arg-type]
                raw_text=el.raw_text,
                bbox_px=BBox(x0=x0 * f, y0=y0 * f, x1=x1 * f, y1=y1 * f),
                room_id=room_id,
                source=el.source,  # type: ignore[arg-type]
                confidence=el.confidence,
            )
        )

    north = getattr(plan_read, "north_angle_deg", None) if plan_read else None
    if north is not None:
        north = float(north) % 360.0

    return FloorPlan(
        source_file=src.source_file,
        source_sha256=src.sha256,
        page=src.page,
        image_width_px=src.original_width,
        image_height_px=src.original_height,
        north_angle_deg=north,
        scale=scale,
        rooms=schema_rooms,
        walls=schema_walls,
        openings=schema_openings,
        elements=schema_elements,
        warnings=warnings,
        unresolved=unresolved,
    )
