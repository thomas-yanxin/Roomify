# Roomify

Robust 2D floor plan → structured JSON.

Roomify converts raw residential floor-plan images and PDFs — the kind found
on Chinese property listings (贝壳/链家-style 户型图) and in decoration
drawings — into a fully structured, validated JSON document: rooms with
polygons and areas, walls, doors, windows and other legend elements,
millimetre calibration, and honest uncertainty reporting.

```bash
roomify floorplan.png -o floorplan.json
```

```python
from roomify import parse

plan = parse("floorplan.png")
plan.rooms[0].name          # "客厅"
plan.rooms[0].area_sqm      # 37.34  (printed on the plan: 37.52 → deviation −0.5%)
plan.openings[3].element_type  # "sliding_door"
plan.model_dump_json()      # the full document
```

## Why hybrid

Pure-VLM extraction of floor-plan *geometry* is unreliable: models read
labels perfectly but return polygons that are 20–50 px off and miss small
rooms. Classical CV is the opposite: pixel-exact boundaries, but no idea
what "卧室 9.65㎡" means. Roomify hard-splits the work:

- **CV owns all geometry.** Wall masks, room polygons, opening positions,
  areas, perimeters, edge lengths. Every coordinate in the output is
  measured, not imagined.
- **The VLM owns all semantics.** Room names and types, printed areas,
  dimension-chain transcription, legend classification (door vs window vs
  passage), and vetoes of false CV detections. It communicates through
  marker-keyed contracts (numbered/lettered overlays burned into the
  image), never through coordinates.
- **Geometry never blocks on the VLM.** Every VLM call has a defined
  degradation path; `parse(..., use_vlm=False)` still emits a valid
  pixel-only document.

Two design details matter more than they look:

- **Anisotropic calibration.** Exported listing images are frequently
  resized non-uniformly (the reference corpus measures ~10% difference
  between axes), so the scale is solved per axis (`px_per_mm_x` /
  `px_per_mm_y`): dimension chains propose per-axis candidates, and the
  median of per-room `area_px / printed_area` ratios elects the pair —
  which makes the calibration robust to individual OCR misreads on either
  source.
- **No invented facts.** Door swing and hinge come exclusively from pixel
  evidence (drawn swing arcs — stroked or tinted). Plans that don't draw
  arcs get `swing: null` plus an entry in `unresolved`, never a guess.
  Everything the parser could not determine is enumerated in `warnings`
  and `unresolved`.

## Install

```bash
pip install -e .          # from a checkout; PyPI release pending
```

Python ≥ 3.11. Core dependencies: OpenCV (headless), NumPy, Shapely,
Pydantic v2, PyMuPDF, and the OpenAI SDK (any OpenAI-compatible
vision-capable endpoint works).

### Configuration

Roomify reads VLM credentials from the environment only — there are no
defaults in code:

```bash
ROOMIFY_VLM_API_KEY=...                    # required for semantic parsing
ROOMIFY_VLM_BASE_URL=https://.../v1        # OpenAI-compatible endpoint
ROOMIFY_VLM_MODEL=qwen3.8-max              # must support image input
```

Copy `.env.example` to `.env` for local work (git-ignored). Without
credentials, `roomify --no-vlm` / `parse(use_vlm=False)` still produce
pixel-only geometry.

## CLI

```
roomify INPUT [-o out.json] [--page N] [--no-vlm] [--debug DIR]
```

- `INPUT`: `.png/.jpg/.tif/.bmp/.webp` or `.pdf` (largest embedded raster
  is used when it is ≥1000 px on both sides, otherwise the page is rendered
  at 300 DPI).
- `--debug DIR` writes per-stage overlays (wall mask, room polygons,
  opening candidates, and the exact images sent to the VLM). With a
  pipeline this threshold-driven, the debug images are part of the
  algorithm — look at them before doubting a result.

## Output document

Top level (`FloorPlan`):

| field | meaning |
|---|---|
| `image_width_px` / `image_height_px` | original input size; **all `*_px` coordinates live in this frame** (x right, y down, origin top-left) |
| `page` | PDF page index, `null` for images |
| `north_angle_deg` | north-arrow direction (0 = up, clockwise), `null` if none |
| `scale` | px↔mm calibration or `null`; `scale: null` ⇔ every `*_mm`/`*_sqm` field is `null` |
| `rooms` / `walls` / `openings` / `elements` | see below |
| `warnings` | machine-readable notes (`code`, `message`, `ref`) about degradations and conflicts |
| `unresolved` | facts the drawing does not contain (`path`, `reason`) — absent, not guessed |

`Scale`: `px_per_mm_x`, `px_per_mm_y`, `anisotropy`, `method`
(`dimension_chains` / `printed_areas` / both), `confidence`
(`high`/`medium`/`low`), and the evidence counts used.

`Room`: `name` (label text exactly as printed, e.g. `"卧室"` — never
translated), `room_type` (normalized enum: `living_room`, `bedroom`,
`kitchen`, `bathroom`, `balcony`, `closet`, `storage`, …), `polygon_px`
(**open ring** — first vertex not repeated; edge *i* runs vertex *i* →
*i+1*, wrapping), `polygon_mm`, `area_px`, `area_sqm`, `perimeter_*`,
`edge_lengths_*`, `printed_area_sqm` (read off the plan),
`area_deviation` and `area_deviation_flag` (|computed − printed|/printed >
10%), `source` (`cv` / `vlm` / `cv+vlm`), `confidence`.

`Wall`: centerline `start_px`/`end_px` (+`_mm`), `thickness_px`/`_mm`, and
`rooms` — the ids on each side, `"exterior"` for outside the building,
`"unknown"` for circulation space no room polygon covered.

`Opening` (wall-hosted elements): `element_type` from the legend
vocabulary — `passage`, `single_door`, `double_door`, `sliding_door`,
`folding_door`, `window`, `casement_window`, `sliding_window`,
`fixed_window`, `bay_window`, `floor_to_ceiling_window`, `blind_window`,
plus `unknown_symbol` as the honest fallback — with `bbox_px`,
`center_px`, `width_px`/`width_mm` (along the wall), `wall_id`,
`connects`, `raw_text` (nearby code like `"C1"`), and `swing`/`hinge_px`
(only when arc pixel-evidence exists).

`Element` (free-standing): `stair`, `railing`, `elevator`, `escalator`,
`equipment_platform`, `column`, `chimney`, `unknown_symbol` — with
`bbox_px` and the containing `room_id`. Elements found only by the VLM
carry approximate bounding-box geometry and `source: "vlm"`.

The element vocabulary follows the `tecton_v1_mapping` naming for common
Chinese residential legend symbols (doors B20–B24, windows B28–B35,
circulation B36–B42).

## How it works

1. **Load** (`io.py`) — image or PDF → working BGR image (≤2000 px);
   original-pixel coordinates are restored at assembly.
2. **Walls** (`walls.py`) — the wall-grey band is auto-estimated from the
   histogram of low-chroma pixels (never hard-coded: real-world corpora
   put walls anywhere from near-black to `#999`), then intersected with a
   stroke-thickness test (`distanceTransform`) so 1 px dimension lines and
   text vanish; a parallel thin-line mask (adaptive threshold ∧ long H/V
   runs, gated to the footprint) keeps door sills and window strokes.
3. **Rooms** (`rooms.py`) — enclosed voids of the combined mask via
   contour-hierarchy holes; a fallback ladder of directional closings
   handles plans without sill strokes; polygons are cleaned by a
   conjunctive collinear-merge (near-straight **and** short-edge) and
   follow the **inner-face (net floor area)** convention that printed ㎡
   labels use.
4. **VLM calls** (`vlm.py`) — three calls, marker-keyed, JSON-mode,
   Pydantic-validated with one error-echo retry each: ① footprint +
   dimension chains + north; ② room semantics over a numbered overlay
   (plus `extra_rooms` recall for anything CV missed); ③ opening
   classification over a lettered overlay + magnified crops (chunked and
   parallelized).
5. **Merge & calibrate** (`merge.py`) — id-keyed fusion, per-axis scale,
   deviation flags, junk filters.
6. **Openings & walls** (`openings.py`) — wall segments are derived from
   room-polygon adjacency; openings are wall-absent intervals scanned
   along each segment (plus a window-stroke signal for windows drawn on
   unbroken walls); swing arcs are detected as tinted quarter-disc sectors
   or stroked arcs, compared strictly within the room they open into.

## Accuracy (reference corpus)

On the two reference listing plans (686×706 / 708×708 px, JPEG-derived):

- 9/9 and 10/10 rooms detected; all room names correct; computed areas
  within **±1.4%** and **±4.4%** of the printed values (no false
  deviation flags).
- Scale confidence `high`, anisotropy (+11%) fully resolved.
- All doors/windows/passages surfaced as openings; the plan that draws no
  swing arcs yields **zero** swing claims; the plan with tinted arcs
  yields swings on its doors.

## Limitations (v1)

- Tuned for residential plans with solid-fill (poché) walls; pure
  line-drawing CAD walls survive via the thin-line path but with less
  redundancy. Colored walls fall back to a wide band (`warnings:
  wall_band_fallback`).
- Wide-open passages between spaces that are drawn with *no* separating
  strokes at all are treated as one room (matching the drawing).
- `bay_window` protrusion polygons are not measured yet
  (`protrusion_polygon_px` stays `null`; classification still works).
- Exterior wall centerlines are approximated at the room's inner face
  (sub-thickness bias).
- One VLM misread of a printed area is tolerated by the median-based
  calibration but will surface as that room's `area_deviation`.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest                      # unit tests: no VLM, no fixtures needed
uv run pytest tests/integration    # live acceptance (needs credentials + example images)
uv run ruff check src tests
```

Integration tests are double-gated on `ROOMIFY_EXAMPLES_DIR` (real plans
are not committed to the repo) and `ROOMIFY_VLM_API_KEY`.

## License

Apache-2.0.
