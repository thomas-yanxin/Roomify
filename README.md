<div align="center">

# Roomify

**Floor plans in. Measured JSON out.**

Roomify parses residential floor-plan images and PDFs into rooms, walls,
openings, and millimetre-scale geometry.

[English](README.md) · [简体中文](README_ZH.md)

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
[![CI](https://github.com/thomas-yanxin/Roomify/actions/workflows/python-package.yml/badge.svg)](https://github.com/thomas-yanxin/Roomify/actions/workflows/python-package.yml)
![Status: Beta](https://img.shields.io/badge/status-beta-6f42c1)
![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-2ea44f)

[Example](#example) · [Install](#install) · [JSON structure](#json-structure) · [Development](#development)

</div>

OpenCV measures the geometry. A vision-language model reads labels and
symbols. The model never supplies coordinates. Without a VLM, Roomify still
returns pixel geometry and records what it could not resolve.

```bash
roomify floorplan.png -o floorplan.json
```

## Example

This plan is included in the repository:

<p align="center">
  <img src="examples/floorplan-1.png" alt="Residential floor plan used by the Roomify example" width="620">
</p>

[`floorplan-1.png`](examples/floorplan-1.png) → run
`roomify examples/floorplan-1.png -o plan.json`

| Rooms | Wall segments | Openings | Scale confidence | Living-room area |
|---:|---:|---:|:---:|:---|
| 9 | 41 | 16 | `high` | 37.88 m² measured / 37.52 m² printed |

Selected fields from the full output:

```json
{
  "schema_version": "1.2",
  "scale": {
    "px_per_mm_x": 0.0431736218444101,
    "px_per_mm_y": 0.048002053563788824,
    "method": "dimension_chains+printed_areas",
    "confidence": "high"
  },
  "rooms": [
    {
      "id": "room_1",
      "name": "客厅",
      "room_type": "living_room",
      "area_sqm": 37.33987625081361,
      "printed_area_sqm": 37.52,
      "area_deviation_flag": false,
      "source": "cv+vlm",
      "confidence": 0.95
    }
  ],
  "openings": [
    {
      "id": "op_1",
      "element_type": "window",
      "width_mm": 1250.763723150358,
      "wall_id": "wall_43",
      "connects": ["room_4", "exterior"],
      "source": "cv+vlm",
      "confidence": 0.9
    }
  ]
}
```

## Install

Roomify requires Python 3.11 or newer. The recommended installer is
[uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/thomas-yanxin/Roomify.git
cd Roomify
uv sync --locked
uv run roomify --help
```

`uv sync` creates `.venv`, installs Roomify in editable mode, and reproduces
the dependency versions in `uv.lock`.

Without uv, use pip in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip check
```

A PyPI package has not been published yet. The different OpenCV wheel variants
all provide `cv2` and must not be mixed in the same environment.

## Quick start

The values below assume the VLM environment variables in the next section
are set.

### Command line

```bash
uv run roomify examples/floorplan-1.png -o plan.json
```

Images and PDFs are supported. Use `--page N` for a PDF page, `--no-vlm` for
CV-only parsing, and `--debug DIR` to save the masks and overlays used by the
pipeline. Metric `nodes` and `rootNodeIds` are populated when calibration
succeeds. If you used the pip fallback and activated `.venv`, omit `uv run`.

### Reviewing results

`examples/gallery.html` is a static review console: it loads whatever
`floorplan-N.json` and its source image it finds beside itself, so a fresh
run shows up on the next refresh and nothing has to be regenerated.

```bash
cd examples && python3 -m http.server 8000
```

Then open `http://localhost:8000/gallery.html`. It overlays room polygons,
wall centrelines, openings and measured swing radii on the drawing, with a
per-room deviation rail and the run's warnings and unresolved fields. (A
`file://` page cannot read its own directory; the console also accepts the
folder through a picker if you would rather not start a server.)

### Python

```python
from roomify import parse

plan = parse("examples/floorplan-1.png")

print(plan.rooms[0].name)       # 客厅
print(plan.rooms[0].area_sqm)   # 37.33987625081361
print(plan.scale.confidence)    # high

plan.model_dump_json(indent=2)  # Complete Roomify JSON, including nodes/rootNodeIds
```

For pixel geometry without VLM calls:

```python
plan = parse("floorplan.png", use_vlm=False)
```

## VLM configuration

Roomify works with an OpenAI-compatible endpoint that accepts images. It
reads credentials from the process environment:

```bash
export ROOMIFY_VLM_API_KEY="..."
export ROOMIFY_VLM_BASE_URL="https://your-endpoint.example/v1"
export ROOMIFY_VLM_MODEL="your-vision-model"
```

Roomify does not load `.env` files itself. `.env.example` lists the supported
variables.

## JSON structure

Every result is a validated `FloorPlan` document:

```text
FloorPlan
├── source_file, source_sha256, page
├── image_width_px, image_height_px, north_angle_deg, level_height_mm
├── scale
│   ├── px_per_mm_x, px_per_mm_y, anisotropy
│   └── method, confidence, evidence counts
├── rooms[]
│   ├── name, room_type, source, confidence
│   ├── polygon_px, area_px, perimeter_px, edge_lengths_px
│   ├── polygon_mm, area_sqm, perimeter_mm, edge_lengths_mm
│   └── printed_area_sqm, area_deviation, area_deviation_flag
├── walls[]
│   ├── start_px, end_px, thickness_px
│   ├── start_mm, end_mm, thickness_mm
│   └── rooms
├── openings[]
│   ├── element_type, bbox_px, center_px, width_px, width_mm
│   ├── sill_height_mm, height_mm, wall_id, connects, swing, hinge_px
│   ├── protrusion_polygon_px, protrusion_polygon_mm
│   └── source, confidence
├── elements[]
├── warnings[]
├── unresolved[]
├── nodes{} (metric node graph; empty without scale)
└── rootNodeIds[] (graph roots; empty without scale)
```

Key rules:

- Pixel coordinates use the original input image. The origin is the top-left;
  x points right and y points down.
- Polygon rings are open: the first point is not repeated at the end.
- If `scale` is `null`, image-derived planar millimetre and square-metre fields
  are `null`; the documented vertical defaults remain present.
- When the 2D drawing has no elevations, defaults are 2800mm storey height,
  2100mm doors, and 900/1500mm window sill/height. CLI flags and `parse()`
  keyword arguments override them.
- `warnings` records degraded or conflicting evidence.
- `unresolved` names fields that could not be established from the drawing.

See [`schema.py`](src/roomify/schema.py) for the complete field and enum
definitions.

## How it works

1. Load the image or PDF and keep a mapping back to its original pixels.
2. Detect walls, enclosed rooms, and wall openings with computer vision. When no real
   diagonal wall family is present, normalize room faces to strict orthogonal polygons and
   reject wall-width pockets; angled plans keep the general polygon path.
3. Send numbered overlays to the VLM for room labels, printed areas,
   dimensions, and symbol classes.
4. Merge both sources, solve the x/y scale, and validate the result with
   Pydantic.

Geometry does not depend on a successful VLM call. Failed calls produce a
warning and leave the affected fields unresolved.

### VLM requests

Room text and scale reads use
[structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
with strict `json_schema` and model thinking disabled. Opening classification
uses one review sheet containing the full plan and magnified crops, then validates
a JSON object with Pydantic. Some compatible endpoints accept the larger opening
schema but time out during constrained decoding; the single-image path avoids that
failure.

Roomify runs at most two VLM calls concurrently. If an endpoint rejects
`json_schema` or `enable_thinking`, that feature is disabled for the session.
A response that fails schema validation gets one schema-guided retry with
thinking enabled.

## Current scope

- The supported release target is orthogonal residential plans with
  solid-filled walls and conventional door/window symbols.
- Thin-line CAD plans use a simpler fallback path.
- Spaces without any separating stroke are returned as one room.
- Bay windows require visible outer and side strokes; incomplete evidence stays
  unresolved instead of receiving an invented depth.
- `floorplan-6`, `floorplan-8`, and `floorplan-14` contain genuine diagonal or
  multi-axis geometry and are explicitly unsupported by the orthogonal mode.
  Their historical regressions are diagnostic only and are not counted as
  supported passes.

## Evaluation and release gate

`examples/` is a frozen development corpus. Exact-pixel assertions under
`tests/integration/` detect regressions on that corpus; they are not evidence of
generalization. Run unit checks and development-corpus checks separately:

```bash
uv run pytest -m "not dev_corpus"
uv run pytest -m dev_corpus
```

Release evidence must come from an unrevealed, cross-source holdout stored
outside the checkout. Generate one Roomify JSON per private image, annotate the
same filenames in a separate directory, and run:

```bash
uv run roomify-eval /secure/predictions /secure/truth \
  --gates evaluation/release-gates-v1.json -o holdout-report.json
```

The evaluator reports room mean IoU, wall boundary F1, exact-type opening F1,
walkable adjacency-graph F1, and physical-violation rate. It applies the fixed
gate to the aggregate and to every `source_group`; v1 requires at least 30 cases
from three independent sources. A truth file uses this minimal schema:

```json
{
  "protocol_version": "1.0",
  "source_group": "independent-provider-a",
  "image_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "image_width_px": 1000,
  "image_height_px": 800,
  "orthogonal": true,
  "rooms": [{"id": "r1", "polygon_px": [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 100, "y": 80}, {"x": 0, "y": 80}]}],
  "walls": [{"start_px": {"x": 0, "y": 0}, "end_px": {"x": 100, "y": 0}}],
  "openings": [{"element_type": "single_door", "bbox_px": {"x0": 30, "y0": 0, "x1": 50, "y1": 8}, "connects": ["r1", "exterior"]}]
}
```

Do not copy, derive, or inspect holdout labels while changing extraction rules
or thresholds. A release is unverified until this command passes; bundled
examples alone can never produce a release claim.

## Development

```bash
uv sync --locked --extra dev
uv run pytest
uv run ruff check src tests
uv run mypy src
```

The live VLM development regression requires the three VLM environment
variables and consumes real calls:

```bash
uv run pytest tests/integration
```

## License

[Apache-2.0](LICENSE)
