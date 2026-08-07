"""Command line: ``roomify plan.png -o plan.json``."""

from __future__ import annotations

import argparse
import sys

from roomify.schema import (
    DEFAULT_DOOR_HEIGHT_MM,
    DEFAULT_LEVEL_HEIGHT_MM,
    DEFAULT_WINDOW_HEIGHT_MM,
    DEFAULT_WINDOW_SILL_HEIGHT_MM,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roomify",
        description="Parse a 2D floor plan (image or PDF) into structured JSON.",
    )
    parser.add_argument("input", help="floor-plan image (.png/.jpg/…) or PDF")
    parser.add_argument("-o", "--output", help="write JSON here (default: stdout)")
    parser.add_argument("--page", type=int, default=None, help="PDF page index (default 0)")
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="CV only: pixel geometry without names/types/mm",
    )
    parser.add_argument("--debug", metavar="DIR", help="write per-stage debug overlays")
    parser.add_argument("--level-height-mm", type=float, default=DEFAULT_LEVEL_HEIGHT_MM)
    parser.add_argument("--door-height-mm", type=float, default=DEFAULT_DOOR_HEIGHT_MM)
    parser.add_argument(
        "--window-sill-height-mm", type=float, default=DEFAULT_WINDOW_SILL_HEIGHT_MM
    )
    parser.add_argument("--window-height-mm", type=float, default=DEFAULT_WINDOW_HEIGHT_MM)
    args = parser.parse_args(argv)

    from roomify.pipeline import parse

    try:
        plan = parse(
            args.input,
            page=args.page,
            use_vlm=not args.no_vlm,
            debug_dir=args.debug,
            level_height_mm=args.level_height_mm,
            door_height_mm=args.door_height_mm,
            window_sill_height_mm=args.window_sill_height_mm,
            window_height_mm=args.window_height_mm,
        )
        payload = plan.model_dump_json(indent=2)
    except (ValueError, FileNotFoundError) as exc:
        print(f"roomify: {exc}", file=sys.stderr)
        return 1

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
