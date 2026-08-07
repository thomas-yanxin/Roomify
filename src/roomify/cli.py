"""Command line: ``roomify plan.png -o plan.json``."""

from __future__ import annotations

import argparse
import sys


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
    args = parser.parse_args(argv)

    from roomify.pipeline import parse

    try:
        plan = parse(
            args.input, page=args.page, use_vlm=not args.no_vlm, debug_dir=args.debug
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"roomify: {exc}", file=sys.stderr)
        return 1

    payload = plan.model_dump_json(indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
