"""Roomify — robust 2D floor plan → structured JSON (hybrid CV + VLM)."""

from typing import Any

from roomify.schema import FloorPlan

__version__ = "0.1.0"
__all__ = ["FloorPlan", "__version__", "parse"]


def __getattr__(name: str) -> Any:
    # Lazy so that schema-only consumers don't pay the cv2/openai import cost.
    if name == "parse":
        from roomify.pipeline import parse

        return parse
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
