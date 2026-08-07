"""Synthetic floor-plan drawing helpers shared by the CV unit tests."""

import cv2
import numpy as np
import pytest

BG = 245
WALL_GREY = 154


def blank(w=700, h=700, bg=BG):
    return np.full((h, w, 3), bg, dtype=np.uint8)


def draw_wall_rect(img, x0, y0, x1, y1, thickness=8, grey=WALL_GREY):
    """Rectangular wall outline drawn as solid poché strokes."""
    cv2.rectangle(img, (x0, y0), (x1, y1), (grey, grey, grey), thickness)


def draw_gap(img, x0, y0, x1, y1, bg=BG):
    """Cut an opening (door/window void) out of previously drawn walls."""
    cv2.rectangle(img, (x0, y0), (x1, y1), (bg, bg, bg), -1)


def draw_sill(img, x0, y0, x1, y1, grey=120):
    """Thin threshold/sill stroke spanning an opening (1px)."""
    cv2.line(img, (x0, y0), (x1, y1), (grey, grey, grey), 1)


def draw_dimension_chain(img, y, x_ticks, grey=90):
    """1px dimension line with tick marks, as found outside the building."""
    cv2.line(img, (x_ticks[0], y), (x_ticks[-1], y), (grey, grey, grey), 1)
    for x in x_ticks:
        cv2.line(img, (x - 3, y + 3), (x + 3, y - 3), (grey, grey, grey), 1)


def add_texture(img, x0, y0, x1, y1, harsh=False):
    """Wood-grain-like colored fill. Real listing plans keep grain contrast
    under the adaptive threshold's C=10; harsh=True exaggerates it to force
    the pathological long-stripe case the fallback ladder must survive.
    """
    img[y0:y1, x0:x1] = (168, 196, 216)  # warm tan (BGR)
    stripe = (150, 180, 205) if harsh else (162, 191, 212)
    for y in range(y0, y1, 7):
        cv2.line(img, (x0, y), (x1, y), stripe, 1)


def add_text_blobs(img, points, grey=30):
    for x, y in points:
        cv2.putText(img, "9.65", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (grey,) * 3, 1)


@pytest.fixture
def simple_plan():
    """One 400x300 'building' with two rooms, a door gap sealed by a sill,
    a window gap with lines, dimension chains outside, texture + text inside.
    """
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    cv2.line(img, (300, 100), (300, 400), (WALL_GREY,) * 3, 8)  # dividing wall
    draw_gap(img, 296, 200, 304, 245)  # door gap in the divider
    draw_sill(img, 296, 200, 296, 245)
    draw_sill(img, 304, 200, 304, 245)
    draw_gap(img, 150, 96, 240, 104)  # window gap in the top wall
    for off in (-2, 0, 2):
        draw_sill(img, 150, 100 + off, 240, 100 + off, grey=130)
    add_texture(img, 110, 110, 290, 390)
    add_text_blobs(img, [(180, 250), (400, 250)])
    draw_dimension_chain(img, 60, [100, 300, 500])
    draw_dimension_chain(img, 440, [100, 500])
    return img
