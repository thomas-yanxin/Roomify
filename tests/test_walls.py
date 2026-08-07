import numpy as np

from conftest import WALL_GREY, blank, draw_dimension_chain, draw_wall_rect
from roomify.walls import estimate_wall_band, extract_walls


def test_band_found_on_grey_walls(simple_plan):
    band = estimate_wall_band(simple_plan)
    assert band is not None
    lo, hi = band
    assert lo <= WALL_GREY <= hi


def test_band_found_on_dark_walls():
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10, grey=100)
    band = estimate_wall_band(img)
    assert band is not None and band[0] <= 100 <= band[1]


def test_band_none_without_wall_mass():
    band = estimate_wall_band(blank())
    assert band is None


def test_solid_excludes_thin_and_textured(simple_plan):
    wx = extract_walls(simple_plan)
    assert not wx.band_fallback
    # dimension chains (1px, outside) must not reach the solid mask
    assert wx.solid[55:65, :].sum() == 0
    assert wx.solid[435:445, :].sum() == 0
    # textured room fill must not be walls
    assert wx.solid[150:250, 130:270].sum() == 0
    # actual walls are present at ~true width
    assert wx.solid[100:110, 350:450].any()
    assert 6 <= wx.thickness_px <= 16


def test_footprint_tight_around_walls(simple_plan):
    wx = extract_walls(simple_plan)
    x0, y0, x1, y1 = wx.footprint
    assert abs(x0 - 95) <= 3 and abs(y0 - 95) <= 3
    assert abs(x1 - 505) <= 3 and abs(y1 - 405) <= 3


def test_lines_carry_sills_and_window_strokes(simple_plan):
    wx = extract_walls(simple_plan)
    assert wx.lines[98:104, 160:230].any(), "window strokes should survive"
    # dimension chains sit outside the footprint gate
    assert wx.lines[55:65, :].sum() == 0


def test_union_is_superset(simple_plan):
    wx = extract_walls(simple_plan)
    assert np.all(wx.union[wx.solid > 0] == 255)
    assert np.all(wx.union[wx.lines > 0] == 255)


def test_fallback_band_no_crash():
    # Thin black strokes only: no band peak, no thick cores — must degrade,
    # not crash, and still expose the line structure.
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=1, grey=20)
    draw_dimension_chain(img, 60, [100, 500])
    wx = extract_walls(img)
    assert wx.solid.sum() == 0
    assert wx.lines.any()
    assert wx.thickness_px == 1.0
    assert wx.footprint[2] > wx.footprint[0]
