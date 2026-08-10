import cv2
import numpy as np

from conftest import WALL_GREY, blank, draw_dimension_chain, draw_wall_rect
from roomify.walls import estimate_wall_bands, extract_walls


def test_band_found_on_grey_walls(simple_plan):
    bands = estimate_wall_bands(simple_plan)
    assert bands
    lo, hi = bands[0]
    assert lo <= WALL_GREY <= hi


def test_band_found_on_dark_walls():
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10, grey=100)
    bands = estimate_wall_bands(img)
    assert bands and bands[0][0] <= 100 <= bands[0][1]


def test_band_none_without_wall_mass():
    assert estimate_wall_bands(blank()) == []


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


def test_light_noisy_fill_never_becomes_a_band():
    # Kitchen-beige / balcony-floor fills sit near the paper tone and carry
    # texture+JPEG speckle. Speckle fragments the fill into stroke-sized
    # chunks that defeat the per-component gate — the band itself must stay
    # clear of near-background tones or the fill erases the room it covers.
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    rng = np.random.default_rng(7)
    coarse = rng.integers(-12, 13, size=(56, 76, 1))  # ~5px speckle after ×5
    noise = cv2.resize(coarse.astype(np.float32), (380, 280), interpolation=cv2.INTER_NEAREST)
    img[110:390, 110:490] = np.clip(210 + noise[..., None], 0, 255).astype(np.uint8)
    # glazing strip in the same light tone: the stroke that legitimizes the band
    cv2.line(img, (150, 96), (240, 96), (208, 208, 208), 6)

    bands = estimate_wall_bands(img)
    assert bands and any(lo <= WALL_GREY <= hi for lo, hi in bands)
    assert all(hi < 198 for _, hi in bands), f"fill tone admitted: {bands}"
    wx = extract_walls(img)
    assert wx.solid[150:350, 150:450].sum() == 0, "fill body entered the solid mask"


def test_diagonal_wall_family_detected_and_sealed():
    # Diamond-wing plans draw thin diagonal walls whose anti-aliased core is
    # too slim for the solid mask's thickness test; the off-axis family must
    # be detected and its strokes carried into the union via rotated
    # long-run passes — an exact-45° line kernel would miss a 43° family.
    # fp6-like structure: thick black walls establish the black band, thin
    # SAME-TONE diagonals then pass the wall-tone acceptance gate.
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10, grey=20)
    cv2.line(img, (150, 150), (290, 290), (20, 20, 20), 3)
    cv2.line(img, (310, 290), (450, 150), (20, 20, 20), 3)
    wx = extract_walls(img)
    assert wx.angles, "off-axis wall family not detected"
    assert any(abs(abs(a) - 45) < 8 for a in wx.angles)
    hits = sum(wx.union[150 + t, 150 + t] > 0 for t in (40, 60, 80, 100))
    assert hits >= 3, "diagonal stroke missing from the room-enclosure union"


def test_decorative_diagonal_does_not_become_a_wall():
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10)
    cv2.line(img, (105, 395), (495, 105), (20, 20, 20), 2)

    wx = extract_walls(img)

    assert wx.angles == ()
    assert wx.union[250, 300] == 0


def test_thick_wall_connected_to_column_keeps_its_band():
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=40, grey=20)
    cv2.rectangle(img, (220, 100), (280, 220), (20, 20, 20), -1)

    bands = estimate_wall_bands(img)
    wx = extract_walls(img)

    assert any(lo <= 20 <= hi for lo, hi in bands)
    assert wx.solid[90:110, 150:200].any()
    assert wx.solid[150:200, 230:270].any()


def test_mixed_black_and_grey_walls_do_not_admit_between_tone_fill():
    # Load-bearing walls black, one partition grey, plus a large fill whose
    # tone lies between them. Only the two qualified bands may enter solid.
    img = blank()
    draw_wall_rect(img, 100, 100, 500, 400, thickness=10, grey=20)
    cv2.line(img, (300, 100), (300, 400), (WALL_GREY,) * 3, 8)
    cv2.rectangle(img, (120, 150), (280, 390), (100, 100, 100), -1)
    bands = estimate_wall_bands(img)
    assert any(lo <= 20 <= hi for lo, hi in bands), bands
    assert any(lo <= WALL_GREY <= hi for lo, hi in bands), bands
    assert not any(lo <= 100 <= hi for lo, hi in bands), bands
    wx = extract_walls(img)
    assert wx.solid[200:210, 296:305].any()  # grey partition present
    assert wx.solid[100:110, 200:220].any()  # black wall present
    assert wx.solid[180:360, 140:260].sum() == 0  # between-tone fill excluded
