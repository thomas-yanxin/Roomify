import cv2
import fitz
import numpy as np
import pytest

from roomify.io import load


def _write_png(path, w=400, h=300):
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (350, 250), (153, 153, 153), 8)
    cv2.imwrite(str(path), img)
    return img


def test_load_png(tmp_path):
    path = tmp_path / "户型图.png"  # exercises the non-ASCII path branch
    original = _write_png(path)
    src = load(path)
    assert src.page is None
    assert src.scale_to_original == 1.0
    assert src.original_width == 400 and src.original_height == 300
    assert np.array_equal(src.bgr, original)
    assert len(src.sha256) == 64


def test_resize_cap(tmp_path):
    path = tmp_path / "big.png"
    _write_png(path, w=4000, h=2000)
    src = load(path, max_dim=1000)
    assert max(src.bgr.shape[:2]) == 1000
    assert src.scale_to_original == pytest.approx(4.0)
    assert (src.original_width, src.original_height) == (4000, 2000)


def test_load_pdf_render_path(tmp_path):
    # A vector-only PDF (no embedded raster) must go through page rendering.
    pdf_path = tmp_path / "plan.pdf"
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.draw_rect(fitz.Rect(20, 20, 180, 80), color=(0, 0, 0), width=3)
    doc.save(pdf_path)
    doc.close()

    src = load(pdf_path, max_dim=8000)
    assert src.page == 0
    # 300 DPI render of a 200x100pt page ≈ 833x416 px
    assert src.original_width > 800
    assert (src.bgr < 128).any(), "drawn rectangle should be visible"


def test_load_pdf_embedded_image(tmp_path):
    # A PDF whose page embeds a large raster should extract that raster.
    img_path = tmp_path / "embed.png"
    _write_png(img_path, w=1400, h=1200)
    pdf_path = tmp_path / "scan.pdf"
    doc = fitz.open()
    page = doc.new_page(width=500, height=400)
    page.insert_image(fitz.Rect(0, 0, 500, 400), filename=str(img_path))
    doc.save(pdf_path)
    doc.close()

    src = load(pdf_path)
    assert (src.original_width, src.original_height) == (1400, 1200)


def test_page_out_of_range(tmp_path):
    pdf_path = tmp_path / "one.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()
    with pytest.raises(ValueError, match="out of range"):
        load(pdf_path, page=3)


def test_page_on_image_rejected(tmp_path):
    path = tmp_path / "img.png"
    _write_png(path)
    with pytest.raises(ValueError, match="non-PDF"):
        load(path, page=2)


def test_unsupported_extension(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hi")
    with pytest.raises(ValueError, match="unsupported"):
        load(path)
