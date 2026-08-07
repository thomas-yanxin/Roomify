"""Input handling: PDF and common image formats → a working BGR image.

CV constants downstream are tuned for working resolutions around 700–2000 px,
so oversized inputs are downscaled and ``scale_to_original`` carries the
factor: the emitted JSON is always in ORIGINAL-image pixels, whatever
resolution the pipeline actually processed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_RENDER_DPI = 300
# Embedded raster smaller than this is likely a thumbnail/logo; render the
# page instead (also the only correct path for vector/CAD PDFs).
MIN_EMBEDDED_PX = 1000


@dataclass(frozen=True)
class SourceImage:
    bgr: np.ndarray  # working image, BGR uint8
    scale_to_original: float  # working px * this = original px
    sha256: str  # of the input file bytes
    original_width: int
    original_height: int
    source_file: str
    page: int | None  # PDF page index; None for images


def load(path: str | Path, page: int | None = None, max_dim: int = 2000) -> SourceImage:
    path = Path(path)
    bgr: np.ndarray
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        page = 0 if page is None else page
        bgr = _load_pdf_page(data, page)
    elif suffix in IMAGE_EXTENSIONS:
        if page not in (None, 0):
            raise ValueError(f"page={page} given for a non-PDF input: {path.name}")
        page = None
        # imdecode instead of imread: robust to non-ASCII paths (常见中文路径).
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"could not decode image: {path}")
        bgr = decoded
    else:
        raise ValueError(f"unsupported input type {suffix!r}: {path.name}")

    oh, ow = bgr.shape[:2]
    scale_to_original = 1.0
    if max(oh, ow) > max_dim:
        factor = max_dim / max(oh, ow)
        bgr = cv2.resize(
            bgr, (round(ow * factor), round(oh * factor)), interpolation=cv2.INTER_AREA
        )
        scale_to_original = 1.0 / factor

    return SourceImage(
        bgr=bgr,
        scale_to_original=scale_to_original,
        sha256=sha,
        original_width=ow,
        original_height=oh,
        source_file=str(path),
        page=page,
    )


def _load_pdf_page(data: bytes, page_index: int) -> np.ndarray:
    import fitz  # pymupdf — imported lazily; image-only users never need it

    with fitz.open(stream=data, filetype="pdf") as doc:
        if not 0 <= page_index < doc.page_count:
            raise ValueError(f"page {page_index} out of range (PDF has {doc.page_count} pages)")
        pdf_page = doc[page_index]

        # Prefer the largest embedded raster (scanned/exported plans keep full
        # fidelity there); fall back to rendering for vector/CAD PDFs.
        best: fitz.Pixmap | None = None
        for img_info in pdf_page.get_images(full=True):
            try:
                pix = fitz.Pixmap(doc, img_info[0])
            except Exception:
                continue
            if pix.n - pix.alpha != 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            if best is None or pix.width * pix.height > best.width * best.height:
                best = pix
        if best is not None and min(best.width, best.height) >= MIN_EMBEDDED_PX:
            pix = fitz.Pixmap(best, 0) if best.alpha else best
        else:
            zoom = PDF_RENDER_DPI / 72
            pix = pdf_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
