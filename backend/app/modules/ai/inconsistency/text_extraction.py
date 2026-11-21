"""Text extraction, OCR only where needed, all local.

A PDF page with a usable text layer is read directly by PyMuPDF. Only a page
without one (a scan glued into a PDF) is rendered to an image and OCR'd — so a
mixed PDF costs one OCR pass per scanned page, not per document. Images always
go to OCR.

OCR engines are optional dependencies, imported lazily:

  * `tesseract` — `pytesseract` + the tesseract binary (default; light).
  * `paddle`    — `paddleocr` (heavier, better on poor scans).

A missing engine yields empty text and a warning, never a cloud fallback.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from functools import lru_cache

import fitz  # PyMuPDF
from PIL import Image, ImageOps

from app.core.config import settings
from app.core.logger import logger

#: Below this many characters a page is treated as scanned.
PAGE_MIN_CHARS = 40
_OCR_DPI = 300


class OcrUnavailable(RuntimeError):
    """No local OCR engine is installed or configured."""


def sanitize(text: str) -> str:
    """Drop the C0 control characters PostgreSQL refuses (NUL first of all)."""
    return "".join(ch for ch in text if ch >= " " or ch in "\t\n\r")


@lru_cache
def _paddle_engine():  # pragma: no cover — optional heavy dependency
    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=True, lang="fr", show_log=False)


def _ocr_tesseract(image: Image.Image) -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise OcrUnavailable("pytesseract n'est pas installé") from exc
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    try:
        return pytesseract.image_to_string(image, lang=settings.ocr_languages)
    except pytesseract.TesseractNotFoundError as exc:
        raise OcrUnavailable("binaire tesseract introuvable (TESSERACT_CMD)") from exc


def _ocr_paddle(image: Image.Image) -> str:  # pragma: no cover — optional dependency
    try:
        engine = _paddle_engine()
    except ImportError as exc:
        raise OcrUnavailable("paddleocr n'est pas installé") from exc
    import numpy as np

    result = engine.ocr(np.array(image.convert("RGB")), cls=True) or []
    lines = [entry[1][0] for page in result if page for entry in page]
    return "\n".join(lines)


def ocr_image(image: Image.Image) -> str:
    """Run the configured local OCR engine on one image. Raises `OcrUnavailable`."""
    backend = settings.ocr_backend.lower()
    # Grayscale + EXIF orientation: cheap, and both engines read better for it.
    prepared = ImageOps.exif_transpose(image).convert("L")
    if backend == "tesseract":
        return sanitize(_ocr_tesseract(prepared))
    if backend == "paddle":
        return sanitize(_ocr_paddle(prepared))
    raise OcrUnavailable(f"OCR local désactivé (OCR_BACKEND={settings.ocr_backend})")


@dataclass
class LocalText:
    text: str
    #: `texte_natif`, `ocr_local`, `mixte` or `aucune`.
    method: str
    ocr_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _pdf_text(data: bytes) -> LocalText:
    pages: list[str] = []
    ocr_pages: list[int] = []
    warnings: list[str] = []
    native_pages = 0

    with fitz.open(stream=data, filetype="pdf") as doc:
        for index, page in enumerate(doc, start=1):
            native = sanitize(page.get_text("text")).strip()
            if len(native) >= PAGE_MIN_CHARS:
                pages.append(native)
                native_pages += 1
                continue
            try:
                pix = page.get_pixmap(dpi=_OCR_DPI)
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                pages.append(ocr_image(image).strip())
                ocr_pages.append(index)
            except OcrUnavailable as exc:
                warnings.append(f"Page {index} sans texte et OCR local indisponible : {exc}")
                pages.append(native)

    text = "\n".join(p for p in pages if p).strip()
    if not text:
        method = "aucune"
    elif ocr_pages and native_pages:
        method = "mixte"
    elif ocr_pages:
        method = "ocr_local"
    else:
        method = "texte_natif"
    return LocalText(text=text, method=method, ocr_pages=ocr_pages, warnings=warnings)


def extract_local_text(data: bytes, mime_type: str) -> LocalText:
    """Text of a PDF or image, without leaving the server. Never raises."""
    try:
        if mime_type == "application/pdf":
            return _pdf_text(data)
        with Image.open(io.BytesIO(data)) as image:
            text = ocr_image(image).strip()
        return LocalText(text=text, method="ocr_local" if text else "aucune", ocr_pages=[1])
    except OcrUnavailable as exc:
        return LocalText(text="", method="aucune", warnings=[f"OCR local indisponible : {exc}"])
    except Exception as exc:  # noqa: BLE001 — a corrupt file must not break the upload
        logger.warn("Extraction de texte locale impossible", {"error": str(exc)})
        return LocalText(text="", method="aucune", warnings=[f"Document illisible : {exc}"])
