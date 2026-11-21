"""Text extraction from an uploaded document.

Local first, so a citizen's document stays on the server whenever possible:

  * `native_pdf` — PyMuPDF reads the PDF's text layer (`ai.inconsistency`'s
    local extractor, shared so OCR never runs twice on one upload).
  * `local_ocr`  — a page or image without a text layer goes through the local
    OCR engine (Tesseract or PaddleOCR).
  * `mistral_ocr` — only when local reading yields too little text *and*
    `OCR_CLOUD_FALLBACK` is on: the file goes to Mistral's OCR endpoint.

Degrades honestly: no engine, or an OCR failure, yields empty text with a
reason, never fabricated content.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import httpx

from app.core.config import settings
from app.core.logger import logger
from app.modules.ai.inconsistency.text_extraction import LocalText, extract_local_text, sanitize
from app.modules.citizen.models import ExtractionMethod

#: The preview shown in the UI is capped — the full text can be large and the
#: panel only needs enough to reassure the citizen the right document was read.
PREVIEW_CHARS = 500

#: Below this many characters, local extraction is treated as failure and the
#: cloud fallback (if enabled) takes over. Matches Squad_B's threshold.
_NATIVE_MIN_CHARS = 40

_OCR_TIMEOUT_S = 60.0


#: Kept under its historical name: callers and tests import it from here.
_sanitize = sanitize


@dataclass
class ExtractionResult:
    method: ExtractionMethod | None
    text: str
    error: str | None = None
    #: The on-server extraction, reused by the inconsistency analysis.
    local: LocalText | None = None

    @property
    def preview(self) -> str | None:
        if not self.text:
            return None
        return self.text[:PREVIEW_CHARS]


def _extract_with_mistral_ocr(data: bytes, mime_type: str) -> str:
    """Mistral OCR on a scan or image. Raises on failure; the caller degrades.

    Ported from Squad_B's `extract_with_mistral_ocr`: the file is sent as a
    base64 data URL to the dedicated OCR model, and the page markdown is joined.
    """
    if not settings.mistral_api_key:
        raise RuntimeError("OCR requis, mais MISTRAL_API_KEY est absente.")

    encoded = base64.b64encode(data).decode("ascii")
    response = httpx.post(
        settings.mistral_ocr_url,
        headers={
            "Authorization": f"Bearer {settings.mistral_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.mistral_ocr_model,
            "document": {
                "type": "document_url",
                "document_url": f"data:{mime_type};base64,{encoded}",
            },
        },
        timeout=_OCR_TIMEOUT_S,
    )
    response.raise_for_status()
    pages = response.json().get("pages", [])
    return _sanitize("\n".join(page.get("markdown", "") for page in pages).strip())


def extract_text(data: bytes, mime_type: str) -> ExtractionResult:
    """Pull text from an upload, choosing the method by content.

    A PDF with a usable text layer, or a scan the local OCR can read, never
    touches the network. Only what stays unreadable locally may go to the cloud
    OCR — and only when `OCR_CLOUD_FALLBACK` allows it and a key is configured.
    """
    local = extract_local_text(data, mime_type)
    if len(local.text) >= _NATIVE_MIN_CHARS:
        method = (
            ExtractionMethod.native_pdf
            if local.method == "texte_natif"
            else ExtractionMethod.local_ocr
        )
        return ExtractionResult(method=method, text=local.text, local=local)

    if not settings.ocr_cloud_fallback:
        reason = "; ".join(local.warnings) or "Aucun texte exploitable localement."
        return ExtractionResult(method=None, text="", error=reason, local=local)

    try:
        text = _extract_with_mistral_ocr(data, mime_type)
    except Exception as exc:  # noqa: BLE001 — OCR failure degrades, never crashes upload
        logger.warn("OCR indisponible", {"error": str(exc)})
        return ExtractionResult(
            method=None, text="", error=f"OCR indisponible : {exc}", local=local
        )

    if not text:
        return ExtractionResult(
            method=ExtractionMethod.mistral_ocr,
            text="",
            error="L'OCR n'a retourné aucun contenu exploitable.",
            local=local,
        )

    return ExtractionResult(method=ExtractionMethod.mistral_ocr, text=text, local=local)
