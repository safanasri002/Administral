"""Document metadata, read locally.

  * PDF   → PyMuPDF's document info (author, title, producer, dates, pages).
  * Image → Pillow's EXIF, including the GPS IFD when present.
  * ExifTool, when the binary is installed, enriches either with fields the
    Python readers miss (camera model on HEIC-converted JPEGs, XMP dates).

Metadata is *descriptive* here — it feeds the structured facts and the report.
Judging it as a forgery signal is `ai.fraud`'s job, not this module's.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
from datetime import datetime

import fitz  # PyMuPDF
from PIL import ExifTags, Image

from app.core.logger import logger
from app.modules.ai.fraud.metadata import exiftool_available
from app.modules.ai.inconsistency.schemas import DocumentMetadata, GpsPosition

_GPS_IFD = 0x8825


def _pdf_date(value: str | None) -> str | None:
    """`D:20240115103000+01'00'` → `2024-01-15T10:30:00`."""
    if not value:
        return None
    digits = re.sub(r"\D", "", value.removeprefix("D:"))[:14]
    for fmt, length in (("%Y%m%d%H%M%S", 14), ("%Y%m%d", 8)):
        if len(digits) >= length:
            try:
                return datetime.strptime(digits[:length], fmt).isoformat()
            except ValueError:
                return None
    return None


def _exif_date(value: object) -> str | None:
    """`2024:01:15 10:30:00` → ISO; anything unparsable is dropped."""
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip()[:19], "%Y:%m:%d %H:%M:%S").isoformat()
    except ValueError:
        return None


def _clean(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _pdf_metadata(data: bytes) -> DocumentMetadata:
    with fitz.open(stream=data, filetype="pdf") as doc:
        info = doc.metadata or {}
        return DocumentMetadata(
            format=_clean(info.get("format")) or "PDF",
            auteur=_clean(info.get("author")),
            titre=_clean(info.get("title")),
            sujet=_clean(info.get("subject")),
            logiciel=_clean(info.get("creator")),
            producteur=_clean(info.get("producer")),
            date_creation=_pdf_date(info.get("creationDate")),
            date_modification=_pdf_date(info.get("modDate")),
            nb_pages=doc.page_count,
            extracteur="pymupdf",
        )


def _dms_to_degrees(dms: object, ref: object) -> float | None:
    try:
        d, m, s = (float(x) for x in dms)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None
    degrees = d + m / 60 + s / 3600
    return -degrees if str(ref).upper() in {"S", "W"} else degrees


def _image_metadata(data: bytes) -> DocumentMetadata:
    with Image.open(io.BytesIO(data)) as image:
        exif = image.getexif()
        tags = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
        # DateTimeOriginal lives in the Exif sub-IFD, not the main one.
        sub = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.get_ifd(0x8769).items()}
        gps_raw = {ExifTags.GPSTAGS.get(k, str(k)): v for k, v in exif.get_ifd(_GPS_IFD).items()}

        gps = None
        if "GPSLatitude" in gps_raw and "GPSLongitude" in gps_raw:
            lat = _dms_to_degrees(gps_raw["GPSLatitude"], gps_raw.get("GPSLatitudeRef", "N"))
            lon = _dms_to_degrees(gps_raw["GPSLongitude"], gps_raw.get("GPSLongitudeRef", "E"))
            if lat is not None and lon is not None:
                gps = GpsPosition(latitude=round(lat, 6), longitude=round(lon, 6))

        appareil = " ".join(x for x in (_clean(tags.get("Make")), _clean(tags.get("Model"))) if x)
        return DocumentMetadata(
            format=image.format,
            auteur=_clean(tags.get("Artist")),
            logiciel=_clean(tags.get("Software")),
            date_modification=_exif_date(tags.get("DateTime")),
            date_prise_vue=_exif_date(sub.get("DateTimeOriginal")),
            appareil=appareil or None,
            gps=gps,
            nb_pages=1,
            extracteur="pillow",
        )


def _exiftool_enrich(path: str, base: DocumentMetadata) -> DocumentMetadata:
    """Fill the gaps the Python readers left. Never overrides a known value."""
    try:
        proc = subprocess.run(
            ["exiftool", "-j", "-n", path], capture_output=True, text=True, timeout=30
        )
        raw = json.loads(proc.stdout)[0] if proc.returncode == 0 else {}
    except (subprocess.SubprocessError, json.JSONDecodeError, IndexError, OSError) as exc:
        logger.warn("ExifTool illisible", {"error": str(exc)})
        return base

    found = {
        "auteur": _clean(raw.get("Author") or raw.get("Artist")),
        "titre": _clean(raw.get("Title")),
        "logiciel": _clean(raw.get("Creator") or raw.get("Software") or raw.get("CreatorTool")),
        "producteur": _clean(raw.get("Producer")),
        "date_creation": _exif_date(raw.get("CreateDate")),
        "date_modification": _exif_date(raw.get("ModifyDate")),
        "date_prise_vue": _exif_date(raw.get("DateTimeOriginal")),
        "appareil": " ".join(
            x for x in (_clean(raw.get("Make")), _clean(raw.get("Model"))) if x
        ) or None,
    }
    updates = {k: v for k, v in found.items() if v and getattr(base, k) is None}
    if base.gps is None and isinstance(raw.get("GPSLatitude"), (int, float)):
        updates["gps"] = GpsPosition(
            latitude=float(raw["GPSLatitude"]), longitude=float(raw.get("GPSLongitude", 0.0))
        )
    if updates:
        updates["extracteur"] = f"{base.extracteur}+exiftool"
    return base.model_copy(update=updates)


def extract_metadata(data: bytes, mime_type: str, path: str | None = None) -> DocumentMetadata:
    """Metadata for one upload. Unreadable files yield an empty record, never raise."""
    try:
        meta = _pdf_metadata(data) if mime_type == "application/pdf" else _image_metadata(data)
    except Exception as exc:  # noqa: BLE001 — a corrupt file is a data problem
        logger.warn("Métadonnées illisibles", {"error": str(exc)})
        meta = DocumentMetadata()
    if path and exiftool_available():
        meta = _exiftool_enrich(path, meta)
    return meta
