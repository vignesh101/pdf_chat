from __future__ import annotations

import io
import os
from typing import Optional


def _decode_text_bytes(content_bytes: bytes) -> str:
    try:
        return content_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_pdf_text(content_bytes: bytes) -> str:
    # Try pypdf first, then fall back to PyPDF2
    reader = None
    exc: Optional[Exception] = None
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(io.BytesIO(content_bytes))
    except Exception as e1:  # pragma: no cover - optional dep
        exc = e1
        try:
            from PyPDF2 import PdfReader  # type: ignore
            reader = PdfReader(io.BytesIO(content_bytes))
            exc = None
        except Exception as e2:  # pragma: no cover - optional dep
            exc = e2

    if reader is None:
        raise RuntimeError(
            "PDF support requires 'pypdf' (or 'PyPDF2'). Install with: pip install pypdf"
        ) from exc

    texts = []
    try:
        for page in getattr(reader, "pages", []):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt:
                texts.append(txt)
    except Exception as e:
        # Fall back to full decode if page iteration fails
        raise RuntimeError("Failed to read PDF content.") from e

    return "\n\n".join(texts)


def extract_text(file_name: str, content_bytes: bytes) -> str:
    """Extract text from uploaded file bytes.

    - For .pdf, attempts to read with pypdf/PyPDF2.
    - For known text-like extensions, UTF-8 decodes.
    - Otherwise, best-effort UTF-8 decode.
    """
    # File extension check
    _, ext = os.path.splitext(file_name.lower())
    if ext == ".pdf" or (content_bytes[:4] == b"%PDF"):
        return _extract_pdf_text(content_bytes)

    if ext in (
        ".txt",
        ".md",
        ".rtf",
        ".json",
        ".csv",
        ".tsv",
        ".log",
        ".ini",
        ".cfg",
        ".yaml",
        ".yml",
        ".py",
        ".html",
        ".htm",
    ):
        return _decode_text_bytes(content_bytes)

    # Default: best-effort decode
    return _decode_text_bytes(content_bytes)

