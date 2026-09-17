from pathlib import Path
import base64
import re

import docx2txt
from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}

# Minimum characters from pypdf before we consider the PDF "scanned"
_OCR_FALLBACK_THRESHOLD = 100


def _normalize(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pdf_text_layer(path: Path) -> str:
    """Extract text from a PDF using its text layer (pypdf)."""
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _pdf_vision_ocr(path: Path) -> str:
    """Fallback: render each page to an image and let the LLM read it."""
    import pymupdf
    from .config import build_model

    model = build_model(temperature=0)

    doc = pymupdf.open(str(path))
    pages_text: list[str] = []

    for page_index in range(doc.page_count):
        page = doc[page_index]
        # 200 dpi gives good OCR quality without exploding token size
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("ascii")

        response = model.invoke([
            (
                "human",
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Transcribe every piece of readable text on this page. "
                            "Preserve line breaks, headings, and bullet structure. "
                            "Ignore decorative elements. Output plain text only."
                        ),
                    },
                ],
            )
        ])
        pages_text.append(str(response.content).strip())

    doc.close()
    return "\n\n".join(pages_text)


def load_document(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        text = _pdf_text_layer(path)
        if len(_normalize(text)) < _OCR_FALLBACK_THRESHOLD:
            text = _pdf_vision_ocr(path)
    else:
        text = docx2txt.process(str(path)) or ""

    text = _normalize(text)
    if not text:
        raise ValueError(f"No readable text found in {path}")
    return text
