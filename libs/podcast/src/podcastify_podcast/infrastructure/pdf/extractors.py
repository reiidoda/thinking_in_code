from __future__ import annotations

import math
import os
from io import BytesIO

from podcastify_podcast.application.ports import PdfExtractor
from podcastify_podcast.domain.models import PageText
from podcastify_podcast.infrastructure.text.normalize import (
    clean_page_text,
    drop_repeated_headers_footers,
)


class PlaceholderPdfExtractor(PdfExtractor):
    """Replace with PyMuPDF/pdfplumber implementation in M1."""

    def extract(self, pdf_bytes: bytes) -> list[PageText]:
        return [
            PageText(
                page_number=1,
                text=(
                    "PLACEHOLDER: PDF extraction not implemented yet. "
                    "This will be replaced with the real extracted research text."
                ),
                source="placeholder.pdf",
            )
        ]


class PdfPlumberExtractor(PdfExtractor):
    """Extract text page-by-page to keep citation metadata."""

    def __init__(self, source_name: str | None = None, min_chars: int = 40) -> None:
        self.source_name = source_name or "uploaded.pdf"
        self.min_chars = min_chars

    def extract(self, pdf_bytes: bytes) -> list[PageText]:
        pages = self._extract_with_pdfplumber(pdf_bytes)
        total_chars = sum(len(p.text) for p in pages)
        if total_chars < self.min_chars:
            try:
                fallback_pages = self._extract_with_pymupdf(pdf_bytes)
                if sum(len(p.text) for p in fallback_pages) > total_chars:
                    pages = fallback_pages
            except Exception:
                # keep original pages
                pass

        pages = drop_repeated_headers_footers(pages)

        if not pages or all(not p.text.strip() for p in pages):
            raise RuntimeError("No pages extracted from PDF.")
        return pages

    def _extract_with_pdfplumber(self, pdf_bytes: bytes) -> list[PageText]:
        try:
            import pdfplumber
        except Exception as e:  # pragma: no cover - import/runtime errors
            raise RuntimeError("pdfplumber is required for PDF extraction. Install with `pip install pdfplumber`.") from e

        pages: list[PageText] = []
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text() or ""
                cleaned = clean_page_text(raw_text)
                pages.append(PageText(page_number=idx, text=cleaned, source=self.source_name))
        return pages

    def _extract_with_pymupdf(self, pdf_bytes: bytes) -> list[PageText]:
        try:
            import fitz  # PyMuPDF
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PyMuPDF is required for fallback extraction. Install with `pip install pymupdf`.") from e

        pages: list[PageText] = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for idx, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                cleaned = clean_page_text(text)
                if len(cleaned) < self.min_chars:
                    ocr_text = self._ocr_page(page)
                    if ocr_text and len(ocr_text) > len(cleaned):
                        cleaned = clean_page_text(ocr_text)
                pages.append(PageText(page_number=idx, text=cleaned, source=self.source_name))
        return pages

    def _ocr_page(self, page) -> str:
        try:
            import pytesseract
            from PIL import Image
        except Exception:
            return ""
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img)


class TopicCorpusExtractor(PdfExtractor):
    """Selects a high-value scientific topic from a local corpus and returns merged pages.

    The scorer rewards: number of PDFs (coverage), math/technical density, and token volume,
    using logarithmic and square-root scaling to balance breadth vs. depth.
    """

    def __init__(self, base_dir: str, top_k_files: int = 3, min_chars: int = 200) -> None:
        self.base_dir = base_dir
        self.top_k_files = top_k_files
        self.min_chars = min_chars

    def extract(self, pdf_bytes: bytes) -> list[PageText]:  # pdf_bytes is ignored; corpus is on disk
        topics = self._discover_topics()
        if not topics:
            raise RuntimeError(f"No topics found under {self.base_dir}")

        scored = [(topic, self._score_topic(topic)) for topic in topics]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_topic, best_score = scored[0]
        pdfs = self._pdfs_for_topic(best_topic)[: self.top_k_files]
        if not pdfs:
            raise RuntimeError(f"No PDFs found for topic {best_topic}")

        pages: list[PageText] = []
        page_offset = 0
        for pdf_path in pdfs:
            file_pages = self._extract_pdf(pdf_path)
            for p in file_pages:
                pages.append(
                    PageText(
                        page_number=p.page_number + page_offset,
                        text=p.text,
                        source=f"{best_topic}/{os.path.basename(pdf_path)}",
                    )
                )
            page_offset += len(file_pages)

        if not pages or sum(len(p.text) for p in pages) < self.min_chars:
            raise RuntimeError(f"Topic '{best_topic}' did not yield enough text (score={best_score:.3f}).")

        return pages

    def _discover_topics(self) -> list[str]:
        if not os.path.isdir(self.base_dir):
            return []
        return [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]

    def _pdfs_for_topic(self, topic: str) -> list[str]:
        topic_dir = os.path.join(self.base_dir, topic)
        return sorted(
            [os.path.join(topic_dir, f) for f in os.listdir(topic_dir) if f.lower().endswith(".pdf")],
            key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0,
            reverse=True,
        )

    def _extract_pdf(self, path: str) -> list[PageText]:
        with open(path, "rb") as f:
            data = f.read()
        return PdfPlumberExtractor(source_name=os.path.basename(path), min_chars=self.min_chars)._extract_with_pdfplumber(data)

    def _score_topic(self, topic: str) -> float:
        pdfs = self._pdfs_for_topic(topic)
        if not pdfs:
            return 0.0
        total_tokens = 0
        math_mass = 0.0
        entropy_mass = 0.0
        docs_considered = 0
        for path in pdfs[: self.top_k_files]:
            try:
                text = self._read_head(path, max_chars=8000)
                tokens = self._tokens(text)
                if not tokens:
                    continue
                total_tokens += len(tokens)
                math_mass += self._math_density(tokens)
                entropy_mass += self._entropy(tokens)
                docs_considered += 1
            except Exception:
                continue
        if docs_considered == 0:
            return 0.0
        avg_math = math_mass / docs_considered
        avg_entropy = entropy_mass / docs_considered
        breadth = math.log1p(len(pdfs))          # more documents per topic
        depth = math.sqrt(total_tokens + 1)      # more tokens per topic
        # Harmonic-style blend to favor balanced math+diversity over sheer length
        balance = (2 * avg_math * avg_entropy) / max(avg_math + avg_entropy, 1e-6)
        return 0.5 * breadth + 0.25 * depth + 0.9 * balance

    def _read_head(self, path: str, max_chars: int = 8000) -> str:
        try:
            import pdfplumber
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pdfplumber is required for PDF extraction. Install with `pip install pdfplumber`.") from e
        out: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = (page.extract_text() or "")[:max_chars]
                out.append(txt)
                if sum(len(t) for t in out) >= max_chars:
                    break
        return "\n".join(out)

    def _tokens(self, text: str) -> list[str]:
        import re

        return re.findall(r"[A-Za-z0-9]{3,}", text)

    def _math_density(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        mathish = sum(1 for t in tokens if any(ch.isdigit() for ch in t) or any(op in t for op in ("+", "-", "*", "/", "^", "=")))
        return mathish / max(len(tokens), 1)

    def _entropy(self, tokens: list[str]) -> float:
        from collections import Counter

        counts = Counter(tokens)
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return -sum((c / total) * math.log(c / total + 1e-9) for c in counts.values())
