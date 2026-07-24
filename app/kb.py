"""The whole KB layer: glob kb_source/, extract text, concatenate, guard.

No retrieval, no embeddings, no chunking. The full text goes in the system
prompt. See §5 of the spec.
"""

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

from app.config import settings

log = logging.getLogger(__name__)

SUPPORTED = {".md", ".txt", ".json", ".pdf"}

# A file that extracts below this is assumed broken (scanned PDF, empty file)
# rather than merely short. Crash instead of serving a half-empty KB.
MIN_CHARS_PER_FILE = 200


class KBError(RuntimeError):
    """Raised at startup when the KB cannot be trusted. Never swallowed."""


@dataclass
class KBFile:
    name: str
    chars: int
    pages: int | None = None


@dataclass
class KB:
    text: str = ""
    files: list[KBFile] = field(default_factory=list)
    tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.files


def count_tokens(text: str) -> int:
    """Token count for the KB guard. Falls back to an estimate if tiktoken
    can't fetch its encoding (it downloads BPE data on first use)."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception as exc:  # network, cache miss, whatever
        log.warning("tiktoken unavailable (%s); estimating tokens as chars/4", exc)
        return len(text) // 4


def _pdf_has_no_text_layer(path: Path) -> bool | None:
    """True if pdffonts reports an empty font table => scanned/raster PDF.

    None when pdffonts (poppler) isn't installed, in which case we fall back to
    the post-extraction char-count guard below.
    """
    if not shutil.which("pdffonts"):
        return None
    try:
        out = subprocess.run(
            ["pdffonts", str(path)], capture_output=True, text=True, timeout=30
        )
    except (subprocess.SubprocessError, OSError):
        return None
    # Header is 2 lines ("name type ..." + a rule); any font => real text layer.
    rows = [ln for ln in out.stdout.splitlines()[2:] if ln.strip()]
    return len(rows) == 0


def _scanned_pdf_error(name: str) -> KBError:
    return KBError(
        f"{name}: no extractable text — this PDF looks scanned/image-based "
        f"(no text layer). Extraction would silently yield an empty KB, so "
        f"refusing to start.\n"
        f"Two options, your call:\n"
        f"  1. Sarvam Vision (document digitization) — same API key, built for "
        f"Indian-language documents.\n"
        f"  2. pytesseract OCR locally.\n"
        f"Tell me which and I'll wire it up."
    )


def _extract_pdf(path: Path) -> tuple[str, int]:
    if _pdf_has_no_text_layer(path):
        raise _scanned_pdf_error(path.name)

    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            body = (page.extract_text() or "").strip()
            if body:
                parts.append(f"=== SOURCE: {path.name} | page {i} ===\n{body}")
    return "\n\n".join(parts), pages


def _extract_json(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise KBError(f"{path.name}: invalid JSON — {exc}") from exc
    # ensure_ascii=False or Devanagari becomes \uXXXX and burns ~6x the tokens.
    body = json.dumps(data, indent=2, ensure_ascii=False)
    return f"=== SOURCE: {path.name} ===\n{body}"


def _extract_text(path: Path) -> str:
    body = path.read_text(encoding="utf-8").strip()
    return f"=== SOURCE: {path.name} ===\n{body}"


def load_kb() -> KB:
    kb_dir = settings.kb_path
    if not kb_dir.is_dir():
        log.warning("KB directory %s does not exist — starting with an empty KB", kb_dir)
        return KB()

    paths = sorted(
        p for p in kb_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    if not paths:
        log.warning(
            "No KB files in %s (looking for %s) — starting with an empty KB. "
            "Drop the Magppie files in and restart.",
            kb_dir,
            ", ".join(sorted(SUPPORTED)),
        )
        return KB()

    blocks: list[str] = []
    files: list[KBFile] = []

    for path in paths:
        suffix = path.suffix.lower()
        pages: int | None = None

        if suffix == ".pdf":
            block, pages = _extract_pdf(path)
        elif suffix == ".json":
            block = _extract_json(path)
        else:
            block = _extract_text(path)

        # Char count of the extracted body, not the delimiter scaffolding.
        body_chars = len(block) - len(f"=== SOURCE: {path.name} ===\n")

        if body_chars < MIN_CHARS_PER_FILE:
            if suffix == ".pdf":
                raise _scanned_pdf_error(path.name)
            raise KBError(
                f"{path.name}: extracted only {body_chars} chars "
                f"(minimum {MIN_CHARS_PER_FILE}). The file is empty or unreadable; "
                f"refusing to start with a half-empty KB."
            )

        blocks.append(block)
        files.append(KBFile(name=path.name, chars=body_chars, pages=pages))

    text = "\n\n".join(blocks)
    tokens = count_tokens(text)

    if tokens > settings.kb_token_limit:
        raise KBError(
            f"KB is {tokens:,} tokens, over KB_TOKEN_LIMIT={settings.kb_token_limit:,}. "
            f"gpt-4o-mini has a 128k window and conversation history needs room. "
            f"Trim the KB or raise the limit."
        )

    return KB(text=text, files=files, tokens=tokens)


def log_kb(kb: KB) -> None:
    """Always print what actually loaded. Trust the answers only after this."""
    if kb.is_empty:
        log.warning("KB EMPTY — /api/turn will refuse until files are added")
        return

    log.info("KB loaded — %d file(s)", len(kb.files))
    for f in kb.files:
        pages = f"{f.pages} page(s)" if f.pages is not None else "-"
        log.info("  %-40s %8s chars  %s", f.name, f"{f.chars:,}", pages)
    log.info("  TOTAL: %s chars, ~%s tokens", f"{len(kb.text):,}", f"{kb.tokens:,}")
