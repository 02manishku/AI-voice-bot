"""The KB guards are what stop the demo from silently lying.

A scanned PDF extracts to nothing, raises no exception, and leaves you with a
healthy-looking server that answers "I don't know" to everything. These tests
prove the guards actually fire.

Run: uv run python tests/test_kb_guards.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import kb as kb_mod
from app.config import settings

PROSE = "Magppie matte finishes are available in 47 shades. " * 12


def make_pdf(path: Path, pages: list[str] | None) -> None:
    """Hand-build a minimal PDF. pages=None writes a page with no text layer
    at all — a stand-in for a scanned/raster PDF.

    Built by hand rather than via a library so the bytes are unambiguous: this
    fixture is the thing proving the guard works, so it can't be the thing in
    doubt.
    """
    objs: list[bytes] = []
    page_texts = pages if pages is not None else [None]
    n_pages = len(page_texts)

    # 1 catalog, 2 pages tree, then per page: page obj + content stream.
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(n_pages))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(
        f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode()
    )

    font_ref = 3 + n_pages * 2
    for i, text in enumerate(page_texts):
        content_id = 4 + i * 2
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_ref} 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode()
        )
        if text is None:
            stream = b""  # no text operators => nothing to extract
        else:
            escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            stream = f"BT /F1 12 Tf 40 700 Td ({escaped}) Tj ET".encode()
        objs.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )

    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()

    path.write_bytes(bytes(out))


def run(label, fn):
    try:
        fn()
        print(f"  PASS  {label}")
        return True
    except AssertionError as e:
        print(f"  FAIL  {label}: {e}")
        return False


def with_kb_dir(d: Path):
    settings.kb_source_dir = str(d)


ok = True

with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    with_kb_dir(d)

    # --- empty dir -> empty KB, no crash (build order steps 1-4 need this) ---
    def t_empty():
        kb = kb_mod.load_kb()
        assert kb.is_empty, "expected empty KB"

    ok &= run("empty dir -> empty KB, server still starts", t_empty)

    # --- healthy text + json ---
    def t_good():
        (d / "catalogue.md").write_text(PROSE, encoding="utf-8")
        (d / "faq.json").write_text(
            json.dumps({"q": "फिनिश कितने हैं?", "a": "47 " + "shades " * 40},
                       ensure_ascii=False),
            encoding="utf-8",
        )
        kb = kb_mod.load_kb()
        names = {f.name for f in kb.files}
        assert names == {"catalogue.md", "faq.json"}, names
        assert "=== SOURCE: catalogue.md ===" in kb.text
        assert kb.tokens > 0, "no tokens counted"
        # ensure_ascii=False: Devanagari stays literal instead of \uXXXX
        assert "फिनिश" in kb.text, "devanagari got escaped -> ~6x tokens"
        assert "\\u0915" not in kb.text

    ok &= run("md + json load, devanagari not escaped", t_good)

    # --- short file -> crash ---
    def t_short():
        (d / "stub.txt").write_text("too short", encoding="utf-8")
        try:
            kb_mod.load_kb()
        except kb_mod.KBError as e:
            assert "stub.txt" in str(e), f"error must name the file: {e}"
            return
        raise AssertionError("expected KBError for a 9-char file")

    ok &= run("file under 200 chars -> crashes, names the file", t_short)
    (d / "stub.txt").unlink()

    # --- token limit -> crash ---
    def t_limit():
        original = settings.kb_token_limit
        settings.kb_token_limit = 10
        try:
            kb_mod.load_kb()
        except kb_mod.KBError as e:
            assert "KB_TOKEN_LIMIT" in str(e), e
            return
        finally:
            settings.kb_token_limit = original
        raise AssertionError("expected KBError when over KB_TOKEN_LIMIT")

    ok &= run("over KB_TOKEN_LIMIT -> crashes", t_limit)

# --- the PDF trap ---
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    with_kb_dir(d)

    def t_scanned():
        make_pdf(d / "scanned.pdf", None)
        (d / "catalogue.md").write_text(PROSE, encoding="utf-8")
        try:
            kb_mod.load_kb()
        except kb_mod.KBError as e:
            msg = str(e)
            assert "scanned.pdf" in msg, f"must name the file: {msg}"
            assert "Sarvam Vision" in msg and "pytesseract" in msg, "must offer both options"
            return
        raise AssertionError("SCANNED PDF LOADED SILENTLY — the exact trap the spec warns about")

    ok &= run("scanned PDF -> crashes loudly, offers both OCR options", t_scanned)
    (d / "scanned.pdf").unlink()

    # The false-positive case: a real text PDF must load, not trip the guard.
    def t_real_pdf():
        make_pdf(
            d / "pricing.pdf",
            [
                "Magppie matte finishes price list. " + "Shade A costs 1200 rupees. " * 6,
                "Gloss finishes page two. " + "Shade B costs 1500 rupees. " * 6,
            ],
        )
        kb = kb_mod.load_kb()
        pdf = next(f for f in kb.files if f.name == "pricing.pdf")
        assert pdf.pages == 2, f"expected 2 pages, got {pdf.pages}"
        assert pdf.chars > 200, f"real PDF extracted only {pdf.chars} chars"
        assert "=== SOURCE: pricing.pdf | page 1 ===" in kb.text, "page 1 not tagged"
        assert "=== SOURCE: pricing.pdf | page 2 ===" in kb.text, "page 2 not tagged"
        assert "Gloss finishes page two" in kb.text, "page 2 body missing"

    ok &= run("real text PDF -> extracts, tags page numbers, no false crash", t_real_pdf)

print("\nOK" if ok else "\nFAILED")
sys.exit(0 if ok else 1)
