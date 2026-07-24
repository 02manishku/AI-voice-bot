"""The partial-JSON prose extractor is the one piece of clever code here.

Run: uv run python tests/test_prose_stream.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Same guard as app/main.py, and for the same reason: this console is cp1252 and
# the cases below are Devanagari. Without it the assertions all pass and the run
# still dies — inside the print that reports they passed.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.llm import _prose_so_far as prose

CASES = [
    {"answer": 'We have 47 finishes across matte, gloss "and" textured.', "citations": []},
    {"answer": "नमस्ते, हमारे पास 47 फिनिश हैं।", "citations": [{"source": "a.pdf", "page": 3}]},
    {"answer": "Line one.\nLine two \\ backslash. Tab\there.", "citations": []},
    {"answer": "", "citations": []},
]


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


ok = True

for case in CASES:
    # ensure_ascii=True is the harsher path: \uXXXX escapes arriving in pieces.
    for ensure_ascii in (False, True):
        full = json.dumps(case, ensure_ascii=ensure_ascii)
        label = f"{'escaped' if ensure_ascii else 'literal'}: {case['answer'][:28]!r}"

        # Feed one char at a time; output must only ever grow, never regress.
        prev = ""
        monotonic = True
        for i in range(len(full) + 1):
            cur = prose(full[:i])
            if not cur.startswith(prev) or len(cur) < len(prev):
                monotonic = False
                print(f"    regressed at {i}: {prev!r} -> {cur!r}")
                break
            prev = cur

        ok &= check(f"{label} — monotonic", monotonic)
        ok &= check(f"{label} — matches json.loads", prose(full) == case["answer"])

print("\nboundaries:")
ok &= check("no answer key yet -> empty", prose('{"citations"') == "")
ok &= check("truncated \\u -> no garbage", prose('{"answer":"a\\u09') == "a")
ok &= check("truncated escape -> no garbage", prose('{"answer":"a\\') == "a")
ok &= check("stops at closing quote", prose('{"answer":"hi","citations":[]}') == "hi")
ok &= check(
    "citations never leak into prose",
    "source" not in prose('{"answer":"hi","citations":[{"source":"x.pdf"}]}'),
)

print("\nOK" if ok else "\nFAILED")
sys.exit(0 if ok else 1)
