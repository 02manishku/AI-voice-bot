"""Speech-to-text has no word for "Magppie" and substitutes the nearest real one.

The repair must be precise, not greedy. A pattern that also matches a real word
silently corrupts genuine questions — "मैट" is "matte", a finish in K-27, and
"MacBook" is an off-topic question the bot is meant to decline. Those failures
are worse than missing a mis-hearing, so they get tested harder than the fix.

Run: uv run python tests/test_brand_repair.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.pronunciation import normalize_transcript as fix

# (input, must contain Magppie?, note)
MUST_REPAIR = [
    ("Tell me about MacPay Kitchens", "the exact failure from the call"),
    ("I only said Mac by Kitchens", "the follow-up, also refused"),
    ("What is Mac Pay?", "spaced"),
    ("Tell me about magpie kitchens", "the bird spelling"),
    ("mag pie wellness kitchens", "spaced bird"),
    ("MACPAY ka price kya hai?", "upper case, hinglish"),
    ("Mac-Pay kitchens", "hyphenated"),
    ("What is MagPy?", "the exact failure from the second call"),
    ("Mag Py kitchens", "spaced Py"),
    ("मैगपाई क्या है?", "devanagari"),
    ("मैग पाई किचन", "spaced devanagari"),
]

# These must come through UNTOUCHED — each has a legitimate meaning.
MUST_NOT_TOUCH = [
    ("मैट फिनिश क्या है?", "मैट = matte, a real finish (K-27)"),
    ("super matt finish", "matte in english"),
    ("Do you sell MacBooks?", "genuinely off-topic; bot must decline, not answer"),
    ("What is macOS?", "off-topic"),
    ("I use a Mac at home", "bare 'Mac' is not the brand"),
    ("Magppie kitchens", "already correct"),
    ("machine ka price", "'mac' inside a word"),
    ("Kitchen ka price kya hai?", "ordinary question"),
    ("mag", "bare fragment"),
]


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


ok = True
print("must repair:")
for text, note in MUST_REPAIR:
    out = fix(text)
    ok &= check(f"{text!r} -> {out!r}   ({note})", "Magppie" in out)

print("\nmust NOT touch (precision — these would corrupt real questions):")
for text, note in MUST_NOT_TOUCH:
    out = fix(text)
    ok &= check(f"{text!r} unchanged   ({note})", out == text)

print("\nedges:")
ok &= check("empty string", fix("") == "")
ok &= check("only rewrites the name, keeps the sentence",
            fix("Tell me about MacPay Kitchens") == "Tell me about Magppie Kitchens")
ok &= check("repairs several in one sentence",
            fix("MacPay or Mac by?").count("Magppie") == 2)

print("\nOK" if ok else "\nFAILED")
sys.exit(0 if ok else 1)
