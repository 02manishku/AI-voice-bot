"""Language resolution + the empty-answer guard, straight from the call logs.

Two real failures drove this:
  - "Tell me that in Hindi" (English words) was tagged en-IN by Saaras, so the
    switch request was invisible and Shubh kept refusing Hindi.
  - "अच्छा अच्छा अच्छा" was tagged bn-IN, so Shubh replied in Bengali.
  - A dead-end abusive turn came back as a bare "…", which TTS spoke as silence.

Run: uv run python tests/test_language.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import main


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


ok = True

# --- explicit "answer me in X", regardless of the language it's asked in -------
print("explicit switch requests (bug 1 — the Hindi refusal):")
ok &= check("'in Hindi or Bhojpuri' -> hi", main._requested_language("Tell me that in Hindi or in Bhojpuri") == "hi-IN")
ok &= check("'भोजपुरी में बताओ' -> hi", main._requested_language("भोजपुरी में बताओ यही चीज़।") == "hi-IN")
ok &= check("'हिंदी में बोल' -> hi", main._requested_language("यार हिंदी में बोल भाई तू।") == "hi-IN")
ok &= check("'say it in English' -> en", main._requested_language("say it in English") == "en-IN")
ok &= check("'angrezi mein bolo' -> en", main._requested_language("bhai angrezi mein bolo") == "en-IN")
ok &= check("plain kitchen question -> None", main._requested_language("kitchen ka price batao") is None)
ok &= check("no false trigger on 'price' -> None", main._requested_language("guarantee kitni hai") is None)

# --- resolve_language: stickiness, Devanagari override, honoring a request ------
print("\nresolve_language (sticky + script override):")
main.SESSION_LANG.clear()
ok &= check("turn1 English stays English", main.resolve_language("s1", "What is Magppie?", "en-IN") == "en-IN")
ok &= check(
    "an explicit Hindi request overrides an English session",
    main.resolve_language("s1", "Tell me that in Hindi or in Bhojpuri", "en-IN") == "hi-IN",
)

main.SESSION_LANG.clear()
main.resolve_language("s2", "What is Magppie?", "en-IN")
ok &= check(
    "'अच्छा अच्छा अच्छा' mis-tagged bn -> Devanagari forces hi (not Bengali)",
    main.resolve_language("s2", "अच्छा अच्छा अच्छा", "bn-IN") == "hi-IN",
)

main.SESSION_LANG.clear()
main.resolve_language("s3", "namaste bhai kaise ho", "hi-IN")
ok &= check(
    "short 'हाँ' mis-tagged bn -> Devanagari still hi",
    main.resolve_language("s3", "हाँ", "bn-IN") == "hi-IN",
)

main.SESSION_LANG.clear()
main.resolve_language("s4", "What is Magppie?", "en-IN")
ok &= check(
    "a one-word 'okay' (en) does not flip an English session",
    main.resolve_language("s4", "okay", "en-IN") == "en-IN",
)

# --- empty / meaningless answer guard (bug 3 — the "…" turn) -------------------
print("\nempty-answer guard:")
for a in ["…", "...", " - ", "", "।", "॥", "—", ".,!?"]:
    ok &= check(f"{a!r} is empty", main._is_empty_answer(a))
for a in ["भाई, बढ़िया हूँ", "3 BHK", "Magppie makes stone kitchens.", "जी, बताइए"]:
    ok &= check(f"{a!r} is a real answer", not main._is_empty_answer(a))

print("\nOK" if ok else "\nFAILED")
sys.exit(0 if ok else 1)
