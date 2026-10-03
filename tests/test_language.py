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
main.SESSION_LANG_PENDING.clear()
main.resolve_language("s2", "What is Magppie?", "en-IN")
ok &= check(
    "'अच्छा अच्छा अच्छा' mis-tagged bn -> NEVER Bengali (stays en, hi now pending)",
    main.resolve_language("s2", "अच्छा अच्छा अच्छा", "bn-IN") == "en-IN",
)
ok &= check(
    "second consecutive Devanagari turn completes the switch to hi",
    main.resolve_language("s2", "अच्छा ये बताओ दाम कितना पड़ेगा", "hi-IN") == "hi-IN",
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

# --- switching AWAY to English needs two turns (2026-08-31 caller complaint) ----
print("\nEnglish-switch hysteresis:")
main.SESSION_LANG.clear()
main.SESSION_LANG_PENDING.clear()
main.resolve_language("s5", "मुझे किचन लगवानी है अपने घर में", "hi-IN")
ok &= check(
    "one English question does NOT flip a Hindi call",
    main.resolve_language("s5", "What about your services?", "en-IN") == "hi-IN",
)
ok &= check(
    "the second consecutive English turn DOES switch",
    main.resolve_language("s5", "And how much does it cost in total?", "en-IN") == "en-IN",
)
ok &= check(
    "one Devanagari turn does NOT flip the now-English call (symmetric rule)",
    main.resolve_language("s5", "अरे भाई मुझे पूरा हिसाब बता दो फिर से", "hi-IN") == "en-IN",
)
ok &= check(
    "…but the second consecutive Hindi turn does",
    main.resolve_language("s5", "हिसाब पूरा चाहिए मुझे भाई सुनो", "hi-IN") == "hi-IN",
)

# --- 2026-09-02 regression: garble imprisonment ---------------------------------
print("\ngarble must not flip, fragments must not block the escape:")
main.SESSION_LANG.clear()
main.SESSION_LANG_PENDING.clear()
main.resolve_language("s7", "Hi, what is your name?", "en-IN")
ok &= check(
    "Marathi/Devanagari GARBLE ('हे पूर्ण आहे ते') cannot flip an English call",
    main.resolve_language("s7", "हे पूर्ण आहे ते.", "mr-IN") == "en-IN",
)
ok &= check(
    "the next real English turn puts the call firmly back",
    main.resolve_language("s7", "So I wanted to understand what all do you have.", "en-IN") == "en-IN",
)

# --- 2026-09-02 evening regression: the FIRST turn must not pin off garble -----
print("\nfirst-turn establishment needs a real sentence:")
main.SESSION_LANG.clear()
main.SESSION_LANG_PENDING.clear()
ok &= check(
    "one-word Devanagari garble ('अपने।') as turn 1 -> English reply, nothing pinned",
    main.resolve_language("s9", "अपने।", "hi-IN") == "en-IN",
)
ok &= check(
    "…session is still unpinned: a real Hindi sentence next establishes hi at once",
    main.resolve_language("s9", "मुझे किचन के बारे में जानना है भाई", "hi-IN") == "hi-IN",
)
main.SESSION_LANG.clear()
main.SESSION_LANG_PENDING.clear()
main.resolve_language("s10", "हेलो?", "hi-IN")  # garbled "Hello" in Devanagari
ok &= check(
    "…and a real English sentence next establishes English at once",
    main.resolve_language("s10", "I want to buy a kitchen for my home", "en-IN") == "en-IN",
)

main.SESSION_LANG.clear()
main.SESSION_LANG_PENDING.clear()
main.resolve_language("s8", "मुझे किचन लगवानी है अपने घर में", "hi-IN")  # wrongly-Hindi call
main.resolve_language("s8", "Can you tell me about your kitchens please?", "en-IN")  # pending en
main.resolve_language("s8", "Okay", "en-IN")  # short fragment — must NOT reset the streak
ok &= check(
    "a short 'Okay' between two English turns does not reset the escape",
    main.resolve_language("s8", "What is the price of Wellness Pro?", "en-IN") == "en-IN",
)

main.SESSION_LANG.clear()
main.SESSION_LANG_PENDING.clear()
main.resolve_language("s6", "मुझे किचन लगवानी है अपने घर में", "hi-IN")
main.resolve_language("s6", "What about your services?", "en-IN")
main.resolve_language("s6", "अच्छा ठीक है, और गारंटी कितनी है?", "hi-IN")
ok &= check(
    "a Hindi turn between two English ones breaks the streak",
    main.resolve_language("s6", "What is the starting price?", "en-IN") == "hi-IN",
)

# --- 2026-09-03: the danda "।" is not Devanagari ---------------------------------
# An Odia fragment (background talk over the greeting) pinned an English
# caller's call to Hindi through its sentence-final danda.
print("\nthe danda alone is not a Hindi signal:")
main.SESSION_LANG.clear()
main.SESSION_LANG_PENDING.clear()
ok &= check(
    "Odia garble 'ହଁ, ଦେଖି ସାରିଲି।' as turn 1 -> English reply",
    main.resolve_language("s11", "ହଁ, ଦେଖି ସାରିଲି।", "od-IN") == "en-IN",
)
ok &= check("…and nothing was pinned", main.SESSION_LANG.get("s11") is None)
ok &= check(
    "the caller's real English sentence then pins English",
    main.resolve_language("s11", "I want to inquire about kitchens", "en-IN") == "en-IN"
    and main.SESSION_LANG.get("s11") == "en-IN",
)
main.SESSION_LANG.clear()
main.SESSION_LANG_PENDING.clear()
ok &= check(
    "real Devanagari with a danda still pins Hindi",
    main.resolve_language("s12", "मुझे किचन के बारे में जानना है।", "hi-IN") == "hi-IN",
)

# --- empty / meaningless answer guard (bug 3 — the "…" turn) -------------------
print("\nempty-answer guard:")
for a in ["…", "...", " - ", "", "।", "॥", "—", ".,!?"]:
    ok &= check(f"{a!r} is empty", main._is_empty_answer(a))
for a in ["भाई, बढ़िया हूँ", "3 BHK", "Magppie makes stone kitchens.", "जी, बताइए"]:
    ok &= check(f"{a!r} is a real answer", not main._is_empty_answer(a))

print("\nOK" if ok else "\nFAILED")
sys.exit(0 if ok else 1)
