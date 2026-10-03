"""The end_call guard: a hang-up is only honoured when the CALLER signalled it.

gpt-4.1-nano sometimes flags end_call while merely declining an off-topic
question ("what is a MacBook?"), which would wrongly drop the call. The guard
corroborates the flag against the caller's own words. It must catch real
goodbyes (so legit hang-ups still work) and reject everything else (so a model
misfire can't end the call). Offline — no API.

Run: uv run python tests/test_exit_guard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.llm import _looks_like_exit as exit_intent

# Real exit intent — the guard must let these hang up.
EXITS = [
    "bye bhai",
    "ok bye, rakhta hoon",
    "nahi karni baat, band karo",
    "mujhe baat nahi karni",
    "not interested",
    "please stop calling",
    "remove my number",
    "call kaat do",
    "अलविदा",
    "नहीं करनी बात",
    "बंद करो",
    "बस करो भाई",
    # the 2026-08-31 real call: guard blocked this and the caller had to cut
    # the call himself after a confused "Hello"
    "नहीं, मैं और कुछ जानना नहीं चाहूंगा। थैंक यू।",
    "और कुछ नहीं चाहिए",
    "aur kuch nahi janna mujhe",
    "बस इतना ही था मेरा",
    "बस हो गया भाई",
    "बहुत हो गया।",          # "enough already" — 2026-09-02 caller, guard blocked it
    "nothing else, thank you",
    "no more questions",
    "I'm done, thanks",
    "that'll be all",
]

# NOT exit intent — a model that flags end_call here is misfiring; guard blocks it.
NOT_EXITS = [
    "What is a MacBook?",
    "Do you sell iPhones?",
    "What is macOS?",
    "Guarantee kitne saal hai?",
    "Mera budget 5 lakh hai",
    "kitchen ka price batao",
    "haan theek hai",          # agreement, not goodbye
    "ok",                       # filler
    "matte finish dikhao",      # 'mat' inside a word must not match "mat call"
    "band gala suit jaisa design",  # 'band' inside a phrase, not "band karo"
    # declining ONE topic mid-call is not "I'm done" — from the same real call
    "तनुष के बारे में नहीं जानना चाहूंगा, आप मुझे ये बताएं कि गारंटी क्या है",
    "बस इतना बताओ कि प्राइस क्या है",   # "just tell me this" is a question
    "और कुछ ऑप्शंस बताओ",              # asking for MORE, not ending
]


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


ok = True
print("real exits (must hang up):")
for t in EXITS:
    ok &= check(f"{t!r}", exit_intent(t))

print("\nnot exits (must NOT hang up):")
for t in NOT_EXITS:
    ok &= check(f"{t!r}", not exit_intent(t))

ok &= check("empty string is not an exit", not exit_intent(""))

print("\nOK" if ok else "\nFAILED")
sys.exit(0 if ok else 1)
