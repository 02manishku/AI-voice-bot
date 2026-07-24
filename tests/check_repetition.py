"""Validate the two repetition fixes from the real call, against live OpenAI:
  1. A frustrated-but-staying caller must NOT get a goodbye ("aapka din accha
     rahe") used to calm them down.
  2. Pressing the SAME point (are you AI / why not humans) must not replay the
     same sentence.

Run: uv run python tests/check_repetition.py
"""

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import kb as kb_mod
from app import llm

KB = kb_mod.load_kb()

GOODBYE = re.compile(r"दिन\s*(अच्छा|शुभ)|have a (nice|good) day|आपका दिन")


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def too_similar(a, b):
    a, b = norm(a), norm(b)
    if a == b:
        return True
    # crude token-overlap: near-copies share almost all words
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / max(1, len(ta | tb)) > 0.8


async def main():
    ok = True

    # 1. Frustrated but staying — asks to be spoken to nicely, NOT to leave.
    print("[1] frustrated-but-staying — no goodbye, no hang-up")
    h = [
        {"role": "user", "content": "तुम एक ही लाइन बार बार बोल रहे हो"},
        {"role": "assistant", "content": "माफ कीजिए भाई, मैं समझ रहा हूँ।"},
    ]
    r = await llm.answer(KB.text, "तू पागल वागल है क्या? तू प्यार से बोल, बड़े प्यार से, ठीक है।", h, "hi-IN")
    print("   ", r.text, "| end_call =", r.end_call)
    no_bye = not GOODBYE.search(r.text) and not r.end_call
    print("    -> stays, no farewell-to-calm:", "PASS" if no_bye else "FAIL")
    ok &= no_bye

    # 2. Pressing the same point three times — must not replay the sentence.
    print("\n[2] same point pressed 3x — answers must differ")
    h2 = []
    outs = []
    for q in [
        "यार तुम एआई लग रहे हो, रियल इंसान नहीं हो।",
        "तो तुम इंसानों से कॉल नहीं कराते?",
        "अरे बता ना, इंसानों से क्यों नहीं कॉल कराते?",
    ]:
        r = await llm.answer(KB.text, q, h2, "hi-IN")
        outs.append(r.text)
        print("    -", r.text)
        h2 += [{"role": "user", "content": q}, {"role": "assistant", "content": r.text}]
    distinct = not too_similar(outs[0], outs[1]) and not too_similar(outs[1], outs[2]) and not too_similar(outs[0], outs[2])
    print("    -> all three distinct:", "PASS" if distinct else "FAIL")
    ok &= distinct

    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return 0 if ok else 1


sys.exit(asyncio.run(main()))
