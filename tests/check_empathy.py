"""Reproduce the upset-caller exchange from the real call and check Shubh now
leads with empathy ("what happened?") instead of a cold kitchen-deflection, and
doesn't repeat a near-identical line. Hits live OpenAI.

Run: uv run python tests/check_empathy.py
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

# A genuine inquiry into what went wrong — the empathy we now want up front.
# Deliberately does NOT include a bare "माफ", which also appears in the case-5
# brush-off ("माफ कीजिए, वो मेरा काम नहीं") we are trying to catch and reject.
EMPATHY = re.compile(
    r"(क्या हुआ|क्या दिक्कत|क्या परेशान|कौन सी (दिक्कत|परेशान)|दिक्कत हो गई|"
    r"परेशानी हुई|क्या गड़बड़|what happened|what'?s wrong|what went wrong|"
    r"what.{0,8}problem|tell me what)",
    re.I,
)
# The case-5 brush-off — treating a complaint as "not my job". Must be ABSENT.
DEFLECT = re.compile(r"(मेरा काम नहीं|किचन वाला बंदा|वो तो मेरा|not my (job|area))", re.I)
# A premature farewell used to calm someone who is still on the line.
GOODBYE = re.compile(r"(दिन शुभ|दिन अच्छा|have a (nice|good) day)", re.I)


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def near_twin(a, b):
    ta, tb = set(norm(a).split()), set(norm(b).split())
    return len(ta & tb) / max(1, len(ta | tb)) > 0.6


async def main():
    ok = True
    h = []

    # Turn 1 — the caller vents / runs the company down (STT's "MacPay" = Magppie).
    q1 = "भाई तुम तो बड़े पागल हो यार, मैक पायर टट्टी है भाई, पता है मैक पायर के एम्प्लॉईस कैसे हैं?"
    r1 = await llm.answer(KB.text, q1, h, "hi-IN")
    print("[1] upset / complaint")
    print("   ", r1.text)
    empathetic = bool(EMPATHY.search(r1.text))
    not_deflected = not DEFLECT.search(r1.text)
    no_bye = not GOODBYE.search(r1.text)
    print("    -> asks what's wrong / shows concern:", "PASS" if empathetic else "FAIL")
    print("    -> does NOT brush off as 'not my job':", "PASS" if not_deflected else "FAIL")
    print("    -> no premature goodbye:", "PASS" if no_bye else "FAIL")
    ok &= empathetic and not_deflected and no_bye
    h += [{"role": "user", "content": q1}, {"role": "assistant", "content": r1.text}]

    # Turn 2 — still dismissive/frustrated; must NOT be a near-twin of turn 1,
    # and must not send them off with a farewell (they haven't asked to leave).
    q2 = "अबे यार तू ना भाई, छोड़ भाई, तेरे बस की नहीं है भाई।"
    r2 = await llm.answer(KB.text, q2, h, "hi-IN")
    print("\n[2] keeps venting")
    print("   ", r2.text)
    fresh = not near_twin(r1.text, r2.text)
    # A farewell is only OK here if it actually ENDS the call (end_call). The bug
    # would be a farewell mid-call (end_call False) — the jarring contradiction.
    coherent = (not GOODBYE.search(r2.text)) or r2.end_call
    print("    -> not a near-twin of the first reply:", "PASS" if fresh else "FAIL")
    print("    -> no farewell-without-ending (coherent):", "PASS" if coherent else "FAIL")
    ok &= fresh and coherent

    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return 0 if ok else 1


sys.exit(asyncio.run(main()))
