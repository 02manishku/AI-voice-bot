"""Reproduce the '…' turns. They all share one trait in the transcripts: the
caller talks about being stressed / family, not kitchens. Find out what the
model actually returns on those turns — empty? an exception? a refusal? — and
whether it varies its identity/goodbye lines across a run. Hits real OpenAI.

Run: uv run python tests/repro_dots.py
"""

import asyncio
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import kb as kb_mod
from app import llm

KB = kb_mod.load_kb()


async def one(label, history, question, lang):
    try:
        r = await llm.answer(KB.text, question, history, lang)
        empty = not (r.text or "").strip() or r.text.strip() in {"…", "..."}
        flag = "  <-- EMPTY/DOTS" if empty else ""
        print(f"[{label}] end_call={r.end_call}{flag}\n    Q: {question}\n    A: {r.text!r}\n")
        return r
    except Exception:
        print(f"[{label}] RAISED — this is what shows as '…' in the UI\n    Q: {question}")
        traceback.print_exc()
        print()
        return None


async def main():
    # The exact 'stressed with my family' turns that produced '…'.
    print("=== stress / family turns (the '…' ones) ===")
    await one("stress-1", [], "I am stressed with my family only bro", "en-IN")
    await one("stress-2", [], "I am stressed with my family only bro", "en-IN")
    await one(
        "stress-3",
        [{"role": "user", "content": "I'm stressed"},
         {"role": "assistant", "content": "A kitchen is a space for health and happiness, sir."}],
        "I'm stressed with my family. They are wanting Magppie kitchens, but your Magppie kitchen is so expensive, bro.",
        "en-IN",
    )
    await one("stress-4", [], "I am stressed with my family, they are wanting Magppie kitchen but your Magppie kitchen is so expensive bro, help me.", "en-IN")

    # The repetition the caller complained about: identity line 3x.
    print("=== repeated identity question (should vary each time) ===")
    h = []
    for i, q in enumerate([
        "यार तुम एआई लग रहे हो यार, रियल इंसान नहीं हो।",
        "तो तुमने एआई से जो कॉल कराई मेरे को, तुम इंसानों से कॉल नहीं कराते?",
        "अरे मेरे को यह बताना तुम इंसानों से क्यों नहीं कॉल कराते?",
    ]):
        r = await one(f"identity-{i+1}", h, q, "hi-IN")
        if r:
            h += [{"role": "user", "content": q}, {"role": "assistant", "content": r.text}]


asyncio.run(main())
