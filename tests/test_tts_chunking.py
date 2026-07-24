"""TTS chunking: start speaking sentence one BEFORE the answer is finished.

This is the time-to-first-audio win. The old behaviour drained the whole LLM
stream, then synthesized — serial. Now the first speakable sentence goes to
Bulbul while the model is still writing the rest, which takes ~1s off every turn.

The test stubs the socket layer (_speak) and asserts both the split and, more
importantly, that synthesis STARTS while the LLM stream still has text pending.

Run: uv run python tests/test_tts_chunking.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import tts

SENT1 = "SilverStone is our patented wellness stone made from porcelain clay."
SENT2 = "It is antibacterial and food grade."
ANSWER = f"{SENT1} {SENT2}"

spoken: list[str] = []          # text handed to the synthesizer, in order
emitted_at_speak: list[int] = []  # how much of the answer the LLM had emitted by then
emitted = 0


def install():
    spoken.clear()
    emitted_at_speak.clear()

    async def fake_speak(text, lang):
        # Mirror the real _speak: empty text synthesizes nothing.
        text = (text or "").strip()
        if not text:
            return
        spoken.append(text)
        emitted_at_speak.append(emitted)
        yield b"\x00\x00" * 8

    tts._speak = fake_speak


async def llm_deltas(text=ANSWER, step=10):
    """Emit the answer gradually, like a real model."""
    global emitted
    emitted = 0
    for i in range(0, len(text), step):
        await asyncio.sleep(0.002)
        emitted = min(i + step, len(text))
        yield text[i : i + step]


async def drive(source):
    chunks = []
    async for pcm in tts.synthesize_stream(source, "en-IN"):
        chunks.append(pcm)
    return chunks


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


ok = True

# --- two sentences: split, and speak #1 early --------------------------------
print("two-sentence answer:")
install()
chunks = asyncio.run(drive(llm_deltas()))
ok &= check("synthesized in 2 pieces", len(spoken) == 2)
ok &= check("piece 1 is the first sentence", spoken[0] == SENT1)
ok &= check("piece 2 is the remainder", spoken[1] == SENT2)
ok &= check("pieces reassemble to the full answer", " ".join(spoken) == ANSWER)
ok &= check("audio was produced", len(chunks) >= 2)
# The whole point: we started speaking before the model finished writing.
ok &= check(
    f"speaking STARTED before the answer finished ({emitted_at_speak[0]} of {len(ANSWER)} chars emitted)",
    emitted_at_speak[0] < len(ANSWER),
)

# --- short one-liner with no boundary: single piece, still spoken ------------
print("\nshort answer with no sentence boundary:")
install()
asyncio.run(drive(llm_deltas("Sure, one moment", step=4)))
ok &= check("spoken as a single piece", len(spoken) == 1)
ok &= check("nothing dropped", spoken[0] == "Sure, one moment")

# --- plain string (greeting / fallback path) --------------------------------
print("\nplain string source:")
install()
asyncio.run(drive("Hi, this is Magppie Wellness Kitchens' support."))
ok &= check("spoken once, verbatim", spoken == ["Hi, this is Magppie Wellness Kitchens' support."])

# --- empty source must not explode ------------------------------------------
print("\nempty source:")
install()
asyncio.run(drive(""))
ok &= check("nothing spoken, no crash", spoken == [])

print("\nOK" if ok else "\nFAILED")
sys.exit(0 if ok else 1)
