"""Drive /api/turn end-to-end with the providers stubbed out.

No Sarvam or OpenAI calls — this exercises the event contract, the ordering,
and the LLM/TTS overlap for free. Timings here are fake by construction; only
the sequencing is under test.

Run: uv run python tests/test_turn_stream.py
"""

import asyncio
import io
import json
import math
import struct
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app import kb as kb_mod
from app import llm, main, stt, tts
from app.config import settings


def tone(secs=0.4, rate=22050):
    n = int(rate * secs)
    d = b"".join(struct.pack("<h", int(6000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(n))
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(d)
    return b.getvalue()


ANSWER = "SilverStone is our patented wellness stone made from porcelain clay. It is antibacterial and food grade."
CITES = [{"source": "magppie-voicebot-kb.md", "page": None}]

tts_calls: list[str] = []


def install_stubs(*, stt_text=ANSWER, fail_tts=False, fail_stream_only=False, streaming=False):
    tts_calls.clear()
    settings.sarvam_api_key = "stub"
    settings.openai_api_key = "stub"
    settings.tts_streaming = streaming
    main.KB = kb_mod.KB(text="stub kb", files=[kb_mod.KBFile("kb.md", 999)], tokens=5)
    main.SESSIONS.clear()
    main.SESSION_LANG.clear()

    async def fake_transcribe(wav):
        await asyncio.sleep(0.01)
        return stt.Transcript(text="What is SilverStone made of?", language_code="en-IN")

    async def fake_stream(kb, q, hist, lang, into=None):
        # Deltas arrive mid-sentence, like the real stream.
        for i in range(0, len(ANSWER), 12):
            await asyncio.sleep(0.005)
            yield ANSWER[i : i + 12]
        if into is not None:
            into.text = ANSWER
            into.citations = [llm.Citation(source="magppie-voicebot-kb.md", page=None)]

    async def fake_synth(text, lang):
        tts_calls.append(text)
        await asyncio.sleep(0.05)
        if fail_tts:
            raise tts.TTSError("bulbul exploded")
        return tone()

    async def fake_synth_stream(text_source, lang):
        # Drain the LLM stream (fills GroundedAnswer), then yield PCM chunks.
        if isinstance(text_source, str):
            text = text_source
        else:
            text = "".join([p async for p in text_source if p])
        tts_calls.append(text)
        await asyncio.sleep(0.05)
        if fail_tts or fail_stream_only:
            # fail_stream_only mimics a real WebSocket blip: the socket dies but
            # the LLM already drained, so the REST fallback can still speak it.
            raise tts.TTSError("socket dropped mid-stream")
        # a few small raw-PCM chunks (silence is fine for the contract test)
        for _ in range(3):
            yield b"\x00\x00" * 512

    stt.transcribe = fake_transcribe
    llm.stream_answer = fake_stream
    tts.synthesize = fake_synth
    tts.synthesize_stream = fake_synth_stream


async def post_turn():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post(
            "/api/turn",
            files={"audio": ("turn.wav", tone(1.0), "audio/wav")},
            data={"session_id": "test"},
        )
        events = [json.loads(line) for line in r.text.strip().splitlines() if line.strip()]
        return r, events


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


ok = True

# --- happy path ---
install_stubs()
r, events = asyncio.run(post_turn())
kinds = [e["type"] for e in events]
print("event sequence:", " -> ".join(kinds))

ok &= check("HTTP 200 + ndjson", r.status_code == 200 and "ndjson" in r.headers["content-type"])
ok &= check("transcript arrives first", kinds[0] == "transcript")
ok &= check("done arrives last", kinds[-1] == "done")
ok &= check("audio streamed in 2 chunks", kinds.count("audio") == 2)
ok &= check("audio indexes ordered 0,1", [e["index"] for e in events if e["type"] == "audio"] == [0, 1])
ok &= check("no error event", "error" not in kinds)

done = events[-1]
ok &= check("answer returned in full", done["answer"] == ANSWER)
ok &= check("citations returned, page null", done["citations"] == CITES)
ok &= check("first_audio_ms reported", "first_audio_ms" in done["timings"])
ok &= check("tts_chunks == 2", done["timings"]["tts_chunks"] == 2)
ok &= check(
    "first_audio_ms <= total_ms",
    done["timings"]["first_audio_ms"] <= done["timings"]["total_ms"],
)

# The whole point: sentence 1 is synthesized separately and first.
ok &= check("TTS called twice (split, not one blob)", len(tts_calls) == 2)
ok &= check(
    "chunk 1 is the first sentence only",
    tts_calls[0] == "SilverStone is our patented wellness stone made from porcelain clay.",
)
ok &= check("chunk 2 is the remainder", tts_calls[1] == "It is antibacterial and food grade.")
ok &= check("chunks reassemble to the full answer", " ".join(tts_calls) == ANSWER)

# --- audio actually concatenates into one playable stream ---
import base64

blobs = [base64.b64decode(e["audio"]) for e in events if e["type"] == "audio"]
joined = tts.concat_wavs(blobs)
with wave.open(io.BytesIO(joined)) as w:
    ok &= check("chunks concat to ~0.8s of playable audio", 0.7 < w.getnframes() / w.getframerate() < 0.9)

# --- mid-stream failure must surface as an error event, not a hang ---
print("\nfailure path:")
install_stubs(fail_tts=True)
r, events = asyncio.run(post_turn())
kinds = [e["type"] for e in events]
ok &= check("still HTTP 200 (headers already sent)", r.status_code == 200)
ok &= check("error event emitted", kinds[-1] == "error")
ok &= check("error text is readable", "bulbul exploded" in events[-1]["error"])

# --- streaming path emits raw PCM chunks ---
print("\nstreaming path:")
install_stubs(streaming=True)
r, events = asyncio.run(post_turn())
kinds = [e["type"] for e in events]
ok &= check("streaming: transcript first, done last", kinds[0] == "transcript" and kinds[-1] == "done")
ok &= check("streaming: emits audio_pcm chunks", kinds.count("audio_pcm") >= 1)
ok &= check(
    "streaming: every pcm chunk carries a sample rate",
    all("rate" in e for e in events if e["type"] == "audio_pcm"),
)
ok &= check("streaming: answer still returned in full", events[-1]["answer"] == ANSWER)
ok &= check("streaming: no wav 'audio' events on this path", "audio" not in kinds)

# --- streaming TTS blips -> must recover via REST, not strand the turn ("…") ---
print("\nstreaming failover (the '…' bug):")
install_stubs(streaming=True, fail_stream_only=True)
r, events = asyncio.run(post_turn())
kinds = [e["type"] for e in events]
ok &= check("failover: no error event (turn survives the socket blip)", "error" not in kinds)
ok &= check("failover: recovers with a REST wav audio event", "audio" in kinds)
ok &= check("failover: done still arrives last", kinds[-1] == "done")
ok &= check("failover: full answer preserved", events[-1]["answer"] == ANSWER)

# --- pre-flight errors stay real HTTP errors ---
print("\npre-flight:")
install_stubs()
settings.sarvam_api_key = ""
transport = httpx.ASGITransport(app=main.app)


async def missing_key():
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        return await c.post("/api/turn", files={"audio": ("t.wav", tone(1.0), "audio/wav")})


r = asyncio.run(missing_key())
ok &= check("missing key -> HTTP 503 JSON, not a stream", r.status_code == 503 and "SARVAM_API_KEY" in r.json()["error"])

print("\nOK" if ok else "\nFAILED")
sys.exit(0 if ok else 1)
