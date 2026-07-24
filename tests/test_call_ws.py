"""Drive the Phase 3 streaming call (/ws/call) with Sarvam + OpenAI stubbed out.

No network: the VAD socket, the REST STT, the LLM and the TTS stream are all
faked, so this exercises the WebSocket contract and the turn flow. The stream is
used only for VAD start/end; the utterance audio is transcribed over REST STT
(that's the reliable path — Sarvam's streaming transcript is not). Real VAD +
barge-in behaviour is validated separately against the live API (scratchpad
probe); here we guard the message contract and the main.* integration.

Run: uv run python tests/test_call_ws.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from starlette.testclient import TestClient

from app import kb as kb_mod
from app import call_ws, llm, main, stt, stt_stream, tts
from app.config import settings

ANSWER = "SilverStone is our patented wellness stone made from porcelain clay. It is food grade."
# Big enough to clear MIN_TURN_BYTES so the buffered "utterance" is transcribed.
UTTERANCE_PCM = b"\x01\x00" * 20000  # 40000 bytes ~= 1.25s of 16k PCM
rest_calls: list[int] = []  # records REST STT fallback invocations


def install_stubs(events, *, stt_text="What is SilverStone made of?", rest_should_run=None):
    settings.sarvam_api_key = "stub"
    settings.openai_api_key = "stub"
    settings.stt_streaming = True
    main.KB = kb_mod.KB(text="stub kb", files=[kb_mod.KBFile("kb.md", 999)], tokens=5)
    main.SESSIONS.clear()
    main.SESSION_LANG.clear()

    class FakeSession:
        async def feed(self, pcm):
            pass

        async def events(self):
            for ev in events:
                await asyncio.sleep(0.08)  # let the mic buffer fill before events fire
                yield ev
            await asyncio.sleep(10)  # keep the "socket" open until the client hangs up

    class FakeConnect:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, *a):
            return False

    stt_stream.connect = lambda language_code="unknown": FakeConnect()

    # REST STT — the FALLBACK, only used when no streaming transcript arrives.
    rest_calls.clear()

    async def fake_rest_transcribe(wav):
        rest_calls.append(len(wav))
        await asyncio.sleep(0.01)
        return stt.Transcript(text=stt_text, language_code="en-IN")

    stt.transcribe = fake_rest_transcribe

    async def fake_stream(kb, q, hist, lang, into=None, context=None):
        for i in range(0, len(ANSWER), 12):
            await asyncio.sleep(0.003)
            yield ANSWER[i : i + 12]
        if into is not None:
            into.text = ANSWER
            into.citations = [llm.Citation(source="kb.md", page=None)]

    llm.stream_answer = fake_stream

    async def fake_synth_stream(source, lang):
        if isinstance(source, str):
            text = source
        else:
            text = "".join([p async for p in source if p])
        for _ in range(3):
            yield b"\x00\x00" * 256

    tts.synthesize_stream = fake_synth_stream

    async def fake_synth(text, lang):
        return b"\x00\x00" * 256

    tts.synthesize = fake_synth


def collect_until(ws, kind, cap=60):
    out = []
    for _ in range(cap):
        m = ws.receive_json()
        out.append(m)
        if m["type"] == kind:
            break
    return out


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


ok = True

# --- FAST PATH: streaming transcript arrives -> no REST round-trip -----------
print("fast path (streaming transcript arrives right after end-of-turn):")
install_stubs(
    [
        stt_stream.SpeechStarted(),
        stt_stream.SpeechEnded(),
        stt_stream.Transcript(text="What is SilverStone made of?", language_code="en-IN"),
    ]
)
client = TestClient(main.app)
with client.websocket_connect("/ws/call") as ws:
    ws.send_json({"type": "hello", "session_id": "t1"})
    ws.send_bytes(UTTERANCE_PCM)
    msgs = collect_until(ws, "done")
    kinds = [m["type"] for m in msgs]
    print("   sequence:", " -> ".join(kinds))
    done = msgs[-1]
    ok &= check("listening emitted on speech start", "listening" in kinds)
    ok &= check("transcript before audio", kinds.index("transcript") < kinds.index("audio_pcm"))
    ok &= check("audio streamed as pcm", kinds.count("audio_pcm") >= 1)
    ok &= check("done arrives last", kinds[-1] == "done")
    ok &= check("answer returned in full", done["answer"] == ANSWER)
    tr = msgs[[m["type"] for m in msgs].index("transcript")]
    ok &= check("language resolved to en-IN", tr["language_code"] == "en-IN")
    ok &= check("citations present", done["citations"] == [{"source": "kb.md", "page": None}])
    ok &= check("end_call false on a question", done["end_call"] is False)
    ok &= check("REST STT was NOT called (no extra round-trip)", rest_calls == [])
    ws.send_json({"type": "bye"})

# --- FALLBACK: no streaming transcript -> REST rescues the turn --------------
print("\nfallback (streaming transcript never arrives -> REST STT):")
install_stubs([stt_stream.SpeechStarted(), stt_stream.SpeechEnded()])
client = TestClient(main.app)
with client.websocket_connect("/ws/call") as ws:
    ws.send_json({"type": "hello", "session_id": "t2"})
    ws.send_bytes(UTTERANCE_PCM)
    msgs = collect_until(ws, "done", cap=80)
    kinds = [m["type"] for m in msgs]
    ok &= check("turn still answered (not stranded)", kinds[-1] == "done")
    ok &= check("REST STT was used as the fallback", len(rest_calls) == 1)
    ok &= check("answer returned in full", msgs[-1]["answer"] == ANSWER)
    ws.send_json({"type": "bye"})

# --- empty transcript (noise): no response, no crash -------------------------
print("\nempty transcript (noise) -> no turn:")
install_stubs([stt_stream.SpeechStarted(), stt_stream.SpeechEnded()], stt_text="")
client = TestClient(main.app)
with client.websocket_connect("/ws/call") as ws:
    ws.send_json({"type": "hello", "session_id": "t3"})
    ws.send_bytes(UTTERANCE_PCM)
    first = ws.receive_json()
    ok &= check("only a listening ping, no response", first["type"] == "listening")
    ws.send_json({"type": "bye"})

# --- barge-in: talking over Shubh AFTER audio was sent still interrupts -------
# is_speaking stays armed through playback (until playback_done), so a late
# SpeechStarted still cuts him — the "he keeps going till the end" fix.
print("\nbarge-in while the reply is still playing:")
install_stubs(
    [
        stt_stream.SpeechStarted(),
        stt_stream.SpeechEnded(),
        stt_stream.Transcript(text="What finishes do you have?", language_code="en-IN"),
        stt_stream.SpeechStarted(),  # talk over him — no playback_done sent, so still "speaking"
    ]
)
client = TestClient(main.app)
with client.websocket_connect("/ws/call") as ws:
    ws.send_json({"type": "hello", "session_id": "t4"})
    ws.send_bytes(UTTERANCE_PCM)
    msgs = collect_until(ws, "interrupted", cap=80)
    kinds = [m["type"] for m in msgs]
    ok &= check("a late barge-in emits 'interrupted' (armed through playback)", "interrupted" in kinds)
    ws.send_json({"type": "bye"})

print("\nOK" if ok else "\nFAILED")
sys.exit(0 if ok else 1)
