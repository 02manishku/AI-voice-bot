"""Phase 3: the whole call over one WebSocket.

The browser opens ONE socket for the call and streams mic PCM continuously.
Sarvam's server-side VAD marks turn boundaries (END_SPEECH), so we never guess
the end of a turn from volume.

Transcription is a race with a safety net. The audio is already streaming, so
Sarvam's own streaming transcript normally lands ~300ms after end-of-turn — that
is the fast path, and it costs no extra round-trip. But those streaming
transcripts can lag many seconds or never arrive, which would strand the turn, so
we also buffer the utterance's raw audio: if the streaming transcript hasn't
shown up within STREAM_TRANSCRIPT_WAIT we transcribe the buffer over the reliable
REST STT instead. Fast when Sarvam is healthy, never stuck when it isn't.

On each turn we run the same LLM + TTS pipeline as POST /api/turn and stream the
audio back — cancellable, so START_SPEECH while Shubh is talking is a barge-in:
we stop his audio mid-sentence and listen, like a real phone call.

Protocol
  browser -> server : binary frames = raw Int16LE mono PCM @16k
                      text JSON      = {"type":"hello","session_id":...}
                                     | {"type":"playback_done"} | {"type":"bye"}
  server  -> browser: text JSON      = transcript | audio_pcm | audio | done
                                       | interrupted | listening | error

This path is gated by settings.stt_streaming; POST /api/turn stays the default.
"""

import asyncio
import base64
import io
import json
import logging
import time
import uuid
import wave

from fastapi import WebSocket, WebSocketDisconnect
from sarvamai.core.api_error import ApiError

from app import llm, prompts, stt, stt_stream, tts
from app.config import settings

log = logging.getLogger("magppie.ws")

MAX_HISTORY_MESSAGES = settings.max_history_messages  # call's working memory (see config)

# How long to wait for Sarvam's streaming transcript before falling back to REST.
# Measured healthy: ~320ms after end-of-turn. Measured degraded: 1.2s-2.7s, or never.
#
# Why 0.8s: if the stream beats the window we pay only T_stream (~0.32s); if it
# misses we pay WAIT + REST (~1s). So the window wants to be comfortably above the
# healthy figure but no higher, because every extra 100ms is added to *every*
# degraded turn. 0.8s is ~2.5x the healthy latency, and keeps the degraded case
# near ~1.8s instead of ~2.2s.
STREAM_TRANSCRIPT_WAIT = 0.8

# The caller's raw 16k PCM is buffered per turn purely as the REST fallback's input.
_BYTES_PER_SEC = stt_stream.STREAM_SAMPLE_RATE * 2  # 16-bit mono
PREROLL_BYTES = int(0.5 * _BYTES_PER_SEC)    # pre-speech kept when a turn starts
MIN_TURN_BYTES = int(0.25 * _BYTES_PER_SEC)  # ignore sub-250ms VAD blips
MAX_BUFFER_BYTES = 30 * _BYTES_PER_SEC       # runaway guard


def _pcm_to_wav(pcm: bytes, rate: int = stt_stream.STREAM_SAMPLE_RATE) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container for the REST STT endpoint."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


async def handle(websocket: WebSocket) -> None:
    """Entry point registered as the /ws/call route in main."""
    from app import main  # deferred: main imports this module to register the route

    await websocket.accept()

    if missing := settings.missing_keys():
        await _send(websocket, type="error", error=f"Server is missing {', '.join(missing)}.")
        await websocket.close()
        return
    if main.KB.is_empty:
        await _send(websocket, type="error", error="The knowledge base is empty on the server.")
        await websocket.close()
        return

    await CallSession(websocket, main).run()


async def _send(ws: WebSocket, **payload) -> None:
    await ws.send_text(json.dumps(payload, ensure_ascii=False))


class CallSession:
    def __init__(self, ws: WebSocket, main) -> None:
        self.ws = ws
        self.main = main
        self.session_id = f"ws-{uuid.uuid4().hex[:12]}"
        self.history: list[dict] = []
        # LEAD CALL mode only: a pinned system note describing the CRM lead, so
        # Shubh knows who he called for the whole conversation. None in assistant
        # mode (the default) — behaviour is then exactly as before.
        self.lead_context: str | None = None

        self.stt: stt_stream.Session | None = None
        self.is_speaking = False              # caller is hearing (or about to hear) Shubh
        self.response_task: asyncio.Task | None = None
        # Resolved by the streaming transcript that follows end-of-turn (fast path).
        self.awaiting_final: asyncio.Future | None = None
        # Raw 16k PCM of the current utterance — input for the REST fallback.
        self.turn_buffer = bytearray()
        self.closed = False

    # -- lifecycle -----------------------------------------------------------

    async def run(self) -> None:
        try:
            async with stt_stream.connect() as stt_session:
                self.stt = stt_session
                mic = asyncio.create_task(self._pump_mic(), name="mic")
                brain = asyncio.create_task(self._pump_stt(), name="stt")
                try:
                    await asyncio.wait({mic, brain}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for t in (mic, brain):
                        t.cancel()
                    await self._cancel_response()
        except ApiError as exc:
            await self._safe_send(type="error", error=f"Sarvam streaming STT failed ({exc.status_code}).")
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("call session crashed")
            await self._safe_send(type="error", error="The call dropped unexpectedly.")
        finally:
            self.closed = True
            try:
                await self.ws.close()
            except Exception:
                pass

    # -- browser -> Sarvam ---------------------------------------------------

    async def _pump_mic(self) -> None:
        """Forward the browser's mic PCM into the Sarvam socket, and keep a copy
        of the current utterance for REST transcription."""
        while True:
            msg = await self.ws.receive()
            if msg["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()
            if (data := msg.get("bytes")) is not None:
                if self.stt is not None:
                    await self.stt.feed(data)      # -> Sarvam VAD (start/end/barge-in)
                self.turn_buffer.extend(data)      # -> our buffer for REST transcription
                if len(self.turn_buffer) > MAX_BUFFER_BYTES:
                    del self.turn_buffer[:-MAX_BUFFER_BYTES]
            elif (text := msg.get("text")) is not None:
                try:
                    payload = json.loads(text)
                except ValueError:
                    continue
                if payload.get("type") == "hello":
                    await self._on_hello(payload)
                elif payload.get("type") == "playback_done":
                    # The caller has now heard the whole reply — stop arming
                    # barge-in until Shubh speaks again.
                    self.is_speaking = False
                elif payload.get("type") == "bye":
                    raise WebSocketDisconnect()

    async def _on_hello(self, payload: dict) -> None:
        """Handshake from the browser: session id + call mode. In LEAD CALL mode we
        claim the lead the greeting stashed (same record, no second fetch) and pin
        it as context so Shubh remembers who he called for the whole conversation."""
        if payload.get("session_id"):
            self.session_id = str(payload["session_id"])

        mode = str(payload.get("mode") or "assistant").strip().lower()
        if mode != "lead":
            return

        lead = self.main.take_session_lead(self.session_id)
        if lead is None:
            # The greeting's stash expired or this call skipped it — fetch fresh so
            # lead mode still works. Soft: a miss just leaves it as a normal call.
            try:
                from app import zoho

                lead = await zoho.latest_lead()
            except Exception:
                lead = None
        if lead and lead.name:
            opening = prompts.outbound_greeting(lead.first_name)
            self.lead_context = prompts.lead_call_context(lead.context_block(), opening)
            log.info("LEAD CALL: pinned context for %s (%s)", lead.name, lead.city or "—")

    # -- Sarvam -> orchestration --------------------------------------------

    async def _pump_stt(self) -> None:
        """Turn-taking from Sarvam's VAD. START/END mark the boundaries; the
        audio itself is transcribed over REST (see _transcribe_and_respond).
        Never blocks on a response — that runs as its own task, so this loop
        stays free to catch a barge-in."""
        async for ev in self.stt.events():
            if isinstance(ev, stt_stream.SpeechStarted):
                # A new utterance is starting — keep only a little pre-speech so
                # the buffer holds THIS turn, not the silence/echo before it.
                if len(self.turn_buffer) > PREROLL_BYTES:
                    del self.turn_buffer[:-PREROLL_BYTES]
                if self.is_speaking:
                    await self._barge_in()
                else:
                    log.info("VAD: caller started speaking")
                    await self._safe_send(type="listening")

            elif isinstance(ev, stt_stream.SpeechEnded):
                log.info("VAD: end of turn")
                if self._busy():
                    continue  # a turn is already being transcribed / answered
                audio = bytes(self.turn_buffer)
                self.turn_buffer.clear()
                if len(audio) >= MIN_TURN_BYTES:
                    self.awaiting_final = asyncio.get_event_loop().create_future()
                    self.response_task = asyncio.create_task(self._handle_turn(audio))

            elif isinstance(ev, stt_stream.Transcript):
                # The fast path: this normally lands ~300ms after end-of-turn.
                if ev.text and self.awaiting_final and not self.awaiting_final.done():
                    self.awaiting_final.set_result((ev.text, ev.language_code))

    def _busy(self) -> bool:
        return self.response_task is not None and not self.response_task.done()

    async def _handle_turn(self, audio: bytes) -> None:
        """Get this utterance's text, then answer it.

        Fast path: use Sarvam's streaming transcript, which is already on its way
        because the audio streamed as the caller spoke — no extra round-trip.
        Fallback: if it doesn't arrive in time, transcribe the buffered audio over
        REST so a lagging/missing streaming transcript can never strand the turn.
        """
        fut = self.awaiting_final
        text: str = ""
        lang: str | None = None
        via = "stream"
        try:
            text, lang = await asyncio.wait_for(fut, timeout=STREAM_TRANSCRIPT_WAIT)
        except asyncio.TimeoutError:
            via = "rest"
        # NOTE: CancelledError is deliberately NOT caught — a barge-in cancels this
        # task, and that must abort the turn rather than fall through to REST.
        finally:
            self.awaiting_final = None  # stop routing transcripts into this turn

        if via == "rest":
            try:
                result = await stt.transcribe(_pcm_to_wav(audio))
                text, lang = result.text, result.language_code
            except Exception as exc:
                log.warning("REST STT fallback failed: %s", exc)
                return

        text = (text or "").strip()
        if not text:
            return  # nothing intelligible — stay listening, say nothing
        log.info("turn via %s: %r (%s)", via, text[:60], lang)

        verdict = self.main.limiter.check(f"ws:{self.session_id}")
        if not verdict.allowed:
            await self._safe_send(type="error", error=verdict.message)
            return
        await self._respond(text, lang)

    async def _respond(self, text: str, detected_lang: str | None) -> None:
        """One turn: LLM + streaming TTS out to the browser.

        Cancellable — a barge-in cancels this task mid-flight. Crucially,
        `is_speaking` is NOT cleared when we finish SENDING the audio: we stream
        faster than realtime, so the caller keeps hearing Shubh for seconds after
        the last chunk leaves. It stays True until the browser reports playback
        actually finished (the `playback_done` message), or a barge-in clears it.
        That is the whole window in which talking over him must stop him.
        """
        t0 = time.perf_counter()
        language = self.main.resolve_language(self.session_id, text, detected_lang)
        await self._safe_send(type="transcript", transcript=text, language_code=language)

        grounded = llm.GroundedAnswer()
        first_audio_ms: int | None = None
        n_chunks = 0
        self.is_speaking = True

        llm_stream = llm.stream_answer(
            self.main.KB.text, text, self.history, language, into=grounded, context=self.lead_context
        )
        try:
            async for pcm in tts.synthesize_stream(llm_stream, language):
                if first_audio_ms is None:
                    first_audio_ms = int((time.perf_counter() - t0) * 1000)
                n_chunks += 1
                await self._safe_send(
                    type="audio_pcm",
                    pcm=base64.b64encode(pcm).decode("ascii"),
                    rate=tts.TTS_SAMPLE_RATE,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Streaming TTS blipped. If nothing was spoken yet, the REST fallback
            # below recovers the turn; if we were mid-sentence, bail.
            log.warning("streaming TTS failed (%s) — falling back to REST", exc)
            if n_chunks:
                raise

        # Nothing reached the caller, or the model said nothing meaningful:
        # recover the answer over the reliable REST path and speak it.
        if n_chunks == 0 or self.main._is_empty_answer(grounded.text):
            if not grounded.text.strip():
                grounded = await llm.answer(
                    self.main.KB.text, text, self.history, language, context=self.lead_context
                )
            if self.main._is_empty_answer(grounded.text):
                grounded.text = self.main._FALLBACK.get(language, self.main._FALLBACK["hi-IN"])
                grounded.citations = []
                grounded.end_call = False
            wav = await tts.synthesize(grounded.text, language)
            if first_audio_ms is None:
                first_audio_ms = int((time.perf_counter() - t0) * 1000)
            await self._safe_send(type="audio", audio=base64.b64encode(wav).decode("ascii"))

        # Fully sent from our side (is_speaking stays True until playback_done).
        # Only reached on a full, uninterrupted turn — a barge-in raises out above.
        self.history.extend(
            [{"role": "user", "content": text}, {"role": "assistant", "content": grounded.text}]
        )
        del self.history[:-MAX_HISTORY_MESSAGES]

        await self._safe_send(
            type="done",
            answer=grounded.text,
            citations=[{"source": c.source, "page": c.page} for c in grounded.citations],
            end_call=grounded.end_call,
            timings={
                "first_audio_ms": first_audio_ms,
                "total_ms": int((time.perf_counter() - t0) * 1000),
                "tts_chunks": n_chunks,
            },
        )

    async def _barge_in(self) -> None:
        """The caller started talking over Shubh. Kill the response, tell the
        browser to stop playback, and go back to listening. Their fresh speech is
        already accumulating in turn_buffer for the next end-of-turn."""
        log.info("BARGE-IN: caller spoke over Shubh — cutting playback")
        await self._cancel_response()
        self.is_speaking = False
        await self._safe_send(type="interrupted")

    async def _cancel_response(self) -> None:
        task = self.response_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # -- helpers -------------------------------------------------------------

    async def _safe_send(self, **payload) -> None:
        if self.closed:
            return
        try:
            await _send(self.ws, **payload)
        except Exception:
            self.closed = True
