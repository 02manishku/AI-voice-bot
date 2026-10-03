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
import random
import re
import time
import uuid
import wave
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect
from sarvamai.core.api_error import ApiError

from app import llm, prompts, pronunciation, stt, stt_stream, tts
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
# When REST finishes first, how long to still wait for the stream transcript —
# it's the cleaner text (VAD-segmented, no preroll). Small on purpose.
STREAM_GRACE = 0.15

# The caller's raw 16k PCM is buffered per turn purely as the REST fallback's input.
_BYTES_PER_SEC = stt_stream.STREAM_SAMPLE_RATE * 2  # 16-bit mono
PREROLL_BYTES = int(0.5 * _BYTES_PER_SEC)    # pre-speech kept when a turn starts
MIN_TURN_BYTES = int(0.25 * _BYTES_PER_SEC)  # ignore sub-250ms VAD blips
MAX_BUFFER_BYTES = 30 * _BYTES_PER_SEC       # runaway guard


# --- junk-turn filter --------------------------------------------------------
# Real phone calls (team test round, 2026-08-31 logs) showed noise and
# backchannel sounds becoming full turns: "Hmm", "So...", "हुम हुम", random
# mis-language fragments — each one fired an LLM reply, made Shubh talk when a
# human would stay quiet, and burned the rate-limit budget so REAL questions got
# dropped. Two tiers, both deliberately conservative:
#
#   FILLERS      non-lexical sounds and bare continuations that are never a real
#                answer in any state ("hmm", "um", "so", "तो"). Dropped always —
#                a human waits through a think-noise, they don't pitch at it.
#   BACKCHANNEL  acknowledgements ("okay", "हाँ", "ठीक है", "अच्छा"). While Shubh
#                is mid-reply these are the caller listening along — dropped.
#                When he's idle they are ANSWERS to his questions ("भेज दूँ?" —
#                "हाँ") and always go through.
#
# Anything not in these sets goes to the LLM untouched. Never add words with a
# standalone meaning ("no", "बस", "wait") — same precision-over-coverage rule as
# the misheard-brand list in app/pronunciation.py.
_FILLER_TOKENS = frozenset(
    """hmm hmmm hm mm mmm um umm uh uhh uhm er err ah aah huh so toh
    हम्म हम्म्म हुम हुम्म अ आ तो और
    ம் ஹ்ம் ఉం ఊ ಹುಂ ഉം ഹും হুম হুঁ ਹੂੰ ਸੋ ହଁ""".split()
)
_BACKCHANNEL_TOKENS = frozenset(
    """ok okay kk yeah ya yes right acha achha accha haan han ji theek thik
    हाँ हां जी अच्छा ठीक है सही ओके हैं
    ਹਾਂ ਅੱਛਾ ਠੀਕ ਹੈ ਜੀ হ্যাঁ আচ্ছা ঠিক আছে சரி ஓகே సరే ಸರಿ ശരി ঠিক""".split()
)
_TOKEN_STRIP = re.compile(r"[.,!?।;:\-–—'\"()।॥]+")

# --- self-echo filter --------------------------------------------------------
# On a speakerphone Shubh's own voice loops back through the caller's mic and
# becomes "caller turns" — on the 2026-08-31 call he literally answered his own
# echoed sentence ("काफी ऑप्शंस होते हैं, तो स्टोन में हमारे पास..."). We KNOW
# what he just said (history + the in-flight reply), so a transcript that is
# near-verbatim his own words, arriving while he is speaking (or just after),
# is his echo — dropped. The bar is deliberately high (≥4 tokens, ≥75% of them
# his): a caller quoting one phrase back ("aapne bola 25 saal?") adds their own
# words around it and stays well under.
_ECHO_MIN_TOKENS = 4
_ECHO_OVERLAP = 0.75
_ECHO_WINDOW = 3.0  # seconds after Shubh stops in which echo can still arrive


def _echo_overlap(text: str, spoken: str) -> float:
    """What fraction of `text`'s tokens appear in Shubh's recent speech."""
    cand = [t for t in _TOKEN_STRIP.sub(" ", text.lower()).split() if t]
    if len(cand) < _ECHO_MIN_TOKENS:
        return 0.0
    spoken_set = set(_TOKEN_STRIP.sub(" ", spoken.lower()).split())
    if not spoken_set:
        return 0.0
    return sum(t in spoken_set for t in cand) / len(cand)


# --- unfinished-turn hold ----------------------------------------------------
# Callers think aloud in pieces: "बता सकते हैं कि" — pause — "बेसिकली किस चीज
# में डील करता है". Sarvam's VAD closes each pause, so Shubh answered fragment
# one, got barged over by fragment two, and sounded like he was restarting the
# same answer (2026-08-31 call). When a turn plainly isn't finished — it ends
# on a connective, a comma, or is a bare "Okay" — hold it briefly and merge
# whatever follows into ONE turn. The delay lands only on turns that were
# useless to answer anyway; finished sentences are untouched.
HOLD_UNFINISHED = 3.5  # ends mid-thought: wait for the rest (callers who think
                       # aloud pause 2-4s; 1.5s flushed mid-pause on real calls,
                       # and 2.5s still missed a 3s pause). The wait only lands
                       # on turns that were unanswerable anyway.
HOLD_ACK = 0.9         # bare "Okay/हाँ" while idle: often a preamble, briefly wait
_HELD_MAX_AGE = 6.0    # a stale held fragment is abandoned, not merged
_HELD_MAX_CHARS = 400  # runaway guard: stop merging, just answer

_CONTINUATION_TAIL = re.compile(
    r"""(?ix)
    (?: \b(?: कि | और | या | पर | लेकिन | मगर | मतलब | जैसे | अगर | जो
          | but | and | or | because | ki | aur | lekin | magar | agar
          | matlab | jaise
          # Dangling prepositions, articles and object-less verbs: "I want to
          # inquire" / "New connection to" / "kitchen के बारे" each closed a
          # turn on the 2026-09-03 call and got answered as half a sentence.
          | to | for | about | regarding | of | in | on | with | at | from
          | into | the | a | an | my | your | our | is | are | was | were
          | inquire | enquire | ask | know | want | need | tell
          | के | की | का | से | को | में | पे | बारे | ke | ka | se | ko | baare )
      | [,–—-]
    ) \s* [।.]* \s* $
    """
)


def _looks_unfinished(text: str) -> bool:
    """True when the turn audibly trails off mid-thought."""
    return bool(_CONTINUATION_TAIL.search(text))


def _is_bare_ack(text: str) -> bool:
    """Only acknowledgement/filler words — nothing to answer yet."""
    tokens = [t for t in _TOKEN_STRIP.sub(" ", text.lower()).split() if t]
    return bool(tokens) and all(
        t in _FILLER_TOKENS or t in _BACKCHANNEL_TOKENS for t in tokens
    )


_TERMINAL_PUNCT = ".?!।"


def _is_short_fragment(text: str) -> bool:
    """A few words with no sentence-final punctuation — "Hi", "About kitchens",
    "New connection to". Sometimes complete, often the first piece of a
    sentence said with pauses: on the 2026-09-03 call "Hi" / "I want to
    inquire" / "About Kitchens" arrived as three turns and each one cut the
    reply to the one before. Worth a short wait for the rest; a question mark
    or a full stop means the caller finished and is not held."""
    t = text.strip()
    if not t or t[-1] in _TERMINAL_PUNCT:
        return False
    tokens = [x for x in _TOKEN_STRIP.sub(" ", t).split() if x]
    return 0 < len(tokens) <= 3


# --- false barge-in recovery -------------------------------------------------
# Barge-in fires on the VAD, before anyone knows what was said. When the
# transcript then turns out to be nothing — background talk in a random
# language, a "hmm", Shubh's own echo — the reply is already cut and the
# caller sits in silence (2026-09-03: cut mid-answer by a Bengali fragment
# nobody said, Shubh went quiet, the caller said "Okay" into the void and hung
# up). So the interrupted reply is remembered and, when the interrupting turn
# proves to be junk, brought back: the phone leg resumes the audio from a
# second before the cut; a leg without buffered audio says the answer again.
FALSE_BARGE_WINDOW = 6.0   # seconds after the cut in which junk still means "resume"
MAX_RECOVERIES = 2         # per answer — a noisy room must not loop him forever


@dataclass
class _Interrupted:
    text: str | None   # the caller's turn the reply was answering (None: greeting)
    lang: str | None
    at: float          # monotonic time of the cut
    complete: bool     # the whole reply had been generated when it was cut


# --- garble-language gate ----------------------------------------------------
# Background speakers and overlapped audio come back from STT as SHORT fragments
# tagged with languages the call is not happening in — the 2026-09-02 calls got
# 'ਤੋਲ ਦੇ ਮਾਇਆ', 'हे पूर्ण आहे ते', 'ಆಂಗದ್ ಮತೀನ್...' as turns, each of which got a
# full reply until the caller begged "will you please talk in English only?".
# Every real question across every logged call was tagged hi-IN or en-IN. A
# SHORT turn in any other language, once the call is under way, is bleed — drop
# it. Longer foreign-language turns still pass (a genuine Punjabi caller's real
# sentences run 6+ tokens and deserve the reply, even if Shubh answers in
# Hindi/English).
_TRUSTED_LANGS = {"en-IN", "hi-IN"}
_DEVANAGARI_RE = re.compile(r"[ऀ-ॣ०-ॿ]")  # letters/digits only — no danda (shared by all Indic scripts)


def _is_garble_turn(text: str, lang: str | None, established: bool) -> bool:
    if not established or lang is None or lang in _TRUSTED_LANGS:
        return False
    # Strip punctuation BEFORE the script test: the danda "।" (U+0964) lives in
    # the Devanagari block but is shared by every Indic script — a Punjabi
    # fragment ending in "।" must not read as Hindi.
    cleaned = _TOKEN_STRIP.sub(" ", text)
    if _DEVANAGARI_RE.search(cleaned):
        return False  # Devanagari mis-tagged as bn/mr = real Hindi, keep
    tokens = cleaned.split()
    return len(tokens) <= 4


def _is_noise_turn(text: str, shubh_speaking: bool) -> bool:
    """True if this transcript should be silently dropped instead of answered."""
    tokens = [t for t in _TOKEN_STRIP.sub(" ", text.lower()).split() if t]
    if not tokens:
        return True
    if all(t in _FILLER_TOKENS for t in tokens):
        return True
    # One bare letter in any script ("A", "अ") is a VAD blip, not a question.
    if len(tokens) == 1 and len(tokens[0]) == 1:
        return True
    if shubh_speaking:
        allowed = _FILLER_TOKENS | _BACKCHANNEL_TOKENS
        return all(t in allowed for t in tokens)
    return False


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
        # Audio rate of THIS call's inbound leg. Browser always captures at 16k;
        # a telephony transport (Exotel) negotiates per call and calls _set_rate.
        self._set_rate(stt_stream.STREAM_SAMPLE_RATE)
        self.is_speaking = False              # caller is hearing (or about to hear) Shubh
        self.response_task: asyncio.Task | None = None
        # Resolved by the streaming transcript that follows end-of-turn (fast path).
        self.awaiting_final: asyncio.Future | None = None
        # Raw 16k PCM of the current utterance — input for the REST fallback.
        self.turn_buffer = bytearray()
        self.closed = False
        # Self-echo filter: when Shubh last stopped speaking, and the reply
        # currently being generated (its text may be mid-stream).
        self._last_speech_end = 0.0
        self._live_answer: llm.GroundedAnswer | None = None
        # Unfinished-turn hold (see _hold_turn): the stashed fragment and the
        # timer that answers it if nothing follows.
        self._held_text = ""
        self._held_lang: str | None = None
        self._held_stt_ms = 0
        self._held_at = 0.0
        self._hold_task: asyncio.Task | None = None
        # Speech that ENDED while a reply was still being produced. Its audio
        # stays in turn_buffer; the moment the reply finishes it becomes a turn
        # of its own instead of being silently discarded (real calls lost the
        # caller's closing words this way).
        self._missed_turn = False
        # False barge-in recovery (see _Interrupted): the reply that was cut,
        # the turn the current reply answers, and how often we've recovered.
        self._interrupted: _Interrupted | None = None
        self._current_turn: tuple[str, str | None] | None = None
        self._reply_complete = False
        self._recoveries = 0
        # The spoken "thinking beat" (see _respond): never the same one twice
        # in a row, and not on every turn.
        self._last_filler: bytes | None = None
        # Between VAD start and end: the caller is mid-utterance right now.
        self._caller_speaking = False

    def _set_rate(self, rate: int) -> None:
        """Pin the inbound sample rate and derive the byte thresholds from it,
        so the preroll/min-turn/runaway windows stay the same DURATIONS at 8k
        telephony as they are at 16k browser audio."""
        self.in_rate = rate
        bps = rate * 2  # 16-bit mono
        self._preroll_bytes = int(0.5 * bps)
        self._min_turn_bytes = int(0.25 * bps)
        self._max_buffer_bytes = 30 * bps

    # -- lifecycle -----------------------------------------------------------

    async def run(self) -> None:
        try:
            async with stt_stream.connect() as stt_session:
                self.stt = stt_session
                mic = asyncio.create_task(self._pump_mic(), name="mic")
                brain = asyncio.create_task(self._pump_stt(), name="stt")
                warm = asyncio.create_task(self._keep_llm_warm(), name="warm")
                try:
                    await asyncio.wait({mic, brain}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for t in (mic, brain, warm):
                        t.cancel()
                    if self._hold_task is not None:
                        self._hold_task.cancel()
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

    async def _keep_llm_warm(self) -> None:
        """While this call is open, never let the OpenAI prompt cache expire.

        The cache prefix lives ~5-10 min. A caller who listens for a while or
        goes quiet past that would pay the cold prefix (+1-2.5s) on their next
        question. One cheap ping only when the cache is about to lapse; on a
        normally-paced call the turns themselves keep it warm and this never
        fires.
        """
        while True:
            await asyncio.sleep(60)
            if not llm.cache_is_warm(ttl=240.0):
                try:
                    await llm.answer(self.main.KB.text, "hello", [], "en-IN")
                    log.info("llm: cache keepalive ping (quiet call)")
                except Exception as exc:
                    log.warning("llm: keepalive skipped (%s)", exc)

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
                if len(self.turn_buffer) > self._max_buffer_bytes:
                    del self.turn_buffer[:-self._max_buffer_bytes]
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
                    self._stop_speaking()
                elif payload.get("type") == "bye":
                    raise WebSocketDisconnect()

    async def _on_hello(self, payload: dict) -> None:
        """Handshake from the browser: session id + call mode. In LEAD CALL mode we
        claim the lead the greeting stashed (same record, no second fetch) and pin
        it as context so Shubh remembers who he called for the whole conversation."""
        if payload.get("session_id"):
            self.session_id = str(payload["session_id"])

        mode = str(payload.get("mode") or "assistant").strip().lower()

        # The spoken greeting goes into history as the model's own first line.
        # Without it the model doesn't know it already introduced itself, so a
        # garbled "कौन बोल रहा?" turn triggered the full "मैं शुभ हूँ, Magppie
        # से..." self-introduction script (reproduced 6/6; seeding fixed 6/6).
        if not self.history:
            self.history.append({"role": "assistant", "content": prompts.GREETING})

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
            # Lead mode spoke the outbound line, not the generic one — history
            # must carry what was actually said.
            self.history[0] = {"role": "assistant", "content": opening}
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
                # UNLESS an unprocessed turn is waiting in the buffer: then the
                # new speech joins it and they resolve as one combined turn.
                if not self._missed_turn and len(self.turn_buffer) > self._preroll_bytes:
                    del self.turn_buffer[:-self._preroll_bytes]
                self._caller_speaking = True
                if self.is_speaking:
                    await self._barge_in()
                else:
                    log.info("VAD: caller started speaking")
                    await self._safe_send(type="listening")

            elif isinstance(ev, stt_stream.SpeechEnded):
                log.info("VAD: end of turn")
                if self._busy():
                    # A reply is mid-flight. Don't discard what the caller just
                    # said — leave it buffered and pick it up the moment the
                    # reply finishes (see _check_missed).
                    self._missed_turn = True
                    self._caller_speaking = False
                    continue
                self._missed_turn = False
                audio = bytes(self.turn_buffer)
                self.turn_buffer.clear()
                if len(audio) >= self._min_turn_bytes:
                    # Force-finalize the transcript NOW rather than at the
                    # server's own pace — shaves the END_SPEECH->Transcript gap
                    # and rescues turns where the transcript would drift late.
                    try:
                        await self.stt.flush()
                    except Exception:
                        pass  # flush is an accelerant, never a requirement
                    self.awaiting_final = asyncio.get_event_loop().create_future()
                    self.response_task = asyncio.create_task(self._handle_turn(audio))
                else:
                    # Too short to be speech — a VAD blip. If it cut Shubh off,
                    # the reply is dead and NO turn is coming to replace it, so
                    # the call would sit silent until the caller gives up (seen
                    # 2026-09-07: a blip killed the sign-off, then 73s of
                    # nothing). Same recovery as a turn dropped as junk.
                    log.info("VAD blip too short to be a turn (%d bytes)", len(audio))
                    await self._after_dropped_turn()
                # Cleared only once this turn's task exists, so a held
                # fragment's flush timer can't slip in between and answer alone.
                self._caller_speaking = False

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

        # RACE the two transcription paths from t=0 instead of waiting out the
        # stream and only then starting REST. The old serial version added the
        # full STREAM_TRANSCRIPT_WAIT (0.8s) to every turn where Sarvam's stream
        # transcript lagged — the exact turns that were already slow. Now REST
        # transcribes the buffered audio concurrently: healthy stream still wins
        # (~320ms) and the REST result is discarded; a lagging/missing stream
        # loses to REST at ~1s instead of 1.8s. Costs one extra STT call per
        # turn, which is pennies next to the ~800ms it buys back.
        async def rest_transcribe():
            result = await stt.transcribe(_pcm_to_wav(audio, self.in_rate))
            return result.text, result.language_code

        t_turn = time.perf_counter()  # SpeechEnded ~= now; measures the STT phase
        rest_task = asyncio.create_task(rest_transcribe())
        try:
            # TRUE first-completed: whichever transcription lands first is taken
            # at the moment it lands. (The previous wait_for(stream, 0.8) held
            # the full window even when REST had finished at ~0.4s.) If the
            # stream wins within the window it is preferred; otherwise whatever
            # REST produced — whenever it produces it — carries the turn.
            done, _pending = await asyncio.wait(
                {fut, rest_task},
                timeout=STREAM_TRANSCRIPT_WAIT,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if fut not in done and rest_task in done:
                # REST landed first. Give the stream a short grace before taking
                # it: the stream's text is the VAD-segmented utterance, cleaner
                # than our buffered audio (which carries preroll/echo). Costs at
                # most STREAM_GRACE, and only on turns where REST was faster.
                try:
                    await asyncio.wait_for(asyncio.shield(fut), timeout=STREAM_GRACE)
                except (asyncio.TimeoutError, Exception):
                    pass  # CancelledError (barge-in) still propagates: BaseException
            if fut.done() and not fut.cancelled() and fut.exception() is None:
                text, lang = fut.result()
            if not (text or "").strip():
                via = "rest"
                try:
                    text, lang = await rest_task
                except Exception as exc:
                    log.warning("REST STT fallback failed: %s", exc)
                    return
        # NOTE: CancelledError is deliberately NOT caught — a barge-in cancels this
        # task, and that must abort the turn rather than fall through to REST.
        finally:
            self.awaiting_final = None  # stop routing transcripts into this turn
            if not rest_task.done():
                rest_task.cancel()
            elif not rest_task.cancelled():
                rest_task.exception()  # retrieve, silencing "never retrieved"
        stt_ms = int((time.perf_counter() - t_turn) * 1000)

        # The brand repair used to run only inside the REST path (stt.transcribe);
        # streaming transcripts — most turns — skipped it. Now both paths get it
        # (it is idempotent, so the REST path applying it twice is harmless).
        text = pronunciation.normalize_transcript((text or "").strip())
        if not text:
            await self._after_dropped_turn()
            return  # nothing intelligible — stay listening, say nothing
        log.info("turn via %s in %dms: %r (%s)", via, stt_ms, text[:60], lang)

        # Noise and listening-along sounds never reach the LLM (or the rate
        # limiter): a human doesn't pitch at a "hmm". If Shubh is mid-reply the
        # audio keeps playing right through it. "Mid-reply" includes the
        # moment right after a barge-in cut him: that transcript IS the
        # interruption, and a bare "okay" then is listening, not a new turn.
        speaking_ctx = self.is_speaking or self._just_cut_off()
        if _is_noise_turn(text, speaking_ctx):
            log.info("turn dropped (noise/backchannel while %s): %r",
                     "speaking" if speaking_ctx else "idle", text[:40])
            await self._after_dropped_turn()
            return

        # Shubh's own voice looping back through a speakerphone.
        if self._is_echo_turn(text):
            log.info("turn dropped as self-echo: %r", text[:40])
            await self._after_dropped_turn()
            return

        # Background-speaker bleed: short fragments in a language this call
        # isn't happening in.
        if _is_garble_turn(text, lang, established=len(self.history) > 1):
            log.info("turn dropped as background garble (%s): %r", lang, text[:40])
            await self._after_dropped_turn()
            return

        # A held mid-thought fragment belongs to the front of this turn.
        if held := self._take_held():
            text = f"{held} {text}"
            log.info("merged with held fragment -> %r", text[:80])

        # An audibly unfinished turn waits briefly for its own continuation
        # instead of being answered as half a question. A bare "Okay" or a
        # two-word unpunctuated fragment ("About kitchens") gets a shorter wait.
        if len(text) <= _HELD_MAX_CHARS:
            if _looks_unfinished(text):
                self._hold_turn(text, lang, stt_ms, HOLD_UNFINISHED)
                return
            if not self.is_speaking and (_is_bare_ack(text) or _is_short_fragment(text)):
                self._hold_turn(text, lang, stt_ms, HOLD_ACK)
                return

        await self._deliver_checked(text, lang, stt_ms)

    async def _deliver_checked(
        self,
        text: str,
        lang: str | None,
        stt_ms: int = 0,
        nudge: str | None = None,
        resuming: bool = False,
    ) -> None:
        """_deliver, then pick up any speech that ended during the reply."""
        await self._deliver(text, lang, stt_ms, nudge, resuming)
        self._check_missed()

    def _check_missed(self) -> None:
        """The caller finished saying something while the reply was in flight —
        its audio is still buffered. Turn it into a turn now, not never."""
        if not self._missed_turn:
            return
        self._missed_turn = False
        audio = bytes(self.turn_buffer)
        self.turn_buffer.clear()
        if len(audio) >= self._min_turn_bytes:
            log.info("processing speech that ended during the last reply")
            self.awaiting_final = asyncio.get_event_loop().create_future()
            self.response_task = asyncio.create_task(self._handle_turn(audio))

    async def _deliver(
        self,
        text: str,
        lang: str | None,
        stt_ms: int = 0,
        nudge: str | None = None,
        resuming: bool = False,
    ) -> None:
        """Rate-limit and answer one finished turn."""
        # Substantive speech while Shubh's audio is still playing, but below
        # the VAD's barge-in threshold (raised on the phone leg): a real talk-
        # over. Cut playback like a barge-in instead of queueing a second reply
        # on top of the one still sounding.
        if self.is_speaking:
            log.info("late barge-in: substantive speech during playback — cutting")
            self._stop_speaking()
            await self._safe_send(type="interrupted")

        verdict = self.main.limiter.check(f"ws:{self.session_id}")
        if not verdict.allowed:
            # kind lets the phone leg SPEAK a busy line — a silent refusal
            # reads as "the bot is broken" (2026-09-03, tripped daily cap).
            await self._safe_send(type="error", error=verdict.message, kind="rate_limit")
            return
        if not resuming:
            # A real new turn supersedes whatever reply was cut earlier.
            self._interrupted = None
            self._recoveries = 0
        await self._respond(text, lang, stt_ms, nudge)

    # -- unfinished-turn hold -------------------------------------------------

    def _take_held(self) -> str:
        """Claim the held fragment for merging (cancels its flush timer)."""
        if self._hold_task is not None and not self._hold_task.done():
            self._hold_task.cancel()
        self._hold_task = None
        text, self._held_text = self._held_text, ""
        if text and time.monotonic() - self._held_at > _HELD_MAX_AGE:
            log.info("held fragment expired unanswered: %r", text[:40])
            return ""
        return text

    def _hold_turn(self, text: str, lang: str | None, stt_ms: int, delay: float) -> None:
        self._held_text, self._held_lang, self._held_stt_ms = text, lang, stt_ms
        self._held_at = time.monotonic()
        log.info("turn sounds unfinished — holding %.1fs for the rest: %r", delay, text[:40])
        self._hold_task = asyncio.create_task(self._flush_held(delay))

    def _rearm_hold(self) -> None:
        """A dropped/empty turn arrived while a fragment was held: keep the
        flush timer alive so the held question still gets its answer."""
        if self._held_text and (self._hold_task is None or self._hold_task.done()):
            self._hold_task = asyncio.create_task(self._flush_held(HOLD_ACK))

    async def _flush_held(self, delay: float) -> None:
        await asyncio.sleep(delay)
        # The caller is mid-sentence at this very moment: the rest of the held
        # thought is on its way. Answering now would talk over it and then get
        # cut by it (seen in replay: the timer fired 0.3s into "about
        # kitchens"). Wait for that utterance to close; it merges the fragment.
        while self._caller_speaking:
            await asyncio.sleep(0.1)
        if self._busy():
            # A newer turn is already being transcribed/answered — IT will
            # merge the held text via _take_held. Firing here anyway would put
            # two reply tasks in flight at once (seen live: doubled answers,
            # and the barge-in machinery can only cancel the tracked one).
            return
        text, lang, stt_ms = self._held_text, self._held_lang, self._held_stt_ms
        self._held_text = ""
        if not text:
            return

        tokens = [t for t in _TOKEN_STRIP.sub(" ", text).split() if t]
        if _looks_unfinished(text) and len(tokens) <= 6:
            if len(tokens) < 3:
                # A dangling connective ("However,") has nothing to answer.
                # Real calls showed the reply colliding with the caller's own
                # continuation. A human just waits — so keep it as context for
                # whatever comes next and stay silent.
                self._held_text = text
                self._held_at = time.monotonic()
                log.info("held fragment has no answerable content — staying quiet: %r", text[:40])
                return
            # "New connection to" — a real start that trailed off, and the
            # wait is over. A person who has sat through the silence asks the
            # caller to finish the thought; staying quiet here left a caller
            # talking to nobody and hanging up (2026-09-03).
            log.info("held fragment trailed off — asking the caller to finish it: %r", text[:40])
            nudge = prompts.NUDGE_TRAILED_OFF.format(said=text)
            self.response_task = asyncio.create_task(
                self._deliver_checked(text, lang, stt_ms, nudge=nudge)
            )
            return

        if _is_bare_ack(text):
            last = next(
                (m["content"] for m in reversed(self.history) if m["role"] == "assistant"), ""
            )
            if not last.rstrip().endswith("?"):
                # No question is pending, so "Okay" is the caller handing the
                # turn back. Answering it as a question buried callers under
                # repeats (2026-08-31); saying NOTHING left one saying "Okay"
                # three times into silence and hanging up (2026-09-03). A
                # person leads here: one short line, the next useful question.
                log.info("bare ack, no pending question — leading the call forward: %r", text[:30])
                nudge = prompts.NUDGE_ADVANCE.format(said=text)
                self.response_task = asyncio.create_task(
                    self._deliver_checked(text, lang, stt_ms, nudge=nudge)
                )
                return

        log.info("nothing followed the held turn — answering it as-is")
        self.response_task = asyncio.create_task(self._deliver_checked(text, lang, stt_ms))

    # -- self-echo filter -----------------------------------------------------

    def _stop_speaking(self) -> None:
        self.is_speaking = False
        self._last_speech_end = time.monotonic()

    def _shubh_recent_text(self) -> str:
        """What Shubh said recently: the last replies plus the one in flight."""
        replies = [m["content"] for m in self.history if m["role"] == "assistant"][-2:]
        if self._live_answer is not None and self._live_answer.text:
            replies.append(self._live_answer.text)
        return " ".join(replies)

    def _is_echo_turn(self, text: str) -> bool:
        if not (self.is_speaking or time.monotonic() - self._last_speech_end < _ECHO_WINDOW):
            return False
        return _echo_overlap(text, self._shubh_recent_text()) >= _ECHO_OVERLAP

    async def _respond(
        self,
        text: str,
        detected_lang: str | None,
        stt_ms: int = 0,
        nudge: str | None = None,
    ) -> None:
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
        # Fire-and-forget: the transcript echo must not sit between us and the
        # LLM kickoff. Ordering to the browser is safe — first audio trails this
        # by hundreds of ms.
        asyncio.create_task(self._safe_send(type="transcript", transcript=text, language_code=language))

        grounded = llm.GroundedAnswer()
        self._live_answer = grounded  # echo filter sees the reply as it streams
        first_audio_ms: int | None = None
        n_chunks = 0
        died_mid_utterance = False
        self.is_speaking = True
        self._current_turn = (text, detected_lang)
        self._reply_complete = False

        # The caller's words enter history NOW, not after the reply survives.
        # History used to be written only at the end of an uninterrupted turn,
        # so a barge-in erased the exchange entirely — Shubh forgot the caller
        # ever said it (live: a caller's budget vanished because their next
        # sentence cut the reply). The LLM gets a snapshot WITHOUT this turn,
        # because the current question is passed to it separately.
        llm_history = list(self.history)
        self.history.append({"role": "user", "content": text})
        del self.history[:-MAX_HISTORY_MESSAGES]

        # Where the wait goes, per turn, in the log: transcript -> first LLM
        # token -> first audio. The model's first token is the single biggest
        # fixed cost (~1.2s, measured 2026-09-03), so it is timed on its own.
        t_first_token: float | None = None

        async def timed_llm():
            nonlocal t_first_token
            async for delta in llm.stream_answer(
                self.main.KB.text, text, llm_history, language,
                into=grounded, context=self.lead_context, nudge=nudge,
            ):
                if t_first_token is None and delta:
                    t_first_token = time.perf_counter()
                yield delta

        # A person who is asked something makes a small sound while they
        # think — "Hmm.", "जी।" — instead of going dead for two seconds. If
        # the answer's first audio hasn't arrived by the deadline, one short
        # pre-rendered beat plays in the reply's own language; the answer
        # follows in the same utterance. Not every turn, never the same beat
        # twice running, and never when the answer is already here.
        # No beat before a goodbye ("Hmm. You're welcome") or when saying a cut
        # answer again ("Sure. As I was saying…") — both read as a tic.
        wants_beat = nudge is not prompts.NUDGE_RESUME and not llm._looks_like_exit(text)
        filler_task = asyncio.create_task(
            self._thinking_beat(language, t0) if wants_beat else asyncio.sleep(0)
        )

        try:
            async for pcm in tts.synthesize_stream(timed_llm(), language):
                if first_audio_ms is None:
                    filler_task.cancel()
                    await self._await_quietly(filler_task)
                    first_audio_ms = int((time.perf_counter() - t0) * 1000)
                    llm_ms = int((t_first_token - t0) * 1000) if t_first_token else -1
                    log.info(
                        "reply: first audio at %dms (stt %dms before it, llm first token %dms)",
                        first_audio_ms, stt_ms, llm_ms,
                    )
                n_chunks += 1
                await self._safe_send(
                    type="audio_pcm",
                    pcm=base64.b64encode(pcm).decode("ascii"),
                    rate=tts.TTS_SAMPLE_RATE,
                )
        except asyncio.CancelledError:
            filler_task.cancel()
            raise
        except Exception as exc:
            # Streaming TTS blipped (seen live: Sarvam's WS returning transient
            # 403s). If nothing was spoken yet, the REST fallback below recovers
            # the whole turn. If we were mid-sentence, DON'T re-speak and DON'T
            # propagate — a raise here used to strand the session half-open
            # (is_speaking stuck True, no playback ack ever coming). Close the
            # utterance cleanly with whatever the caller already heard.
            log.warning("streaming TTS failed after %d chunks (%s)", n_chunks, exc)
            if n_chunks:
                died_mid_utterance = True
        finally:
            if not filler_task.done():
                filler_task.cancel()

        # Nothing reached the caller, or the model said nothing meaningful:
        # recover the answer over the reliable REST path and speak it.
        if not died_mid_utterance and (n_chunks == 0 or self.main._is_empty_answer(grounded.text)):
            try:
                if not grounded.text.strip():
                    grounded = await llm.answer(
                        self.main.KB.text, text, self.history, language,
                        context=self.lead_context, nudge=nudge,
                    )
                if self.main._is_empty_answer(grounded.text):
                    grounded.text = self.main._FALLBACK.get(language, self.main._FALLBACK["hi-IN"])
                    grounded.citations = []
                    grounded.end_call = False
                wav = await tts.synthesize(grounded.text, language)
                if first_audio_ms is None:
                    first_audio_ms = int((time.perf_counter() - t0) * 1000)
                await self._safe_send(type="audio", audio=base64.b64encode(wav).decode("ascii"))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Even the REST path failed — give the turn up, but leave the
                # SESSION healthy: clear the speaking flag so barge-in/echo
                # logic doesn't stay armed against audio that never played.
                log.warning("REST recovery failed too (%s) — dropping this turn", exc)
                self._stop_speaking()
                await self._safe_send(type="error", error="TTS unavailable for this turn")
                return

        # Fully sent from our side (is_speaking stays True until playback_done).
        # Only reached on a completed turn — a barge-in raises out above, which
        # leaves the caller's words (already recorded) without a reply, exactly
        # like a person who was cut off before answering.
        self.history.append({"role": "assistant", "content": grounded.text})
        del self.history[:-MAX_HISTORY_MESSAGES]

        await self._safe_send(
            type="done",
            answer=grounded.text,
            citations=[{"source": c.source, "page": c.page} for c in grounded.citations],
            end_call=grounded.end_call,
            timings={
                "stt_ms": stt_ms,
                "first_audio_ms": first_audio_ms,
                "total_ms": int((time.perf_counter() - t0) * 1000),
                "tts_chunks": n_chunks,
            },
        )
        self._reply_complete = True
        log.info("reply: all audio sent at %dms", int((time.perf_counter() - t0) * 1000))

    async def _thinking_beat(self, language: str, t0: float) -> None:
        """Play one short pre-rendered beat if the answer is slow to arrive.
        Cancelled the moment real audio shows up (see _respond)."""
        if not settings.reply_filler:
            return
        deadline = settings.reply_filler_after_ms / 1000.0
        await asyncio.sleep(max(0.0, deadline - (time.perf_counter() - t0)))
        if random.random() > settings.reply_filler_chance:
            return
        options = [f for f in self.main.filler_clips(language) if f != self._last_filler]
        if not options:
            return
        clip = random.choice(options)
        self._last_filler = clip
        log.info("reply: thinking beat at %dms", int((time.perf_counter() - t0) * 1000))
        await self._safe_send(
            type="audio_pcm",
            pcm=base64.b64encode(clip).decode("ascii"),
            rate=tts.TTS_SAMPLE_RATE,
        )

    @staticmethod
    async def _await_quietly(task: asyncio.Task) -> None:
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    # -- false barge-in recovery -----------------------------------------------

    def _just_cut_off(self) -> bool:
        return (
            self._interrupted is not None
            and time.monotonic() - self._interrupted.at < FALSE_BARGE_WINDOW
        )

    def _note_interruption(self) -> None:
        text, lang = self._current_turn or (None, None)
        self._interrupted = _Interrupted(text, lang, time.monotonic(), self._reply_complete)

    async def _after_dropped_turn(self) -> None:
        """A transcript was thrown away (noise, echo, garble, empty)."""
        self._rearm_hold()
        await self._recover_false_barge_in()

    async def _recover_false_barge_in(self) -> None:
        """The speech that cut Shubh off turned out to be nothing. Bring the
        reply back: resume its audio where the transport can, else say it again."""
        info, self._interrupted = self._interrupted, None
        if info is None or time.monotonic() - info.at > FALSE_BARGE_WINDOW:
            return
        if self._recoveries >= MAX_RECOVERIES:
            log.info("false barge-in: already recovered %d times — leaving it", self._recoveries)
            return
        self._recoveries += 1
        if await self._resume_interrupted_audio(info):
            return
        if info.text is None:
            return  # the greeting or a canned line: nothing to generate again
        log.info("false barge-in: the interruption was nothing — saying the answer again")
        self._forget_dangling_turn(info.text)
        self.response_task = asyncio.create_task(
            self._deliver_checked(info.text, info.lang, nudge=prompts.NUDGE_RESUME, resuming=True)
        )

    async def _resume_interrupted_audio(self, info: _Interrupted) -> bool:
        """Transport hook: continue the cut audio from just before the cut.
        True = handled (resumed, or nothing worth resuming). The browser leg
        keeps no audio, so it answers again instead."""
        return False

    def _forget_dangling_turn(self, text: str) -> None:
        """Drop the user turn a cut reply left unanswered, so saying the answer
        again doesn't record the caller as having asked twice."""
        if self.history and self.history[-1]["role"] == "user" and self.history[-1]["content"] == text:
            self.history.pop()

    async def _barge_in(self) -> None:
        """The caller started talking over Shubh. Kill the response, tell the
        browser to stop playback, and go back to listening. Their fresh speech is
        already accumulating in turn_buffer for the next end-of-turn."""
        log.info("BARGE-IN: caller spoke over Shubh — cutting playback")
        self._note_interruption()
        await self._cancel_response()
        self._stop_speaking()
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
