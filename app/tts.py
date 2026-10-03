"""Sarvam Bulbul v3. Two paths:

  synthesize()        REST, returns a whole WAV. Used for the greeting and as
                      the fallback when streaming is off.
  synthesize_stream() WebSocket, yields raw PCM as it generates — first audio in
                      ~300ms instead of ~1800ms. This is the latency win.
"""

import asyncio
import base64
import io
import logging
import re
import wave
from functools import lru_cache
from typing import AsyncIterator

import websockets
from sarvamai import AsyncSarvamAI

from app import pronunciation
from app.config import settings

log = logging.getLogger(__name__)

# Streaming emits linear16 (raw PCM) — the browser schedules it gaplessly via
# Web Audio. This is the sample rate the browser must reconstruct it at.
TTS_SAMPLE_RATE = 22050

# Exactly the 11 languages bulbul accepts, read off the installed SDK's
# TextToSpeechLanguage literal. Saaras understands 24, so STT can hand us a
# language TTS cannot speak (as-IN, ur-IN, ne-IN, sa-IN...). Fall back, never 500.
BULBUL_LANGUAGES = frozenset(
    {
        "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN",
        "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN",
    }
)
FALLBACK_LANGUAGE = "en-IN"

# v3 hard limit. Answers are 2-3 sentences, so this should never trip.
MAX_TTS_CHARS = 2500

# --- sentence splitting: how we start speaking before the answer is finished ---
# Synthesis costs ~700ms fixed + ~25ms/char, so speaking the first sentence as its
# own call overlaps the rest of the LLM stream. Below MIN_CHUNK_CHARS the 700ms
# floor outweighs the saving, so don't bother splitting.
#
# 10: low enough that a short natural opener ("Bilkul!", "जी हाँ, बताइए।") speaks
# the moment it exists instead of waiting to merge with sentence two (~150-400ms
# earlier first audio on those turns; 20 still held back a 14-char "जी हाँ,
# बताइए।"). Safe now that the tail synthesizes concurrently on its own
# pre-opened socket — a short head no longer implies a gap before the rest. The
# head is always a COMPLETE sentence, so prosody at the boundary stays natural.
MIN_CHUNK_CHARS = 10
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")
# A period here is punctuation, not a sentence end. "Rs. 8,400" must not split.
_ABBREVIATIONS = {"rs", "mr", "mrs", "ms", "dr", "sq", "ft", "no", "vs", "approx", "etc"}


def split_first_sentence(text: str) -> tuple[str | None, str]:
    """Return (first chunk, rest) once a clean SENTENCE boundary exists, else
    (None, text).

    Sentence ends only — no comma/clause splitting. Two TTS calls have no
    prosodic continuity, so a mid-sentence seam ("मैं शुभ हूँ, [seam] Magppie
    से...") sounds like the line dropped: the first half gets end-of-sentence
    intonation and any tail delay lands as a dead gap inside a thought. A full
    sentence is a natural place for a beat; a comma never is. With the ~30-word
    answer cap a sentence boundary always arrives, so nothing is lost.
    """
    for m in _SENTENCE_END.finditer(text):
        chunk = text[: m.start() + 1].strip()
        if len(chunk) < MIN_CHUNK_CHARS:
            continue
        last_word = re.split(r"[\s(]+", chunk[:-1])[-1].lower().strip(".,")
        if last_word in _ABBREVIATIONS:
            continue
        return chunk, text[m.end() :]

    return None, text


# Set once at startup from app.pronunciation. Applied to every synthesis so the
# brand name is said the same way everywhere — greeting included. None means the
# dictionary couldn't be created; calls still work, "Magppie" just may not.
DICT_ID: str | None = None


class TTSError(RuntimeError):
    pass


def to_tts_language(language_code: str | None) -> str:
    """Map an STT language onto something Bulbul can actually speak."""
    if language_code in BULBUL_LANGUAGES:
        return language_code
    if language_code:
        log.warning(
            "stt returned %s which bulbul cannot speak — falling back to %s",
            language_code,
            FALLBACK_LANGUAGE,
        )
    return FALLBACK_LANGUAGE


@lru_cache(maxsize=1)
def _client() -> AsyncSarvamAI:
    return AsyncSarvamAI(api_subscription_key=settings.sarvam_api_key)


async def synthesize(text: str, language_code: str | None, pace: float | None = None) -> bytes:
    """REST synthesis. `pace` overrides the per-language default — used by the
    greeting, which plays deliberately slower so callers register the brand."""
    lang = to_tts_language(language_code)
    text = pronunciation.normalize_speech_text((text or "").strip(), lang)
    if not text:
        raise TTSError("Nothing to speak.")
    if len(text) > MAX_TTS_CHARS:
        log.warning("answer is %d chars, truncating to %d", len(text), MAX_TTS_CHARS)
        text = text[:MAX_TTS_CHARS]

    extra = {"dict_id": DICT_ID} if DICT_ID else {}
    resp = await _client().text_to_speech.convert(
        text=text,
        target_language_code=lang,
        model=settings.sarvam_tts_model,
        speaker=settings.sarvam_tts_speaker,
        pace=pace if pace is not None else _tts_pace(lang),
        # NOTE: bulbul:v3 REJECTS pitch/loudness with a 400 ("currently not
        # supported") — volume shaping happens downstream (exotel gain).
        # Sarvam serves an exact-repeat from its own cache in ~170ms vs ~1800ms.
        # Dynamic answers rarely repeat, but greetings, acknowledgements and
        # common lines ("25 saal ki guarantee...") hit it — free win, no downside.
        enable_cached_responses=True,
        enable_preprocessing=True,  # smooth code-mixed (Devanagari+Latin) rendering
        **extra,
    )

    if not resp.audios:
        raise TTSError("Bulbul returned no audio.")

    # audios is a list of base64-encoded WAV strings, not raw binary.
    wav = base64.b64decode("".join(resp.audios))
    log.info("tts: %d chars -> %d wav bytes", len(text), len(wav))
    return wav


def _tts_pace(lang: str) -> float:
    """Per-language pace so Hindi keeps the same energy as English.

    Bulbul reads Devanagari with a slower, more deliberate prosody than Latin
    text from the same speaker — callers hear it as a different, draggy voice.
    A slightly higher pace for hi-IN brings the two deliveries level.
    """
    if lang == "hi-IN":
        return settings.sarvam_tts_pace_hi
    return settings.sarvam_tts_pace


def _tts_config(lang: str) -> dict:
    cfg = dict(
        target_language_code=lang,
        speaker=settings.sarvam_tts_speaker,
        pace=_tts_pace(lang),
        output_audio_codec="linear16",
        speech_sample_rate=TTS_SAMPLE_RATE,
        min_buffer_size=30,  # smaller -> first audio sooner
        # Sarvam's normalizer for mixed-language text — answers are deliberately
        # code-mixed (Devanagari Hindi + Latin English loanwords), which is
        # exactly what this flag exists to render smoothly.
        enable_preprocessing=True,
    )
    if DICT_ID:
        cfg["dict_id"] = DICT_ID
    return cfg


async def _open_socket(lang: str):
    """Open AND configure a Bulbul socket before there is text to say.

    The connect + configure round-trips cost ~200-400ms; doing them while the
    LLM is still writing means the first sentence starts synthesizing the moment
    its text exists. Returns (cm, ws); the caller owns closing cm.
    """
    # send_completion_event: Bulbul emits a 'final' event after the last audio
    # chunk. Without it, the socket lingers open ~60s before closing, which would
    # stall the whole turn — we break on 'final' instead of waiting for the close.
    cm = _client().text_to_speech_streaming.connect(
        model=settings.sarvam_tts_model, send_completion_event="true"
    )
    ws = await cm.__aenter__()
    await ws.configure(**_tts_config(lang))
    return cm, ws


async def _speak(text: str, lang: str, ready=None) -> AsyncIterator[bytes]:
    """Synthesize one piece of text on its own socket, yielding raw linear16 PCM.

    One socket per piece is deliberate: Bulbul closes the connection if more text
    is sent while audio is being read back, so a piece cannot be appended to a
    socket that is already speaking. Pass `ready=(cm, ws)` from _open_socket to
    skip the connect cost; this function closes the socket either way.
    """
    text = pronunciation.normalize_speech_text((text or "").strip(), lang)
    if not text:
        if ready is not None:  # nothing to say on a socket we were handed — close it
            await ready[0].__aexit__(None, None, None)
        return
    if len(text) > MAX_TTS_CHARS:
        log.warning("answer is %d chars, truncating to %d", len(text), MAX_TTS_CHARS)
        text = text[:MAX_TTS_CHARS]

    if ready is None:
        cm, ws = await _open_socket(lang)
    else:
        cm, ws = ready
    try:
        await ws.convert(text)
        await ws.flush()
        try:
            while True:
                msg = await ws.recv()
                data = getattr(msg, "data", None)
                audio = getattr(data, "audio", None)
                if audio:
                    yield base64.b64decode(audio)
                elif getattr(data, "event_type", None) == "final":
                    break  # all audio delivered — stop before the idle close
        except (websockets.ConnectionClosedOK, websockets.ConnectionClosed):
            pass  # 1000-close is the normal end-of-stream signal
    finally:
        await cm.__aexit__(None, None, None)


async def synthesize_stream(
    text_source: "str | AsyncIterator[str]", language_code: str
) -> AsyncIterator[bytes]:
    """Yield raw linear16 PCM, starting as early as possible.

    `text_source` is either a finished string (greeting / fallback) or the LLM's
    prose-delta stream. For the stream we do NOT wait for the whole answer: as
    soon as the model has produced a first speakable sentence we start
    synthesizing it, while the rest of the answer is still being written. That
    turns a serial "generate everything, then speak" into an overlap and takes
    roughly a second off time-to-first-audio.

    The remainder is drained concurrently, so by the time sentence one's audio has
    been sent the tail is usually ready to synthesize with no gap in playback.
    Raw PCM, not WAV; the browser reconstructs it at TTS_SAMPLE_RATE.
    """
    lang = to_tts_language(language_code)

    if isinstance(text_source, str):
        async for pcm in _speak(text_source, lang):
            yield pcm
        return

    source = text_source.__aiter__()
    pending = ""
    head: str | None = None

    # 0) Open the Bulbul socket NOW, while the model is still writing — its
    #    ~200-400ms of connect+configure runs under the LLM's own latency, so
    #    the first sentence starts synthesizing the instant its text exists.
    sock_task = asyncio.create_task(_open_socket(lang))
    sock_used = False

    async def claim_socket():
        """Hand the pre-opened socket to a _speak, or None if opening failed
        (that _speak then opens its own — same behaviour as before)."""
        nonlocal sock_used
        try:
            ready = await sock_task
        except Exception as exc:
            log.warning("tts: pre-open failed (%s) — connecting inline", exc)
            return None
        sock_used = True
        return ready

    try:
        # 1) Pull deltas only until there's a sentence worth speaking.
        async for delta in source:
            if not delta:
                continue
            pending += delta
            chunk, rest = split_first_sentence(pending)
            if chunk:
                head, pending = chunk, rest
                break

        if head is None:
            # The model finished without a clean boundary (a short one-liner) —
            # the whole answer is already in hand, so just speak it.
            async for pcm in _speak(pending, lang, ready=await claim_socket()):
                yield pcm
            return
    except BaseException:
        # Cancelled (barge-in) or the LLM stream failed before any speech: don't
        # leak the pre-opened socket.
        if not sock_task.done():
            sock_task.cancel()
        elif not sock_used and not sock_task.cancelled() and not sock_task.exception():
            cm, _ws = sock_task.result()
            await cm.__aexit__(None, None, None)
        raise

    # 2) Keep draining the rest of the answer while sentence one is synthesized.
    #    (Draining to completion also lets llm.stream_answer fill its GroundedAnswer.)
    async def drain_tail() -> str:
        buf = ""
        async for d in source:
            if d:
                buf += d
        return buf

    tail_task = asyncio.create_task(drain_tail())

    # 3) Synthesize the tail CONCURRENTLY with the head, not after it. The tail
    #    text usually completes while sentence one is still streaming, so its
    #    ~700ms socket+synthesis cost happens during head playback and the gap
    #    between sentence one and the rest disappears. Chunks buffer in a queue
    #    (a few hundred KB of PCM at most) and drain in order after the head.
    tail_q: asyncio.Queue = asyncio.Queue()

    async def pump_tail() -> None:
        # Open the tail's socket while its text is still being drained, mirroring
        # the head's pre-open — with a short head, the tail's connect cost would
        # otherwise land as an audible seam right after sentence one.
        tail_sock = asyncio.create_task(_open_socket(lang))
        try:
            tail = (pending + await tail_task).strip()
            ready = None
            try:
                ready = await tail_sock
            except Exception as exc:
                log.warning("tts: tail pre-open failed (%s) — connecting inline", exc)
            if tail:
                async for pcm in _speak(tail, lang, ready=ready):
                    await tail_q.put(pcm)
            elif ready is not None:
                await ready[0].__aexit__(None, None, None)  # nothing to say
        finally:
            if not tail_sock.done():
                tail_sock.cancel()
            # put_nowait, not put: this also runs when the pump is CANCELLED
            # (barge-in), and a fresh await inside cancellation is asking for
            # trouble. The queue is unbounded, so nowait always succeeds.
            tail_q.put_nowait(None)  # sentinel: tail finished (or empty/failed)

    pump = asyncio.create_task(pump_tail())
    try:
        # 4) Speak sentence one on the pre-opened socket — first audio now costs
        #    only the synthesis itself, not connect+configure.
        async for pcm in _speak(head, lang, ready=await claim_socket()):
            yield pcm

        # 5) The tail, already synthesized (or synthesizing) in parallel.
        while (pcm := await tail_q.get()) is not None:
            yield pcm
        await pump  # propagate a tail synthesis failure instead of going silent
    finally:
        for t in (pump, tail_task):
            if not t.done():
                t.cancel()
        # A cancellation before claim_socket ran would leak the pre-opened
        # socket; close it here (BaseException guard above covers phase 1).
        if not sock_task.done():
            sock_task.cancel()
        elif not sock_used and not sock_task.cancelled() and not sock_task.exception():
            cm, _ws = sock_task.result()
            await cm.__aexit__(None, None, None)


def concat_wavs(chunks: list[bytes]) -> bytes:
    """Join WAVs produced by separate synthesize() calls into one stream.

    Synthesis costs roughly 700ms + 25ms/char, so splitting an answer across
    concurrent calls pays the per-char part in parallel. Bulbul returns a
    consistent format for a given request, but verify rather than assume —
    silently concatenating mismatched rates yields chipmunk audio.
    """
    chunks = [c for c in chunks if c]
    if not chunks:
        raise TTSError("No audio to join.")
    if len(chunks) == 1:
        return chunks[0]

    params = None
    frames: list[bytes] = []
    for chunk in chunks:
        with wave.open(io.BytesIO(chunk)) as w:
            here = (w.getnchannels(), w.getsampwidth(), w.getframerate())
            if params is None:
                params = here
            elif here != params:
                log.warning("tts chunk format %s != %s; returning first chunk only", here, params)
                return chunks[0]
            frames.append(w.readframes(w.getnframes()))

    channels, width, rate = params
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"".join(frames))
    return out.getvalue()
