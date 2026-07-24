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
# 35, not 60: a first sentence in Hindi is routinely ~55 chars, so a 60-char floor
# silently disabled the split on most real answers.
MIN_CHUNK_CHARS = 35
# Past this length with no sentence end in sight, break at a clause boundary
# instead — a long comma-run sentence otherwise can't split, so the whole thing
# goes as one call and first audio lands seconds late.
CLAUSE_SPLIT_CHARS = 70
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:—])\s+")
# A period here is punctuation, not a sentence end. "Rs. 8,400" must not split.
_ABBREVIATIONS = {"rs", "mr", "mrs", "ms", "dr", "sq", "ft", "no", "vs", "approx", "etc"}


def split_first_sentence(text: str) -> tuple[str | None, str]:
    """Return (first chunk, rest) once a clean boundary exists, else (None, text).

    Prefers a full sentence; falls back to a clause boundary once the text has run
    long without one, so TTS starts speaking sooner.
    """
    for m in _SENTENCE_END.finditer(text):
        chunk = text[: m.start() + 1].strip()
        if len(chunk) < MIN_CHUNK_CHARS:
            continue
        last_word = re.split(r"[\s(]+", chunk[:-1])[-1].lower().strip(".,")
        if last_word in _ABBREVIATIONS:
            continue
        return chunk, text[m.end() :]

    if len(text) >= CLAUSE_SPLIT_CHARS:
        for m in _CLAUSE_END.finditer(text):
            chunk = text[: m.start() + 1].strip()
            if len(chunk) >= MIN_CHUNK_CHARS:
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


async def synthesize(text: str, language_code: str | None) -> bytes:
    text = (text or "").strip()
    if not text:
        raise TTSError("Nothing to speak.")
    if len(text) > MAX_TTS_CHARS:
        log.warning("answer is %d chars, truncating to %d", len(text), MAX_TTS_CHARS)
        text = text[:MAX_TTS_CHARS]

    extra = {"dict_id": DICT_ID} if DICT_ID else {}
    resp = await _client().text_to_speech.convert(
        text=text,
        target_language_code=to_tts_language(language_code),
        model=settings.sarvam_tts_model,
        speaker=settings.sarvam_tts_speaker,
        pace=settings.sarvam_tts_pace,
        # Sarvam serves an exact-repeat from its own cache in ~170ms vs ~1800ms.
        # Dynamic answers rarely repeat, but greetings, acknowledgements and
        # common lines ("25 saal ki guarantee...") hit it — free win, no downside.
        enable_cached_responses=True,
        **extra,
    )

    if not resp.audios:
        raise TTSError("Bulbul returned no audio.")

    # audios is a list of base64-encoded WAV strings, not raw binary.
    wav = base64.b64decode("".join(resp.audios))
    log.info("tts: %d chars -> %d wav bytes", len(text), len(wav))
    return wav


async def _speak(text: str, lang: str) -> AsyncIterator[bytes]:
    """Synthesize one piece of text on its own socket, yielding raw linear16 PCM.

    One socket per piece is deliberate: Bulbul closes the connection if more text
    is sent while audio is being read back, so a piece cannot be appended to a
    socket that is already speaking.
    """
    text = (text or "").strip()
    if not text:
        return
    if len(text) > MAX_TTS_CHARS:
        log.warning("answer is %d chars, truncating to %d", len(text), MAX_TTS_CHARS)
        text = text[:MAX_TTS_CHARS]

    cfg = dict(
        target_language_code=lang,
        speaker=settings.sarvam_tts_speaker,
        pace=settings.sarvam_tts_pace,
        output_audio_codec="linear16",
        speech_sample_rate=TTS_SAMPLE_RATE,
        min_buffer_size=30,  # smaller -> first audio sooner
    )
    if DICT_ID:
        cfg["dict_id"] = DICT_ID

    # send_completion_event: Bulbul emits a 'final' event after the last audio
    # chunk. Without it, the socket lingers open ~60s before closing, which would
    # stall the whole turn — we break on 'final' instead of waiting for the close.
    async with _client().text_to_speech_streaming.connect(
        model=settings.sarvam_tts_model, send_completion_event="true"
    ) as ws:
        await ws.configure(**cfg)
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
        # The model finished without a clean boundary (a short one-liner) — the
        # whole answer is already in hand, so just speak it.
        async for pcm in _speak(pending, lang):
            yield pcm
        return

    # 2) Keep draining the rest of the answer while sentence one is synthesized.
    #    (Draining to completion also lets llm.stream_answer fill its GroundedAnswer.)
    async def drain_tail() -> str:
        buf = ""
        async for d in source:
            if d:
                buf += d
        return buf

    tail_task = asyncio.create_task(drain_tail())
    try:
        # 3) Speak sentence one immediately — this is the latency win.
        async for pcm in _speak(head, lang):
            yield pcm

        # 4) Then the remainder, which by now has almost certainly finished.
        tail = (pending + await tail_task).strip()
        if tail:
            async for pcm in _speak(tail, lang):
                yield pcm
    finally:
        if not tail_task.done():
            tail_task.cancel()


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
