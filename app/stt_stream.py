"""Sarvam Saaras v3 streaming STT over a persistent WebSocket.

Thin async wrapper around the SDK socket: feed raw 16k PCM with `feed()`, read
typed events by iterating the session. Sarvam's own VAD decides turn boundaries
and emits them as events, so the caller never guesses end-of-speech from volume.

Event timeline for one utterance (measured against the live API):

    SpeechStarted  -> ... audio ... -> SpeechEnded -> Transcript

The Transcript lands ~150ms AFTER SpeechEnded, so an orchestrator waits a beat
past SpeechEnded for it. The socket tolerates concurrent send + recv (feeding
audio while reading events) and stays open across turns.
"""

import base64
import logging
from dataclasses import dataclass
from typing import AsyncIterator

import websockets

from app.config import settings
from app.stt import _client  # reuse the one AsyncSarvamAI instance

log = logging.getLogger(__name__)

STREAM_SAMPLE_RATE = 16000  # what the browser captures at; the only rate we send


@dataclass
class SpeechStarted:
    """VAD saw speech begin. Mid-response, this is a barge-in."""


@dataclass
class SpeechEnded:
    """VAD saw the turn end. The final Transcript follows shortly after."""


@dataclass
class Transcript:
    text: str
    language_code: str | None


class STTStreamError(RuntimeError):
    pass


class Session:
    """One open STT socket. `feed()` sends audio; `events()` yields typed events."""

    def __init__(self, ws, sample_rate: int = STREAM_SAMPLE_RATE):
        self._ws = ws
        self._sample_rate = sample_rate

    async def feed(self, pcm: bytes) -> None:
        """Send a chunk of raw 16-bit little-endian mono PCM at the session rate."""
        if not pcm:
            return
        # encoding is a fixed literal in the SDK; the real codec is set at connect
        # time via input_audio_codec="pcm_s16le". So this just ships the bytes.
        await self._ws.transcribe(
            audio=base64.b64encode(pcm).decode("ascii"),
            encoding="audio/wav",
            sample_rate=self._sample_rate,
        )

    async def flush(self) -> None:
        """Force-finalize whatever audio is buffered (rarely needed — the VAD
        finalizes on its own when the caller pauses)."""
        await self._ws.flush()

    async def events(self) -> AsyncIterator[object]:
        """Yield SpeechStarted / SpeechEnded / Transcript until the socket closes."""
        try:
            while True:
                msg = await self._ws.recv()
                data = getattr(msg, "data", None)
                if data is None:
                    continue
                signal = getattr(data, "signal_type", None)
                if signal == "START_SPEECH":
                    yield SpeechStarted()
                elif signal == "END_SPEECH":
                    yield SpeechEnded()
                    continue
                text = getattr(data, "transcript", None)
                if text is not None:
                    yield Transcript(text=text.strip(), language_code=getattr(data, "language_code", None))
        except (websockets.ConnectionClosedOK, websockets.ConnectionClosed):
            return  # end of stream


class _Connect:
    """`async with connect() as session:` — opens the socket, yields a Session."""

    def __init__(
        self,
        language_code: str,
        sample_rate: int = STREAM_SAMPLE_RATE,
        interrupt_min_speech_frames: int | None = None,
    ):
        self._language_code = language_code
        # Telephony calls negotiate their own rate (Exotel: 8k default, 16k on
        # request); Sarvam's SDK accepts 8000/16000, verified. Browser stays 16k.
        self._sample_rate = sample_rate
        # Per-transport barge-in resistance: the phone leg passes a higher value
        # than the browser's settings default (noisy rooms, speakerphone echo).
        self._interrupt_frames = (
            interrupt_min_speech_frames
            if interrupt_min_speech_frames is not None
            else settings.stt_interrupt_min_speech_frames
        )
        self._cm = None

    async def __aenter__(self) -> Session:
        extra = {}
        if settings.stt_negative_frames_count is not None:
            # Shorter end-of-speech hangover: fewer silence frames close the turn
            # sooner, cutting the dead air before END_SPEECH (see config).
            extra["negative_frames_count"] = str(settings.stt_negative_frames_count)
        self._cm = _client().speech_to_text_streaming.connect(
            # "unknown" lets Saaras auto-detect per turn (returned on each Transcript).
            language_code=self._language_code,
            model=settings.sarvam_stt_model,
            mode=settings.sarvam_stt_mode,
            sample_rate=str(self._sample_rate),
            vad_signals="true",
            # Lets Session.flush() force-finalize the transcript the moment
            # END_SPEECH fires, instead of waiting out the server's own pace.
            flush_signal="true",
            high_vad_sensitivity="true" if settings.stt_high_vad_sensitivity else "false",
            interrupt_min_speech_frames=str(self._interrupt_frames),
            input_audio_codec="pcm_s16le",
            **extra,
        )
        ws = await self._cm.__aenter__()
        return Session(ws, self._sample_rate)

    async def __aexit__(self, *exc) -> None:
        if self._cm is not None:
            await self._cm.__aexit__(*exc)


def connect(
    language_code: str = "unknown",
    sample_rate: int = STREAM_SAMPLE_RATE,
    interrupt_min_speech_frames: int | None = None,
) -> _Connect:
    return _Connect(language_code, sample_rate, interrupt_min_speech_frames)
