"""Sarvam Saaras v3, REST. transcribe(wav_bytes) -> Transcript."""

import io
import logging
from dataclasses import dataclass
from functools import lru_cache

from sarvamai import AsyncSarvamAI

from app.config import settings
from app.pronunciation import normalize_transcript

log = logging.getLogger(__name__)

# Hard API limit is 30s; the browser caps at MAX_RECORDING_SECONDS (25).
MAX_AUDIO_SECONDS = 30


class STTError(RuntimeError):
    pass


@dataclass
class Transcript:
    text: str
    language_code: str | None


@lru_cache(maxsize=1)
def _client() -> AsyncSarvamAI:
    return AsyncSarvamAI(api_subscription_key=settings.sarvam_api_key)


async def transcribe(wav_bytes: bytes) -> Transcript:
    if not wav_bytes:
        raise STTError("Empty audio — nothing to transcribe.")

    # A file object, not a path. Named tuple form so the SDK sends a filename.
    file = ("turn.wav", io.BytesIO(wav_bytes), "audio/wav")

    resp = await _client().speech_to_text.transcribe(
        file=file,
        model=settings.sarvam_stt_model,
        mode=settings.sarvam_stt_mode,
    )

    # Repair the brand name before anything downstream reads it: the model treats
    # "MacPay Kitchens" as a rival company and refuses the customer.
    text = normalize_transcript((resp.transcript or "").strip())
    log.info("stt: lang=%s chars=%d", resp.language_code, len(text))
    return Transcript(text=text, language_code=resp.language_code)
