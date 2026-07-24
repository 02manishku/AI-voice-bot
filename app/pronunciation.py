"""Sarvam pronunciation dictionary — how the brand name gets said correctly.

The brand is pronounced **mag-pai** — like the bird "magpie". Confirmed directly
by Magppie, 2026-07-17.

NOTE: this contradicts A7 of the knowledge base, which states "mag-pee". A7 is
wrong and should be corrected; the value here is the authoritative one. If you
change one, change the other, or the bot and the sales team will say the brand
two different ways.

Left alone, Bulbul spells out the double-p or guesses. This uploads a dictionary
once, gets a dict_id, and passes it on every TTS call; Sarvam substitutes
matching words before synthesis.

Values are plain-text replacements, not IPA — a respelling for Latin script, the
native script for Devanagari.

Sarvam's guidance: only add words that actually mispronounce. Bulbul v3 already
handles ordinary English, numbers and Hinglish. Keep this list short.
"""

import hashlib
import io
import json
import logging
import re
from pathlib import Path

from sarvamai import AsyncSarvamAI

from app.config import settings

log = logging.getLogger(__name__)

# --- the other direction: how the brand name gets HEARD -----------------------
#
# Saaras does not know "Magppie" and reaches for the nearest real word, so the
# transcript arrives as "MacPay Kitchens" or "Mac by Kitchens". The model then
# reads that as a DIFFERENT company and politely refuses the customer. Telling
# the model "mis-hearings happen" is too weak — it still pattern-matches "not
# Magppie" to a refusal. So repair the transcript before the model sees it.
#
# Precision matters more than coverage here. Every entry must be a word that has
# no other plausible meaning on a Magppie call:
#   - "मैट" is NOT here: it is "matte", a real finish in K-27.
#   - "MacBook"/"Mac" are NOT here: those are genuinely off-topic questions the
#     bot is supposed to decline, and rewriting them would break that.
# Adding a word that has a legitimate meaning silently corrupts real questions,
# which is far worse than missing one mis-hearing.
_MISHEARD_BRAND = re.compile(
    r"""(?<!\w)(
          mac \s*[-–]?\s* (?: pay | pai | bye? | pie | pe | py )   # MacPay, Mac by, Mac pie
        | mag \s*[-–]?\s* (?: pie | pai | pay | pee | py | p )     # magpie, mag pie, MagPy
        | मैगपाई | मैग \s* पाई | मैकपे | मैक \s* पे | मैगपी | मैग्पी
    )(?!\w)""",
    re.IGNORECASE | re.VERBOSE,
)


def normalize_transcript(text: str) -> str:
    """Repair speech-to-text's guesses at the brand name.

    Runs on every transcript before it reaches the model, so "Tell me about
    MacPay Kitchens" becomes a question about Magppie instead of a refusal.
    """
    if not text:
        return text
    fixed, n = _MISHEARD_BRAND.subn("Magppie", text)
    if n:
        log.info("stt: repaired %d misheard brand name(s) -> %r", n, fixed)
    return fixed

# Keyed by target_language_code; only entries matching the synthesis language
# are applied. Keys are matched against the text the model wrote, so cover the
# spellings it actually produces — including how it transliterates into
# Devanagari, where it will not write the Latin "Magppie" at all.
PRONUNCIATIONS: dict[str, dict[str, str]] = {
    "en-IN": {
        # "Magpie" is the ordinary English word for the bird, so Bulbul already
        # says it mag-pai. Borrow that rather than invent a respelling.
        "Magppie": "Magpie",
        "magppie": "Magpie",
        "MAGPPIE": "Magpie",
    },
    "hi-IN": {
        "Magppie": "मैगपाई",
        "मैगपी": "मैगपाई",    # "mag-pee" — what A7 wrongly specifies
        "मैग्पी": "मैगपाई",
        "मैगप्पी": "मैगपाई",   # double-p carried over from the spelling
    },
}


def _payload() -> bytes:
    # ensure_ascii=False or the Devanagari ships as \uXXXX escapes.
    return json.dumps({"pronunciations": PRONUNCIATIONS}, ensure_ascii=False, indent=2).encode("utf-8")


def _cache_file(cache_dir: Path) -> Path:
    digest = hashlib.sha256(_payload()).hexdigest()[:16]
    return cache_dir / f"pronunciation_{digest}.id"


async def ensure_dict_id(cache_dir: Path) -> str | None:
    """Upload the dictionary once and remember its id.

    Cached on disk against a hash of the dictionary itself, so editing
    PRONUNCIATIONS re-uploads and nothing else does. Returns None if it can't be
    created — a mispronounced brand name is not worth failing a call over.
    """
    cache = _cache_file(cache_dir)
    if cache.exists():
        dict_id = cache.read_text(encoding="utf-8").strip()
        if dict_id:
            log.info("pronunciation: using cached dictionary %s", dict_id)
            return dict_id

    if settings.missing_keys():
        return None

    try:
        client = AsyncSarvamAI(api_subscription_key=settings.sarvam_api_key)
        resp = await client.pronunciation_dictionary.create(
            file=("pronunciations.json", io.BytesIO(_payload()), "application/json")
        )
        dict_id = resp.dictionary_id
    except Exception as exc:
        log.warning(
            "pronunciation: could not create dictionary (%s) — "
            "the brand name may be mispronounced, but calls still work",
            exc,
        )
        return None

    cache.write_text(dict_id, encoding="utf-8")
    words = sum(len(v) for v in PRONUNCIATIONS.values())
    log.info("pronunciation: created dictionary %s (%d words)", dict_id, words)
    return dict_id
