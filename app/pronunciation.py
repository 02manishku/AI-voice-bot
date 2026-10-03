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
        # From the 2026-08-31 real call: "मैप पे की किचन", "मैक भाई की प्राइसिंग",
        # "मैक पाई", "मैप पाइ के किचन". "मैप पे" alone could genuinely mean "on
        # the map" ("showroom मैप पे भेज दो"), so it only counts when what
        # follows makes it possessive — की/का/के/किचन/वाल.
        | मैक \s* पाई | मैक \s* भाई | मैप \s* पा[ईइय]
        | मैप \s* पे (?= \s* (?: की | का | के | किचन | वाल ) )
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

# --- unit abbreviations and year counts: what the model writes vs what Shubh SAYS
#
# The model abbreviates units on its own ("25 yrs की guarantee", "per sq. ft.")
# even though the KB spells them out — and Bulbul then spells the abbreviation
# letter by letter. A real caller literally asked "What is 25 YEAR?" on a team
# test call (2026-08-31). Expand them to full words just before synthesis; this
# runs on every TTS path (streaming head/tail, REST, greeting). Keep the list
# to abbreviations that only ever mean the unit — same precision rule as the
# misheard-brand list above.
_UNIT_REWRITES: list[tuple[re.Pattern, object]] = [
    # "25 yrs" / "25-yr" / "25yr" -> "25 years" (singular for exactly 1). The
    # dot after "yrs." stays: at sentence end it IS the sentence period.
    (
        re.compile(r"(?<!\w)(\d+)\s*[-–]?\s*yrs?\b", re.IGNORECASE),
        lambda m: f"{m.group(1)} {'year' if m.group(1) == '1' else 'years'}",
    ),
    # a bare "yrs" with no number still reads as the unit
    (re.compile(r"(?<!\w)yrs\b", re.IGNORECASE), "years"),
    # "25-year guarantee" -> "25 year guarantee". Bulbul's preprocessor takes a
    # digit-hyphen-letters token as an alphanumeric code and reads it out
    # character by character — the team heard "two five Y-A-R guarantee" on a
    # call (2026-09-03). The KB wrote "25-year" everywhere, so the model did too.
    (re.compile(r"(?<!\w)(\d+)\s*[-–]\s*(years?|साल|वर्ष)", re.IGNORECASE), r"\1 \2"),
    # "25+ years" -> "25 plus years" (the "+" is silent or spelled otherwise)
    (re.compile(r"(?<!\w)(\d+)\s*\+\s*(?=(?:years?|साल|वर्ष))", re.IGNORECASE), r"\1 plus "),
    # "sq. ft." / "sq ft" / "sq.ft" / "sqft" -> "square feet"
    (re.compile(r"(?<!\w)sq\.?\s*ft\b\.?", re.IGNORECASE), "square feet"),
    (re.compile(r"(?<!\w)sqft\b\.?", re.IGNORECASE), "square feet"),
    (re.compile(r"(?<!\w)sq\.?\s*feet\b", re.IGNORECASE), "square feet"),
]

# Year counts are then written out in WORDS in the synthesis language. Digits
# are right everywhere else (prices, sizes, phone numbers — Bulbul reads them
# correctly and the prompt asks for them), but a count of years is the one
# number every guarantee answer turns on, so it gets the deterministic
# treatment: "twenty five years" / "पच्चीस साल". Nothing left for the engine to
# guess. Only 1-3 digit counts directly before a year word are touched.
_EN_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_EN_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_HI_WORDS = (
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस",
    "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस",
    "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस",
    "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस", "चालीस",
    "इकतालीस", "बयालीस", "तैंतालीस", "चौवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास", "पचास",
    "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ", "साठ",
    "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर", "सत्तर",
    "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी", "अस्सी",
    "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अठासी", "नवासी", "नब्बे",
    "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे", "सौ",
)
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# A 1-3 digit count right before a year word. Not a piece of a bigger number
# ("1,300 years", "2.5 years") and not one end of a range ("2-3 years") — those
# stay digits. The unit itself is kept exactly as the model wrote it.
_YEAR_COUNT = re.compile(
    r"(?<![\w,.\-–])(\d{1,3})(?=\s+(?:plus\s+)?(?:years?|साल|वर्ष)(?![A-Za-z]))",
    re.IGNORECASE,
)


def _en_number(n: int) -> str | None:
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _EN_TENS[tens] + ("" if ones == 0 else " " + _EN_ONES[ones])
    if n == 100:
        return "one hundred"
    return None


def _hi_number(n: int) -> str | None:
    return _HI_WORDS[n] if n <= 100 else None


def _spell_year_counts(text: str, lang: str | None) -> str:
    if lang == "hi-IN" and _DEVANAGARI.search(text):
        words = _hi_number
    elif lang in (None, "en-IN", "hi-IN"):
        # hi-IN with no Devanagari at all is an English sentence on a Hindi call
        words = _en_number
    else:
        return text  # Marathi, Gujarati, ...: leave Bulbul's own number reading alone

    def repl(m: re.Match) -> str:
        spelled = words(int(m.group(1)))
        return spelled if spelled is not None else m.group(0)

    return _YEAR_COUNT.sub(repl, text)


def normalize_speech_text(text: str, lang: str | None = None) -> str:
    """Expand unit abbreviations and spell out year counts, so TTS speaks
    words — never letter salad. `lang` is the synthesis language (Bulbul code)."""
    if not text:
        return text
    for pattern, repl in _UNIT_REWRITES:
        text = pattern.sub(repl, text)
    return _spell_year_counts(text, lang)


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
