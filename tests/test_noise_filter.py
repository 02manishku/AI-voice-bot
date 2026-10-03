"""The junk-turn filter and the spoken-unit normalizer, born from the first
team test round on real Exotel calls (2026-08-31 logs).

The filter must be precise, not greedy: dropping a real answer ("हाँ" to "भेज
दूँ?") is worse than letting a "hmm" through, so the keep-cases get tested
harder than the drop-cases — same philosophy as test_brand_repair.

Run: uv run python tests/test_noise_filter.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.call_ws import (
    _echo_overlap,
    _is_bare_ack,
    _is_garble_turn,
    _is_noise_turn,
    _is_short_fragment,
    _looks_unfinished,
)
from app.pronunciation import normalize_speech_text, normalize_transcript

failures = []


def check(cond: bool, note: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + note)
    if not cond:
        failures.append(note)


# --- dropped ALWAYS: non-lexical fillers, from the real call logs -------------
for text in [
    "Hmm", "hm", "Umm", "uh", "So...", "तो", "हम्म.", "हुम हुम हुम हुम",
    "ம்.", "ഉം.", "হুম হুম", "ਹੂੰ", "A", "अ", "Hmm, umm", "so",
]:
    check(_is_noise_turn(text, False), f"idle: drop {text!r}")
    check(_is_noise_turn(text, True), f"speaking: drop {text!r}")

# --- backchannels: dropped while Shubh speaks, ANSWERS when he is idle --------
for text in ["Okay", "yeah", "हाँ", "ठीक है", "ਅੱਛਾ।", "अच्छा", "ok ok", "जी",
             "হুম ঠিক আছে", "Yeah yeah", "hmm okay"]:
    check(_is_noise_turn(text, True), f"speaking: drop backchannel {text!r}")
    check(not _is_noise_turn(text, False), f"idle: KEEP {text!r} (may answer a question)")

# --- substantive turns must ALWAYS pass, in both states -----------------------
for text in [
    "What is your starting price?",
    "ये वुडन किचन से कैसे अच्छा है?",
    "No",                      # a real answer, never junk
    "नहीं",
    "Wait wait",               # a real interruption
    "रुको",
    "बस",                      # "enough/stop" — has standalone meaning
    "Bye",
    "धन्यवाद",
    "Okay, but the price?",    # backchannel + substance = substance
    "हाँ बताओ ना",
    "Two design options",
    "15 lakhs is not a small amount.",
]:
    check(not _is_noise_turn(text, False), f"idle: KEEP {text!r}")
    check(not _is_noise_turn(text, True), f"speaking: KEEP {text!r}")

# --- spoken-unit normalizer ---------------------------------------------------
# Year counts come out in WORDS (the 2026-09-03 "two five Y-A-R guarantee" call):
# the synthesis language picks the words, the unit stays exactly as written.
for raw, lang, want in [
    ("25 yrs की guarantee", "hi-IN", "पच्चीस years की guarantee"),
    ("It lasts 25 yrs.", "en-IN", "It lasts twenty five years."),
    ("a 25-yr unconditional guarantee", "en-IN", "a twenty five years unconditional guarantee"),
    ("1 yr of service", "en-IN", "one year of service"),
    ("guarantee is for yrs together", "en-IN", "guarantee is for years together"),
    # the hyphenated form the KB used to write — Bulbul spelled it out
    ("a 25-year unconditional guarantee", "en-IN", "a twenty five year unconditional guarantee"),
    ("we offer a 25-year warranty on the cabinets", None, "we offer a twenty five year warranty on the cabinets"),
    ("Twenty-five years, sir. Ten years on the hardware.", "en-IN", "Twenty-five years, sir. Ten years on the hardware."),
    ("10 years on the hardware, 2 years on lighting.", "en-IN", "ten years on the hardware, two years on lighting."),
    ("It's 5 years on Wellness First.", "en-IN", "It's five years on Wellness First."),
    ("over 50 years", "en-IN", "over fifty years"),
    ("20+ years of kitchens", "en-IN", "twenty plus years of kitchens"),
    ("25 years of trust", "en-IN", "twenty five years of trust"),
    # Hindi call: a Devanagari sentence gets Hindi number words, unit untouched
    ("25 साल की guarantee", "hi-IN", "पच्चीस साल की guarantee"),
    ("stone पे 25-साल की guarantee", "hi-IN", "stone पे पच्चीस साल की guarantee"),
    ("5 साल की guarantee, और 5 complimentary services", "hi-IN", "पाँच साल की guarantee, और 5 complimentary services"),
    ("10 साल hardware पे, 2 साल lighting पे।", "hi-IN", "दस साल hardware पे, दो साल lighting पे।"),
    ("पिछले 20 सालों से", "hi-IN", "पिछले बीस सालों से"),
    # an English sentence on a Hindi call gets English words
    ("The guarantee is 25 years.", "hi-IN", "The guarantee is twenty five years."),
    ("Rs. 5,900 per sq. ft.", "en-IN", "Rs. 5,900 per square feet"),
    ("9,943 per sq ft, cabinetry only", "en-IN", "9,943 per square feet, cabinetry only"),
    ("around 500 rupees per sqft", "en-IN", "around 500 rupees per square feet"),
    ("120 sq.ft. kitchen", "en-IN", "120 square feet kitchen"),
]:
    got = normalize_speech_text(raw, lang)
    check(got == want, f"normalize {raw!r} -> {got!r}" + ("" if got == want else f" (want {want!r})"))

# --- normalizer must NOT touch these ------------------------------------------
for text, lang in [
    ("years", "en-IN"),
    ("Mr. Sharma", "en-IN"),
    ("square feet", "en-IN"),
    ("she says yrsomething", "en-IN"),   # embedded, not a word
    ("5,900 rupees", "en-IN"),
    ("a 10x10 kitchen", "en-IN"),
    ("baked at 1300 degrees", "en-IN"),
    ("installed in 2016", "en-IN"),
    ("1,300 years of stone", "en-IN"),   # part of a bigger number — digits stay
    ("2.5 years", "en-IN"),
    ("2-3 years", "en-IN"),              # a range — left to Bulbul
    ("3 BHK", "hi-IN"),
    ("25 वर्ष", "mr-IN"),                 # Marathi: Hindi words would be wrong, digits stay
    ("25 years", "ta-IN"),
]:
    got = normalize_speech_text(text, lang)
    check(got == text, f"untouched {text!r}" + ("" if got == text else f" -> {got!r}"))

# --- fillers/backchannels added after the 2026-08-31 real call ----------------
for text in ["और।"]:
    check(_is_noise_turn(text, False), f"idle: drop {text!r}")
for text in ["ठीक है।", "ਠੀਕ ਹੈ।", "হ্যাঁ।"]:
    check(_is_noise_turn(text, True), f"speaking: drop backchannel {text!r}")
    check(not _is_noise_turn(text, False), f"idle: KEEP {text!r}")

# --- self-echo filter ---------------------------------------------------------
SHUBH_SAID = (
    "जी बिल्कुल! हमारे पास काफी ऑप्शंस होते हैं — स्टोन में हमारे पास आपके "
    "Magppie में 40 से ज़्यादा finishes मिलती हैं। Wellness First और Wellness "
    "Pro दोनों में stone doors आते हैं।"
)
# near-verbatim echo (the smoking gun from the real call) must be caught
for echo in [
    "काफी ऑप्शंस होते हैं, तो स्टोन में हमारे पास आपके Magppie मे",
    "Wellness First और Wellness Pro दोनों में stone doors आते हैं",
]:
    check(_echo_overlap(echo, SHUBH_SAID) >= 0.75, f"echo caught: {echo[:40]!r}")
# the caller quoting one phrase inside their OWN question must pass
for real in [
    "आपने बोला 25 साल की गारंटी? सच में?",
    "stone doors मतलब क्या हुआ भाई समझाओ",
    "What is the price of Wellness Pro?",
    "अच्छा briefly बताओ फिर से क्या क्या मिलता है",
]:
    check(_echo_overlap(real, SHUBH_SAID) < 0.75, f"not echo: {real[:40]!r}")
# too short to judge = never echo (junk filter owns short fragments)
check(_echo_overlap("काफी ऑप्शंस", SHUBH_SAID) == 0.0, "short fragment never echo-matched")

# --- unfinished-turn detection ------------------------------------------------
for text in [
    "तो आप मुझे वैसे थोड़ा सा टेंटेटिवली बता सकते हैं कि",
    "और इसमें स्टोन की जो",
    "बेसिकली किस चीज में डील करता है, वो है क्या और",
    "पर",
    "Okay, but",
    "I wanted to ask about the price and",
    "मुझे एक बात बताओ,",
]:
    check(_looks_unfinished(text), f"unfinished: {text[:40]!r}")
for text in [
    "What is your starting price?",
    "इसकी गारंटी क्या होती है आपके Magppie के किचन की?",
    "ठीक है, समझ गया।",
    "I think so.",
    "What is that about?",
    "मुझे वुडन किचन चाहिए",
    "Which city are you in?",       # ends on a preposition BUT is a question
    "Thank you",
    "About Kitchens",
    "Bye bye",
]:
    check(not _looks_unfinished(text), f"finished: {text[:40]!r}")

# --- 2026-09-03: dangling prepositions / object-less verbs ---------------------
for text in [
    "I want to inquire",
    "New connection to",
    "I wanted to ask",
    "Do you have a showroom in",
    "kitchen के बारे",
    "मुझे Magppie से",
]:
    check(_looks_unfinished(text), f"unfinished (dangling): {text!r}")

# --- short unpunctuated fragments: held briefly for the rest -------------------
for text in ["Hi", "About Kitchens", "Thank you", "Bye bye", "Wellness Pro price"]:
    check(_is_short_fragment(text), f"short fragment (brief hold): {text!r}")
for text in ["Hi.", "What is the price?", "I want a kitchen for my flat", "ठीक है।", "Yes!"]:
    check(not _is_short_fragment(text), f"not a short fragment: {text!r}")

# --- bare-ack detection (the "Okay"-then-continue churn) ----------------------
for text in ["Okay", "हाँ", "ok ok", "ठीक है"]:
    check(_is_bare_ack(text), f"bare ack: {text!r}")
for text in ["Okay, send it", "हाँ बताओ", "yes please do that"]:
    check(not _is_bare_ack(text), f"not bare ack: {text!r}")

# --- garble-language gate (background-speaker bleed, 2026-09-02 calls) --------
for text, lang in [
    ("ਤੋਲ ਦੇ ਮਾਇਆ।", "pa-IN"),
    ("ও আর কি?", "bn-IN"),
    ("ಆಂಗದ್ ಮತೀನ್ ಹೋದ ನಂತರ", "kn-IN"),
    ("ഇത് നോക്ക്", "ml-IN"),
]:
    check(_is_garble_turn(text, lang, True), f"garble dropped: {text[:25]!r} ({lang})")
for text, lang, note in [
    ("ਯਾਰ ਮੇਰਾ ਬਜਟ ਤਾਂ ਦਸ ਹਜ਼ਾਰ ਰੁਪਏ ਦਾ ਹੈ।", "pa-IN", "real Punjabi sentence, 6+ tokens"),
    ("What is your price?", "en-IN", "trusted language"),
    ("क्या रेट है?", "bn-IN", "Devanagari mis-tagged as Bengali = real Hindi"),
    ("ਤੋਲ ਦੇ ਮਾਇਆ।", "pa-IN", "FIRST turn of a call — gate off") ,
]:
    established = note != "FIRST turn of a call — gate off"
    check(not _is_garble_turn(text, lang, established), f"kept ({note}): {text[:30]!r}")

# --- brand repairs from the 2026-08-31 call ----------------------------------
for raw, note in [
    ("सर, मुझे मैप पे की किचन लगवानी है", "map pe + possessive"),
    ("प्राइसिंग मैक भाई की प्राइसिंग", "mac bhai"),
    ("मैक पाई किचन के बारे में", "mac pai"),
    ("मैप पाइ के किचन की गारंटी", "map pai short-i"),
]:
    got = normalize_transcript(raw)
    check("Magppie" in got, f"brand repaired ({note}): {got[:45]!r}")
for raw, note in [
    ("showroom मैप पे भेज दो", "'on the map' must survive"),
    ("लोकेशन मैप पे दिखा दो", "'show on map' must survive"),
]:
    got = normalize_transcript(raw)
    check("Magppie" not in got, f"untouched ({note}): {got[:45]!r}")

print()
if failures:
    print(f"{len(failures)} FAILURES")
    sys.exit(1)
print("ALL PASSED")
sys.exit(0)
