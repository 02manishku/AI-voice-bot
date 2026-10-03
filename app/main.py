"""FastAPI: serves static/, POST /api/turn, GET /api/health, GET /api/kb/debug."""

import asyncio
import base64
import hashlib
import json
import logging
import re
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sarvamai.core.api_error import ApiError

from app import kb as kb_mod
from app import call_ws, llm, prompts, pronunciation, stt, tts, zoho
from app.config import PROJECT_ROOT, settings
from app.limits import RateLimiter

# The Windows console is cp1252, which cannot encode Devanagari — and this app
# is entirely about Devanagari. A KB file named फिनिश.pdf would otherwise take
# down startup inside the logger. errors="replace" keeps a stray glyph from
# ever being fatal.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)-12s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("magppie")

STATIC_DIR = PROJECT_ROOT / "static"
DEBUG_DIR = PROJECT_ROOT / ".debug"
CACHE_DIR = PROJECT_ROOT / ".cache"
LAST_UPLOAD = DEBUG_DIR / "last_upload.wav"

GREETING_WAV: bytes | None = None
# Spoken when the rate limiter refuses a turn on a PHONE call. Before this, a
# tripped limiter meant the greeting played and then dead silence — the whole
# team read it as "the bot is broken" (2026-09-03). Rendered once, disk-cached;
# playing it costs no Sarvam credits, which is the limiter's whole point.
BUSY_WAV: bytes | None = None
BUSY_LINE = (
    "I'm really sorry, our lines are very busy right now. "
    "कृपया थोड़ी देर बाद दोबारा कॉल करें। Thank you!"
)
# Personalised (per-lead) greeting audio, keyed by a hash of the exact text.
# Same lead on a repeat call => instant, no re-synthesis. In-memory only.
PERSONAL_GREETINGS: dict[str, bytes] = {}

# The lead a "Lead Call" greeting fetched, handed to the matching /ws/call session
# so both use the SAME record (no second fetch, no race with a newer lead). Keyed
# by the browser's session_id; short-lived — a greeting is followed by its call
# within seconds. One-shot: the call takes it and it's gone.
LEAD_TTL_SECONDS = 300
LEAD_BY_SESSION: dict[str, tuple[float, "zoho.Lead"]] = {}
limiter = RateLimiter(settings.max_turns_per_minute, settings.max_turns_per_day)


def _stash_lead(session_id: str, lead: "zoho.Lead") -> None:
    now = time.monotonic()
    for k in [k for k, (t, _) in LEAD_BY_SESSION.items() if now - t > LEAD_TTL_SECONDS]:
        LEAD_BY_SESSION.pop(k, None)  # sweep anything the call never claimed
    LEAD_BY_SESSION[session_id] = (now, lead)


def take_session_lead(session_id: str) -> "zoho.Lead | None":
    """Claim (and remove) the lead the greeting stashed for this session."""
    entry = LEAD_BY_SESSION.pop(session_id or "", None)
    if not entry:
        return None
    when, lead = entry
    return lead if time.monotonic() - when <= LEAD_TTL_SECONDS else None

# The call's working memory: how many past messages ride along in the prompt.
# In-memory only; dies with the process. Configurable — and cheap, because the
# growing history is append-only and sits in OpenAI's cached prefix (see config).
MAX_HISTORY_MESSAGES = settings.max_history_messages

# Sentence splitting lives in tts.py now — both the REST path here and the
# streaming path use the same rule for "what's a chunk worth speaking yet".
_split_first_sentence = tts.split_first_sentence

KB = kb_mod.KB()
SESSIONS: dict[str, list[dict]] = {}

# The reply language, sticky per session. Saaras guesses a language from each
# turn, but a one-word "okay" or "achha" gives it almost nothing — it returned
# en-IN for "okay" (so Shubh re-answered in English) and bn-IN for "achha
# achha achha" (so he replied in Bengali). Anchor to the language the caller has
# actually been speaking; only switch on a substantial turn in Hindi or English.
SESSION_LANG: dict[str, str] = {}
# One substantial foreign-language turn is NOT enough to switch an established
# session — Saaras tags Hinglish sentences as en-IN, and a single English
# question in a Hindi call flipped Shubh to English (real caller complained,
# 2026-08-31: "आपने अपनी भाषा को हिंदी से अंग्रेजी क्यों कर लिया अचानक से?").
# This remembers the one candidate turn; a second consecutive one confirms it.
SESSION_LANG_PENDING: dict[str, str] = {}
_REPLY_LANGUAGES = {"hi-IN", "en-IN"}

# Any Devanagari LETTER is a definitive Hindi signal — stronger than Saaras's
# language_code, which mis-tags short Devanagari turns as Bengali/Marathi.
# Letters only: the danda "।" (U+0964) sits in the Devanagari block but ends
# sentences in every Indic script — an Odia fragment 'ହଁ, ଦେଖି ସାରିଲି।' pinned an
# English caller's call to Hindi through its danda (2026-09-03 call).
_DEVANAGARI = re.compile(r"[ऀ-ॣ०-ॿ]")

# An explicit request to switch language, spoken in EITHER language. Saaras tags
# "tell me that in Hindi" as en-IN (it's English words), so the ask is invisible
# to language_code — we read the request from the words themselves. We only speak
# Hindi and English, so a request for any Hindi-family language maps to Hindi.
_ASK_HINDI = re.compile(r"(?i)(हिंदी|हिन्दी|\bhindi\b|भोजपुरी|bhojpuri|\bhinglish\b|हिंग्लिश)")
_ASK_ENGLISH = re.compile(r"(?i)(अंग्रे|इंग्लिश|\benglish\b|angre[jz])")


def _requested_language(text: str) -> str | None:
    """A caller explicitly asking to be answered in a language, or None."""
    if not text:
        return None
    wants_hi = bool(_ASK_HINDI.search(text))
    wants_en = bool(_ASK_ENGLISH.search(text))
    if wants_hi and not wants_en:
        return "hi-IN"
    if wants_en and not wants_hi:
        return "en-IN"
    if wants_hi and wants_en:
        # "in Hindi, not English" / "Hindi or Bhojpuri" — they named a target and
        # a foil; the target they want is almost always the non-English one.
        return "hi-IN"
    return None

# Said when the model returns nothing (a rare blip), so the caller hears a human
# re-ask instead of dead air / a "…" that never resolves.
_FALLBACK = {
    "hi-IN": "माफ कीजिए भाई, ठीक से सुनाई नहीं दिया — दोबारा बोलेंगे?",
    "en-IN": "Sorry, I didn't catch that — could you say it again?",
}

# nano sometimes answers an abusive/dead-end turn with just an ellipsis or a
# stray dash — TTS speaks near-silence and the caller hears nothing. Anything
# that is only punctuation/whitespace counts as no answer, so the fallback fires.
_MEANINGLESS = re.compile(r"^[\s.…·•,;:!?\-–—_'\"()।॥]*$")


def _is_empty_answer(text: str) -> bool:
    return not text or bool(_MEANINGLESS.match(text.strip()))


def resolve_language(session_id: str, text: str, detected: str | None) -> str:
    key = session_id or "default"
    established = SESSION_LANG.get(key)

    # An explicit "answer me in X" outranks everything, including stickiness —
    # this is the one thing that must always be honoured, even mid-English.
    requested = _requested_language(text)
    if requested:
        SESSION_LANG[key] = requested
        SESSION_LANG_PENDING.pop(key, None)
        return requested

    if _DEVANAGARI.search(text):
        candidate, substantial = "hi-IN", True   # Devanagari present = Hindi signal
        # …but for ESTABLISHING a fresh session, script alone is not enough:
        # a one-word garble ('अपने।', noise while the greeting played) pinned a
        # real English call to Hindi (2026-09-02). Pinning needs real words.
        weighty = len(text.split()) >= 3
    else:
        candidate = detected if detected in _REPLY_LANGUAGES else None
        substantial = weighty = len(text.split()) >= 4 or len(text) >= 15

    if established is None:
        # English is the baseline: open in English unless the caller's first
        # REAL sentence is clearly in another language we speak. A short first
        # turn establishes nothing — answer it in the greeting's language and
        # let the caller's first proper sentence decide.
        if weighty and candidate:
            SESSION_LANG[key] = candidate
            return candidate
        return "en-IN"

    # SYMMETRIC two-turn hysteresis. One turn must never flip an established
    # call in EITHER direction: a lone English line flipped a Hindi call
    # (2026-08-31 complaint), and a lone Devanagari GARBLE — background noise
    # transcribed as Marathi 'हे पूर्ण आहे ते.' — flipped an English call to
    # Hindi (2026-09-02 complaint). Only two consecutive substantial turns in
    # the other language switch. Short/uncertain turns advance nothing AND
    # break nothing — a fragmented speaker ("Okay", "down") must still be able
    # to complete the two-turn escape.
    if substantial and candidate:
        if candidate == established:
            SESSION_LANG_PENDING.pop(key, None)  # back on the call's language
        elif SESSION_LANG_PENDING.get(key) == candidate:
            SESSION_LANG_PENDING.pop(key, None)
            SESSION_LANG[key] = candidate
            log.info("language: switched %s -> %s (second consecutive turn)", established, candidate)
            return candidate
        else:
            SESSION_LANG_PENDING[key] = candidate
            log.info("language: %s turn in a %s call — staying, one more confirms", candidate, established)
    return established


async def _load_greeting() -> bytes | None:
    """Render the greeting once and cache it on disk.

    Keyed by the text and every voice setting that affects the audio, so
    changing the speaker or the wording re-renders, and nothing else does.
    Disk-backed because --reload restarts constantly and each render costs real
    TTS characters.
    """
    key = "|".join(
        [
            prompts.GREETING,
            prompts.GREETING_LANGUAGE,
            settings.sarvam_tts_model,
            settings.sarvam_tts_speaker,
            str(settings.sarvam_tts_pace),
            str(settings.sarvam_tts_pace_greeting),
            str(settings.sarvam_tts_loudness),
            # The greeting says "Magppie", so the pronunciation dictionary
            # changes the audio. Without this the cache would serve the old
            # mispronounced take forever.
            tts.DICT_ID or "no-dict",
        ]
    )
    cached = CACHE_DIR / f"greeting_{hashlib.sha256(key.encode()).hexdigest()[:16]}.wav"

    if cached.exists():
        log.info("greeting: loaded from cache (%s)", cached.name)
        return cached.read_bytes()

    if settings.missing_keys():
        return None

    try:
        # The intro plays at its own (slower) pace so "Magppie" registers —
        # team feedback 2026-08-31: the greeting flew by too fast to place.
        wav = await tts.synthesize(
            prompts.GREETING, prompts.GREETING_LANGUAGE, pace=settings.sarvam_tts_pace_greeting
        )
    except Exception as exc:
        # Never fatal: the call can still open, just without an instant greeting.
        log.warning("greeting: could not pre-render (%s) — will retry on request", exc)
        return None

    cached.write_bytes(wav)
    log.info("greeting: rendered and cached (%s, %d bytes)", cached.name, len(wav))
    return wav


async def _load_busy_line() -> bytes | None:
    """Render the rate-limit apology once and cache it on disk — same pattern
    as the greeting. Soft: without it, a tripped limiter is silent again."""
    key = "|".join(
        [
            BUSY_LINE,
            settings.sarvam_tts_model,
            settings.sarvam_tts_speaker,
            str(settings.sarvam_tts_pace),
            tts.DICT_ID or "no-dict",
        ]
    )
    cached = CACHE_DIR / f"busyline_{hashlib.sha256(key.encode()).hexdigest()[:16]}.wav"
    if cached.exists():
        return cached.read_bytes()
    if settings.missing_keys():
        return None
    try:
        wav = await tts.synthesize(BUSY_LINE, "en-IN")
    except Exception as exc:
        log.warning("busy line: could not pre-render (%s)", exc)
        return None
    cached.write_bytes(wav)
    log.info("busy line: rendered and cached (%s, %d bytes)", cached.name, len(wav))
    return wav


# Short "thinking beats" (see call_ws._thinking_beat), pre-rendered per reply
# language and kept as raw PCM at tts.TTS_SAMPLE_RATE so either leg plays them
# as-is. Plain, neutral sounds a consultant makes before answering — nothing
# that presumes the answer ("Yes.", "Great question.").
FILLER_LINES: dict[str, list[str]] = {
    "en-IN": ["Hmm.", "Right.", "Okay.", "Sure."],
    "hi-IN": ["जी।", "हम्म।", "अच्छा।", "ठीक है।"],
}
FILLER_CLIPS: dict[str, list[bytes]] = {}


def filler_clips(language: str) -> list[bytes]:
    return FILLER_CLIPS.get(language) or FILLER_CLIPS.get("en-IN") or []


def _wav_to_pcm(wav_bytes: bytes, rate: int) -> bytes:
    """Mono 16-bit PCM at `rate` from a Bulbul WAV."""
    import audioop
    import io
    import wave

    with wave.open(io.BytesIO(wav_bytes)) as w:
        src = w.getframerate()
        frames = w.readframes(w.getnframes())
        if w.getsampwidth() != 2:
            frames = audioop.lin2lin(frames, w.getsampwidth(), 2)
        if w.getnchannels() == 2:
            frames = audioop.tomono(frames, 2, 0.5, 0.5)
    if src != rate:
        frames, _ = audioop.ratecv(frames, 2, 1, src, rate, None)
    return frames


async def _load_fillers() -> dict[str, list[bytes]]:
    """Render the thinking beats once and cache them on disk, like the greeting.
    Soft: a beat that fails to render is simply not in the rotation."""
    out: dict[str, list[bytes]] = {}
    if settings.missing_keys():
        return out
    for lang, lines in FILLER_LINES.items():
        clips: list[bytes] = []
        for line in lines:
            key = "|".join(
                [line, lang, settings.sarvam_tts_model, settings.sarvam_tts_speaker,
                 str(settings.sarvam_tts_pace), tts.DICT_ID or "no-dict"]
            )
            cached = CACHE_DIR / f"filler_{hashlib.sha256(key.encode()).hexdigest()[:16]}.wav"
            try:
                if cached.exists():
                    wav = cached.read_bytes()
                else:
                    wav = await tts.synthesize(line, lang)
                    cached.write_bytes(wav)
                clips.append(_wav_to_pcm(wav, tts.TTS_SAMPLE_RATE))
            except Exception as exc:
                log.warning("filler: could not render %r (%s)", line, exc)
        out[lang] = clips
    log.info("filler: %d thinking beats ready", sum(len(v) for v in out.values()))
    return out


async def _personal_greeting_wav(text: str) -> bytes | None:
    """Synthesize a per-lead greeting, memoised by its exact text.

    The lead's name only changes the audio when the name changes, so keying on
    the text means a repeat call for the same lead is an instant cache hit and
    costs no TTS. Soft: any synthesis failure returns None and the caller drops
    to the generic greeting.
    """
    key = hashlib.sha256(
        f"{text}|{settings.sarvam_tts_loudness}|{settings.sarvam_tts_pace_greeting}".encode("utf-8")
    ).hexdigest()[:16]
    hit = PERSONAL_GREETINGS.get(key)
    if hit is not None:
        return hit
    try:
        wav = await tts.synthesize(
            text, prompts.GREETING_LANGUAGE, pace=settings.sarvam_tts_pace_greeting
        )
    except Exception as exc:
        log.warning("greeting: personalised synth failed (%s) — generic", exc)
        return None
    PERSONAL_GREETINGS[key] = wav
    return wav


async def _warm_lead_greeting() -> None:
    """Pre-fetch the freshest lead and pre-synthesize its greeting at startup, so
    the first personalised call is instant. Best-effort: never raises."""
    if not zoho.enabled():
        return
    try:
        lead = await zoho.latest_lead()
        if lead and lead.name:
            text = prompts.outbound_greeting(lead.first_name)
            if await _personal_greeting_wav(text):
                log.info("greeting: warmed outbound greeting for lead %s", lead.name)
    except Exception as exc:
        log.warning("greeting: lead warm-up skipped (%s)", exc)


async def _prewarm_llm() -> None:
    """Prime OpenAI's prompt cache with the ~10k-token KB prefix at startup.

    The first turn after any restart is otherwise cold: cached=0, ~11s, because
    the whole KB is processed from scratch. One throwaway call now populates the
    cached prefix, so the caller's real first turn is warm (~1s). Best-effort and
    fire-and-forget — never blocks startup, never fatal.
    """
    if settings.missing_keys() or KB.is_empty:
        return
    try:
        t0 = time.perf_counter()
        await llm.answer(KB.text, "hello", [], prompts.GREETING_LANGUAGE)
        log.info("llm: prompt cache warmed in %dms — first real turn will be warm",
                 int((time.perf_counter() - t0) * 1000))
    except Exception as exc:
        log.warning("llm: prewarm skipped (%s)", exc)


def prewarm_llm_if_stale() -> None:
    """Fire a background prewarm if the prompt cache has likely expired.

    OpenAI's cache prefix lives ~5-10 min. A call that starts after the server
    idled longer than that would pay the cold prefix on the caller's FIRST
    question (+1-2.5s at exactly the worst moment). The greeting runs seconds
    before that first question, so warming here hides the whole cost inside the
    greeting playback. No-op when the cache is already warm, so back-to-back
    calls cost nothing extra.
    """
    if llm.cache_is_warm():
        return
    log.info("llm: cache likely cold — prewarming behind the greeting")
    asyncio.create_task(_prewarm_llm())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global KB, GREETING_WAV, BUSY_WAV, FILLER_CLIPS
    # Any KBError here is fatal by design: a half-empty KB looks healthy and
    # answers "I don't know" to everything.
    KB = kb_mod.load_kb()
    kb_mod.log_kb(KB)

    missing = settings.missing_keys()
    if missing:
        log.warning("Missing env vars: %s — /api/turn will fail until set", ", ".join(missing))

    DEBUG_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    # Must precede the greeting: it says "Magppie", and the greeting's cache key
    # includes the dict id.
    tts.DICT_ID = await pronunciation.ensure_dict_id(CACHE_DIR)

    # Pre-rendered so turn one is instant (§10) and costs nothing per call.
    GREETING_WAV = await _load_greeting()
    BUSY_WAV = await _load_busy_line()
    if settings.reply_filler:
        FILLER_CLIPS = await _load_fillers()

    # If Zoho is wired up, warm the freshest lead's greeting now so the first
    # personalised call is instant too. Soft — a Zoho hiccup never blocks startup.
    if zoho.enabled():
        log.info("zoho: enabled — greeting will open by lead name")
        await _warm_lead_greeting()

    # Warm the LLM prompt cache in the background so the first real turn isn't the
    # ~11s cold one. Fire-and-forget: startup returns immediately; the warm-up
    # finishes within a couple of seconds, before the caller can dial.
    asyncio.create_task(_prewarm_llm())

    yield


app = FastAPI(title="Magppie voice demo", lifespan=lifespan)


def _fail(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail=message)


def _readable_sarvam_error(exc: ApiError, stage: str) -> HTTPException:
    codes = {
        403: "Sarvam rejected the API key (403). Check SARVAM_API_KEY.",
        429: "Sarvam rate limit hit (429). Wait a moment and try again.",
        422: f"Sarvam rejected the {stage} request (422) — bad audio, or over the limit.",
        503: "Sarvam is overloaded (503). Try again in a few seconds.",
    }
    msg = codes.get(exc.status_code or 0, f"Sarvam {stage} failed ({exc.status_code}).")
    return _fail(502, msg)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    # The UI renders `error` as text — never leave it hanging on a spinner.
    # Keep exc.headers: Retry-After on a 429 is the useful half of the answer.
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
        headers=getattr(exc, "headers", None),
    )


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "kb_files": len(KB.files),
        "kb_tokens": KB.tokens,
        "missing_env": settings.missing_keys(),
        "max_recording_seconds": settings.max_recording_seconds,
        "stt_streaming": settings.stt_streaming,
        "zoho": zoho.enabled(),
        "greeting_ready": GREETING_WAV is not None,
        "turns_today": limiter.spent_today(),
        "turns_per_day": settings.max_turns_per_day,
    }


@app.get("/api/kb/debug")
async def kb_debug():
    """Per-file char counts + total tokens. Confirm every file actually
    extracted before trusting a single answer."""
    return {
        "source_dir": str(settings.kb_path),
        "files": [
            {"name": f.name, "chars": f.chars, "pages": f.pages} for f in KB.files
        ],
        "total_chars": len(KB.text),
        "total_tokens": KB.tokens,
        "token_limit": settings.kb_token_limit,
        "empty": KB.is_empty,
    }


@app.post("/api/greeting")
async def greeting(request: Request):
    """The opening line, per call mode. Also resets the conversation: a new call
    starts with no history.

    mode "lead"      -> outbound: open by the freshest Zoho lead's name and stash
                        that lead for the matching /ws/call session, so Shubh
                        knows who he called for the whole conversation.
    mode "assistant" -> inbound (default): the generic help-desk greeting,
                        pre-rendered so it's instant and free.

    Body: {"mode": "lead"|"assistant", "session_id": "..."} — all optional; a
    missing/invalid body is treated as assistant mode. Any Zoho miss in lead mode
    silently falls back to the assistant greeting.
    """
    global GREETING_WAV

    # A greeting means a real turn is ~10s away — make sure it lands warm.
    prewarm_llm_if_stale()

    mode, session_id = "assistant", ""
    try:
        body = await request.json()
        if isinstance(body, dict):
            mode = (body.get("mode") or "assistant").strip().lower()
            session_id = str(body.get("session_id") or "")
    except Exception:
        pass  # no/invalid JSON body -> assistant mode

    # --- lead call: outbound open by freshest lead (best-effort, never blocks) --
    if mode == "lead" and zoho.enabled():
        lead = await zoho.latest_lead()
        if lead and lead.name:
            if session_id:
                _stash_lead(session_id, lead)  # the /ws/call session will claim it
            text = prompts.outbound_greeting(lead.first_name)
            wav = await _personal_greeting_wav(text)
            if wav is not None:
                return {
                    "text": text,
                    "language_code": prompts.GREETING_LANGUAGE,
                    "audio": base64.b64encode(wav).decode("ascii"),
                    "mode": "lead",
                    "lead": {
                        "name": lead.name,
                        "first_name": lead.first_name,
                        "city": lead.city,
                        "budget": lead.budget,
                        "timeline": lead.timeline,
                        "interest": lead.interest,
                    },
                }

    # --- assistant greeting (default, and the fallback for every lead miss) -----
    if GREETING_WAV is None:
        # Startup couldn't render it (keys arrived late, or Sarvam blipped).
        GREETING_WAV = await _load_greeting()
    if GREETING_WAV is None:
        raise _fail(503, "Couldn't load the greeting audio. Check SARVAM_API_KEY and restart.")

    return {
        "text": prompts.GREETING,
        "language_code": prompts.GREETING_LANGUAGE,
        "audio": base64.b64encode(GREETING_WAV).decode("ascii"),
        "mode": "assistant",
    }


@app.get("/api/lead/preview")
async def lead_preview():
    """The freshest Zoho lead's details, for the sidebar to show before a Lead
    Call — so you can see who Shubh is about to call and everything on their file.
    No audio, no side effects. Soft: {"enabled": false} or {"lead": null} on any
    miss, never a 500."""
    if not zoho.enabled():
        return {"enabled": False, "lead": None}
    lead = await zoho.latest_lead()
    if not lead or not lead.name:
        return {"enabled": True, "lead": None}
    return {
        "enabled": True,
        "lead": {
            "name": lead.name,
            "first_name": lead.first_name,
            "city": lead.city,
            "budget": lead.budget,
            "timeline": lead.timeline,
            "interest": lead.interest,
            "source": lead.source,
            "status": lead.status,
        },
    }


@app.get("/api/debug/last-upload.wav")
async def last_upload():
    """Whatever the browser sent last. If this doesn't play, the WAV is wrong
    and everything downstream is lying to you."""
    if not LAST_UPLOAD.exists():
        raise _fail(404, "No upload recorded yet.")
    return FileResponse(LAST_UPLOAD, media_type="audio/wav")


def _event(**payload) -> bytes:
    """One NDJSON line. Newline-delimited so the browser can parse incrementally."""
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


@app.post("/api/turn")
async def turn(request: Request, audio: UploadFile, session_id: str = Form(default="")):
    """Stream the turn as NDJSON events so the browser can start playing the
    first sentence while the rest is still being synthesized.

    Validation happens here, before the response starts, so these still surface
    as real HTTP errors. Anything that fails mid-stream arrives as an `error`
    event instead — the status line is long gone by then.
    """
    # Check before spending anything: STT, the LLM and TTS all cost money.
    caller = request.client.host if request.client else "unknown"
    verdict = limiter.check(f"{caller}:{session_id or 'default'}")
    if not verdict.allowed:
        raise HTTPException(
            status_code=429,
            detail=verdict.message,
            headers={"Retry-After": str(verdict.retry_after)},
        )

    if missing := settings.missing_keys():
        raise _fail(503, f"Server is missing {', '.join(missing)}. Set it in .env and restart.")
    if KB.is_empty:
        raise _fail(
            503,
            "The knowledge base is empty — drop the Magppie files into "
            f"{settings.kb_path.name}/ and restart the server.",
        )

    wav = await audio.read()
    if len(wav) < 1000:
        raise _fail(400, "That recording was empty. Hold the button and speak.")

    DEBUG_DIR.mkdir(exist_ok=True)
    LAST_UPLOAD.write_bytes(wav)

    return StreamingResponse(
        _turn_events(wav, session_id),
        media_type="application/x-ndjson",
        # Stop any proxy from buffering the whole body and undoing the point.
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


async def _turn_events(wav: bytes, session_id: str):
    t0 = time.perf_counter()
    timings: dict[str, int] = {}
    tts_tasks: list[asyncio.Task] = []

    def mark(stage: str, since: float) -> float:
        now = time.perf_counter()
        timings[f"{stage}_ms"] = int((now - since) * 1000)
        return now

    def since_start() -> int:
        return int((time.perf_counter() - t0) * 1000)

    try:
        # --- STT ---
        try:
            transcript = await stt.transcribe(wav)
        except ApiError as exc:
            yield _event(type="error", error=_readable_sarvam_error(exc, "speech-to-text").detail)
            return
        mark_from = mark("stt", t0)

        if not transcript.text:
            yield _event(type="error", error="Didn't catch that — nothing was transcribed. Try again.")
            return

        # No retrieval layer: the whole KB is already in the system prompt (§5).
        timings["retrieval_ms"] = 0
        language = resolve_language(session_id, transcript.text, transcript.language_code)
        history = SESSIONS.setdefault(session_id or "default", [])

        # Show the transcript now — it's ~4s until audio, don't sit on this.
        yield _event(
            type="transcript",
            transcript=transcript.text,
            language_code=language,
            spoken_language_code=tts.to_tts_language(language),
            stt_ms=timings["stt_ms"],
        )

        grounded = llm.GroundedAnswer()

        if settings.tts_streaming:
            # STREAMING: pipe the LLM's prose into the TTS socket, forward PCM as
            # it comes back. First audio lands in ~300ms instead of ~1800ms.
            llm_stream = llm.stream_answer(
                KB.text, transcript.text, history, language, into=grounded
            )
            n_chunks = 0
            try:
                async for pcm in tts.synthesize_stream(llm_stream, language):
                    if n_chunks == 0:
                        timings["first_audio_ms"] = since_start()
                    n_chunks += 1
                    yield _event(
                        type="audio_pcm",
                        pcm=base64.b64encode(pcm).decode("ascii"),
                        rate=tts.TTS_SAMPLE_RATE,
                    )
            except Exception as exc:
                # The streaming-TTS WebSocket blipped. This is the "…" bug: a
                # socket hiccup here used to throw all the way out and strand the
                # turn, even though the answer itself is perfectly good. Swallow
                # it — the n_chunks==0 branch below speaks the answer over the
                # reliable REST path. Only bail out if we'd already spoken part of
                # it, since REST would replay the opening on top of live audio.
                log.warning("streaming TTS failed (%s) — falling back to REST", exc)
                if n_chunks:
                    raise
            timings["tts_chunks"] = n_chunks

            if n_chunks == 0:
                # No audio reached the caller — the socket died early, or the
                # model said nothing. The socket can die before the LLM even
                # drained, so `grounded` may be empty; re-run over REST to be sure
                # we have the answer, then speak it. A slower turn beats a dead one.
                if not grounded.text.strip():
                    grounded = await llm.answer(KB.text, transcript.text, history, language)
                if _is_empty_answer(grounded.text):
                    log.warning("empty/meaningless answer from model — using fallback re-ask")
                    grounded.text = _FALLBACK.get(language, _FALLBACK["hi-IN"])
                    grounded.citations = []
                    grounded.end_call = False
                wav = await tts.synthesize(grounded.text, language)
                timings.setdefault("first_audio_ms", since_start())
                yield _event(type="audio", index=0, audio=base64.b64encode(wav).decode("ascii"))
        else:
            # REST fallback: first sentence + rest, synthesized concurrently.
            pending = ""
            async for delta in llm.stream_answer(
                KB.text, transcript.text, history, language, into=grounded
            ):
                pending += delta
                if not tts_tasks:
                    first, rest = _split_first_sentence(pending)
                    if first:
                        tts_tasks.append(asyncio.create_task(tts.synthesize(first, language)))
                        pending = rest
            mark_from = mark("llm", mark_from)

            if _is_empty_answer(grounded.text):
                log.warning("empty/meaningless answer from model — using fallback re-ask")
                grounded.text = _FALLBACK.get(language, _FALLBACK["hi-IN"])
                grounded.citations = []
                grounded.end_call = False
                pending = grounded.text

            if pending.strip():
                tts_tasks.append(asyncio.create_task(tts.synthesize(pending.strip(), language)))

            for i, task in enumerate(tts_tasks):
                chunk = await task
                if i == 0:
                    timings["first_audio_ms"] = since_start()
                yield _event(type="audio", index=i, audio=base64.b64encode(chunk).decode("ascii"))
            timings["tts_chunks"] = len(tts_tasks)

        mark("tts", mark_from)

        history.extend(
            [
                {"role": "user", "content": transcript.text},
                {"role": "assistant", "content": grounded.text},
            ]
        )
        del history[:-MAX_HISTORY_MESSAGES]

        timings["total_ms"] = since_start()
        log.info("turn: %s", " ".join(f"{k}={v}" for k, v in timings.items()))

        yield _event(
            type="done",
            answer=grounded.text,
            citations=[{"source": c.source, "page": c.page} for c in grounded.citations],
            end_call=grounded.end_call,
            timings=timings,
        )

    except ApiError as exc:
        yield _event(type="error", error=_readable_sarvam_error(exc, "text-to-speech").detail)
    except (llm.LLMError, tts.TTSError, stt.STTError) as exc:
        yield _event(type="error", error=str(exc))
    except Exception as exc:
        log.exception("turn failed")
        yield _event(type="error", error=f"The turn failed: {type(exc).__name__}")
    finally:
        for t in tts_tasks:
            if not t.done():
                t.cancel()


@app.websocket("/ws/call")
async def ws_call(websocket: WebSocket):
    """Phase 3 streaming call: mic in, Sarvam VAD turn-taking, audio out, barge-in.
    Gated by STT_STREAMING; the browser only dials this when health says it's on."""
    await call_ws.handle(websocket)


@app.websocket("/exotel/stream")
async def ws_exotel(websocket: WebSocket):
    """Exotel Voicebot Applet endpoint: a real phone call's audio, both ways.
    Configure the applet URL as wss://KEY:TOKEN@host/exotel/stream?sample-rate=16000."""
    from app import exotel_ws

    await exotel_ws.handle(websocket)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
