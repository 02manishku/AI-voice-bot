"""Typed settings. Every env var lands here; no os.getenv() anywhere else."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sarvam_api_key: str = ""
    openai_api_key: str = ""

    # Which model is Shubh's brain on the live call.
    #   "openai"  -> gpt-4o-mini. Reliable on this heavily-ruled prompt: enforces
    #               Magppie-only, follows the no-repeat / garble rules, 99%
    #               prompt-cached. The default, and the right choice today.
    #   "sarvam"  -> sarvam-30b/105b. Tested 2026-07-18 and NOT viable as the
    #               brain yet: 30b (a reasoning model) spirals into empty answers
    #               unpredictably; 105b is fast but leaks off-topic questions
    #               (answered "capital of France", did arithmetic, told jokes),
    #               breaking the Magppie-only guard. The code path is kept and
    #               config-switchable for the future (a non-reasoning Sarvam
    #               model, or once speculative streaming makes 30b affordable AND
    #               its spiral is solved). Sarvam still powers STT + TTS.
    llm_provider: str = "openai"

    # gpt-4.1-nano: ~45% faster to answer than gpt-4o-mini and ~33% cheaper,
    # and — with the end_call guard in llm.py — just as reliable on the
    # Magppie-only guard. Measured 2026-07-18. gpt-4o-mini remains a safe
    # fallback (set OPENAI_MODEL=gpt-4o-mini) if a future prompt trips nano up.
    openai_model: str = "gpt-4.1-nano"

    sarvam_llm_model: str = "sarvam-30b"
    # low/medium/high — a reasoning model can't be turned off, so use the
    # cheapest setting on the live call.
    sarvam_reasoning_effort: str = "low"

    sarvam_stt_model: str = "saaras:v3"
    sarvam_stt_mode: str = "transcribe"

    sarvam_tts_model: str = "bulbul:v3"
    sarvam_tts_speaker: str = "shubh"
    sarvam_tts_pace: float = 1.0
    # Kept only for the greeting cache keys. bulbul:v3 REJECTS a loudness param
    # (400: "currently not supported"), so output level is shaped downstream:
    # the phone leg applies exotel_tts_gain with a clip-proof limiter.
    sarvam_tts_loudness: float = 1.0
    # Hindi pace. 1.0 = Bulbul's natural delivery, unchanged — a 1.1 boost was
    # tried to level Hindi with English and the caller heard it as "too fast";
    # the speaker's own pacing wins. Knob kept (SARVAM_TTS_PACE_HI) in case a
    # future speaker needs levelling, but the default is hands-off.
    sarvam_tts_pace_hi: float = 1.0
    # The pre-built intro ONLY. Team feedback (2026-08-31): the greeting flew by
    # too fast for callers to register "this is Magppie". 0.9 = a touch slower,
    # medium pace — the intro lands, everything after speaks at normal pace.
    sarvam_tts_pace_greeting: float = 0.9

    # Stream TTS audio over a WebSocket (first audio ~300ms) instead of the REST
    # call that returns the whole clip (~1800ms). Set TTS_STREAMING=false to fall
    # straight back to the REST path — an instant rollback if streamed playback
    # ever misbehaves.
    tts_streaming: bool = True

    # Phase 3: stream the MIC to Sarvam's streaming STT over a persistent
    # WebSocket, and let Sarvam's own VAD decide when a turn ends — instead of
    # recording a whole clip, guessing the end with a client-side RMS threshold,
    # and uploading a file. Removes ~1.6s/turn (silence wait + upload), survives
    # background noise, and enables barge-in (interrupting Shubh mid-sentence).
    #
    # Default OFF: this rewires the microphone path, so the known-good POST /turn
    # flow stays live until the streaming call is confirmed by ear. Flip
    # STT_STREAMING=true to test; set it back to false for an instant rollback.
    stt_streaming: bool = False

    # Sarvam streaming-STT VAD knobs (strings — the API wants them as query
    # params). high sensitivity catches soft/quick speech; interrupt frames set
    # how much speech is needed to count as a barge-in — higher resists Shubh's
    # own audio leaking back through the mic and falsely interrupting him.
    stt_high_vad_sensitivity: bool = False
    stt_interrupt_min_speech_frames: int = 12
    # End-of-speech hangover: how many consecutive silence frames close the turn.
    # This window is pure dead air between the caller's last word and END_SPEECH —
    # the single largest piece of perceived latency. Measured live: 5 frames ≈
    # 340ms (so ~68ms/frame), vs ~770ms at the server default — but 340ms proved
    # TOO eager, splitting a natural mid-sentence pause into two turns. 7 ≈
    # ~480ms: still ~300ms faster than default, without jumping on a caller who
    # pauses to think. None = don't send, keep the server default.
    stt_negative_frames_count: int | None = 7

    # Exotel Voicebot stream — optional. When both are set, /exotel/stream
    # requires Basic auth (the applet URL carries wss://KEY:TOKEN@host and
    # Exotel forwards it as an Authorization header). Blank = auth off, for
    # local simulator testing only; the endpoint logs a warning per call.
    exotel_ws_key: str = ""
    exotel_ws_token: str = ""
    # RNNoise on the phone leg's inbound audio (app/denoise.py). Real calls come
    # from noisy rooms and speakerphones; the first team test round (2026-08-31)
    # showed background noise opening Sarvam's VAD constantly — junk turns,
    # barge-in storms cutting Shubh mid-reply, the rate limiter burning its
    # budget on noise. Costs ~0.8ms of CPU per 20ms chunk, no added latency.
    exotel_denoise: bool = True
    # Barge-in resistance for the PHONE leg only. The browser leg keeps
    # stt_interrupt_min_speech_frames (12); a phone in a noisy Indian street
    # needs sustained speech, not a horn blast, before Shubh shuts up. Real
    # interruptions ("wait wait", "नहीं नहीं रुको") are a second or more of
    # speech and still cut him within a beat.
    exotel_interrupt_min_speech_frames: int = 24
    # Loudness on the phone leg. Bulbul already peaks at full scale (-0 dBFS)
    # but with speech RMS around -14 dBFS — dynamic studio audio that reads as
    # QUIET on a phone earpiece (first real call, 2026-08-29). This gain is the
    # STARTING gain; a soft-knee clip keeps peaks from buzzing.
    exotel_tts_gain: float = 2.0
    # AGC: Bulbul's level also varies per synthesis call (head and tail of one
    # answer render on separate sockets; Hindi vs English differ too), which
    # callers heard as "volume keeps fluctuating" (2026-09-02 team round). The
    # phone leg now steers gain continuously toward this speech-RMS target
    # (11500 ≈ the level the team called comfortable). 0 = AGC off, fixed gain.
    exotel_tts_target_rms: int = 11500
    # Per-caller memory (app/caller_memory.py): Shubh remembers a phone number
    # across calls — notes rewritten by one cheap LLM call AFTER each call, one
    # local file read at pickup. False = every call is a stranger again.
    caller_memory_enabled: bool = True
    # The spoken "thinking beat". The model's first token alone is ~1.2s and
    # first audio ~1.7s after the transcript (measured 2026-09-03) — dead air a
    # person would fill with a "Hmm." / "जी।". If the answer's audio hasn't
    # started by reply_filler_after_ms, one short pre-rendered beat plays in
    # the reply's language, on reply_filler_chance of turns, never the same
    # beat twice running. REPLY_FILLER=false switches it off entirely.
    reply_filler: bool = True
    reply_filler_after_ms: int = 700
    reply_filler_chance: float = 0.75
    # Background-voice gate on the caller's audio. RNNoise removes NOISE but
    # passes any human speech — including people talking near the caller, which
    # became junk turns in random languages (2026-09-02 calls). Frames far
    # quieter than the caller's own running speech level get ducked before
    # Sarvam hears them. False = off.
    exotel_speech_gate: bool = True

    # Zoho CRM — optional. When these three secrets are set, the greeting opens
    # by the freshest lead's name ("Hi Rahul, ... your modular kitchen ..."). All
    # blank (the default) => Zoho is simply off and the generic greeting is used;
    # a fetch failure never blocks a call. Get the refresh token from a Zoho
    # Self Client with scope ZohoCRM.modules.READ (or .leads.READ).
    zoho_client_id: str = ""
    zoho_client_secret: str = ""
    zoho_refresh_token: str = ""
    # Data-centre endpoints. Default to the India DC (.in); use .com / .eu / .com.au
    # for other regions. accounts_domain issues tokens; api_domain serves records.
    zoho_accounts_domain: str = "https://accounts.zoho.in"
    zoho_api_domain: str = "https://www.zohoapis.in"
    # Which module and which field API names to read. Full_Name/Description are
    # Zoho's stock Leads fields; override if the "interested in" data lives in a
    # custom field (e.g. Kitchen_Interest, Product_Interest).
    zoho_module: str = "Leads"
    zoho_name_field: str = "Full_Name"
    zoho_interest_field: str = "Description"
    # The extra fields the outbound "Lead Call" mode reads so Shubh knows who he's
    # calling. Defaults are this account's real Leads field API names (discovered
    # 2026-07-24); override per deploy if yours differ.
    zoho_city_field: str = "City"
    zoho_budget_field: str = "Est_Budget"
    zoho_timeline_field: str = "How_Soon_Do_You_Require_Magppie"
    # Hard ceiling on the whole Zoho round trip so a slow CRM can't delay the
    # greeting — past this we fall back to the generic line.
    zoho_timeout: float = 4.0

    kb_source_dir: str = "kb_source"
    kb_token_limit: int = 100_000

    max_recording_seconds: int = 25

    # How many past messages (user + assistant) to keep in the prompt as the
    # conversation's working memory. 30 ~= 15 exchanges, enough to remember a
    # whole call — so Shubh tracks what the caller wants and never repeats a line
    # from earlier. Latency cost is ~nil: history is append-only, so like the KB
    # it sits in OpenAI's cached prefix; only the caller's newest turn is uncached.
    max_history_messages: int = 30

    # Rate limits. Every turn spends real credits (~Rs 0.6, mostly TTS), so
    # these exist to stop a stuck client or an idle open tab draining the
    # account. Per-caller stops one bad session; daily caps total exposure.
    # 20/min: real phone calls (2026-08-31 logs) legitimately split questions
    # into rapid fragments and hit the old 12 — which then silently dropped a
    # REAL question ("What is the silver stone?"). The junk-turn filter now
    # keeps noise from spending the budget; this cap only stops runaways.
    max_turns_per_minute: int = 20
    # 1000: sized for real phone-testing days, not browser demos — the team
    # burned the old 300 in one afternoon (2026-09-03) and every call after
    # played the greeting then went MUTE (the refusal had no voice). Worst-case
    # exposure at ~Rs 0.6/turn is ~Rs 600/day. .env carries the live override.
    max_turns_per_day: int = 1000

    @property
    def kb_path(self) -> Path:
        p = Path(self.kb_source_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def missing_keys(self) -> list[str]:
        # Sarvam is always needed (STT + TTS). OpenAI is only needed when it is
        # the conversation model — a Sarvam-only call must not be blocked on it.
        missing = []
        if not self.sarvam_api_key:
            missing.append("SARVAM_API_KEY")
        if self.llm_provider == "openai" and not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        return missing


settings = Settings()
