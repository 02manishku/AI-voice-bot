"""Per-caller memory: Shubh remembers people across phone calls.

Exotel hands us the caller's number in the `start` event, which gives every
phone call a stable identity — so there is no reason for Shubh to greet a
returning customer like a stranger (team feedback 2026-09-02: "the bot has no
memory, bro").

How it works, and what it costs:
  - AT CALL START: one tiny local JSON read keyed by the number's last 10
    digits (<1ms). If the caller is known, their notes ride into the prompt as
    a per-call system note (the same `context` slot LEAD CALL mode uses), and
    their language from last time is pre-pinned — a returning English speaker
    gets English from turn one. Adds ~150 prompt tokens; nothing else touches
    the call path.
  - AT CALL END: fire-and-forget. One cheap LLM call rewrites the caller's
    notes from (old notes + this call's transcript) and saves the file. The
    call is already over, so this costs the caller nothing.

Failure is always soft: no file, bad JSON, a dead LLM — the call simply runs
memory-less, exactly like before.
"""

import asyncio
import json
import logging
import re
import time

from openai import AsyncOpenAI

from app.config import PROJECT_ROOT, settings

log = logging.getLogger(__name__)

CALLERS_DIR = PROJECT_ROOT / ".cache" / "callers"

# Notes are capped so a chatty history can't bloat the prompt (or the file).
MAX_NOTES_CHARS = 900
MAX_TRANSCRIPT_CHARS = 6000

_SUMMARY_PROMPT = (
    "You maintain a sales consultant's private notes about one customer of "
    "Magppie Wellness Kitchens, updated after each phone call.\n"
    "EXISTING NOTES (may be empty):\n{notes}\n\n"
    "TRANSCRIPT OF THE CALL THAT JUST ENDED:\n{transcript}\n\n"
    "Rewrite the notes: merge what is still true with what this call added.\n"
    "The notes are about THE PERSON, never about the products. The consultant "
    "already knows his own catalog — NEVER store prices, ranges, specs or any "
    "product fact the consultant quoted. Store only what the CALLER revealed:\n"
    "- name, city, language they spoke\n"
    "- their budget, kitchen/wardrobe size, which range or design THEY liked\n"
    "- their situation: new home or renovation, timeline, stage (exploring / "
    "comparing / ready to buy)\n"
    "- what they asked about, objections they raised, what was promised to them\n"
    "- anything personal they shared (family, profession, past bad experience)\n"
    "Every one of those the transcript contains MUST appear in the notes — "
    "losing the caller's budget or city is the one unforgivable error. Ignore "
    "garbled fragments. Plain text, no headings, at most 120 words. There is "
    "always something to write — at minimum what they asked and the language "
    "they spoke. Reply with the notes only."
)


# asyncio only holds WEAK references to tasks — a fire-and-forget task with no
# other reference can vanish mid-execution. Every post-call update is parked
# here until it finishes.
_PENDING: set = set()


def schedule_update(number: str, history: list[dict], language: str | None) -> None:
    """Fire-and-forget the post-call notes rewrite, safely."""
    task = asyncio.get_event_loop().create_task(update_after_call(number, history, language))
    _PENDING.add(task)
    task.add_done_callback(_PENDING.discard)


def _path(number: str):
    digits = re.sub(r"\D", "", number or "")[-10:]
    if len(digits) < 7:  # masked/absent caller id — no stable identity
        return None
    return CALLERS_DIR / f"{digits}.json"


def load(number: str) -> dict | None:
    """The caller's memory file, or None. Local disk only — sub-millisecond."""
    if not settings.caller_memory_enabled:
        return None
    path = _path(number)
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data.get("notes") else None
    except Exception as exc:
        log.warning("caller memory: unreadable file for %s (%s)", path.name, exc)
        return None


def context_block(mem: dict) -> str:
    """The system note that makes Shubh remember this caller."""
    when = mem.get("last_call_at_text") or "recently"
    return (
        "=== RETURNING CALLER — YOU HAVE SPOKEN BEFORE ===\n"
        f"This number has called {mem.get('calls', 1)} time(s) before; the last "
        f"call was {when}. Your private notes from those calls:\n"
        f"{mem['notes']}\n\n"
        "Use this the way a real consultant uses a client file: recognise them "
        "warmly in your first reply if natural (by name if you know it), pick "
        "up where you left off, and never re-ask what the notes already answer. "
        "Do NOT volunteer the notes unprompted, do not mention 'notes' or "
        "'records' — BUT if they directly ask what you remember about them "
        "('what is my name?', 'what was my budget?', 'which city am I from?'), "
        "answer plainly and warmly with the fact. Remembering out loud when "
        "ASKED is good service; only unprompted recitation is off-putting. If "
        "they seem to be a different person on the same number, just follow "
        "the conversation and trust what they say now."
    )


def remembered_language(mem: dict) -> str | None:
    lang = mem.get("language")
    return lang if lang in ("hi-IN", "en-IN") else None


def _transcript(history: list[dict]) -> str:
    lines = []
    for m in history:
        role = "Caller" if m.get("role") == "user" else "Shubh"
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)[-MAX_TRANSCRIPT_CHARS:]


async def update_after_call(number: str, history: list[dict], language: str | None) -> None:
    """Rewrite and save this caller's notes. Runs AFTER the call — never on
    the latency path. Any failure is logged and swallowed."""
    if not settings.caller_memory_enabled:
        return
    path = _path(number)
    if path is None or len(history) < 3:  # greeting + at least one exchange
        return
    log.info("caller memory: rewriting notes for %s", path.stem[-4:].rjust(4, "*"))
    try:
        old = load(number) or {}
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        resp = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[{
                "role": "user",
                "content": _SUMMARY_PROMPT.format(
                    notes=old.get("notes", "(none)"),
                    transcript=_transcript(history)
                    + (f"\n\n(System note: the caller's detected language was {language}.)"
                       if language else ""),
                ),
            }],
            max_tokens=220,
            temperature=0.2,
        )
        notes = (resp.choices[0].message.content or "").strip()[:MAX_NOTES_CHARS]
        if not notes or notes.lower().strip("()[].,'\" ") in {"none", "nothing", "n/a"}:
            return  # nothing worth remembering — leave any existing notes alone
        CALLERS_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "notes": notes,
            "language": language or old.get("language"),
            "calls": int(old.get("calls", 0)) + 1,
            "last_call_at": time.time(),
            "last_call_at_text": time.strftime("%d %b %Y"),
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        log.info("caller memory: saved notes for %s (call #%d)", path.stem[-4:].rjust(4, "*"), data["calls"])
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.warning("caller memory: update failed (%s) — next call runs without it", exc)
