"""OpenAI, streaming. Grounded answer + structured citations.

The model returns JSON. `stream_answer` yields prose deltas as they arrive by
reading the partially-complete "answer" string out of the buffer, so the §10
"send sentence 1 to TTS immediately" lever is a swap, not a rewrite. v1 just
drains it.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import AsyncIterator

from openai import AsyncOpenAI
from sarvamai import AsyncSarvamAI

from app.config import settings
from app.prompts import build_system_prompt, build_user_message

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass
class Citation:
    source: str
    page: int | None = None


@dataclass
class GroundedAnswer:
    text: str = ""
    citations: list[Citation] = field(default_factory=list)
    end_call: bool = False  # the caller wants to hang up — say bye once, then stop


# Corroborates the model's end_call flag against the caller's own words. A
# smaller model (gpt-4.1-nano) sometimes flags end_call while merely declining
# an off-topic question ("what is a MacBook?"), which would wrongly hang up the
# call. We only honour a hang-up when the caller actually signalled they're done
# — a pure safety guard that can only PREVENT false hang-ups, never cause one.
_EXIT_INTENT = re.compile(
    r"""(?ix)
      \b(bye|goodbye|ta-?ta|alvida)\b
    | \bsee\s+you\b | \bthat'?s\s+(all|it)\b | \bno\s+thanks?\b
    | \bnot\s+interested\b | \bstop\s+(calling|it|now)\b | \bhang\s+up\b
    | (remove|delete)\s+my\s+number | (don'?t|do\s+not|mat)\s+call
    | rakh(ta|ti)\s+h(oo|u)?n | baat\s+nah(i|in) | nah(i|in)\s+kar(ni|na)
    | band\s+kar | call\s+kaat | kaat\s+(do|de|dijiye) | bas\s+kar
    | (chhod|chod)(o|iye|na) | \bjaane?\s+do\b | reh?ne\s+(do|de)
    | अलविदा | बाय | बात\s*नहीं | नहीं\s*कर | बंद\s*कर
    | रख(ता|ती)\s*हूँ | बस\s*कर | काट\s*(दो|दीजिए|दे) | छोड़ | रहने\s*(दे|दो)
    """
)


def _looks_like_exit(text: str) -> bool:
    return bool(_EXIT_INTENT.search(text or ""))


# "answer" is first on purpose: it generates first, so prose streams before
# citations do.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The spoken reply. 2-3 sentences, plain prose, "
            "in the user's language. No markdown, no filenames.",
        },
        "citations": {
            "type": "array",
            "description": "Source of every claim. Empty if the KB "
            "does not cover the question.",
            "items": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Exact filename from a === SOURCE: === delimiter.",
                    },
                    "page": {
                        "type": ["integer", "null"],
                        "description": "Page number for PDFs, null otherwise.",
                    },
                },
                "required": ["source", "page"],
                "additionalProperties": False,
            },
        },
        "end_call": {
            "type": "boolean",
            "description": "True ONLY when the CALLER clearly wants to end the "
            "call — a goodbye, 'not interested', 'stop', 'call kaat do', or they "
            "have plainly stopped engaging. When true, 'answer' is a short warm "
            "sign-off. False for every normal turn, including questions and small "
            "talk. CRITICAL: declining an off-topic question is NOT ending the "
            "call — end_call stays false when you refuse or redirect. You never "
            "hang up on your own; only the caller ends the call.",
        },
    },
    "required": ["answer", "citations", "end_call"],
    "additionalProperties": False,
}

_ANSWER_KEY = re.compile(r'"answer"\s*:\s*"')
_ESCAPES = {
    '"': '"', "\\": "\\", "/": "/",
    "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t",
}


def _prose_so_far(buf: str) -> str:
    """Decode as much of the JSON "answer" string as has arrived.

    Returns "" until the key shows up. Stops cleanly on a truncated escape so a
    half-arrived \\uXXXX never emits a broken character.
    """
    m = _ANSWER_KEY.search(buf)
    if not m:
        return ""

    out: list[str] = []
    i = m.end()
    while i < len(buf):
        c = buf[i]
        if c == '"':
            break  # string closed
        if c != "\\":
            out.append(c)
            i += 1
            continue

        if i + 1 >= len(buf):
            break  # escape still in flight
        esc = buf[i + 1]
        if esc == "u":
            if i + 6 > len(buf):
                break  # \uXXXX incomplete
            try:
                out.append(chr(int(buf[i + 2 : i + 6], 16)))
            except ValueError:
                break
            i += 6
            continue
        if esc not in _ESCAPES:
            break
        out.append(_ESCAPES[esc])
        i += 2

    return "".join(out)


@lru_cache(maxsize=1)
def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key)


# ---- Sarvam path -------------------------------------------------------------
# sarvam-30b has no strict-JSON mode and is a reasoning model, so the OpenAI
# JSON approach doesn't fit. Instead: plain-text answer (streams straight to
# TTS, no '{"answer":"' prefix to wait through), with end_call carried by a
# sentinel appended to the very end and held back so it is never spoken.

@lru_cache(maxsize=1)
def _sarvam_client() -> AsyncSarvamAI:
    return AsyncSarvamAI(api_subscription_key=settings.sarvam_api_key)


_END_SENTINEL = "<<BYE>>"
# Hold back this many trailing chars while streaming so a half-formed sentinel
# (or the whitespace before it) is never spoken. Slack above the token length.
_HOLD = len(_END_SENTINEL) + 2

_SARVAM_OUTPUT_OVERRIDE = (
    "\n\n=== OUTPUT FORMAT (this overrides any earlier instruction about JSON) ===\n"
    "Reply with ONLY the spoken answer, as plain text — no JSON, no braces, no "
    "field names, no citations, no filenames. Just what Shubh says out loud.\n"
    f"If, and ONLY if, the caller wants to end the call, append the exact token "
    f"{_END_SENTINEL} at the very end of your reply. Never say it aloud or use it "
    "in any other situation."
)

_SOURCE_RE = re.compile(r"=== SOURCE: (.+?)(?: \| page \d+)? ===")


def _kb_sources(kb_text: str) -> list[Citation]:
    """The KB files, in first-seen order — used to cite a grounded answer.

    Sarvam replies in plain text, so there are no per-claim citations like the
    OpenAI path produces. With a small KB this is honest enough: a substantive
    answer is grounded in these files. Refusals get [] (handled by the caller).
    """
    seen: list[str] = []
    for m in _SOURCE_RE.finditer(kb_text):
        name = m.group(1).strip()
        if name not in seen:
            seen.append(name)
    return [Citation(source=n, page=None) for n in seen]


async def _stream_sarvam(
    kb_text: str,
    question: str,
    history: list[dict],
    language_code: str,
    into: GroundedAnswer | None,
    context: str | None = None,
) -> AsyncIterator[str]:
    system = build_system_prompt(kb_text) + _SARVAM_OUTPUT_OVERRIDE
    messages = [{"role": "system", "content": system}]
    if context:
        messages.append({"role": "system", "content": context})
    messages.extend(history)
    messages.append({"role": "user", "content": build_user_message(question, language_code)})

    stream = await _sarvam_client().chat.completions(
        model=settings.sarvam_llm_model,
        messages=messages,
        stream=True,
        temperature=0.2,
        max_tokens=800,  # room for hidden reasoning + the answer
        reasoning_effort=settings.sarvam_reasoning_effort,
    )

    buf = ""
    emitted = 0
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # reasoning_content is the model's hidden thinking — never speak it.
        piece = getattr(delta, "content", None)
        if not piece:
            continue
        buf += piece
        safe = len(buf) - _HOLD  # keep a tail back in case a sentinel is forming
        if safe > emitted:
            yield buf[emitted:safe]
            emitted = safe

    # Stream done: pull the sentinel off the end, flush whatever's left.
    text = buf.rstrip()
    end_call = text.endswith(_END_SENTINEL)
    if end_call:
        text = text[: -len(_END_SENTINEL)].rstrip()
    if len(text) > emitted:
        yield text[emitted:]

    if into is not None:
        into.text = text.strip()
        into.end_call = end_call
        into.citations = _kb_sources(kb_text) if into.text else []


def _messages(kb_text: str, question: str, history, language_code: str, context: str | None = None):
    # Static prefix first (KB + rules), variable content last, or prefix
    # caching never hits.
    msgs = [{"role": "system", "content": build_system_prompt(kb_text)}]
    # A per-call note (e.g. the CRM lead in an outbound call) goes in its own
    # system message AFTER the big cached one — so the ~10k-token KB prefix still
    # caches, only this short note is uncached, and it's pinned for the whole call
    # rather than ageing out of the trimmed history.
    if context:
        msgs.append({"role": "system", "content": context})
    msgs.extend(history)
    msgs.append({"role": "user", "content": build_user_message(question, language_code)})
    return msgs


async def _stream_deltas(
    kb_text: str,
    question: str,
    history: list[dict],
    language_code: str,
    raw: list[str] | None = None,
    context: str | None = None,
) -> AsyncIterator[str]:
    """Yield prose deltas; append every raw JSON delta to `raw` if given."""
    stream = await _client().chat.completions.create(
        model=settings.openai_model,
        messages=_messages(kb_text, question, history, language_code, context),
        stream=True,
        temperature=0.2,
        # A hard ceiling on runaway answers (answer + citations + end_call JSON).
        # ~30 words is the target; 200 tokens leaves room without truncating.
        max_tokens=200,
        # The KB is ~10k tokens on every turn. Report whether the static prefix
        # is actually being cached — if it isn't, the prompt ordering is broken
        # and we're paying full price and full latency for it every time.
        stream_options={"include_usage": True},
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "grounded_answer",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            },
        },
    )

    buf = ""
    emitted = 0
    async for chunk in stream:
        if chunk.usage:
            cached = getattr(chunk.usage.prompt_tokens_details, "cached_tokens", 0) or 0
            total = chunk.usage.prompt_tokens or 0
            log.info(
                "llm: prompt=%d cached=%d (%d%%) output=%d",
                total,
                cached,
                round(100 * cached / total) if total else 0,
                chunk.usage.completion_tokens or 0,
            )
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if not delta:
            continue
        if raw is not None:
            raw.append(delta)
        buf += delta
        prose = _prose_so_far(buf)
        if len(prose) > emitted:
            yield prose[emitted:]
            emitted = len(prose)


def _parse_into(buf: str, prose: str, into: GroundedAnswer) -> GroundedAnswer:
    """Fill `into` from a completed JSON buffer, falling back to streamed prose."""
    try:
        data = json.loads(buf)
    except json.JSONDecodeError:
        # Streamed prose is still good even if the JSON tail got cut off.
        if prose:
            log.warning("LLM JSON did not parse; using streamed prose without citations")
            into.text = prose
            return into
        raise LLMError("LLM returned unparseable output.") from None

    into.text = (data.get("answer") or prose).strip()
    into.end_call = bool(data.get("end_call"))

    # The model happily cites the same file once per claim. With a single-file
    # KB that renders as "kb.md, kb.md". Dedupe, keeping first-seen order.
    seen: set[tuple[str, int | None]] = set()
    citations: list[Citation] = []
    for c in data.get("citations") or []:
        source = c.get("source")
        if not source:
            continue
        key = (source, c.get("page"))
        if key in seen:
            continue
        seen.add(key)
        citations.append(Citation(source=source, page=c.get("page")))
    into.citations = citations
    return into


async def stream_answer(
    kb_text: str,
    question: str,
    history: list[dict],
    language_code: str,
    into: GroundedAnswer | None = None,
    context: str | None = None,
) -> AsyncIterator[str]:
    """Yield prose deltas as they arrive.

    Pass `into` to have the full text and citations written to it once the
    stream completes — lets a caller start speaking sentence one while the
    model is still writing the citations.

    Pass `context` to pin a per-call system note (e.g. the CRM lead on an
    outbound call) that rides along every turn without ageing out of history.

    Provider is chosen by settings.llm_provider. Both branches expose the same
    contract: yield prose deltas, fill `into` at the end.
    """
    if settings.llm_provider == "sarvam":
        async for delta in _stream_sarvam(kb_text, question, history, language_code, into, context):
            yield delta
    else:
        raw: list[str] = []
        chunks: list[str] = []
        async for delta in _stream_deltas(kb_text, question, history, language_code, raw, context):
            chunks.append(delta)
            yield delta
        if into is not None:
            _parse_into("".join(raw), "".join(chunks).strip(), into)

    # Only hang up if the caller's own words back it up (guards any model's misfire).
    if into is not None and into.end_call and not _looks_like_exit(question):
        log.info("end_call suppressed — no caller exit intent in %r", question[:50])
        into.end_call = False


async def answer(
    kb_text: str,
    question: str,
    history: list[dict],
    language_code: str,
    context: str | None = None,
) -> GroundedAnswer:
    """Drain the stream and parse out prose + citations."""
    result = GroundedAnswer()
    async for _ in stream_answer(kb_text, question, history, language_code, into=result, context=context):
        pass
    return result
