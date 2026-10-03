"""OpenAI, streaming. Grounded answer + structured citations.

The model returns JSON. `stream_answer` yields prose deltas as they arrive by
reading the partially-complete "answer" string out of the buffer, so the §10
"send sentence 1 to TTS immediately" lever is a swap, not a rewrite. v1 just
drains it.
"""

import asyncio
import json
import logging
import re
import time
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
    # "I'm done" phrasings — a real caller said "मैं और कुछ जानना नहीं चाहूंगा।
    # थैंक यू।" (2026-08-31) and the guard blocked the hang-up; he then said a
    # confused "Hello" and had to cut the call himself. The "और कुछ" anchor is
    # what keeps this safe: a mid-call "उसके बारे में नहीं जानना चाहूंगा, ये
    # बताएं..." has no "और कुछ" and still cannot hang up.
    | और\s*कुछ\s*(भी\s*)?(जानना|पूछना|सुनना)?\s*नह[ीि]
    | aur\s+kuch\s+(bhi\s+)?(jaan|pooch|sun)\S*\s+nah(i|in) | aur\s+kuch\s+nah(i|in)
    | बस\s*इतना\s*(ही|काफी) | bas\s+itna\s+(hi|k?aafi|kafi)
    | बस\s*हो\s*गया | bas\s+ho\s+gaya | बहुत\s*हो\s*गया | bah?ut\s+ho\s+gaya
    | \bnothing\s+else\b | \bno\s+more\s+questions?\b | \bi'?m\s+done\b
    | \bthat('?ll|\s+will)\s+be\s+all\b
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


# monotonic() of the last OpenAI call — lets main/call_ws decide whether the
# prompt cache (TTL ~5-10 min) is likely still warm before spending on a ping.
last_call_at: float = 0.0

# OpenAI routes requests to cache shards partly by this key. Without it,
# routing roulette occasionally lands a turn on a cold shard: measured live as
# the rare cached=0 turn that costs +1-2.5s. A constant key pins every call —
# prewarm included — to the same warm shard. (Priority tier was benchmarked
# 2026-08-17 for the same purpose and REJECTED: no faster, and switching tiers
# lost the warm shard entirely.)
PROMPT_CACHE_KEY = "magppie-kb"


def cache_is_warm(ttl: float = 240.0) -> bool:
    """True if an OpenAI call happened recently enough that the prompt-cache
    prefix is very likely still resident."""
    return last_call_at > 0 and (time.monotonic() - last_call_at) < ttl


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
    nudge: str | None = None,
) -> AsyncIterator[str]:
    system = build_system_prompt(kb_text) + _SARVAM_OUTPUT_OVERRIDE
    messages = [{"role": "system", "content": system}]
    if context:
        messages.append({"role": "system", "content": context})
    messages.extend(history)
    if nudge:
        messages.append({"role": "system", "content": nudge})
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


def _messages(
    kb_text: str,
    question: str,
    history,
    language_code: str,
    context: str | None = None,
    nudge: str | None = None,
):
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
    # A one-turn director's note (prompts.NUDGE_*) sits right next to the turn
    # it steers — adjacent beats buried, for a small model — and is never
    # stored in history, so it steers exactly one reply.
    if nudge:
        msgs.append({"role": "system", "content": nudge})
    msgs.append({"role": "user", "content": build_user_message(question, language_code)})
    return msgs


# Tail-latency hedge. The first token normally lands in ~1.0-1.3s, but OpenAI
# sometimes sits on a request for 3-7s (two such turns in one replayed call,
# 2026-09-03) — on a phone that is the difference between "quick" and "dead".
# If no first chunk has arrived by HEDGE_AFTER, an identical second request is
# raced against the first and whichever answers first is used; the other is
# closed. Costs one extra (99%-cached, cheap) call on the slow turns only.
HEDGE_AFTER = 1.8


async def _open(kwargs: dict):
    """Open one streaming request and wait for its FIRST chunk (that wait IS
    the time-to-first-token). Returns (stream, iterator, first_chunk); closes
    the stream if cancelled while waiting."""
    stream = await _client().chat.completions.create(**kwargs)
    try:
        ait = stream.__aiter__()
        first = await ait.__anext__()
    except BaseException:
        await stream.close()
        raise
    return stream, ait, first


async def _hedged_chunks(kwargs: dict):
    """Yield the chunks of whichever request produces a first token first."""
    t1 = asyncio.create_task(_open(kwargs))
    done, _ = await asyncio.wait({t1}, timeout=HEDGE_AFTER)
    if t1 in done and not t1.exception():
        winner, loser = t1, None
    else:
        if t1 in done:  # the first request FAILED outright — retry, don't race
            log.warning("llm: first request failed (%s) — retrying", t1.exception())
            t1 = asyncio.create_task(_open(kwargs))
            winner, loser = t1, None
            await asyncio.wait({t1})
        else:
            log.info("llm: no first token after %.1fs — hedging with a second request", HEDGE_AFTER)
            t2 = asyncio.create_task(_open(kwargs))
            done, _ = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
            winner = t1 if t1 in done and not t1.exception() else t2
            if winner not in done or winner.exception():
                # the one that finished first failed; fall back to the other
                winner = t2 if winner is t1 else t1
                await asyncio.wait({winner})
            loser = t2 if winner is t1 else t1
            log.info("llm: hedge — %s request won", "first" if winner is t1 else "second")
    if loser is not None:
        if loser.done() and not loser.cancelled() and not loser.exception():
            await loser.result()[0].close()
        else:
            loser.cancel()
    stream, ait, first = winner.result()
    try:
        yield first
        async for chunk in ait:
            yield chunk
    finally:
        await stream.close()


async def _stream_deltas(
    kb_text: str,
    question: str,
    history: list[dict],
    language_code: str,
    raw: list[str] | None = None,
    context: str | None = None,
    nudge: str | None = None,
) -> AsyncIterator[str]:
    """Yield prose deltas; append every raw JSON delta to `raw` if given."""
    global last_call_at
    last_call_at = time.monotonic()
    kwargs = dict(
        model=settings.openai_model,
        messages=_messages(kb_text, question, history, language_code, context, nudge),
        stream=True,
        # 0.6, not 0.2: at 0.2 the model is near-deterministic, so the same
        # question gets the same recited sentence every time — the #1 "it's a bot"
        # tell callers noticed. Structured output (json_schema, strict) holds the
        # format regardless. (0.7 + frequency_penalty 0.4 was tried and produced
        # occasional off-key word choices in Hindi — the penalty was steering the
        # model away from the domain's own natural words. Keep both gentle; the
        # press-again ladder in the prompt is what really prevents repeats.)
        temperature=0.6,
        frequency_penalty=0.15,
        # A hard ceiling on runaway answers (answer + citations + end_call JSON).
        # ~30 words is the target; 200 tokens leaves room without truncating.
        max_tokens=200,
        # The KB is ~10k tokens on every turn. Report whether the static prefix
        # is actually being cached — if it isn't, the prompt ordering is broken
        # and we're paying full price and full latency for it every time.
        stream_options={"include_usage": True},
        # Pin cache-shard routing (see PROMPT_CACHE_KEY above).
        extra_body={"prompt_cache_key": PROMPT_CACHE_KEY},
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
    async for chunk in _hedged_chunks(kwargs):
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
    nudge: str | None = None,
) -> AsyncIterator[str]:
    """Yield prose deltas as they arrive.

    Pass `into` to have the full text and citations written to it once the
    stream completes — lets a caller start speaking sentence one while the
    model is still writing the citations.

    Pass `context` to pin a per-call system note (e.g. the CRM lead on an
    outbound call) that rides along every turn without ageing out of history.

    Pass `nudge` for a one-turn director's note (prompts.NUDGE_*) about the
    moment — it steers this reply only and never enters history.

    Provider is chosen by settings.llm_provider. Both branches expose the same
    contract: yield prose deltas, fill `into` at the end.
    """
    if settings.llm_provider == "sarvam":
        async for delta in _stream_sarvam(
            kb_text, question, history, language_code, into, context, nudge
        ):
            yield delta
    else:
        raw: list[str] = []
        chunks: list[str] = []
        async for delta in _stream_deltas(
            kb_text, question, history, language_code, raw, context, nudge
        ):
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
    nudge: str | None = None,
) -> GroundedAnswer:
    """Drain the stream and parse out prose + citations."""
    result = GroundedAnswer()
    async for _ in stream_answer(
        kb_text, question, history, language_code, into=result, context=context, nudge=nudge
    ):
        pass
    return result
