# Magppie voice

Browser push-to-talk voice agent. Hold the button, ask in English or Hinglish,
hear an answer grounded strictly in the Magppie knowledge base.

```
[Browser] hold-to-talk -> 16kHz mono WAV
    -> POST /api/turn                      (streams NDJSON events back)
        -> Sarvam Saaras v3 (STT)   -> transcript + language_code
             <- {"type":"transcript", ...}                    ~0.5s
        -> OpenAI gpt-4o-mini       -> grounded answer + citations  [KB is in the system prompt]
             |  first sentence goes to Bulbul as soon as it exists,
             |  while the model is still writing the rest
        -> Sarvam Bulbul v3 (TTS)   -> WAV per chunk, synthesized concurrently
             <- {"type":"audio","index":0, ...}               ~3.7s  <- playback starts here
             <- {"type":"audio","index":1, ...}
             <- {"type":"done", answer, citations, timings}
```

The browser plays chunk 0 while chunk 1 is still being synthesized. Pre-flight
failures (missing keys, empty KB, silent recording) are plain HTTP errors;
anything failing mid-stream arrives as an `{"type":"error"}` event, because the
status line is long gone by then.

**Latency.** ~6s per turn, ~3.7s to first sound. TTS costs roughly 700ms fixed +
25ms/character, so answers are capped at 2 sentences / 35 words — that cap is a
latency and cost decision, not a style one. Getting to ~1s needs streaming STT
and TTS over WebSockets, which is the telephony build, not this demo.

## Run

```bash
cp .env.example .env      # fill in SARVAM_API_KEY and OPENAI_API_KEY
# drop the Magppie KB files into kb_source/
uv run uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 — `getUserMedia` needs HTTPS or localhost.

## Check the KB before trusting an answer

Startup prints every file, its extracted char count, and the total token count.
Same data at **`GET /api/kb/debug`**. If a file isn't listed with a sane char
count, fix that before believing anything the bot says.

A scanned PDF extracts to nothing, raises no exception, and leaves a
healthy-looking server that answers "I don't know" to everything. So startup
**crashes** if any file yields under 200 chars, naming the file. If that's a
scanned PDF, the fix is Sarvam Vision or local `pytesseract` OCR — the code
deliberately doesn't pick one for you.

`GET /api/debug/last-upload.wav` plays back the last WAV the browser sent. If
that sounds wrong, everything downstream is lying to you.

## Tests

All three run offline — no API keys, no credits spent.

```bash
uv run python tests/test_prose_stream.py   # partial-JSON prose extraction
uv run python tests/test_kb_guards.py      # PDF trap + KB guards
uv run python tests/test_turn_stream.py    # /api/turn event contract, providers stubbed
```

## Notes for this machine

TLS here is intercepted (corporate proxy / AV), so the trusted root lives in the
Windows cert store rather than in certifi. Two places handle it, and both are
required — without them every call fails with `CERTIFICATE_VERIFY_FAILED`, which
looks exactly like a bad API key:

- `uv.toml` sets `system-certs = true` for `uv` itself.
- `app/__init__.py` calls `truststore.inject_into_ssl()` for the Sarvam and
  OpenAI SDKs.

## Layout

```
app/
  main.py     FastAPI: static/, POST /api/turn, GET /api/health, GET /api/kb/debug
  config.py   pydantic-settings; every env var typed
  stt.py      Saaras v3   -> Transcript(text, language_code)
  llm.py      OpenAI streaming -> GroundedAnswer(text, citations)
  tts.py      Bulbul v3   -> WAV bytes (decodes base64) + language routing
  kb.py       load_kb() — glob, extract, concatenate, guard. No retrieval.
  prompts.py  system prompt + grounding rules
kb_source/    drop KB files here (.md / .txt / .json / .pdf)
static/       index.html, app.js, pcm-worklet.js
```

There is no retrieval layer, by design: the whole KB goes in the system prompt,
static content first so OpenAI's prefix cache hits. Anything variable
(language, history, the question) goes in `messages`, never the system prompt.
