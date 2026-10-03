# Exotel Voicebot Applet — spec digest + build plan

Source: "Working with the Stream and Voicebot Applet", Exotel Support Center, modified 2025-09-16
(support.exotel.com article 3000108630; PDF received from Exotel sales 2026-08-27).
Provider decision context: `telephony-provider-selection.md`. Protocol sibling: `ozonetel-integration.md`.

Verdict unchanged from the provider research: **adapter, not rewrite.** Everything hard (Sarvam VAD
turn-taking, transcript race, LLM→TTS overlap, barge-in logic, Zoho, memory) is transport-independent.
Exotel does NOT provide VAD or turn-taking — none needed: our turn-taking is Sarvam's server-side VAD
on our own STT stream, which works on any audio source.

---

## 1. What this doc settles (questions we'd been carrying)

1. **WS authentication — SOLVED** (OzoneTel's spec had none). Two supported options:
   - Basic auth: configure the applet URL as `wss://KEY:TOKEN@host/path` — Exotel strips the
     credentials and sends them as an `Authorization: Basic base64(KEY:TOKEN)` header.
   - IP whitelisting: mail hello@exotel.com for their outbound IP ranges.
   We'll do Basic auth (checked in the adapter's connection handler), IP allowlist later at the proxy.
2. **16 kHz confirmed, and it's their own best practice**: "Use 16 kHz for most voicebot
   integrations." Requested via query param on the applet URL: `?sample-rate=16000`. Default is 8k;
   24k is "HD". **At 16k, inbound audio matches our STT input rate exactly (STREAM_SAMPLE_RATE=16000)
   — zero inbound resampling.** Outbound: one resample, Bulbul 22050 → 16000.
3. **A simulator already exists** — github.com/exotel/voice-streaming ("Simulator to make streaming
   calls with dummy bot"), plus reference bots github.com/exotel/Agent-Stream and Agent-Stream-echobot.
   We don't need to build our own protocol simulator from scratch; validate against theirs and keep a
   thin pytest harness for regression.
4. **Dynamic per-call routing**: the applet URL may be an HTTPS endpoint that returns
   `{"url": "wss://..."}` per call — lets us route/inject per-call context server-side. Start static;
   the dynamic hook exists when we need it.
5. **Recording**: a checkbox on the applet; the recording URL surfaces in a **Passthru applet placed
   after the Voicebot applet** → wire that to a webhook for CRM logging later.
6. **No explicit stop needed**: for bidirectional streams, closing the WS (or call end) auto-ends the
   stream and the flow advances to the next applet. Flow design: Voicebot → Hangup (later: Voicebot →
   Passthru (recording/CDR) → Hangup).

## 2. Protocol (exact)

JSON messages over one WSS connection. Audio: **raw/slin 16-bit mono PCM little-endian, base64** in
`media.payload` — same byte format as our browser path. Actual `encoding`/`sample_rate`/`bit_rate`
arrive in the `start` message's `media_format` — read it, don't assume.

**Exotel → us:** `connected` (first), `start` (once: stream_sid, call_sid, account_sid, **from** =
caller number, to, custom_parameters, media_format), `media` (chunk, timestamp ms-from-start,
payload), `dtmf` (digit + duration), `mark` (echo of ours, = playback processed), `stop` (reason:
"stopped or callended"), `clear`.

**Us → Exotel:** `media` (same shape, must carry stream_sid), `mark` (named; sent after media, comes
back when that audio has been processed → this is our playback-completion signal), `clear` (flush
un-played audio = barge-in).

**Outbound chunk discipline (hard rules):** multiples of **320 bytes**; min **3.2 KB** ("100ms
data"); max 100 KB. Below min → jitter gaps; non-multiple of 320 → platform waits 20ms and you get
audio gaps; above max → timeouts.

**Field notes:** `sequence_number` orders messages; custom params max 3 keys / 256 chars total (don't
put secrets there — that's what Basic auth is for); stream is mono (no diarization needed — it's the
caller's leg).

## 3. Mapping onto our session engine

| Ours (browser `/ws/call`) | Exotel | Adapter work |
|---|---|---|
| `hello` {session_id, mode} | `connected` + `start` | new session per WS; `start.from` = caller number → **Zoho lookup of the actual caller** (better than freshest-lead) |
| binary Int16 PCM frames in | `media` base64 payload | b64decode → `stt.feed()`; at 16k, no resample |
| `audio_pcm` chunks out | `media` messages out | resample 22050→16k once, re-chunk to 320-byte multiples ≥3.2KB, b64encode |
| `playback_done` from browser | our `mark` → their `mark` echo | send named mark after each utterance's last media; mark echo drives `is_speaking=False` |
| `interrupted` → browser stops audio | send `clear` | on Sarvam VAD START_SPEECH while speaking (our existing barge-in trigger) |
| `bye` / hangup | close WS (or their `stop`) | on `end_call`, close after mark-confirmed sign-off |
| — | `dtmf` | ignore v1 (log it); future: menu shortcuts |

**Design consequence of `clear` semantics:** clear only flushes *queued* media — a huge chunk already
picked up keeps playing. The doc explicitly advises small chunks. So the adapter sends ~100–200ms
pieces (3.2–6.4 KB at 16k) instead of whole-utterance blobs, and barge-in stays snappy like the
browser. Do NOT ship one giant media message per sentence.

**Bug-class carryover from the browser path:** a `mark` echo will still arrive for audio that played
before a `clear` — same "ack after interrupt looks like natural completion" trap we fixed in app.js
(`if (p && p.stopped) return`). Guard marks by utterance id: ignore mark echoes for utterances that
were cleared.

## 4. Applet configuration (dashboard, once enabled)

- App Bazaar → Custom App → flow: **[incoming call] → Voicebot applet → Hangup**.
- Voicebot URL: `wss://KEY:TOKEN@our-host/exotel/stream?sample-rate=16000`
- Record: ON (recording URL later via Passthru).
- The applet is **not available by default** — enablement via hello@exotel.in / account manager
  (request already part of our sales thread with Naman).

## 5. Build plan

1. `app/transport.py` — extract the transport seam from `CallSession` (in: PCM frames + lifecycle;
   out: audio chunks + control events). Browser transport = current behaviour, zero regression.
2. `app/exotel_ws.py` — FastAPI WS route `/exotel/stream`: Basic-auth check, `start` handling,
   b64 media in → session; session audio out → resample/re-chunk/b64 `media` + `mark`; `clear` on
   barge-in; mark-echo playback tracking with cleared-utterance guard; `stop`/disconnect teardown.
3. Local proof: run Exotel's simulator (github.com/exotel/voice-streaming) against us + a pytest
   harness speaking the same JSON protocol (greeting, 2 turns, barge-in mid-reply, hangup).
4. Deploy Mumbai host (public wss://), configure the applet, first real call on the trial ExoPhone.
5. Telephony tuning pass: confirm 16k is actually granted (read `start.media_format`), STT accuracy
   by ear, greeting timing on `start`.
6. Outbound (phase 2): originate via Exotel call API into this same flow; DLT per sales guidance.

## 6. Open items (for the Exotel thread)

- Confirm per-minute rate + any Voicebot-applet surcharge, in writing (still unpublished).
- Confirm 16k sample-rate is honoured on our account/trunk type (doc says supported; verify in
  `start.media_format` on the first test call).
- Their outbound IP ranges, if/when we add IP allowlisting.
- Whether the trial account can get the applet enabled, or only paid plans.
