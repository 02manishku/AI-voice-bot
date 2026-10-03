# OzoneTel telephony integration — spec digest + build plan

Status: **PARKED, not started.** Analysed 2026-08-04 from "Bot Telephony
Integration — Complete Reference" v2.0 (2026-06-10, `webrtcserver` v1.0.0+).
Resume by reading this file; no code has been written yet.

Goal: run the existing Shubh voice bot on real phone calls instead of the
browser. The verdict from the analysis: **their protocol is the same shape as our
browser protocol, so this is an adapter, not a rewrite.**

---

## 1. Why this is cheap for us

Everything hard is transport-independent and already built: Sarvam streaming STT
with VAD turn-taking, barge-in, the transcript race, the LLM→TTS overlap, Zoho
lead lookup, conversation memory, and the whole personality/prompt layer.

Their protocol maps onto ours almost 1:1:

| Our browser `/ws/call`            | OzoneTel                                  |
|-----------------------------------|-------------------------------------------|
| binary Int16 PCM frames from mic  | `media` event, PCM-16 ints in `data.samples` |
| `audio_pcm` chunks out            | `{seqid, data:{samples, sampleRate}}`     |
| `playback_done` from browser      | `mark` ack (better — server-confirmed)    |
| `interrupted` → stop playback     | `clearBuffer` command                     |
| `hello` handshake                 | `start` event                             |
| `bye`                             | `stop` / `callDisconnect`                 |

**Coupling check (done):** all outbound traffic already funnels through
`CallSession._safe_send`, and the sample rate appears in only 5 places
(`stt_stream.STREAM_SAMPLE_RATE`, `call_ws._BYTES_PER_SEC`, `_pcm_to_wav`).
That is the seam — extract a transport interface, keep the brain.

---

## 2. Part 1 — IVR webhook (HTTP GET, XML replies)

One endpoint, e.g. `GET /api/ivr/webhook`. Three events per call, in order.

### NewCall
Query params: `event, sid, cid` (caller number), `called_number` (our DID),
`cid_e164, operator, cid_country, cid_countryname, cid_type, circle, request_time`.

Must return:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<response>
  <start-record></start-record>
  <stream is_sip="true" moh="silence"
          url="wss://OUR-HOST/ozonetel/stream"
          x-uui='{"sid":"...","cid":"...","called_number":"..."}'>
    5142XXX
  </stream>
</response>
```
- `url` = our bot WebSocket.
- `x-uui` = arbitrary JSON we pass through; it comes back on the WS `start`
  event as `x_headers`. **This is how we correlate webhook → WS session.**
- Inner text = the **SIP extension** assigned to the call (provisioned by
  OzoneTel — we don't have this value yet).

### Stream
Fired when the bot's audio session ends. Decision point:
- transfer → `<cctransfer uui="sales" timeout="30" moh="..." >SalesSkill</cctransfer>`
- end → `<hangup></hangup>`

### Hangup
Final event. Carries `call_recording_url` / `data` (S3 mp3), `callduration`,
`total_call_duration`, `telco_code`. Reply `<hangup></hangup>`. No further callbacks.

---

## 3. Part 2 — WebSocket media protocol

**Server → Bot**
- `start` — on open. Fields: `ucid` (correlation id), `did`, `call_id` (**caller's
  number**), `x_account`, `x_headers`, and `media: {encoding:"PCMU",
  sampleRate, channels, bitsPerSample:16, payloadType}`.
  `sampleRate` is negotiated per call: **8000 / 16000 / 24000 / 48000 — do not
  hardcode 8k.**
- `media` + `type:"media"` — inbound audio. `data.samples` is a **JSON array of
  signed 16-bit ints** (already μ-law-decoded), `numberOfFrames` 80 @ 8 kHz
  (10 ms frames, ~100 packets/sec), `channelCount` always 1.
- `media` + `type:"dtmf"` — `signal` is `0-9`, `*`, `#`. Deduped by the server.
- `mark` + `type:"ack"` — the last frame of a given `seqid` was transmitted.
- `stop` — call ended. `cause: 433` = normal caller hangup; **field omitted on
  SIP failure**. WS is closed by them immediately after.

**Bot → Server**
- Audio: `{"seqid":"utt-001","data":{"samples":[...],"sampleRate":8000}}`
  - `seqid` optional but **required to get `mark` acks** + dedup.
  - `sampleRate` optional; **their server auto-resamples** (16k TTS → 8k call).
- Barge-in: `{"command":"clearBuffer","extension":"1001","sessionId":"<ucid>"}`
- Hang up: `{"command":"callDisconnect","causeCode":200}` (default 1433)

**Limits:** 3 concurrent calls per extension (else SIP 486 Busy, no WS opened);
60 s audio queue per session (excess silently dropped); dedup window 3000 seqids.

**Error handling:** WS error *or* bot closing the WS **terminates the live SIP
call** (cause 16). Invalid JSON / unknown command = ignored + logged.

---

## 4. Sharp edges (found in analysis — do not rediscover the hard way)

1. **`clearBuffer` needs the SIP `extension`, not just the session id.** So the
   webhook must persist `sid → extension → ucid`. Miss this and barge-in
   silently never works.
2. **A `mark` is also emitted when `clearBuffer` fires** (for the last seqid
   actually transmitted). So a post-barge-in ack looks identical to natural
   completion — this is the *exact* bug class already fixed in the browser path
   (`if (p && p.stopped) return;` in app.js). Carry that guard over.
3. **Sample rate is per-call**, read it from `start`.
4. **Concurrency = extensions × 3**, and the app has never been load-tested with
   concurrent calls (`SESSIONS`/`SESSION_LANG` dicts, rate limiter are per
   session id — should be fine, but unverified).
5. **Our WS dying drops a real customer call**, not a refreshable page.
6. Audio is **JSON int arrays, not binary** — verbose. Downsample our 22 kHz TTS
   to 8 kHz before sending (~2.7× smaller payload; their server would resample
   to 8k anyway since the SIP leg is PCMU 8k).
7. Sarvam's streaming STT **accepts `sample_rate="8000"`** (verified in the
   installed SDK), so phone audio can be fed directly — no upsampling. Expect a
   real STT accuracy drop on 8 kHz narrowband; budget a tuning pass.
8. RNNoise does **not** apply (browser-only worklet, 48 kHz).

---

## 5. Two upgrades telephony unlocks

- `start.call_id` gives the **caller's phone number** → look that specific person
  up in Zoho instead of "freshest lead". Strictly better than what we do now.
- The `Stream` event is a real transfer point → our A8 escalation ("let me check
  with our team") can actually `<cctransfer>` to a live agent queue instead of
  only promising a callback.

---

## 6. The gap: this document is INBOUND ONLY

Every flow starts with "a call arrives on your DID". There is **no outbound /
click-to-call API here**, which is what the flagship "Lead Call" mode needs (bot
dials the lead). Requires a separate OzoneTel outbound/campaign API, plus
DLT/TRAI compliance for automated outbound calling in India. **Inbound is fully
buildable from this doc alone.**

---

## 7. Blockers — asked of OzoneTel, awaiting answers

1. Which **SIP extension number(s)** are provisioned? (and how many → concurrency cap)
2. **Webhook authentication** — the doc shows none. IP allowlist? Shared secret?
   As specified, anyone hitting the URL could drive call flows.
3. **Test DID + sandbox** to validate one call end-to-end.
4. **Outbound calling API**, for Lead Call mode.

Also required: a **public deploy** (`https://` webhook + `wss://` bot URL) —
their cloud cannot reach localhost. Render/Railway/Fly, not Vercel (serverless
cannot host a persistent WebSocket). User has not yet approved hosting.

---

## 8. Build plan when resumed

1. Deploy the current app to a public host (prerequisite for everything).
2. `app/ivr_webhook.py` — the 3 lifecycle events + XML replies + the
   `sid → extension → ucid` store.
3. Refactor `call_ws.CallSession` to take a transport object (the seam is
   `_pump_mic` in / `_safe_send` out); browser transport keeps current behaviour.
4. `app/ozonetel_ws.py` — their JSON frame format, seqid/mark handling,
   `clearBuffer` on barge-in, `callDisconnect` on end_call, greeting sent on
   `start`, Zoho lookup by `call_id`.
5. **A simulator** that impersonates OzoneTel (sends `start`, streams frames,
   expects audio back, fires `clearBuffer`) so the whole flow is provable before
   any real phone call.
6. Real inbound test call → telephony audio tuning pass (8 kHz STT accuracy).
7. Outbound, once that API exists.
