# Shubh — an AI voice sales agent that answers the phone

Shubh is a production voice agent for **Magppie Wellness Kitchens**. Someone dials a
real phone number, a real voice picks up, and they have a normal sales conversation in
English, Hindi or a mix of both. Shubh qualifies the lead, answers questions from a
fixed knowledge base, and tries to get the caller onto WhatsApp with their kitchen
layout.

It is not a chatbot with a microphone bolted on. The hard part of a voice agent is not
transcription or generation, both of which are API calls. The hard part is
**turn-taking**: knowing when the caller has finished speaking, when they are only
thinking out loud, when they are interrupting you, whether the sound you just heard was
even a person, and what to do when you were cut off mid-sentence by nothing at all.
Most of this repository is that problem.

```
 Caller's phone
      |                                       Exotel Voicebot Applet
      |  PSTN                                 (bidirectional WebSocket)
      v
 +----------------+   16 kHz PCM   +--------------------------------------+
 | Exotel         | -------------> |  /exotel/stream                      |
 | telephony      | <------------- |  app/exotel_ws.py  (transport)       |
 +----------------+   16 kHz PCM   |  app/call_ws.py    (the call brain)  |
                                   +--------------------------------------+
                                        |             |             ^
                            RNNoise +   |             | question    | audio
                            speech gate |             v             |
                                        v      +--------------+     |
                                 +-----------+ | OpenAI       |     |
                                 | Sarvam    | | gpt-4.1-nano |     |
                                 | Saaras v3 | | + whole KB   |     |
                                 | STT + VAD | | in the prompt|     |
                                 +-----------+ +--------------+     |
                                                      |             |
                                                      v             |
                                               +--------------+     |
                                               | Sarvam       | ----+
                                               | Bulbul v3    |  AGC + limiter
                                               | TTS, stream  |
                                               +--------------+
```

The same brain also serves a browser demo at `/`, so you can try it without a phone
line.

---

## Contents

- [How one turn actually works](#how-one-turn-actually-works)
- [Making it sound like a person](#making-it-sound-like-a-person)
- [Staying on the facts](#staying-on-the-facts)
- [Audio quality on a real phone line](#audio-quality-on-a-real-phone-line)
- [Languages](#languages)
- [Memory across calls](#memory-across-calls)
- [The telephony leg](#the-telephony-leg)
- [Running it](#running-it)
- [Keeping it running](#keeping-it-running)
- [Configuration](#configuration)
- [Tests](#tests)
- [Project layout](#project-layout)
- [Cost and limits](#cost-and-limits)
- [Known limits](#known-limits)

---

## How one turn actually works

A turn is one round: the caller says something, Shubh answers. There are four stages,
and every one of them is overlapped with the next wherever physics allows.

### 1. Hearing

Audio arrives from Exotel as base64 16-bit PCM inside JSON, at whatever sample rate
that particular call negotiated, which is read from the `start` event and never
assumed. Before anything else sees it, the audio is cleaned:

- **RNNoise** removes steady background noise. It runs on the server for phone calls
  (`app/denoise.py`, a raw ctypes binding to pyrnnoise) and in the browser for the web
  demo (a WASM worklet). It costs about 0.8 ms per 20 ms chunk, so it adds no
  perceptible latency.
- A **speech gate** then ducks frames that are far quieter than the caller's own
  running speech level. RNNoise removes *noise* but happily passes *other humans*, and
  a colleague talking across the room was becoming full conversational turns in random
  languages.

The cleaned audio is streamed to Sarvam Saaras v3 over a persistent WebSocket, and
buffered locally at the same time.

### 2. Knowing when the caller stopped

Sarvam's server-side voice activity detection emits `START_SPEECH` and `END_SPEECH`.
End of turn is never guessed from volume. The silence window before `END_SPEECH` is the
single largest piece of felt latency, so it is tuned down from the server default of
roughly 770 ms to about 480 ms. Going lower was tried and reverted: at about 340 ms it
split a caller's natural mid-sentence pause into two separate turns.

### 3. Thinking

Transcription is a **race with a safety net**. The streaming transcript normally lands
about 320 ms after end of turn. But it can lag for seconds or never arrive at all,
which would strand the turn in silence, so the buffered audio is also sent to the REST
endpoint concurrently from the start, and whichever returns first wins. A healthy
stream wins and the REST result is thrown away. A sick stream loses to REST at about
1 second instead of 1.8.

The transcript then runs a gauntlet of filters, described in the next section, before
it is allowed to become a question. If it survives, it goes to OpenAI with the entire
knowledge base in the system prompt.

Two latency tricks live here:

- **Prompt caching.** The knowledge base is roughly 14k tokens on every single turn. It
  sits first in the prompt, is never modified, and carries a fixed cache key, so OpenAI
  reports a 98 to 99 percent cache hit and we pay neither the money nor the latency for
  it. A background keepalive pings the cache during quiet moments, so a caller who
  pauses for several minutes does not pay the cold-prefix penalty on their next
  question.
- **Hedged requests.** First token normally arrives in 1.0 to 1.3 seconds, but OpenAI
  occasionally sits on a request for 4 to 7 seconds. If no first token has arrived
  after 1.8 seconds, an identical second request is raced against the first and the
  loser is closed. One extra cheap, fully cached call buys back several seconds on
  exactly the turns that were about to feel broken.

### 4. Speaking

The reply is **not** generated and then spoken. As soon as the model has produced one
complete sentence, that sentence goes to Bulbul and starts playing while the model is
still writing the rest. The remainder is synthesized concurrently on its own
pre-opened socket, so the seam between sentence one and the rest is inaudible.

Synthesis costs roughly 700 ms fixed plus 25 ms per character, which is why answers are
capped near 30 words. That cap is a latency decision, not a style one.

If the answer is still not ready after 700 ms, Shubh makes a small human noise while he
thinks: a pre-rendered "Hmm.", "Right.", or in Hindi "जी।", "अच्छा।". It never repeats
the same one twice in a row, never fires before a goodbye, and costs nothing because
the clips are rendered once and cached on disk. A person filling a gap sounds alive.
Two seconds of dead air sounds broken.

### Latency budget

Measured on real calls, medians:

| Stage | Time |
|---|---|
| Caller stops speaking, to end of turn detected | about 480 ms |
| Speech to text | 100 to 400 ms |
| Model first token | 1.0 to 1.3 s |
| Transcript to first reply audio | about 1.5 s |
| Best observed, browser leg | 640 ms |
| Whole reply sent | 2.3 to 3.5 s |

---

## Making it sound like a person

Every mechanism below exists because a real caller was let down in a logged call. This
is the part that took the most work, and it is almost all in `app/call_ws.py`.

| Problem seen on a real call | What happens now |
|---|---|
| Noise and "hmm" became questions, so Shubh pitched at a cough | A two-tier junk filter. Non-lexical fillers are always dropped. Acknowledgements like "okay" or "हाँ" are dropped while he is speaking, but honoured when he is idle, because then they are answers to his own question. |
| On speakerphone, Shubh answered his own echo | Self-echo filter. A transcript that is 75 percent or more his own recent words, arriving while he speaks or within 3 seconds after, is his voice looping back through the caller's handset. |
| A colleague talking nearby produced turns in Punjabi, Bengali and Odia | Garble gate. Once the call's language is established, a short turn tagged in a language the call is not happening in is bleed, and is dropped. |
| A caller said "I want to inquire", paused, got answered mid-thought, then interrupted his own answer | Unfinished-turn hold. A turn ending on a dangling word ("to", "about", "के बारे") waits 3.5 seconds for its continuation and merges the two into one question. The timer refuses to fire while the caller is still mid-word. |
| A caller said "Okay" three times into silence, then hung up | When a bare acknowledgement arrives with no question pending, Shubh now leads with exactly one unasked discovery question instead of staying mute. |
| **Background noise cut Shubh off mid-answer and he never spoke again** | False barge-in recovery. The interrupted reply is remembered. If the thing that interrupted turns out to be junk, the phone leg resumes the audio from one second before the cut, or says the answer again with a natural "as I was saying". The rule this taught: every path that cancels a reply must end in either a new reply or a recovery. |
| A sub-250 ms noise blip cut the sign-off, produced no turn at all, and left 73 seconds of silence | That path now triggers the same recovery. |
| Shubh kept talking over a caller trying to interrupt | Barge-in cuts playback within a beat. The phone leg requires sustained speech, not a car horn, before it yields. |
| A caller's closing words were lost when they spoke during a reply | Missed-turn recovery. Speech that ends while a reply is in flight is picked up the moment that reply finishes, instead of being discarded. |
| A caller said goodbye and Shubh kept selling | Exit-intent guard, in both languages, including the exact phrasings real callers used. It can only prevent a wrong hang-up, never cause one. |

---

## Staying on the facts

A voice agent that invents a price destroys trust in the whole system, so grounding is
strict.

**There is no retrieval layer, by design.** The entire knowledge base goes into the
system prompt. At this size, retrieval would add both latency and a new failure mode,
the right chunk not being retrieved, to solve a problem we do not have. Static content
goes first so the prefix cache hits. Anything variable, the question, the language
steer, the history, goes last.

The knowledge base lives in [`kb_source/`](kb_source/) as plain Markdown, and every
file in that folder is loaded at startup. It carries the call script, both price
ranges, the guarantee terms per range, objection handling, showroom addresses, and the
exact sentences to say and never say.

The guards that matter:

- **The scanned-PDF trap.** A scanned PDF extracts to nothing, raises no exception, and
  leaves a healthy-looking server that answers "I do not know" to everything. Startup
  therefore *crashes* if any file yields under 200 characters, and names the file.
- **Startup prints every file with its character and token count**, and the same data
  is at `GET /api/kb/debug`. If a file is not listed with a sane count, fix that before
  believing anything the bot says.
- **Numbers are never re-derived.** The prompt forbids translating, converting or
  re-wording any figure.
- **Spoken-form normalisation.** The model writes for the eye, and Bulbul reads exactly
  what is written. "25 yrs" was spoken as letters, and "25-year" came out as "two five
  Y-A-R" on a real call, because the speech engine reads a digit-hyphen-letters token
  as a code. Year counts are therefore rewritten into words in the synthesis language
  ("twenty five years", "पच्चीस साल") just before synthesis, and the knowledge base is
  kept free of such hyphens. The lesson generalises: to prove a pronunciation fix,
  synthesize the line and transcribe it back, rather than trusting the text.
- **Brand mishearing repair.** Saaras does not know "Magppie" and returns "MacPay" or
  "मैक पाई", which the model then read as a different company and politely refused the
  caller. Known mishearings are repaired in the transcript before the model sees it.

---

## Audio quality on a real phone line

Studio-quality text-to-speech sounds wrong on a phone earpiece, and callers notice.

- **Automatic gain control.** Bulbul's output level varies between calls, between
  languages, and even between the head and tail of a single answer, because those
  render on separate sockets. Callers reported that the volume kept going up and down.
  The phone leg now steers gain continuously toward a target speech level in 100 ms
  windows, so the *spoken loudness* stays constant rather than the gain.
- **A soft-knee limiter**, not plain gain. Bulbul already peaks at full scale, so
  multiplying would hard-clip into buzz. Everything below the knee is multiplied
  cleanly, and only the loudest instants are squashed.
- **Paced outbound audio.** Sending a whole reply at once saturated the uplink: the
  voice audibly cut, and the inbound microphone stream jittered too, inflating latency.
  The first 0.8 seconds leaves immediately for an instant start, then chunks flow at
  playback rate. A pleasant side effect is that a barge-in then has almost nothing
  queued remotely to flush.
- **The greeting is pre-rendered** and cached on disk, so the call opens instantly and
  costs nothing per call. It plays slightly slower than normal speech, because the team
  found the brand name otherwise flew by too fast to register.

---

## Languages

Shubh speaks English and Hindi, and mixes them the way people actually do on an Indian
sales call: Hindi words in Devanagari for native pronunciation, English loanwords left
in English, as in "देखिए, हमारा पूरा kitchen stone का बनता है".

Choosing the language for each turn is deceptively hard, because the speech engine tags
short Hinglish fragments as English and short Devanagari fragments as Bengali or
Marathi. The rules:

- An explicit request, "Hindi mein batao", wins instantly and always.
- Otherwise the call's language is **sticky, with two-turn hysteresis in both
  directions**. A single stray turn never flips the call. One caller complained after a
  lone English question flipped his Hindi call; later, one Devanagari garble flipped an
  English call. Both are now impossible.
- Short or uncertain turns neither advance nor reset the switch, so a caller speaking in
  fragments can still complete a deliberate change of language.
- Establishing the language on the *first* turn requires a real sentence, because a
  one-word garble during the greeting once pinned an entire English call to Hindi.
- The Devanagari test matches letters only. The danda "।" lives inside the Devanagari
  Unicode block but ends sentences in every Indic script, and an Odia fragment ending in
  one pinned a call to Hindi.

---

## Memory across calls

Shubh remembers people by phone number. After each call, one cheap model call rewrites
a short note about *the person*, their city, budget, what they are building, which range
they liked, into `.cache/callers/<number>.json`. On the next call that note is injected
as context and the remembered language is pre-pinned, so a returning English speaker is
never greeted in Hindi.

Some deliberate choices here. Notes are about the caller, never about prices, which the
knowledge base already knows. The caller's words enter conversation history at the
*start* of a turn rather than after a reply survives, because a barge-in used to erase
the exchange entirely and make Shubh forget a budget he had just been told. And when a
caller asks directly what Shubh remembers about them, he answers plainly instead of
pretending not to know.

Caller files are gitignored. They contain real phone numbers.

---

## The telephony leg

Exotel was chosen over OzoneTel and Plivo after a written comparison, mainly for 16 kHz
PCM, self-serve onboarding and India-local media. The reasoning is in
[`docs/telephony-provider-selection.md`](docs/telephony-provider-selection.md), and the
wire protocol in [`docs/exotel-integration.md`](docs/exotel-integration.md).

`app/exotel_ws.py` is only a transport adapter. It subclasses the browser call session
and inherits the entire brain, overriding nothing but the wire format:

| Direction | Events |
|---|---|
| Exotel to us | `connected`, `start` (carries the caller's number and the sample rate), `media`, `mark`, `dtmf`, `stop` |
| Us to Exotel | `media` (re-chunked to their 320-byte rule), `mark` (playback tracking), `clear` (barge-in flush) |

Two details cause real bugs if missed. A `mark` echo can still arrive for audio that a
barge-in already cleared, and it looks identical to natural completion, so cleared marks
are tracked and ignored. And hanging up *is* closing the socket, which advances Exotel's
flow to the next applet, so the socket is closed only after the final mark confirms the
sign-off actually played.

Configure the Voicebot Applet URL as:

```
wss://KEY:TOKEN@your-host/exotel/stream?sample-rate=16000
```

Exotel strips those credentials and sends them as an `Authorization: Basic` header,
which is checked with a constant-time comparison.

---

## Running it

**Prerequisites:** Python 3.12 or newer, [uv](https://docs.astral.sh/uv/), a Sarvam API
key and an OpenAI API key.

```bash
git clone https://github.com/02manishku/AI-voice-bot.git
cd AI-voice-bot

cp .env.example .env     # then paste your two keys into it
uv sync

uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000> and talk. The browser demo needs `getUserMedia`, which
browsers only permit on **HTTPS or localhost**, so a plain `http://192.168.x.x` link
will load the page and then fail to get the microphone.

**To take real phone calls** you need a public HTTPS address. For a quick test:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Put the resulting host into the Exotel applet URL shown above. A quick tunnel gets a new
random hostname on every restart, so the applet URL has to be updated each time. A
permanent deployment avoids that.

**Verify through the public URL, not localhost.** A 200 from `/api/health` only proves
that HTTP works, not that the WebSocket call path does.

---

## Keeping it running

On Windows, processes started from a terminal session die along with it. Two scheduled
tasks keep the bot and the tunnel alive independently of any session:

```powershell
Start-ScheduledTask MagppieBot-Server
Start-ScheduledTask MagppieBot-Tunnel
```

They have no boot trigger, so after a reboot they must be started again. Logs and the
launcher scripts live in `%LOCALAPPDATA%/magppie-bot/`. Search `tunnel.log` for
`trycloudflare.com` to find the current public hostname.

---

## Configuration

Everything is typed in `app/config.py` and overridable from `.env`. There is no
`os.getenv()` anywhere else in the codebase. The settings worth knowing:

| Variable | Default | What it does |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4.1-nano` | The conversation model |
| `STT_STREAMING` | `false`, set it `true` | WebSocket VAD turn-taking and barge-in |
| `STT_NEGATIVE_FRAMES_COUNT` | `7` | Silence frames that close a turn. The biggest latency lever |
| `EXOTEL_DENOISE` | `true` | RNNoise on inbound phone audio |
| `EXOTEL_SPEECH_GATE` | `true` | Duck background talkers |
| `EXOTEL_INTERRUPT_MIN_SPEECH_FRAMES` | `24` | Barge-in resistance on the phone leg |
| `EXOTEL_TTS_TARGET_RMS` | `11500` | AGC target loudness. `0` disables AGC |
| `EXOTEL_TTS_GAIN` | `2.0` | Starting gain, before the limiter |
| `REPLY_FILLER` | `true` | The spoken thinking beat |
| `REPLY_FILLER_AFTER_MS` | `700` | How long to wait before filling the gap |
| `CALLER_MEMORY_ENABLED` | `true` | Remember callers between calls |
| `MAX_HISTORY_MESSAGES` | `30` | Working memory, about 15 exchanges |
| `MAX_TURNS_PER_DAY` | `1000` | Spend guard. Trips audibly, never silently |

---

## Tests

Twelve suites run **completely offline**, with no API keys and no credits spent:

```bash
for f in tests/test_*.py; do uv run python "$f"; done
```

| Suite | What it guards |
|---|---|
| `test_noise_filter.py` | Junk, echo, garble, unfinished turns, spoken-number normalisation |
| `test_call_ws.py` | The WebSocket contract, barge-in, false-barge-in recovery, the bare-"Okay" rule |
| `test_language.py` | Stickiness, hysteresis, the danda bug, first-turn pinning |
| `test_caller_memory.py` | Note writing and context injection |
| `test_exit_guard.py` | Hang-up intent in both languages |
| `test_kb_guards.py` | The scanned-PDF trap |
| `test_limits.py` | Rate limiting |
| `test_tts_chunking.py` | Sentence splitting and streaming overlap |
| `test_prose_stream.py` | Extracting prose from partial JSON |
| `test_turn_stream.py` | The `/api/turn` event contract |
| `test_brand_repair.py` | Brand mishearing repair |
| `test_zoho.py` | CRM lead handling |

The `tests/check_*.py` scripts are behavioural probes against the live APIs and do spend
credits. They are judgement calls, not pass or fail.

Every filter was written against a transcript of a real call that went wrong, and the
keep-cases are tested harder than the drop-cases, because wrongly dropping a real
question is far worse than letting one "hmm" through.

---

## Project layout

```
app/
  main.py          FastAPI app, routes, greeting cache, language resolution
  call_ws.py       The call brain: turn-taking, filters, barge-in, recovery
  exotel_ws.py     Exotel wire format. Subclasses the brain, adds nothing to it
  stt_stream.py    Saaras v3 streaming speech-to-text over a persistent socket
  stt.py           Saaras v3 REST, the reliable half of the transcript race
  llm.py           OpenAI streaming, grounded answers, citations, hedged requests
  tts.py           Bulbul v3, streaming and REST, sentence-level overlap
  prompts.py       System prompt, grounding rules, language steers, turn nudges
  kb.py            Loads kb_source/ with guards. No retrieval, by design
  denoise.py       RNNoise plus the background-speech gate
  caller_memory.py Per-number notes, written after the call ends
  pronunciation.py Brand pronunciation, mishearing repair, spoken numbers
  zoho.py          CRM lead lookup for outbound calls
  limits.py        Per-caller and per-day spend guards
  config.py        Every setting, typed
kb_source/         The knowledge base. Everything Shubh knows
static/            Browser demo: worklet mic capture, RNNoise WASM, player
docs/              Telephony protocol notes and the provider comparison
tests/             Twelve offline suites plus live behavioural probes
```

---

## Cost and limits

Roughly **Rs 0.6 per turn**, dominated by text-to-speech. The knowledge base is 98 to
99 percent prompt-cached, so the model is a minor line item despite 14k tokens riding
on every turn.

Spend guards are deliberately *audible*. A silent refusal is the worst possible failure
mode: the greeting is cached on disk and plays without touching any API, so an
exhausted limiter once produced a bot that greeted callers warmly and then went dead,
which looked exactly like a crash. The limiter now speaks a pre-rendered apology
instead.

---

## Known limits

- **A quick tunnel changes hostname on every restart**, so the Exotel applet URL has to
  be updated each time. A permanent deployment is the fix.
- **It runs on a laptop.** If the machine sleeps, the bot is unreachable.
- **Very short words at the edge of voice detection** can be missed, for example a lone
  "Okay", or a "bye bye" said over Shubh, which needs about a second of speech to count
  as an interruption.
- **Escalation cannot transfer a live call** yet. It reads the escalation line.
- **Leads bind by recency rather than by phone number** on the outbound Zoho path.

---

## Credits

Built with [Sarvam AI](https://www.sarvam.ai/) for Indic speech recognition and
synthesis, OpenAI for conversation, [Exotel](https://exotel.com/) for telephony,
FastAPI, and [RNNoise](https://github.com/xiph/rnnoise) for noise suppression.

The knowledge base content belongs to Magppie Wellness Kitchens.
