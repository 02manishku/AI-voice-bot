# Telephony provider selection — research record

2026-08-26. 30 providers assessed by an 11-agent research workflow (7 researchers, 4 adversarial
verifiers, 333 web fetches), every claim below checked against primary developer docs. Full
structured results: the workflow output (`tasks/w2fd3j2tl.output` in the session scratchpad);
this file keeps what matters for the decision.

**Hard requirement:** raw bidirectional media streaming between a live Indian PSTN call and OUR
server (we bring Sarvam STT/TTS + OpenAI — the provider must not force its own bot brain).

## Verdict

| # | Provider | Why | Watch out |
|---|---|---|---|
| 1 | **Exotel (AgentStream / Voicebot Applet)** | India-native media; public protocol docs; **16kHz PCM supported** (better STT accuracy than 8k-only rivals); self-serve trial (₹1000 credits + trial ExoPhone, verify own phone pre-KYC); echo-bot reference repo; outbound connect API. Verified strong_fit. | Voicebot applet needs manual enablement — email hello@exotel.in (confirmed in their KB). Chunk rules: multiples of 320 bytes, ≥3.2KB, ≤100KB. 60-min stream cap. Per-minute rates unpublished — get a quote. Plans: Dabbler ₹9,999/5mo, Believer ₹19,999/11mo. |
| 2 | **Plivo (AudioStream)** | Verified strong_fit. Best docs of the lot; Mumbai PoP + India media anchoring; streaming **free** (listed on India pricing page); ₹0.38/min, DID ₹200/mo; L16 16k also supported. | Indian DIDs need a registered Indian entity (COI **or** Udyam/MSME + PAN **or** GST). **Data region is permanent** — sign up in India region or never get Indian numbers on that account. No auto-recharge on INR accounts (balance monitoring needed). Signup risk review can reject; use business-domain email. |
| 3 | **Acefone (ex-Servetel)** | GA, public Twilio-style WS docs (connected/start/media/mark/clear); Indian operator; a Pipecat serializer for it was merged upstream — proof third-party bots run on it. | 8k mulaw only; Channels Hub enablement + KYC; pricing unpublished; WS auth details undocumented. |
| 4 | **Vobiz (vobiz.ai)** | Purpose-built for AI bots: L16 8/16k in, clearAudio/checkpoint barge-in primitives, self-serve console, ~₹0.45/min indicative, sub-80ms claims. Sarvam publishes an official Vobiz guide. | Young company (2025-era), track record thin; pricing not confirmed on their own pages. |
| 5 | **EnableX** | Clean public docs; signed-JWT auth on the WSS (nicest auth story); outbound API examples literally use Indian numbers; INR pricing. | ulaw 8k only; India DC asserted, not proven. |
| 6 | **FreJun Teler** | Real BYO-brain streaming, open-source bridges (incl. Devnagri), L16 8k, sub-250ms claims, Indian company. | Very young; pricing/signup unverified; protocol lives in SDK code, not a spec page. |
| — | **OzoneTel** | Capability confirmed (we hold the partner spec; digest in `ozonetel-integration.md`) but **sales-led**: no public docs/sandbox, our 4 questions still unanswered, spec is inbound-only, streaming absent from their public docs portal. | Keep warm as fallback; don't wait on them. |

## Disqualified (with the one-line reason)

- **Twilio** — flawless protocol, but **no Indian local/mobile numbers** (toll-free only, non-India address required); outbound to India = foreign caller ID now force-labeled "International Call"; media in US/IE/AU only; ~₹4+/min. Dev-reference only.
- **Vapi / Retell / Bland** — no raw media channel your server owns, and Vapi's own docs: "Indian phone numbers cannot be used… TRAI regulations require SIP termination via an Indian server, which Vapi does not support." US/EU media = +265ms+ RTT.
- **Airtel IQ, Tata (Kaleyra/DIGO), Gupshup/Knowlarity** — no public bidirectional media streaming; managed-bot or enterprise-sales territory.
- **MyOperator, CallHippo, C-Zentrix, Ameyo, Fonada, Sarv, Voxbay, SuperBot, Smallest.ai, ElevenLabs Agents** — closed bot platforms or no streaming at all.
- **LiveKit / Pipecat Cloud** — good infra, but they'd replace our own FastAPI orchestrator; they still need Exotel/Plivo underneath for Indian numbers. Useful as references (both have Exotel/Plivo transports + Sarvam plugins), not as our stack.

## Latency reality check

Exotel / Plivo / Vobiz / Acefone all anchor media in India; the telephony leg adds roughly
100–300ms (one ~100ms frame each way + PSTN). Our felt-latency floor stays dominated by the LLM
(~1s TTFT). Provider choice cannot get us under 1s; it decides whether we ADD ~150ms (India-anchored)
or ~600ms+ (US-anchored). Bot server should sit in Mumbai (AWS ap-south-1 / equivalent) at deploy.

Protocol accuracy bonus: Exotel's 16kHz PCM beats 8k mulaw for Sarvam STT accuracy — the only
top-tier provider where the phone leg isn't narrowband-capped.

## Compliance (any provider)

- Inbound (customer calls the bot): no special registration.
- Outbound (bot dials leads): TRAI/TCCCPR — promotional robocalls need DLT registration + 140-series
  numbers (1600-series = transactional/service); calls 9AM–9PM; warm leads who submitted a form are
  arguably service/consented calls, but treat DLT as a real workstream. Indian providers handle
  registration during onboarding.

## Action plan

1. Create the Exotel trial (self-serve, ₹1000 credits) **and immediately email hello@exotel.in**
   asking to enable the Voicebot applet — that round-trip is the critical path.
2. Meanwhile build the transport seam + adapter. Exotel/Plivo/Acefone/Vobiz all speak the same
   Twilio-ish dialect (base64 audio in JSON events, `clear` for barge-in, `mark` for playback acks) —
   one adapter shape covers the market. OzoneTel's JSON-int-array protocol is the outlier.
3. Prove the flow against a protocol simulator locally; deploy to a Mumbai host; point a flow at it.
4. Plivo is the live fallback if Exotel's enablement stalls — signup needs Magppie's entity docs
   (or Udyam) and MUST pick the India data region at account creation.
