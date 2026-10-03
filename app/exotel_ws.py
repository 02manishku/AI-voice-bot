"""Exotel Voicebot Applet transport: the same call brain, on a real phone line.

Exotel's bidirectional stream (support article 3000108630, digested in
docs/exotel-integration.md) maps ~1:1 onto the browser protocol, so this module
is a transport adapter around call_ws.CallSession — every hard part (Sarvam VAD
turn-taking, the transcript race, LLM→TTS overlap, barge-in, Zoho context,
history) is inherited untouched. What this file owns:

  in   their JSON events    connected / start / media(b64 slin PCM) / dtmf /
                            mark(echo) / stop
  out  their JSON messages  media (re-chunked to their 320-byte rule),
                            mark (playback tracking), clear (barge-in flush)

Wire facts this code must honour (from the spec):
  - Audio both ways: raw/slin 16-bit mono PCM little-endian, base64 in JSON.
  - The call's sample rate is per-call: read start.media_format, never assume.
  - Outbound chunks: multiples of 320 bytes, >= 3.2 KB, <= 100 KB. We send
    3.2 KB pieces — small chunks are what make `clear` (barge-in) actually cut
    promptly, per the spec's own note.
  - `clear` only flushes UN-played audio; a `mark` echo can still arrive for a
    cleared utterance — same ack-after-interrupt trap as the browser player,
    guarded here with the _cleared set.
  - Closing the WS ends the stream and advances Exotel's flow to the next
    applet (our Hangup) — so hanging up IS closing the socket, after the final
    mark echo confirms the sign-off audio was played.
"""

import array
import asyncio
import audioop
import base64
import binascii
import hmac
import io
import json
import logging
import wave

from fastapi import WebSocket, WebSocketDisconnect

from app import call_ws, caller_memory, denoise, prompts, stt_stream, tts
from app.config import settings

log = logging.getLogger("magppie.exotel")

# One outbound piece: 3200 bytes = the spec's minimum ("100ms data"), a
# multiple of 320. At 16k that is 100ms of audio, at 8k 200ms — small enough
# that a barge-in `clear` cuts within a beat.
CHUNK_BYTES = 3200
# The `start` event must arrive promptly after connect, else this isn't a call.
START_TIMEOUT = 20.0
# Outbound pacing: keep at most this much un-played audio queued at Exotel.
# Bursting a whole reply at once saturated the uplink on the first real call
# (2026-08-29): the caller heard the voice "cutting", and the inbound mic
# stream jittered too, inflating latency. The first LEAD seconds go out
# immediately (instant start), then chunks flow at playback rate.
LEAD_SECONDS = 0.8
# False barge-in recovery (call_ws): when the "interruption" proves to be
# nothing, the cut reply resumes from this far BEFORE the cut — the caller
# hears the last phrase again, like "as I was saying", and Exotel's own
# playout delay (which we can only estimate) is covered.
RESUME_REWIND = 1.0


def _authorized(websocket: WebSocket) -> bool:
    """Basic-auth check. Exotel sends `Authorization: Basic b64(key:token)`
    when the applet URL is configured as wss://KEY:TOKEN@host/... . Blank
    settings = auth off (local testing); warn so it's never silently open."""
    key, token = settings.exotel_ws_key, settings.exotel_ws_token
    if not (key and token):
        log.warning("exotel: EXOTEL_WS_KEY/TOKEN unset — accepting unauthenticated stream")
        return True
    header = websocket.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        return False
    try:
        decoded = base64.b64decode(header.split(None, 1)[1]).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, IndexError):
        return False
    return hmac.compare_digest(decoded, f"{key}:{token}")


async def handle(websocket: WebSocket) -> None:
    """Entry point for the /exotel/stream route."""
    from app import main  # deferred, same as call_ws: main registers the routes

    await websocket.accept()
    if not _authorized(websocket):
        log.warning("exotel: rejected stream with bad/missing credentials")
        await websocket.close(code=4401)
        return
    if settings.missing_keys() or main.KB.is_empty:
        log.error("exotel: refusing call — missing keys or empty KB")
        await websocket.close(code=1011)
        return

    await ExotelCallSession(websocket, main).run()


class ExotelCallSession(call_ws.CallSession):
    """CallSession with Exotel's wire format on both ends.

    Inherits the whole brain; overrides the mic pump (their JSON in), the send
    path (_safe_send translated to their messages), and run() — which must wait
    for `start` to learn the call's sample rate BEFORE opening the STT socket.
    """

    def __init__(self, ws: WebSocket, main) -> None:
        super().__init__(ws, main)
        self.stream_sid: str | None = None
        self.caller: str = ""
        self.out_rate = 8000            # overwritten by start.media_format
        self._denoiser: denoise.Denoiser | None = None  # created after `start`
        self._agc: _Agc | None = None                   # created after `start`
        self._busy_played = False   # the rate-limit apology plays at most once
        self._outbuf = bytearray()      # PCM at out_rate, waiting to fill a chunk
        self._rs_state = None           # audioop.ratecv streaming state
        self._rs_active = False         # reset the state at each utterance start
        self._utt = 0
        self._awaiting_mark: str | None = None   # mark sent, echo not yet seen
        self._cleared: set[str] = set()          # marks killed by a barge-in
        self._hangup_mark: str | None = None     # close the call when this echoes
        # Paced outbound queue (see LEAD_SECONDS). Items: ("media", gen, bytes)
        # or ("mark", gen, name). A barge-in bumps _gen; the sender drops any
        # stale-generation item, so cancelled audio can never leak out late.
        self._sendq: asyncio.Queue = asyncio.Queue()
        self._gen = 0
        self._playout = 0.0                      # monotonic playout deadline
        # The current utterance's audio, at the call rate and already
        # levelled, so a false barge-in can resume it (_resume_interrupted_audio).
        self._utt_idx = 0
        self._utt_audio = bytearray()
        self._utt_complete = False               # all of it has been queued
        self._utt_end_call = False
        self._utt_sent_idx = -1                  # utterance whose first chunk went out
        self._utt_first_sent_at: float | None = None
        self._resume: dict | None = None         # snapshot taken by _clear()

    # -- lifecycle -----------------------------------------------------------

    async def run(self) -> None:
        try:
            try:
                ok = await asyncio.wait_for(self._await_start(), timeout=START_TIMEOUT)
            except asyncio.TimeoutError:
                log.warning("exotel: no start event within %ss — closing", START_TIMEOUT)
                return
            if not ok:
                return

            # The greeting is history's first line for the same reason as the
            # browser path: without it the model re-introduces itself.
            self.history.append({"role": "assistant", "content": prompts.GREETING})

            # A returning caller gets their file: notes into the prompt, and
            # last call's language pre-pinned so an English regular is never
            # greeted with a Hindi first reply. One local file read.
            mem = caller_memory.load(self.caller)
            if mem is not None:
                self.lead_context = caller_memory.context_block(mem)
                if lang := caller_memory.remembered_language(mem):
                    self.main.SESSION_LANG[self.session_id] = lang
                log.info(
                    "exotel: returning caller (%d prior call(s), language %s)",
                    mem.get("calls", 1), mem.get("language") or "unknown",
                )

            # The greeting takes ~7s to play — exactly enough cover to re-warm
            # a lapsed prompt cache so the caller's FIRST question isn't the
            # cold one. (The browser path gets this via /api/greeting; a phone
            # call never touches that endpoint. First real call paid cached=0.)
            self.main.prewarm_llm_if_stale()

            # Phone audio gets denoised before Sarvam hears it — see app/denoise.
            if settings.exotel_denoise:
                self._denoiser = denoise.maybe_create(
                    self.in_rate, gate=settings.exotel_speech_gate
                )
                if self._denoiser is None:
                    log.warning(
                        "exotel: denoise unavailable (%s) — raw phone audio",
                        denoise.unavailable_reason() or "init failed",
                    )
                else:
                    log.info(
                        "exotel: RNNoise active on inbound audio (speech gate %s)",
                        "on" if settings.exotel_speech_gate else "off",
                    )
            if settings.exotel_tts_target_rms > 0:
                self._agc = _Agc(
                    self.out_rate,
                    float(settings.exotel_tts_target_rms),
                    settings.exotel_tts_gain,
                )

            async with stt_stream.connect(
                sample_rate=self.in_rate,
                interrupt_min_speech_frames=settings.exotel_interrupt_min_speech_frames,
            ) as stt_session:
                self.stt = stt_session
                sender = asyncio.create_task(self._media_sender(), name="sender")
                greet = asyncio.create_task(self._speak_greeting(), name="greet")
                mic = asyncio.create_task(self._pump_mic(), name="mic")
                brain = asyncio.create_task(self._pump_stt(), name="stt")
                warm = asyncio.create_task(self._keep_llm_warm(), name="warm")
                try:
                    await asyncio.wait({mic, brain}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for t in (mic, brain, warm, greet, sender):
                        t.cancel()
                    if self._hold_task is not None:
                        self._hold_task.cancel()
                    await self._cancel_response()
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("exotel call session crashed")
        finally:
            self.closed = True
            if self._denoiser is not None:
                self._denoiser.close()
            # Write down what we learned about this caller — fire-and-forget,
            # the call is already over so this costs the caller nothing.
            if self.caller and len(self.history) >= 3:
                caller_memory.schedule_update(
                    self.caller,
                    list(self.history),
                    self.main.SESSION_LANG.get(self.session_id),
                )
            try:
                await self.ws.close()
            except Exception:
                pass

    async def _await_start(self) -> bool:
        """Consume messages until `start` arrives; learn stream_sid, the
        caller's number, and the negotiated sample rate."""
        while True:
            msg = await self.ws.receive()
            if msg["type"] == "websocket.disconnect":
                return False
            if (text := msg.get("text")) is None:
                continue
            try:
                payload = json.loads(text)
            except ValueError:
                continue
            event = payload.get("event")
            if event == "connected":
                continue
            if event == "start":
                start = payload.get("start") or {}
                self.stream_sid = payload.get("stream_sid") or start.get("stream_sid")
                self.caller = str(start.get("from") or "")
                fmt = start.get("media_format") or {}
                try:
                    rate = int(fmt.get("sample_rate") or 8000)
                except (TypeError, ValueError):
                    rate = 8000
                self._set_rate(rate)   # STT-side thresholds follow the call rate
                self.out_rate = rate   # we must speak back at the same rate
                self.session_id = f"exo-{self.stream_sid or 'unknown'}"
                log.info(
                    "exotel: call started sid=%s from=%s rate=%d",
                    self.stream_sid, self.caller[-4:].rjust(4, "*"), rate,
                )
                return True
            # Anything else before start (media from an eager sender) is dropped.

    async def _speak_greeting(self) -> None:
        """Speak the opening line the moment the stream is up. The browser plays
        a pre-rendered greeting via REST before opening its socket; on a phone
        call the socket IS the call, so the greeting goes down the same pipe."""
        try:
            wav = self.main.GREETING_WAV
            if wav is None:
                wav = await tts.synthesize(
                    prompts.GREETING,
                    prompts.GREETING_LANGUAGE,
                    pace=settings.sarvam_tts_pace_greeting,
                )
            frames, rate = _wav_pcm(wav)
            self.is_speaking = True
            await self._push_pcm(frames, rate)
            await self._finish_utterance(end_call=False)
        except Exception as exc:
            log.warning("exotel: greeting failed (%s) — call continues silent", exc)

    # -- Exotel -> brain -----------------------------------------------------

    async def _pump_mic(self) -> None:
        """Their media/mark/dtmf/stop events, mapped onto the parent's inputs."""
        while True:
            msg = await self.ws.receive()
            if msg["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()
            if (text := msg.get("text")) is None:
                continue
            try:
                payload = json.loads(text)
            except ValueError:
                continue
            event = payload.get("event")

            if event == "media":
                b64 = (payload.get("media") or {}).get("payload") or ""
                try:
                    pcm = base64.b64decode(b64)
                except binascii.Error:
                    continue
                if self._denoiser is not None and pcm:
                    # Clean it BEFORE Sarvam's VAD or the REST fallback hear it,
                    # so both paths judge the same (noise-free) audio.
                    pcm = self._denoiser.process(pcm)
                if self.stt is not None and pcm:
                    await self.stt.feed(pcm)
                    self.turn_buffer.extend(pcm)
                    if len(self.turn_buffer) > self._max_buffer_bytes:
                        del self.turn_buffer[:-self._max_buffer_bytes]

            elif event == "mark":
                name = (payload.get("mark") or {}).get("name") or ""
                if name in self._cleared:
                    # Echo for audio that played before a barge-in `clear` —
                    # looks identical to natural completion; must not re-arm.
                    self._cleared.discard(name)
                    continue
                if name == self._awaiting_mark:
                    self._awaiting_mark = None
                    # The caller has HEARD the whole reply (their server has
                    # transmitted the last frame) — browser's playback_done.
                    self._stop_speaking()
                if name and name == self._hangup_mark:
                    log.info("exotel: sign-off played — hanging up")
                    raise WebSocketDisconnect()

            elif event == "dtmf":
                digit = (payload.get("dtmf") or {}).get("digit")
                log.info("exotel: dtmf %r (ignored)", digit)

            elif event == "stop":
                reason = (payload.get("stop") or {}).get("reason")
                log.info("exotel: stream stopped (%s)", reason)
                raise WebSocketDisconnect()

    # -- brain -> Exotel -----------------------------------------------------

    async def _safe_send(self, **payload) -> None:
        """Translate the browser-protocol events the parent emits into Exotel's
        messages. transcript/listening/error have no wire equivalent on a phone
        call — they are logged, not sent."""
        if self.closed:
            return
        t = payload.get("type")
        try:
            if t == "audio_pcm":
                pcm = base64.b64decode(payload["pcm"])
                await self._push_pcm(pcm, int(payload.get("rate") or tts.TTS_SAMPLE_RATE))
            elif t == "audio":
                # REST-fallback path: a whole WAV instead of a PCM stream.
                frames, rate = _wav_pcm(base64.b64decode(payload["audio"]))
                await self._push_pcm(frames, rate)
            elif t == "done":
                await self._finish_utterance(end_call=bool(payload.get("end_call")))
            elif t == "interrupted":
                await self._clear()
            elif t == "error":
                log.warning("exotel: turn error: %s", payload.get("error"))
                # A rate-limited caller must HEAR something — dead air after the
                # greeting made the whole team think the bot was broken. The
                # apology is pre-rendered (main.BUSY_WAV), so playing it spends
                # no Sarvam credits. Once per call; repeats stay silent.
                if payload.get("kind") == "rate_limit" and not self._busy_played:
                    wav = getattr(self.main, "BUSY_WAV", None)
                    if wav and not self.is_speaking:
                        self._busy_played = True
                        frames, rate = _wav_pcm(wav)
                        self.is_speaking = True
                        await self._push_pcm(frames, rate)
                        await self._finish_utterance(end_call=False)
                        log.info("exotel: spoke the busy line")
            else:
                log.debug("exotel: %s event (browser-only, dropped)", t)
        except WebSocketDisconnect:
            self.closed = True
        except Exception as exc:
            log.warning("exotel: send failed (%s) — closing session", exc)
            self.closed = True

    async def _push_pcm(self, pcm: bytes, from_rate: int) -> None:
        """Resample one piece of Shubh's audio to the call rate, apply the
        optional telephony gain, and queue every full chunk for the paced
        sender. ratecv keeps streaming state so consecutive TTS chunks stay
        continuous across the resample."""
        if not pcm:
            return
        if not self._rs_active:
            self._begin_utterance()
        if from_rate != self.out_rate:
            pcm, self._rs_state = audioop.ratecv(pcm, 2, 1, from_rate, self.out_rate, self._rs_state)
        if self._agc is not None:
            pcm = self._agc.process(pcm)
        else:
            pcm = _compress(pcm, settings.exotel_tts_gain)
        self._queue_raw(pcm)

    def _begin_utterance(self) -> None:
        self._rs_state = None
        self._rs_active = True
        self._utt_idx += 1
        self._utt_audio = bytearray()
        self._utt_complete = False
        self._utt_end_call = False

    def _queue_raw(self, pcm: bytes) -> None:
        """Queue audio that is already at the call rate and already levelled."""
        self._utt_audio.extend(pcm)
        self._outbuf.extend(pcm)
        while len(self._outbuf) >= CHUNK_BYTES:
            piece = bytes(self._outbuf[:CHUNK_BYTES])
            del self._outbuf[:CHUNK_BYTES]
            self._sendq.put_nowait(("media", self._gen, self._utt_idx, piece))

    async def _finish_utterance(self, end_call: bool) -> None:
        """Queue the tail (padded to the 320-byte rule with silence), then a
        named mark — its echo is this utterance's playback_done."""
        if self._outbuf:
            tail = bytes(self._outbuf)
            self._outbuf.clear()
            pad = max(CHUNK_BYTES, ((len(tail) + 319) // 320) * 320) - len(tail)
            tail += b"\x00" * pad
            self._utt_audio.extend(tail)
            self._sendq.put_nowait(("media", self._gen, self._utt_idx, tail))
        self._rs_active = False
        self._utt_complete = True
        self._utt_end_call = end_call
        self._utt += 1
        name = f"utt-{self._utt}"
        self._awaiting_mark = name
        if end_call:
            self._hangup_mark = name
        self._sendq.put_nowait(("mark", self._gen, self._utt_idx, name))

    async def _resume_interrupted_audio(self, info) -> bool:
        """The interruption was nothing: carry on from just before the cut.
        Only when the whole reply had been generated — a reply cut while still
        streaming has no tail to resume, so the parent says it again."""
        snap, self._resume = self._resume, None
        if not snap or not snap["complete"]:
            return False
        bps = 2 * self.out_rate
        start = int(max(0.0, snap["played"] - RESUME_REWIND) * bps) & ~1
        rest = snap["audio"][start:]
        if len(rest) < int(0.6 * bps):
            log.info("exotel: the cut reply had all but played out — nothing to resume")
            return True
        log.info(
            "exotel: resuming the cut reply from %.1fs (%.1fs to go)", start / bps, len(rest) / bps
        )
        self.is_speaking = True
        self._begin_utterance()
        self._queue_raw(rest)
        await self._finish_utterance(end_call=snap["end_call"])
        return True

    async def _clear(self) -> None:
        """Barge-in: drop everything we haven't sent, tell Exotel to flush
        everything we have. Bumping the generation makes the sender discard
        any stale chunk it was already holding."""
        # Remember where the cut landed, in case the interruption proves to
        # be nothing and the reply should carry on (call_ws false barge-in).
        played = 0.0
        if self._utt_first_sent_at is not None and self._utt_sent_idx == self._utt_idx:
            played = max(0.0, asyncio.get_event_loop().time() - self._utt_first_sent_at)
        self._resume = {
            "audio": bytes(self._utt_audio),
            "played": played,
            "complete": self._utt_complete,
            "end_call": self._utt_end_call,
        }
        self._gen += 1
        self._playout = 0.0
        while not self._sendq.empty():
            try:
                self._sendq.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._outbuf.clear()
        self._rs_active = False
        if self._awaiting_mark:
            # Its echo may still arrive for the part that already played.
            self._cleared.add(self._awaiting_mark)
            self._awaiting_mark = None
        await self._send_json(event="clear", stream_sid=self.stream_sid)

    async def _media_sender(self) -> None:
        """The only task that writes media/mark frames, pacing them so Exotel
        holds at most LEAD_SECONDS of un-played audio. The first LEAD_SECONDS
        of a reply leave immediately (fast start); after that, chunks flow at
        playback rate — smooth on the uplink, and a barge-in has almost
        nothing queued remotely to flush."""
        loop = asyncio.get_event_loop()
        try:
            while True:
                kind, gen, utt, item = await self._sendq.get()
                if gen != self._gen:
                    continue  # cancelled by a barge-in while queued
                if kind == "mark":
                    await self._send_json(
                        event="mark", stream_sid=self.stream_sid, mark={"name": item}
                    )
                    continue
                now = loop.time()
                if utt != self._utt_sent_idx:
                    # First chunk of a new utterance leaves now: the clock a
                    # barge-in's "how much had played" estimate runs from.
                    self._utt_sent_idx = utt
                    self._utt_first_sent_at = now
                self._playout = max(self._playout, now) + len(item) / (2 * self.out_rate)
                ahead = self._playout - now - LEAD_SECONDS
                if ahead > 0:
                    await asyncio.sleep(ahead)
                    if gen != self._gen:
                        continue  # barge-in happened during the sleep
                await self._send_json(
                    event="media",
                    stream_sid=self.stream_sid,
                    media={"payload": base64.b64encode(item).decode("ascii")},
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("exotel: media sender died (%s) — closing session", exc)
            self.closed = True

    async def _send_json(self, **message) -> None:
        await self.ws.send_text(json.dumps(message))


def _compress(pcm: bytes, gain: float) -> bytes:
    """Telephony loudness: boost the speech body, soft-clip only the peaks.

    Bulbul's output already peaks at full scale, so plain gain would hard-clip
    into buzz. Instead: everything below the knee (26000) is multiplied
    cleanly — that's where nearly all speech energy lives — and whatever the
    boost pushes past the knee is squashed to a quarter slope. Perceived
    loudness rises ~6 dB at gain 2.0; distortion is confined to the loudest
    instants, which a phone speaker masks anyway.
    """
    if gain <= 1.01 or not pcm:
        return pcm
    samples = array.array("h")
    samples.frombytes(pcm)
    knee = 26000.0
    for i, s in enumerate(samples):
        v = s * gain
        if v > knee:
            v = knee + (v - knee) * 0.25
            if v > 32700.0:
                v = 32700.0
        elif v < -knee:
            v = -knee + (v + knee) * 0.25
            if v < -32700.0:
                v = -32700.0
        samples[i] = int(v)
    return samples.tobytes()


class _Agc:
    """Keep Shubh's SPOKEN level constant, not his gain.

    A fixed gain leaves Bulbul's own level swings audible: the head and tail of
    one answer synthesize on separate sockets, Hindi and English render at
    different energies, and the cached greeting differs from streamed replies —
    callers heard "volume keeps going up and down" (2026-09-02 round). This
    steers gain smoothly toward a target speech RMS in ~100ms windows: quiet
    sentences get lifted more, loud ones less, and the soft-knee clip from
    _compress still catches the peaks. Silent windows neither adapt (silence
    says nothing about level) nor pump.
    """

    def __init__(self, rate: int, target: float, gain: float = 2.0) -> None:
        self.rate = max(1, rate)
        self.target = target
        self.gain = min(max(gain, 1.0), 5.0)

    def process(self, pcm: bytes) -> bytes:
        if not pcm:
            return pcm
        out = bytearray()
        window = 2 * (self.rate // 10) or 3200  # ~100ms of 16-bit mono
        for i in range(0, len(pcm), window):
            piece = pcm[i : i + window]
            rms = audioop.rms(piece, 2)
            if rms > 500:  # only speech-bearing audio teaches the AGC
                desired = min(max(self.target / rms, 1.0), 5.0)
                self.gain += (desired - self.gain) * 0.20
            out.extend(_compress(piece, self.gain))
        return bytes(out)


def _wav_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    """Extract raw mono 16-bit PCM + rate from a WAV blob (Bulbul output)."""
    with wave.open(io.BytesIO(wav_bytes)) as w:
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
        if w.getsampwidth() != 2:
            frames = audioop.lin2lin(frames, w.getsampwidth(), 2)
        if w.getnchannels() == 2:
            frames = audioop.tomono(frames, 2, 0.5, 0.5)
    return frames, rate
