const TARGET_RATE = 16000;

// --- silence detection ------------------------------------------------------
// Replaces hold-to-talk: the browser decides when you've stopped speaking.
// Thresholds are the fragile part of a call UI, so they adapt to the room
// rather than trusting a fixed constant.
const CALIBRATE_MS = 400;   // sample the room before listening for speech
const SILENCE_MS = 800;     // hush this long after speech ends the turn
const NO_SPEECH_MS = 9000;  // said nothing at all -> re-arm, don't send
const MIN_SPEECH_MS = 250;  // shorter than this is a cough, not a question
const FLOOR_RMS = 0.008;    // absolute floor for a silent room
const SPEECH_FACTOR = 3.0;  // speech must beat the noise floor by this much
// Keep a little silence after the last word so STT hears a clean ending, but
// don't upload the full SILENCE_MS of nothing — that's dead weight to transcribe.
const KEEP_TAIL_MS = 250;

const els = {
  scope: document.getElementById("scope"),
  clock: document.getElementById("clock"),
  state: document.getElementById("state"),
  begin: document.getElementById("begin"),
  end: document.getElementById("end"),
  hint: document.getElementById("hint"),
  error: document.getElementById("error"),
  feed: document.getElementById("feed"),
  count: document.getElementById("count"),
  net: document.getElementById("net"),
  netLabel: document.getElementById("netLabel"),
  kbStat: document.getElementById("kbStat"),
  modeLead: document.getElementById("modeLead"),
  modeAssistant: document.getElementById("modeAssistant"),
  leadcard: document.getElementById("leadcard"),
  lcName: document.getElementById("lcName"),
  lcFacts: document.getElementById("lcFacts"),
  lcEmpty: document.getElementById("lcEmpty"),
};

const STATE_TEXT = {
  off: "Not connected",
  idle: "On call",
  work: "On call",
  live: "Listening",
  talk: "Shubh speaking",
};

const sessionId = crypto.randomUUID();
let maxSeconds = 25;
let inCall = false;
let listening = false;
let mic = null;          // capture graph, pinned to 16k for STT
let stream = null;
let node = null;
let micAnalyser = null;
let out = null;          // playback graph, native rate
let outAnalyser = null;
let player = null;
let callStart = 0;
let timerId = null;
let turns = 0;
let streamingMode = false; // Phase 3: server-VAD streaming call (set from /api/health)
let callWs = null;         // the persistent /ws/call socket when streamingMode is on
let mode = "lead";         // "lead" (outbound, Zoho) | "assistant" (inbound). Sidebar-driven.
let zohoEnabled = false;   // from /api/health — gates Lead Call mode

// ---- the meter -------------------------------------------------------------
// Fixed in place: nothing travels. Bands run low to high across the panel and
// each bar rises where the voice actually has energy, so the shape is the
// spectrum of whoever is talking rather than a decorative blob. Stop talking
// and the bars settle onto the axis and hold there.
//
// It reads the analyser on its own clock rather than being driven by the
// worklet, which posts ~125 times a second — that used to be ~125 style writes
// per second to animate a single div.

const BARS = 56;

class Meter {
  constructor(canvas) {
    this.canvas = canvas;
    this.g = canvas.getContext("2d");
    this.level = new Float32Array(BARS);
    this.target = new Float32Array(BARS);
    this.analyser = null;
    this.bins = null;
    this.topBin = 8;
    this.mode = "off";
    this.raf = null;
    this.dpr = 1;
    this.reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.loop = this.loop.bind(this);

    this.readTheme();
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  // Colours live in CSS so the palette stays in one place. Cache them —
  // getComputedStyle every frame would cost more than the drawing does.
  readTheme() {
    const cs = getComputedStyle(document.documentElement);
    const pick = (n, f) => cs.getPropertyValue(n).trim() || f;
    this.colors = {
      ink: pick("--ink", "#000000"),
      ink2: pick("--ink-2", "#5C5C5C"),
      rule: pick("--rule", "#DCDCDC"),
    };
  }

  resize() {
    const r = this.canvas.getBoundingClientRect();
    if (!r.width) return;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(r.width * this.dpr);
    this.canvas.height = Math.round(r.height * this.dpr);
    this.draw();
  }

  reset() {
    this.level.fill(0);
    this.target.fill(0);
  }

  set(mode, analyser) {
    this.mode = mode;
    this.analyser = analyser || null;
    if (analyser) {
      // The default dB window is -100..-30, which is far wider than speech
      // occupies: measured against real Bulbul audio every band pinned at
      // ~0.9 and the meter drew a solid block. Narrowing the window to where a
      // voice actually lives is what gives the bars their travel.
      analyser.minDecibels = -72;
      analyser.maxDecibels = -18;
      this.bins = new Uint8Array(analyser.frequencyBinCount);
      // A voice lives under ~4kHz. Map onto that band whatever the context's
      // rate is — capture runs at 16k, playback at whatever the device uses.
      const nyquist = analyser.context.sampleRate / 2;
      this.topBin = Math.max(8, Math.floor((4200 / nyquist) * analyser.frequencyBinCount));
    }
    this.start();
  }

  start() {
    if (this.raf === null) this.raf = requestAnimationFrame(this.loop);
  }

  measure() {
    if (!this.analyser) return this.target.fill(0);
    this.analyser.getByteFrequencyData(this.bins);

    const top = this.topBin;
    // Bands are spaced on a curve, not evenly: a voice sits at the bottom of
    // the range, and a linear split would cram all of it into the first few
    // bars and leave the rest dead.
    const edge = (i) => Math.min(top - 1, 1 + Math.round((i / BARS) ** 2 * (top - 1)));

    for (let i = 0; i < BARS; i++) {
      const lo = edge(i);
      const hi = Math.max(lo + 1, edge(i + 1));
      let sum = 0;
      for (let b = lo; b < hi; b++) sum += this.bins[b];
      // A gentle lift toward the top — speech thins out up there, but the byte
      // data is already dB-scaled, so this stays small. Anything more and the
      // whole meter saturates.
      const tilt = 1 + (i / BARS) * 0.3;
      this.target[i] = Math.min(1, (sum / (hi - lo) / 255) * tilt);
    }
  }

  loop() {
    this.raf = null;
    this.measure();

    // Fast attack, slow release — how a real meter behaves. Instant decay reads
    // as a glitch; slow attack loses the transient that makes it feel live.
    let moving = false;
    for (let i = 0; i < BARS; i++) {
      const d = this.target[i] - this.level[i];
      this.level[i] += d * (d > 0 ? 0.5 : 0.12);
      if (this.level[i] > 0.003) moving = true;
    }
    this.draw();
    // Keep polling while the line is open — silence still has to be noticed.
    // Once the call is over and the bars have settled, stop burning frames.
    if (this.mode !== "off" || moving) this.start();
  }

  draw() {
    const { g, canvas, dpr } = this;
    const W = canvas.width;
    const H = canvas.height;
    if (!W || !H) return;
    g.clearRect(0, 0, W, H);

    const mid = Math.round(H / 2);
    const unit = Math.max(1, Math.round(dpr));

    // The axis. Silence is this line, and the bars rest on it.
    g.fillStyle = this.colors.rule;
    g.fillRect(0, mid - unit / 2, W, unit);

    const voice = this.mode === "talk" ? this.colors.ink : this.colors.ink2;

    if (this.reduced) {
      // Reduced motion gets a plain level meter: one bar, no per-band spring.
      let sum = 0;
      for (let i = 0; i < BARS; i++) sum += this.level[i];
      const w = (sum / BARS) * W;
      g.fillStyle = voice;
      g.beginPath();
      if (g.roundRect) g.roundRect(0, mid - unit * 2, w, unit * 4, unit * 2);
      else g.rect(0, mid - unit * 2, w, unit * 4);
      g.fill();
      return;
    }

    const gap = 5 * dpr;
    const bw = Math.max(2 * dpr, (W - gap * (BARS - 1)) / BARS);
    const maxH = H * 0.84;

    g.fillStyle = voice;
    for (let i = 0; i < BARS; i++) {
      const h = Math.max(unit * 2, this.level[i] * maxH);
      const x = i * (bw + gap);
      g.beginPath();
      // Rounded caps, and the radius never exceeds half the bar so a resting
      // bar is a dot rather than a clipped rectangle.
      if (g.roundRect) g.roundRect(x, mid - h / 2, bw, h, Math.min(bw / 2, h / 2));
      else g.rect(x, mid - h / 2, bw, h);
      g.fill();
    }
  }
}

const scope = new Meter(els.scope);

// ---- health ----------------------------------------------------------------

fetch("/api/health")
  .then((r) => r.json())
  .then((h) => {
    maxSeconds = h.max_recording_seconds ?? 25;
    streamingMode = !!h.stt_streaming;
    zohoEnabled = !!h.zoho;
    initModes();
    els.kbStat.textContent =
      `kb ${h.kb_files} file${h.kb_files === 1 ? "" : "s"} · ` +
      `${h.kb_tokens.toLocaleString()} tokens · ${h.turns_today}/${h.turns_per_day} turns today`;

    const broken = h.missing_env?.length
      ? `Server is missing ${h.missing_env.join(", ")}.`
      : !h.kb_files
        ? "The knowledge base is empty — add files to kb_source/ and restart."
        : null;

    els.net.dataset.ok = broken ? "no" : "yes";
    els.netLabel.textContent = broken ? "line down" : "line ready";
    if (broken) {
      fail(broken);
      els.begin.disabled = true;
    }
  })
  .catch(() => {
    els.net.dataset.ok = "no";
    els.netLabel.textContent = "no server";
    fail("Can't reach the server.");
    els.begin.disabled = true;
  });

// ---- call mode (sidebar) ---------------------------------------------------

function initModes() {
  if (!zohoEnabled) {
    // No Zoho wired up -> Lead Call can't work; disable it, fall to Assistant.
    els.modeLead.disabled = true;
    els.modeLead.querySelector(".mode-d").textContent =
      "Unavailable — connect Zoho CRM to call live leads";
    mode = "assistant";
  }
  els.modeLead.addEventListener("click", () => setMode("lead"));
  els.modeAssistant.addEventListener("click", () => setMode("assistant"));
  setMode(mode);
}

function setMode(m) {
  if (inCall) return;                       // can't switch mid-call
  if (m === "lead" && !zohoEnabled) return; // guarded
  mode = m;
  els.modeLead.setAttribute("aria-pressed", String(m === "lead"));
  els.modeAssistant.setAttribute("aria-pressed", String(m === "assistant"));
  // The call button says what it will do (the last text node, after the icon).
  const label = els.begin.childNodes[els.begin.childNodes.length - 1];
  if (label && label.nodeType === Node.TEXT_NODE) {
    label.textContent = m === "lead" ? " Call lead" : " Call Shubh";
  }
  els.leadcard.hidden = m !== "lead";
  if (m === "lead") refreshLeadPreview();
}

function lockModes(locked) {
  els.modeAssistant.disabled = locked;
  els.modeLead.disabled = locked || !zohoEnabled;
}

async function refreshLeadPreview() {
  els.lcName.textContent = "Fetching…";
  els.lcFacts.innerHTML = "";
  els.lcEmpty.hidden = true;
  try {
    const d = await (await fetch("/api/lead/preview")).json();
    renderLead(d.lead);
  } catch {
    renderLead(null);
  }
}

function renderLead(lead) {
  els.lcFacts.innerHTML = "";
  if (!lead) {
    els.lcName.textContent = "No lead found";
    els.lcEmpty.hidden = false;
    els.lcEmpty.textContent = "Zoho returned no lead. Shubh will open without a name.";
    return;
  }
  els.lcEmpty.hidden = true;
  els.lcName.textContent = lead.name || "—";
  const rows = [
    ["City", lead.city],
    ["Budget", lead.budget],
    ["Timeline", lead.timeline],
    ["Interest", lead.interest],
    ["Status", lead.status],
  ].filter(([, v]) => v);
  for (const [k, v] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    els.lcFacts.append(dt, dd);
  }
}

// ---- state -----------------------------------------------------------------

function setPhase(phase) {
  els.state.dataset.phase = phase;
  els.state.textContent = STATE_TEXT[phase];
  scope.set(phase, phase === "live" ? micAnalyser : phase === "talk" ? outAnalyser : null);
}

function clock() {
  const s = callStart ? Math.floor((Date.now() - callStart) / 1000) : 0;
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function startTimer() {
  callStart = Date.now();
  els.clock.dataset.idle = "no";
  const tick = () => (els.clock.textContent = clock());
  tick();
  timerId = setInterval(tick, 1000);
}

function stopTimer() {
  clearInterval(timerId);
  timerId = null;
  els.clock.dataset.idle = "yes";
}

function fail(msg) {
  els.error.textContent = msg;
  els.error.hidden = false;
}

function clearError() {
  els.error.hidden = true;
}

function addTurn(who, text, cls) {
  els.feed.querySelector(".empty")?.remove();

  const wrap = document.createElement("div");
  wrap.className = `turn ${cls}`;

  const head = document.createElement("div");
  head.className = "head";
  const tag = document.createElement("span");
  tag.className = "who-tag";
  tag.textContent = who;
  const at = document.createElement("span");
  at.className = "at";
  at.textContent = clock();
  head.append(tag, at);

  const said = document.createElement("p");
  said.className = "said";
  said.textContent = text;

  wrap.append(head, said);
  els.feed.append(wrap);
  // Scroll the panel, not the page — scrollIntoView used to drag the whole
  // document down every turn.
  els.feed.scrollTop = els.feed.scrollHeight;
  return wrap;
}

function countTurns() {
  els.count.textContent = turns ? `${turns} exchange${turns > 1 ? "s" : ""}` : "";
}

// ---- WAV encoding ----------------------------------------------------------

function resample(input, from, to) {
  if (from === to) return input;
  const ratio = from / to;
  const out = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const pos = i * ratio;
    const left = Math.floor(pos);
    const right = Math.min(left + 1, input.length - 1);
    out[i] = input[left] + (input[right] - input[left]) * (pos - left);
  }
  return out;
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const str = (off, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  str(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  str(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

// ---- playback --------------------------------------------------------------

class VoicePlayer {
  /** Decodes each chunk and schedules it against the audio clock. */
  constructor(ctx, dest) {
    this.ctx = ctx;
    this.dest = dest;
    this.chain = Promise.resolve();  // decode is async; this keeps chunks in order
    this.nextAt = 0;
    this.live = new Set();
    this.pending = 0;
    this.stopped = false;
    this.idle = Promise.resolve();
    this._resolve = null;
  }

  push(b64) {
    if (this.stopped) return;
    if (this.pending === 0 && this.live.size === 0) {
      this.idle = new Promise((r) => (this._resolve = r));
    }
    this.pending++;
    this.chain = this.chain.then(() => this._run(b64));
  }

  // Raw linear16 PCM from the streaming path — no container, so decodeAudioData
  // can't parse it. Build the AudioBuffer straight from samples and schedule it
  // against the same clock as decoded chunks, so the two mix gaplessly.
  pushPCM(b64, rate) {
    if (this.stopped) return;
    if (this.pending === 0 && this.live.size === 0) {
      this.idle = new Promise((r) => (this._resolve = r));
    }
    this.pending++;
    this.chain = this.chain.then(() => this._runPCM(b64, rate));
  }

  _runPCM(b64, rate) {
    try {
      if (this.stopped) return;
      let bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      // PCM samples are 2 bytes; carry an odd trailing byte to the next chunk
      // so a sample never gets split across a boundary.
      if (this.pcmCarry?.length) {
        const merged = new Uint8Array(this.pcmCarry.length + bytes.length);
        merged.set(this.pcmCarry);
        merged.set(bytes, this.pcmCarry.length);
        bytes = merged;
      }
      const usable = bytes.length - (bytes.length % 2);
      this.pcmCarry = usable < bytes.length ? bytes.slice(usable) : null;
      if (usable === 0) return;

      const int16 = new Int16Array(bytes.buffer, 0, usable / 2);
      const f32 = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 32768;

      const buf = this.ctx.createBuffer(1, f32.length, rate);
      buf.copyToChannel(f32, 0);
      const src = this.ctx.createBufferSource();
      src.buffer = buf;
      src.connect(this.dest);
      const at = Math.max(this.ctx.currentTime + 0.02, this.nextAt);
      src.start(at);
      this.nextAt = at + buf.duration;
      this.live.add(src);
      src.onended = () => {
        this.live.delete(src);
        this._settle();
      };
    } catch (err) {
      console.warn("dropped a pcm chunk:", err);
    } finally {
      this.pending--;
      this._settle();
    }
  }

  async _run(b64) {
    try {
      if (this.stopped) return;
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      const buf = await this.ctx.decodeAudioData(bytes.buffer);
      if (this.stopped) return;

      const src = this.ctx.createBufferSource();
      src.buffer = buf;
      src.connect(this.dest);
      // Butt chunk N+1 against the end of chunk N on the audio clock. Waiting
      // for onended instead — which is what this used to do — left an audible
      // hole in the middle of a sentence that was only split for latency.
      const at = Math.max(this.ctx.currentTime + 0.02, this.nextAt);
      src.start(at);
      this.nextAt = at + buf.duration;
      this.live.add(src);
      src.onended = () => {
        this.live.delete(src);
        this._settle();
      };
    } catch (err) {
      console.warn("dropped an audio chunk:", err);
    } finally {
      this.pending--;
      this._settle();
    }
  }

  _settle() {
    if (!this.stopped && (this.pending > 0 || this.live.size > 0)) return;
    this._resolve?.();
    this._resolve = null;
  }

  stop() {
    this.stopped = true;
    for (const src of this.live) {
      try {
        src.stop();
      } catch {}
    }
    this.live.clear();
    this._settle();
  }

  done() {
    return this.idle;
  }
}

// ---- neural noise suppression (RNNoise) ------------------------------------
// A 48kHz RNNoise model compiled to WASM, run as an AudioWorklet in the mic
// chain. It strips background NOISE (traffic, fan, hum, keyboard) far better
// than the browser's built-in suppressor — it does NOT remove other people's
// voices (that's speaker isolation, a different, paid thing). Vendored under
// /vendor from @sapphi-red/web-noise-suppressor (MIT; RNNoise BSD).

let rnnoiseWasm = null; // ArrayBuffer, fetched once and reused across calls

async function makeDenoiser(ctx, source) {
  try {
    if (!rnnoiseWasm) {
      const r = await fetch("/vendor/rnnoise.wasm");
      if (!r.ok) throw new Error(`wasm ${r.status}`);
      rnnoiseWasm = await r.arrayBuffer();
    }
    await ctx.audioWorklet.addModule("/vendor/rnnoise-worklet.js");
    const denoise = new AudioWorkletNode(ctx, "@sapphi-red/web-noise-suppressor/rnnoise", {
      // structured-cloned into the worklet; slice so the original stays usable.
      processorOptions: { wasmBinary: rnnoiseWasm.slice(0), maxChannels: 1 },
    });
    source.connect(denoise);
    console.info("RNNoise: denoiser active");
    return denoise;
  } catch (e) {
    console.warn("RNNoise unavailable — using the raw mic:", e);
    return null; // caller falls back to the undenoised source
  }
}

// ---- the call --------------------------------------------------------------

async function beginCall() {
  clearError();
  els.begin.disabled = true;

  try {
    // One getUserMedia + one AudioContext for the whole call — re-acquiring the
    // mic each turn costs hundreds of ms and re-prompts on some browsers.
    stream = await navigator.mediaDevices.getUserMedia({
      // The browser's own DSP: cancel Shubh's echo, suppress steady noise, and
      // hold the input level steady so soft speech still transcribes. These are
      // the free, zero-latency wins; heavier denoise (RNNoise) would sit on top.
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (e) {
    els.begin.disabled = false;
    setPhase("off");
    fail(`Microphone blocked: ${e.message}. Needs HTTPS or localhost.`);
    return;
  }

  // Capture at 48k: RNNoise is a 48kHz model, and the mic frames are resampled
  // to 16k per-frame before they're sent to Sarvam anyway (see resample()). If a
  // browser refuses 48k we still work — the resample handles whatever rate we get.
  mic = new AudioContext({ sampleRate: 48000 });
  await mic.audioWorklet.addModule("/pcm-worklet.js");
  const source = mic.createMediaStreamSource(stream);
  node = new AudioWorkletNode(mic, "pcm-recorder");
  micAnalyser = mic.createAnalyser();
  micAnalyser.fftSize = 256;
  micAnalyser.smoothingTimeConstant = 0.72;

  // Neural noise suppression (RNNoise) sits right after the mic, so BOTH the
  // meter and the audio sent to Sarvam are denoised. Best-effort: if it can't
  // load, we fall back to the raw mic — a call must never break for this.
  const denoise = await makeDenoiser(mic, source);
  const head = denoise || source; // everything downstream taps the denoised signal
  head.connect(micAnalyser);
  const mute = mic.createGain();
  mute.gain.value = 0;
  head.connect(node).connect(mute).connect(mic.destination);

  // Playback gets its own context at the device's own rate. Sharing the capture
  // context would resample Bulbul's 22k output to the capture rate for nothing.
  out = new AudioContext();
  outAnalyser = out.createAnalyser();
  outAnalyser.fftSize = 256;
  outAnalyser.smoothingTimeConstant = 0.72;
  outAnalyser.connect(out.destination);

  inCall = true;
  lockModes(true);            // can't switch call mode mid-call
  els.begin.hidden = true;
  els.end.hidden = false;
  els.hint.textContent = "The line is open. Speak, then pause.";
  startTimer();
  scope.reset(); // each call gets a clean record; the last one shouldn't bleed in
  setPhase("idle");

  // Pre-rendered server-side, so this is instant and costs nothing.
  try {
    const g = await (
      await fetch("/api/greeting", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, session_id: sessionId }),
      })
    ).json();
    if (g.error) throw new Error(g.error);
    addTurn("Shubh", g.text, "bot");
    setPhase("talk");
    player = new VoicePlayer(out, outAnalyser);
    player.push(g.audio);
    await player.done();
  } catch (e) {
    fail(`Couldn't play the greeting: ${e.message}`);
  }

  // Streaming (Phase 3) keeps the mic open and lets the server's VAD decide when
  // a turn ends; the classic path records a clip and posts it. Either way we only
  // start after the greeting has finished, so it never echoes into the mic.
  if (inCall) {
    if (streamingMode) startStreamingCall();
    else listen();
  }
}

function endCall() {
  inCall = false;
  listening = false;
  if (callWs) {
    try {
      callWs.send(JSON.stringify({ type: "bye" }));
    } catch {}
    callWs.onclose = null;
    callWs.close();
    callWs = null;
  }
  player?.stop();
  if (node) node.port.onmessage = null;
  node?.disconnect();
  stream?.getTracks().forEach((t) => t.stop());
  mic?.close();
  out?.close();
  mic = out = stream = node = micAnalyser = outAnalyser = null;
  stopTimer();
  lockModes(false);          // free to pick a mode for the next call
  els.end.hidden = true;
  els.begin.hidden = false;
  els.begin.disabled = false;
  els.hint.textContent = "Call ended.";
  setPhase("off");
}

// Listens until you stop talking, then sends. The whole hold-to-talk button
// collapses into this.
function listen() {
  if (!inCall || listening) return;
  listening = true;
  setPhase("live");

  const chunks = [];
  const started = performance.now();
  const rate = mic.sampleRate;
  let noiseFloor = 0;
  let calibrated = false;
  let calibSum = 0;
  let calibCount = 0;
  let speechStart = 0;
  let lastVoice = 0;
  let samplesAtLastVoice = 0;
  let samples = 0;

  const finish = (send) => {
    if (!listening) return;
    listening = false;
    node.port.onmessage = null;
    if (!send || !inCall) {
      if (inCall) listen(); // heard nothing — re-arm rather than send silence
      return;
    }

    const flat = new Float32Array(samples);
    let off = 0;
    for (const c of chunks) {
      flat.set(c, off);
      off += c.length;
    }
    // Drop the trailing silence we spent detecting the pause: it adds nothing
    // to the transcript and STT is billed per second of audio.
    const keep = Math.min(samples, samplesAtLastVoice + Math.floor((KEEP_TAIL_MS / 1000) * rate));
    sendTurn(encodeWAV(resample(flat.subarray(0, keep), rate, TARGET_RATE), TARGET_RATE));
  };

  node.port.onmessage = (e) => {
    if (!listening) return;
    const buf = e.data;
    chunks.push(buf);
    samples += buf.length;

    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    const rms = Math.sqrt(sum / buf.length);
    const now = performance.now();

    // Learn the room's noise floor before deciding what counts as speech.
    if (!calibrated) {
      calibSum += rms;
      calibCount++;
      if (now - started >= CALIBRATE_MS) {
        noiseFloor = calibSum / Math.max(calibCount, 1);
        calibrated = true;
      }
      return;
    }

    // This runs ~125 times a second. It decides when your turn ends and does
    // nothing else — the scope reads the analyser on its own clock.
    const threshold = Math.max(FLOOR_RMS, noiseFloor * SPEECH_FACTOR);
    if (rms > threshold) {
      if (!speechStart) speechStart = now;
      lastVoice = now;
      samplesAtLastVoice = samples;
    }

    if (!speechStart) {
      if (now - started > NO_SPEECH_MS) finish(false);
      return;
    }
    if (now - started > maxSeconds * 1000) return finish(true);
    if (now - lastVoice > SILENCE_MS) finish(now - speechStart >= MIN_SPEECH_MS);
  };
}

async function sendTurn(blob) {
  setPhase("work"); // no "thinking" — a pause on a call speaks for itself
  const body = new FormData();
  body.append("audio", blob, "turn.wav");
  body.append("session_id", sessionId);

  player = new VoicePlayer(out, outAnalyser);
  let wrap = null;
  let hangUp = false; // Shubh decided the call is over — hang up after the sign-off

  try {
    const res = await fetch("/api/turn", { method: "POST", body });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Server error ${res.status}`);
    }

    for await (const evt of ndjson(res)) {
      if (evt.type === "transcript") {
        turns++;
        countTurns();
        addTurn("You", evt.transcript, "you");
        wrap = addTurn("Shubh", "…", "bot");
        wrap.querySelector(".said").classList.add("waiting");
      } else if (evt.type === "audio_pcm") {
        setPhase("talk");
        player.pushPCM(evt.pcm, evt.rate);
      } else if (evt.type === "audio") {
        setPhase("talk");
        player.push(evt.audio);
      } else if (evt.type === "done") {
        if (wrap) {
          const said = wrap.querySelector(".said");
          said.classList.remove("waiting");
          said.textContent = evt.answer;

          if (evt.citations?.length) {
            const c = document.createElement("div");
            c.className = "cites";
            c.textContent =
              "Source: " +
              evt.citations.map((x) => (x.page ? `${x.source} p${x.page}` : x.source)).join(", ");
            wrap.append(c);
          }
          const t = evt.timings || {};
          const m = document.createElement("div");
          m.className = "tel";
          m.textContent = `first audio ${t.first_audio_ms ?? "—"}ms · total ${t.total_ms ?? "—"}ms`;
          wrap.append(m);
          els.feed.scrollTop = els.feed.scrollHeight;
        }
        if (evt.end_call) hangUp = true; // caller said bye — don't re-arm the mic
      } else if (evt.type === "error") {
        throw new Error(evt.error);
      }
    }
  } catch (e) {
    player.stop();
    fail(e.message);
    // Don't strand the "…" bubble — that's the orphaned dot the caller sees when
    // a turn dies. Mark it as dropped so the transcript stays honest.
    if (wrap) {
      const said = wrap.querySelector(".said");
      said.classList.remove("waiting");
      said.textContent = "(that turn didn't go through — please say it again)";
    }
    if (inCall) setTimeout(listen, 400); // a bad turn shouldn't kill the call
    return;
  }

  await player.done();
  // Let the sign-off finish, then hang up like a person would — rather than
  // re-opening the mic and forcing the caller to say goodbye a second time.
  if (hangUp) {
    if (inCall) endCall();
    els.hint.textContent = "Shubh ended the call.";
    return;
  }
  if (inCall) listen();
}

async function* ndjson(res) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) yield JSON.parse(line);
    }
  }
  if (buf.trim()) yield JSON.parse(buf);
}

// ---- streaming call (Phase 3) ----------------------------------------------
// The mic streams continuously to /ws/call; Sarvam's VAD on the server marks the
// end of each turn (no client-side silence guess), and START_SPEECH while Shubh
// is talking is a barge-in — the server stops his audio and we cut playback.

function floatToPCM16(f32) {
  const buf = new ArrayBuffer(f32.length * 2);
  const view = new DataView(buf);
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

function renderDone(wrap, m) {
  if (!wrap) return;
  const said = wrap.querySelector(".said");
  said.classList.remove("waiting");
  said.textContent = m.answer;
  if (m.citations?.length) {
    const c = document.createElement("div");
    c.className = "cites";
    c.textContent =
      "Source: " + m.citations.map((x) => (x.page ? `${x.source} p${x.page}` : x.source)).join(", ");
    wrap.append(c);
  }
  const t = m.timings || {};
  const tel = document.createElement("div");
  tel.className = "tel";
  tel.textContent = `first audio ${t.first_audio_ms ?? "—"}ms · total ${t.total_ms ?? "—"}ms`;
  wrap.append(tel);
  els.feed.scrollTop = els.feed.scrollHeight;
}

function startStreamingCall() {
  const url = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/call";
  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  callWs = ws;

  const micRate = mic.sampleRate; // requested 16k; resample per-frame if the browser ignored it
  let wrap = null;
  let hangUp = false;

  ws.onopen = () => {
    ws.send(JSON.stringify({ type: "hello", session_id: sessionId, mode }));
    setPhase("live");
    els.hint.textContent = "The line is open — just talk, pause when you're done.";
    // Feed every mic frame straight to the server for the whole call.
    node.port.onmessage = (e) => {
      if (ws.readyState !== WebSocket.OPEN) return;
      const frame = micRate === TARGET_RATE ? e.data : resample(e.data, micRate, TARGET_RATE);
      ws.send(floatToPCM16(frame));
    };
  };

  ws.onmessage = (ev) => {
    let m;
    try {
      m = JSON.parse(ev.data);
    } catch {
      return;
    }

    if (m.type === "listening") {
      if (!player || player.stopped) setPhase("live");
    } else if (m.type === "transcript") {
      turns++;
      countTurns();
      addTurn("You", m.transcript, "you");
      wrap = addTurn("Shubh", "…", "bot");
      wrap.querySelector(".said").classList.add("waiting");
      player = new VoicePlayer(out, outAnalyser); // a fresh player per reply
    } else if (m.type === "audio_pcm") {
      setPhase("talk");
      player?.pushPCM(m.pcm, m.rate);
    } else if (m.type === "audio") {
      setPhase("talk");
      player?.push(m.audio);
    } else if (m.type === "interrupted") {
      // Barge-in: the caller talked over Shubh. Cut his audio at once, and drop
      // the reply he never finished — its "…" placeholder would otherwise linger.
      player?.stop();
      if (wrap && wrap.querySelector(".said")?.classList.contains("waiting")) {
        wrap.remove();
      }
      wrap = null;
      setPhase("live");
    } else if (m.type === "done") {
      renderDone(wrap, m);
      wrap = null;
      if (m.end_call) hangUp = true;
      // Wait for the audio to actually FINISH PLAYING (it's buffered and plays
      // for seconds after the last chunk arrives), then tell the server so it can
      // stop arming barge-in. Until this fires, the server keeps listening for an
      // interrupt — that's what lets you talk over him and have him stop.
      const p = player;
      (p ? p.done() : Promise.resolve()).then(() => {
        if (!inCall) return;
        if (p && p.stopped) return; // barged-in mid-playback; the interrupt path owns cleanup
        if (callWs?.readyState === WebSocket.OPEN) {
          callWs.send(JSON.stringify({ type: "playback_done" }));
        }
        if (hangUp) {
          endCall();
          els.hint.textContent = "Shubh ended the call.";
        } else {
          setPhase("live");
        }
      });
    } else if (m.type === "error") {
      fail(m.error);
    }
  };

  ws.onerror = () => {
    if (inCall) fail("Call connection error.");
  };
  ws.onclose = () => {
    if (inCall && !hangUp) {
      fail("The call connection dropped.");
      endCall();
    }
  };
}

els.begin.addEventListener("click", beginCall);
els.end.addEventListener("click", endCall);
setPhase("off");
