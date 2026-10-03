// Ships raw mono Float32 frames to the main thread, batched.
//
// The engine calls process() every render quantum — 128 samples, i.e. every
// 2.7ms at 48k. Posting each quantum individually meant ~375 messages/sec, and
// every one became a resample + WebSocket send to the server and a base64+JSON
// message to Sarvam: pure per-message overhead on the path that decides how
// fast a turn ends. Batching to 1024 samples (~21ms at 48k) cuts that 8x for
// at most one batch (~21ms) of added capture delay — under Sarvam's own VAD
// frame size, so turn endings are unaffected.
//
// The input buffer is recycled by the engine every quantum, so always copy.
const BATCH_SAMPLES = 1024;

class PCMRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(BATCH_SAMPLES);
    this.fill = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel || !channel.length) return true;

    let read = 0;
    while (read < channel.length) {
      const take = Math.min(channel.length - read, BATCH_SAMPLES - this.fill);
      this.buf.set(channel.subarray(read, read + take), this.fill);
      this.fill += take;
      read += take;
      if (this.fill === BATCH_SAMPLES) {
        this.port.postMessage(this.buf);
        this.buf = new Float32Array(BATCH_SAMPLES);
        this.fill = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-recorder", PCMRecorder);
