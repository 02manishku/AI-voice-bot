// Ships raw mono Float32 frames to the main thread. The input buffer is
// recycled by the engine every render quantum, so copy before posting.
class PCMRecorder extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel && channel.length) {
      this.port.postMessage(new Float32Array(channel));
    }
    return true;
  }
}

registerProcessor("pcm-recorder", PCMRecorder);
