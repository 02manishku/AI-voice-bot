"""RNNoise for the telephony leg — clean the caller's audio BEFORE Sarvam hears it.

Real phone calls come from kitchens, streets and speakerphones. On the first
team test round (2026-08-31) that background noise kept opening Sarvam's VAD:
junk fragments became turns, barge-ins fired every second and cut Shubh
mid-sentence, and the rate limiter burned its budget on noise while real
questions got dropped. The browser leg already runs RNNoise in the mic chain
(static/); this gives the phone leg the same treatment, server-side.

Implementation notes:
  - Uses pyrnnoise's raw ctypes binding (create/destroy/process_frame), NOT its
    audiolab Graph wrapper — the wrapper drags PyAV filter graphs into a hot
    path that runs every 20ms. Here: audioop.ratecv up to RNNoise's native
    48kHz, one C call per 10ms frame, ratecv back down. Both resamplers keep
    streaming state, so chunk boundaries stay artifact-free.
  - Latency added is only the framing remainder (<10ms of buffered audio) plus
    ~0.1ms of compute per chunk — measured, see scratchpad/rnnoise_probe.py.
  - Import is guarded. If pyrnnoise is missing or its DLL fails to load, calls
    run exactly as before (pass-through) and one warning is logged at startup.
"""

import audioop
import logging

log = logging.getLogger(__name__)

try:
    import numpy as _np
    from pyrnnoise import rnnoise as _rn

    _AVAILABLE = True
except Exception as _exc:  # pragma: no cover - environment-dependent
    _AVAILABLE = False
    _IMPORT_ERROR = _exc

_FRAME_SAMPLES = 480 if not _AVAILABLE else _rn.FRAME_SIZE  # 10ms @ 48k
_RN_RATE = 48000
_FRAME_BYTES = _FRAME_SAMPLES * 2


def available() -> bool:
    return _AVAILABLE


def unavailable_reason() -> str:
    return "" if _AVAILABLE else str(_IMPORT_ERROR)


class Denoiser:
    """Streaming RNNoise for one call. Feed 16-bit mono PCM at `rate`, get the
    same format back, denoised. Stateful — one instance per call, never shared.

    With `gate=True` it also ducks OTHER PEOPLE'S voices: RNNoise passes any
    human speech, so someone talking near the caller still reached Sarvam and
    became junk turns (2026-09-02 calls). The gate tracks the caller's own
    speech level (EMA of frame RMS on confident-speech frames) and attenuates
    frames far quieter than it — a background talker is farther from the mic,
    so their frames sit well below the caller's level. Deliberately gentle:
    duck ×0.15, never mute, and only once a reliable caller level exists."""

    _GATE_RATIO = 0.22      # frames below this fraction of the caller's level duck
    _GATE_DUCK = 0.15       # how much of the frame survives the duck
    _GATE_FLOOR = 250.0     # absolute near-silence floor
    _GATE_MIN_LEVEL = 1500.0  # don't judge until the caller level is this solid

    def __init__(self, rate: int, gate: bool = False) -> None:
        self.rate = rate
        self._state = _rn.create()
        self._up_state = None    # ratecv state: call rate -> 48k
        self._down_state = None  # ratecv state: 48k -> call rate
        self._pending = bytearray()  # 48k samples waiting to fill a frame
        self._gate = gate
        self._speech_level = 0.0  # EMA of the caller's own speech RMS (48k domain)

    def close(self) -> None:
        if self._state is not None:
            _rn.destroy(self._state)
            self._state = None

    def process(self, pcm: bytes) -> bytes:
        """Denoise one chunk. Returns the cleaned chunk — possibly a few ms
        shorter or longer than the input while the framing remainder shifts,
        but sample-continuous across calls."""
        if not pcm or self._state is None:
            return pcm
        if self.rate != _RN_RATE:
            up, self._up_state = audioop.ratecv(pcm, 2, 1, self.rate, _RN_RATE, self._up_state)
        else:
            up = pcm
        self._pending.extend(up)

        out48 = bytearray()
        while len(self._pending) >= _FRAME_BYTES:
            frame = _np.frombuffer(bytes(self._pending[:_FRAME_BYTES]), dtype=_np.int16)
            del self._pending[:_FRAME_BYTES]
            denoised, prob = _rn.process_frame(self._state, frame)
            denoised = denoised.astype(_np.int16, copy=False)
            if self._gate:
                denoised = self._apply_gate(denoised, prob)
            out48.extend(denoised.tobytes())

        if not out48:
            return b""
        if self.rate != _RN_RATE:
            down, self._down_state = audioop.ratecv(
                bytes(out48), 2, 1, _RN_RATE, self.rate, self._down_state
            )
            return down
        return bytes(out48)


    def _apply_gate(self, frame: "_np.ndarray", prob) -> "_np.ndarray":
        """Duck frames that are clearly not the caller (see class docstring)."""
        try:
            p = float(_np.ravel(_np.asarray(prob, dtype=_np.float64))[0])
        except Exception:
            p = 1.0  # can't read the probability -> never duck on its account
        rms = float(_np.sqrt(_np.mean(frame.astype(_np.float64) ** 2))) if frame.size else 0.0

        # Learn the caller's level only from confident, non-quiet speech:
        # fast attack upward (a real caller asserts their level in a beat),
        # slow release downward (a pause must not collapse the reference).
        if p > 0.6 and rms > self._GATE_FLOOR:
            step = 0.30 if rms > self._speech_level else 0.02
            self._speech_level += (rms - self._speech_level) * step

        if self._speech_level < self._GATE_MIN_LEVEL:
            return frame  # no reliable reference yet — touch nothing
        if rms < self._GATE_FLOOR or rms < self._GATE_RATIO * self._speech_level:
            return (frame.astype(_np.float64) * self._GATE_DUCK).astype(_np.int16)
        return frame


def maybe_create(rate: int, gate: bool = False) -> "Denoiser | None":
    """A Denoiser if RNNoise is usable, else None (caller passes audio through)."""
    if not _AVAILABLE:
        return None
    try:
        return Denoiser(rate, gate=gate)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("denoise: could not start (%s) — passing audio through", exc)
        return None
