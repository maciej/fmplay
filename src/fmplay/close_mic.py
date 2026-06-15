from __future__ import annotations

import argparse
import math
import random
import subprocess
import sys
import wave
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

SAMPLE_RATE = 16_000
INTENSITIES = ("subtle", "normal", "hot", "abusive")
DEFAULT_STREAM_CHUNK_SAMPLES = 4096
FloatArray = NDArray[np.float64]


class CloseMicError(RuntimeError):
    """Raised when close-mic processing cannot complete."""


@dataclass(frozen=True)
class CloseMicPreset:
    proximity: tuple[float, float]
    bad_technique: tuple[float, float]
    drive_db: tuple[float, float]
    pop_probability: tuple[float, float]
    pop_gain: tuple[float, float]
    burst_gain: tuple[float, float]
    breath_probability: tuple[float, float]
    click_probability: tuple[float, float]
    hard_clip_probability: tuple[float, float]
    compressor_ratio: tuple[float, float]
    channel_lowpass: tuple[float, float]


PRESETS: dict[str, CloseMicPreset] = {
    "subtle": CloseMicPreset(
        proximity=(0.25, 0.48),
        bad_technique=(0.05, 0.24),
        drive_db=(2.5, 5.5),
        pop_probability=(0.12, 0.26),
        pop_gain=(0.18, 0.34),
        burst_gain=(0.012, 0.030),
        breath_probability=(0.05, 0.15),
        click_probability=(0.01, 0.03),
        hard_clip_probability=(0.00, 0.04),
        compressor_ratio=(2.2, 3.0),
        channel_lowpass=(3150, 3400),
    ),
    "normal": CloseMicPreset(
        proximity=(0.42, 0.75),
        bad_technique=(0.18, 0.52),
        drive_db=(5.0, 9.0),
        pop_probability=(0.28, 0.54),
        pop_gain=(0.38, 0.78),
        burst_gain=(0.025, 0.062),
        breath_probability=(0.10, 0.26),
        click_probability=(0.02, 0.06),
        hard_clip_probability=(0.03, 0.12),
        compressor_ratio=(2.8, 4.2),
        channel_lowpass=(2900, 3250),
    ),
    "hot": CloseMicPreset(
        proximity=(0.68, 0.92),
        bad_technique=(0.42, 0.80),
        drive_db=(8.0, 13.0),
        pop_probability=(0.52, 0.82),
        pop_gain=(0.72, 1.30),
        burst_gain=(0.045, 0.105),
        breath_probability=(0.18, 0.42),
        click_probability=(0.04, 0.11),
        hard_clip_probability=(0.10, 0.26),
        compressor_ratio=(3.4, 5.3),
        channel_lowpass=(2700, 3150),
    ),
    "abusive": CloseMicPreset(
        proximity=(0.86, 1.0),
        bad_technique=(0.70, 1.0),
        drive_db=(12.0, 17.0),
        pop_probability=(0.72, 0.96),
        pop_gain=(1.05, 1.75),
        burst_gain=(0.08, 0.16),
        breath_probability=(0.28, 0.58),
        click_probability=(0.08, 0.18),
        hard_clip_probability=(0.24, 0.46),
        compressor_ratio=(4.3, 7.0),
        channel_lowpass=(2450, 2950),
    ),
}


@dataclass(frozen=True)
class CloseMicState:
    seed: int
    intensity: str
    proximity: float
    bad_technique: float
    drive_db: float
    pop_probability: float
    pop_gain: float
    burst_gain: float
    breath_probability: float
    click_probability: float
    hard_clip_probability: float
    compressor_ratio: float
    channel_lowpass: float
    presence_db: float


@dataclass(frozen=True)
class Onset:
    index: int
    strength: float
    centroid: float
    preceded_by_pause: bool


def render_file(
    source_path: Path,
    output_path: Path,
    *,
    seed: int,
    intensity: str,
    ffmpeg_command: str = "ffmpeg",
) -> None:
    samples = decode_audio(source_path, ffmpeg_command=ffmpeg_command)
    output = process_array(samples, seed=seed, intensity=intensity)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(output_path, output)


def stream_file(
    source_path: Path,
    output: BinaryIO,
    *,
    seed: int,
    intensity: str,
    ffmpeg_command: str = "ffmpeg",
    chunk_samples: int = DEFAULT_STREAM_CHUNK_SAMPLES,
) -> None:
    processor = CloseMicStreamProcessor(seed=seed, intensity=intensity)
    for samples in decode_audio_chunks(
        source_path,
        ffmpeg_command=ffmpeg_command,
        chunk_samples=chunk_samples,
    ):
        output.write(float_to_s16le(processor.process_chunk(samples)))
    tail = processor.finish()
    if tail.size:
        output.write(float_to_s16le(tail))


class OnePoleFilter:
    def __init__(self, cutoff: float) -> None:
        alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff / SAMPLE_RATE)
        self.b = [alpha]
        self.a = [1.0, -(1.0 - alpha)]
        self.zi = np.array([0.0], dtype=np.float64)

    def process(self, samples: FloatArray) -> FloatArray:
        output, self.zi = lfilter(self.b, self.a, samples, zi=self.zi)
        return output


class HighpassFilter:
    def __init__(self, cutoff: float) -> None:
        self.lowpass = OnePoleFilter(cutoff)

    def process(self, samples: FloatArray) -> FloatArray:
        return samples - self.lowpass.process(samples)


class BandpassFilter:
    def __init__(self, low: float, high: float) -> None:
        self.highpass = HighpassFilter(low)
        self.lowpass = OnePoleFilter(high)

    def process(self, samples: FloatArray) -> FloatArray:
        return self.lowpass.process(self.highpass.process(samples))


class StreamingRmsNormalizer:
    def __init__(self, target_db: float) -> None:
        self.target = db_to_amp(target_db)
        self.level = db_to_amp(target_db)

    def process(self, samples: FloatArray) -> FloatArray:
        if samples.size == 0:
            return samples
        chunk_rms = rms(samples)
        if chunk_rms > db_to_amp(-50):
            self.level = 0.92 * self.level + 0.08 * chunk_rms
        gain = self.target / max(self.level, 1e-9)
        gain = min(gain, db_to_amp(18.0))
        return samples * gain


class StreamingCompressor:
    def __init__(self, state: CloseMicState) -> None:
        self.state = state
        self.envelope = 0.0
        self.threshold = db_to_amp(-17.0 + 3.0 * state.bad_technique)
        self.attack = math.exp(-1.0 / (0.007 * SAMPLE_RATE))
        self.release = math.exp(
            -1.0 / ((0.070 + 0.090 * state.proximity) * SAMPLE_RATE)
        )

    def process(self, samples: FloatArray) -> FloatArray:
        output = np.empty_like(samples)
        for index, sample in enumerate(samples):
            level = abs(sample)
            coeff = self.attack if level > self.envelope else self.release
            self.envelope = coeff * self.envelope + (1.0 - coeff) * level
            if self.envelope <= self.threshold:
                gain = 1.0
            else:
                over_db = amp_to_db(self.envelope / self.threshold)
                gain_db = -over_db * (1.0 - 1.0 / self.state.compressor_ratio)
                gain = db_to_amp(gain_db)
            output[index] = sample * gain * 1.35
        return output


class StreamingLimiter:
    def __init__(self, *, limit: float) -> None:
        self.limit = limit
        self.release = math.exp(-1.0 / (0.025 * SAMPLE_RATE))
        self.gain = 1.0

    def process(self, samples: FloatArray) -> FloatArray:
        output = np.empty_like(samples)
        for index, sample in enumerate(samples):
            target = min(1.0, self.limit / max(abs(sample), 1e-9))
            if target < self.gain:
                self.gain = target
            else:
                self.gain = self.release * self.gain + (1.0 - self.release) * target
            output[index] = max(-self.limit, min(self.limit, sample * self.gain))
        return output


class StreamingOutputMatcher:
    def __init__(self, *, target_db: float) -> None:
        self.target = db_to_amp(target_db)
        self.level = self.target
        self.gain = 1.0

    def process(self, samples: FloatArray) -> FloatArray:
        if samples.size == 0:
            return samples
        chunk_rms = rms(samples)
        if chunk_rms > db_to_amp(-55):
            self.level = 0.96 * self.level + 0.04 * chunk_rms
            target_gain = self.target / max(self.level, 1e-9)
            self.gain = 0.98 * self.gain + 0.02 * target_gain
        return samples * min(self.gain, db_to_amp(12.0))


class CloseMicStreamProcessor:
    """Chunk-oriented approximation of the close-mic profile.

    The offline renderer can use whole-clip statistics and place events with
    full lookahead. This processor keeps bounded state instead: running gain,
    persistent IIR filters, compressor/limiter envelopes, and local onset
    context. That makes it suitable for pipes while preserving the same seeded
    close-mic character.
    """

    def __init__(self, *, seed: int, intensity: str) -> None:
        if intensity not in PRESETS:
            raise CloseMicError(
                f"unknown close-mic intensity {intensity!r}; expected "
                f"{', '.join(INTENSITIES)}"
            )
        self.rng = random.Random(seed)
        self.state = sample_state(self.rng, seed=seed, intensity=intensity)
        self.input_gain = StreamingRmsNormalizer(target_db=-22.0)
        self.proximity_low = OnePoleFilter(320.0)
        self.proximity_low_mid = BandpassFilter(170.0, 620.0)
        self.proximity_presence = BandpassFilter(1400.0, 3000.0)
        self.compressor = StreamingCompressor(self.state)
        self.channel_highpass = HighpassFilter(205.0)
        self.channel_lowpass = OnePoleFilter(self.state.channel_lowpass)
        self.channel_low_mid = BandpassFilter(430.0, 850.0)
        self.channel_presence = BandpassFilter(1450.0, 2500.0)
        self.pre_limiter = StreamingLimiter(limit=0.966)
        self.output_matcher = StreamingOutputMatcher(
            target_db=target_output_rms(self.state)
        )
        self.final_limiter = StreamingLimiter(limit=0.966)
        self.context = np.array([], dtype=np.float64)
        self.context_samples = int(0.24 * SAMPLE_RATE)
        self.position = 0
        self.last_onset_index = -10**9

    def process_chunk(
        self, samples: Sequence[float] | NDArray[np.floating]
    ) -> FloatArray:
        x = np.asarray(samples, dtype=np.float64)
        if x.size == 0:
            return x.copy()

        x = self.input_gain.process(x)
        x = self._apply_proximity(x)
        onsets = self._detect_stream_onsets(x)
        x = inject_plosives_and_breaths(x, onsets, self.state, self.rng)
        x = apply_event_drive_and_clip(x, onsets, self.state, self.rng)
        x = saturate(x, self.state)
        x = self.compressor.process(x)
        x = self._channel_filter(x)
        x = inject_radio_pop_residue(x, onsets, self.state, self.rng)
        x = self.pre_limiter.process(x)
        x = self.output_matcher.process(x)
        x = self.final_limiter.process(x)
        self._remember_context(x)
        self.position += x.size
        return x

    def finish(self) -> FloatArray:
        return np.array([], dtype=np.float64)

    def _apply_proximity(self, samples: FloatArray) -> FloatArray:
        low_gain = 0.45 + 1.15 * self.state.proximity
        low_mid_gain = 0.16 + 0.55 * self.state.proximity
        presence_gain = 0.06 + 0.12 * (1.0 - self.state.bad_technique)
        return (
            samples
            + low_gain * self.proximity_low.process(samples)
            + low_mid_gain * self.proximity_low_mid.process(samples)
            + presence_gain * self.proximity_presence.process(samples)
        )

    def _channel_filter(self, samples: FloatArray) -> FloatArray:
        x = self.channel_highpass.process(samples)
        x = self.channel_lowpass.process(x)
        low_mid = self.channel_low_mid.process(x)
        presence = self.channel_presence.process(x)
        presence_gain = db_to_amp(self.state.presence_db) - 1.0
        return x - 0.18 * low_mid + presence_gain * 0.42 * presence

    def _detect_stream_onsets(self, samples: FloatArray) -> list[Onset]:
        combined = np.concatenate((self.context, samples))
        context_size = self.context.size
        onsets: list[Onset] = []
        for onset in detect_onsets(combined):
            local_index = onset.index - context_size
            if local_index < 0 or local_index >= samples.size:
                continue
            absolute_index = self.position + local_index
            if absolute_index - self.last_onset_index <= int(0.045 * SAMPLE_RATE):
                continue
            self.last_onset_index = absolute_index
            onsets.append(
                Onset(
                    index=local_index,
                    strength=onset.strength,
                    centroid=onset.centroid,
                    preceded_by_pause=onset.preceded_by_pause,
                )
            )
        return onsets

    def _remember_context(self, samples: FloatArray) -> None:
        combined = np.concatenate((self.context, samples))
        self.context = combined[-self.context_samples :]


def process_samples(
    samples: Sequence[float] | NDArray[np.floating],
    *,
    seed: int,
    intensity: str,
) -> list[float]:
    return process_array(samples, seed=seed, intensity=intensity).tolist()


def process_array(
    samples: Sequence[float] | NDArray[np.floating],
    *,
    seed: int,
    intensity: str,
) -> FloatArray:
    if intensity not in PRESETS:
        raise CloseMicError(
            f"unknown close-mic intensity {intensity!r}; expected "
            f"{', '.join(INTENSITIES)}"
        )
    rng = random.Random(seed)
    state = sample_state(rng, seed=seed, intensity=intensity)
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return x.copy()

    x = normalize_active(x, target_db=-22.0)
    x = apply_proximity(x, state)
    onsets = detect_onsets(x)
    x = inject_plosives_and_breaths(x, onsets, state, rng)
    x = apply_event_drive_and_clip(x, onsets, state, rng)
    x = saturate(x, state)
    x = compress(x, state)
    x = channel_filter(x, state)
    x = inject_radio_pop_residue(x, onsets, state, rng)
    x = limiter(x, limit=0.966)
    x = match_rms(x, target_db=target_output_rms(state))
    return limiter(x, limit=0.966)


def sample_state(rng: random.Random, *, seed: int, intensity: str) -> CloseMicState:
    preset = PRESETS[intensity]
    proximity = rng.uniform(*preset.proximity)
    bad_technique = rng.uniform(*preset.bad_technique)
    correlated = 0.58 * proximity + 0.42 * bad_technique
    return CloseMicState(
        seed=seed,
        intensity=intensity,
        proximity=proximity,
        bad_technique=bad_technique,
        drive_db=lerp_range(preset.drive_db, correlated, rng, 0.8),
        pop_probability=lerp_range(preset.pop_probability, correlated, rng, 0.7),
        pop_gain=lerp_range(preset.pop_gain, correlated, rng, 0.7),
        burst_gain=lerp_range(preset.burst_gain, correlated, rng, 0.7),
        breath_probability=lerp_range(
            preset.breath_probability, bad_technique, rng, 0.5
        ),
        click_probability=lerp_range(preset.click_probability, bad_technique, rng, 0.5),
        hard_clip_probability=lerp_range(
            preset.hard_clip_probability, correlated, rng, 0.75
        ),
        compressor_ratio=lerp_range(preset.compressor_ratio, correlated, rng, 0.4),
        channel_lowpass=lerp_range(preset.channel_lowpass, 1.0 - proximity, rng, 0.35),
        presence_db=1.3 + 2.1 * correlated + rng.uniform(-0.3, 0.3),
    )


def lerp_range(
    range_values: tuple[float, float],
    control: float,
    rng: random.Random,
    jitter: float,
) -> float:
    low, high = range_values
    control = min(1.0, max(0.0, control + rng.uniform(-0.15, 0.15) * jitter))
    return low + (high - low) * control


def target_output_rms(state: CloseMicState) -> float:
    if state.intensity == "subtle":
        return -15.6
    if state.intensity == "normal":
        return -13.8
    if state.intensity == "hot":
        return -12.4
    return -11.6


def decode_audio(source_path: Path, *, ffmpeg_command: str) -> FloatArray:
    command = [
        ffmpeg_command,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        raw = subprocess.check_output(command)
    except FileNotFoundError as exc:
        raise CloseMicError(f"{ffmpeg_command} was not found") from exc
    except subprocess.CalledProcessError as exc:
        raise CloseMicError(f"{ffmpeg_command} failed decoding {source_path}") from exc
    return np.frombuffer(raw, dtype="<f4").astype(np.float64)


def decode_audio_chunks(
    source_path: Path,
    *,
    ffmpeg_command: str,
    chunk_samples: int = DEFAULT_STREAM_CHUNK_SAMPLES,
) -> Iterator[FloatArray]:
    command = [
        ffmpeg_command,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise CloseMicError(f"{ffmpeg_command} was not found") from exc

    assert process.stdout is not None
    bytes_per_chunk = max(1, chunk_samples) * 4
    remainder = b""
    while True:
        raw = process.stdout.read(bytes_per_chunk)
        if not raw:
            break
        raw = remainder + raw
        aligned = len(raw) - (len(raw) % 4)
        remainder = raw[aligned:]
        if aligned:
            yield np.frombuffer(raw[:aligned], dtype="<f4").astype(np.float64)

    if remainder:
        raise CloseMicError(f"{ffmpeg_command} produced partial f32 samples")

    stderr = process.stderr.read() if process.stderr is not None else b""
    return_code = process.wait()
    if return_code:
        details = stderr.decode(errors="replace").strip()
        suffix = f": {details.splitlines()[-1]}" if details else ""
        raise CloseMicError(f"{ffmpeg_command} failed decoding {source_path}{suffix}")


def write_wav(path: Path, samples: Sequence[float] | NDArray[np.floating]) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(float_to_s16le(samples))


def float_to_s16le(samples: Sequence[float] | NDArray[np.floating]) -> bytes:
    array = np.asarray(samples, dtype=np.float64)
    pcm = np.rint(np.clip(array, -1.0, 1.0) * 32767.0).astype("<i2")
    return pcm.tobytes()


def normalize_active(samples: FloatArray, *, target_db: float) -> FloatArray:
    frame = int(0.025 * SAMPLE_RATE)
    hop = int(0.010 * SAMPLE_RATE)
    levels = frame_rms(samples, frame, hop)
    if levels.size == 0:
        return samples
    threshold = max(percentile(levels, 55) * 0.75, db_to_amp(-45))
    active = levels[levels >= threshold]
    active_rms = percentile(active if active.size else levels, 70)
    gain = db_to_amp(target_db) / max(active_rms, 1e-9)
    return samples * gain


def apply_proximity(samples: FloatArray, state: CloseMicState) -> FloatArray:
    low = lowpass_onepole(samples, 320.0)
    low_mid = bandpass_cheap(samples, 170.0, 620.0)
    presence = bandpass_cheap(samples, 1400.0, 3000.0)
    low_gain = 0.45 + 1.15 * state.proximity
    low_mid_gain = 0.16 + 0.55 * state.proximity
    presence_gain = 0.06 + 0.12 * (1.0 - state.bad_technique)
    return samples + low_gain * low + low_mid_gain * low_mid + presence_gain * presence


def detect_onsets(samples: FloatArray) -> list[Onset]:
    frame = int(0.010 * SAMPLE_RATE)
    hop = int(0.005 * SAMPLE_RATE)
    levels = frame_rms(samples, frame, hop)
    if levels.size < 5:
        return []
    level_db = 20.0 * np.log10(np.maximum(levels, 1e-12))
    onsets: list[Onset] = []
    median_level = percentile(levels, 55)
    active_peak = percentile(levels, 76)
    for index in range(2, levels.size - 2):
        rise = level_db[index] - level_db[index - 1]
        previous_floor = min(levels[max(0, index - 5) : index])
        if rise < 3.2 or levels[index] < median_level * 0.72:
            continue
        if previous_floor > levels[index] * 0.78:
            continue
        sample_index = index * hop
        window = samples[sample_index : sample_index + int(0.050 * SAMPLE_RATE)]
        strength = min(1.0, max(0.0, (rise - 4.0) / 14.0))
        strength *= min(1.0, levels[index] / max(percentile(levels, 85), 1e-9))
        centroid = zero_crossing_rate(window)
        pause = previous_floor < median_level * 0.34
        if not onsets or sample_index - onsets[-1].index > int(0.045 * SAMPLE_RATE):
            onsets.append(Onset(sample_index, strength, centroid, pause))

    for index in range(3, levels.size - 3):
        if levels[index] < active_peak:
            continue
        neighbors = np.concatenate(
            (levels[index - 2 : index], levels[index + 1 : index + 3])
        )
        if levels[index] < max(neighbors):
            continue
        sample_index = index * hop
        too_close = any(
            abs(sample_index - onset.index) < int(0.16 * SAMPLE_RATE)
            for onset in onsets
        )
        if too_close:
            continue
        window = samples[sample_index : sample_index + int(0.045 * SAMPLE_RATE)]
        strength = min(1.0, levels[index] / max(percentile(levels, 92), 1e-9))
        onsets.append(
            Onset(
                index=sample_index,
                strength=0.45 + 0.55 * strength,
                centroid=zero_crossing_rate(window),
                preceded_by_pause=False,
            )
        )
    onsets.sort(key=lambda onset: onset.index)
    return onsets


def inject_plosives_and_breaths(
    samples: FloatArray,
    onsets: list[Onset],
    state: CloseMicState,
    rng: random.Random,
) -> FloatArray:
    output = samples.copy()
    for onset in onsets:
        probability = state.pop_probability
        if onset.preceded_by_pause:
            probability *= 1.35
        if rng.random() < probability:
            inject_air_pop(output, onset, state, rng)
            inject_release_burst(output, onset, state, rng)
        if onset.preceded_by_pause and rng.random() < state.breath_probability:
            inject_breath(output, onset.index, state, rng)
        if rng.random() < state.click_probability:
            inject_click(output, onset.index, state, rng)
    return output


def inject_air_pop(
    samples: FloatArray, onset: Onset, state: CloseMicState, rng: random.Random
) -> None:
    duration = int(rng.uniform(0.028, 0.080) * SAMPLE_RATE)
    start = max(0, onset.index - int(rng.uniform(0.002, 0.010) * SAMPLE_RATE))
    closure = int(rng.uniform(0.010, 0.028) * SAMPLE_RATE)
    closure_start = max(0, start - closure)
    if start > closure_start:
        n = np.arange(start - closure_start, dtype=np.float64)
        t = n / max(1, closure)
        attenuation = 1.0 - (
            0.24 + 0.36 * state.bad_technique
        ) * np.sin(math.pi * t)
        samples[closure_start:start] *= attenuation
    frequency = rng.uniform(38.0, 85.0)
    radio_frequency = rng.uniform(265.0, 360.0)
    decay = rng.uniform(0.010, 0.030)
    amplitude = state.pop_gain * (0.45 + onset.strength) * rng.uniform(0.75, 1.22)
    polarity = -1.0 if rng.random() < 0.55 else 1.0
    end = min(samples.size, start + duration)
    if end <= start:
        return
    n = np.arange(end - start, dtype=np.float64)
    t = n / SAMPLE_RATE
    envelope = np.exp(-t / decay)
    pressure = np.sin(2.0 * math.pi * frequency * t + 0.7)
    thump = np.sin(2.0 * math.pi * frequency * 2.2 * t)
    radio_thud = np.sin(2.0 * math.pi * radio_frequency * t + 0.2)
    samples[start:end] += (
        polarity
        * amplitude
        * envelope
        * (0.58 * pressure + 0.17 * thump + 0.25 * radio_thud)
    )


def inject_release_burst(
    samples: FloatArray, onset: Onset, state: CloseMicState, rng: random.Random
) -> None:
    duration = int(rng.uniform(0.012, 0.050) * SAMPLE_RATE)
    start = onset.index + int(rng.uniform(0.000, 0.012) * SAMPLE_RATE)
    amplitude = state.burst_gain * (0.7 + onset.strength) * rng.uniform(0.8, 1.4)
    brightness = 0.18 + min(0.65, onset.centroid * 2.5)
    end = min(samples.size, start + duration)
    if end <= start:
        return
    length = end - start
    t = np.arange(length, dtype=np.float64) / max(1, duration)
    envelope = (1.0 - t) ** 2
    noise = np.fromiter(
        (rng.uniform(-1.0, 1.0) for _ in range(length)),
        dtype=np.float64,
        count=length,
    )
    burst = lfilter([brightness], [1.0, -(1.0 - brightness)], noise)
    samples[start:end] += burst * amplitude * envelope


def inject_breath(
    samples: FloatArray, onset_index: int, state: CloseMicState, rng: random.Random
) -> None:
    duration = int(rng.uniform(0.035, 0.180) * SAMPLE_RATE)
    start = max(
        0,
        onset_index - duration - int(rng.uniform(0.010, 0.050) * SAMPLE_RATE),
    )
    amplitude = (0.008 + 0.025 * state.bad_technique) * rng.uniform(0.5, 1.3)
    end = min(samples.size, start + duration)
    if end <= start:
        return
    length = end - start
    t = np.arange(length, dtype=np.float64) / max(1, duration)
    envelope = np.sin(math.pi * t) ** 0.8
    noise = np.fromiter(
        (rng.uniform(-1.0, 1.0) for _ in range(length)),
        dtype=np.float64,
        count=length,
    )
    breath = lfilter([0.38], [1.0, -0.62], noise)
    samples[start:end] += breath * amplitude * envelope


def inject_click(
    samples: FloatArray, onset_index: int, state: CloseMicState, rng: random.Random
) -> None:
    duration = int(rng.uniform(0.0015, 0.005) * SAMPLE_RATE)
    start = max(0, onset_index - int(rng.uniform(0.015, 0.060) * SAMPLE_RATE))
    amplitude = (0.025 + 0.08 * state.bad_technique) * rng.uniform(0.5, 1.0)
    polarity = -1.0 if rng.random() < 0.5 else 1.0
    end = min(samples.size, start + duration)
    if end <= start:
        return
    n = np.arange(end - start, dtype=np.float64)
    samples[start:end] += polarity * amplitude * (1.0 - n / max(1, duration))


def apply_event_drive_and_clip(
    samples: FloatArray,
    onsets: list[Onset],
    state: CloseMicState,
    rng: random.Random,
) -> FloatArray:
    output = samples.copy()
    for onset in onsets:
        if rng.random() > state.hard_clip_probability * (0.5 + onset.strength):
            continue
        start = max(0, onset.index - int(0.006 * SAMPLE_RATE))
        duration = int(rng.uniform(0.030, 0.090) * SAMPLE_RATE)
        drive = db_to_amp(rng.uniform(2.0, 7.5) * (0.4 + state.bad_technique))
        positive_limit = rng.uniform(0.78, 0.94)
        negative_limit = rng.uniform(0.70, 0.92)
        end = min(output.size, start + duration)
        if end <= start:
            continue
        t = np.arange(end - start, dtype=np.float64) / max(1, duration)
        envelope = np.sin(math.pi * t) ** 0.7
        value = output[start:end] * (1.0 + (drive - 1.0) * envelope)
        output[start:end] = np.clip(value, -negative_limit, positive_limit)
    return output


def saturate(samples: FloatArray, state: CloseMicState) -> FloatArray:
    drive = db_to_amp(state.drive_db)
    asymmetry = 0.05 + 0.12 * state.bad_technique
    threshold = 0.78 - 0.18 * state.bad_technique
    biased = samples * drive + asymmetry
    shaped = np.tanh(biased / max(0.20, threshold)) * threshold
    return shaped - asymmetry * 0.72


def compress(samples: FloatArray, state: CloseMicState) -> FloatArray:
    threshold = db_to_amp(-17.0 + 3.0 * state.bad_technique)
    attack = math.exp(-1.0 / (0.007 * SAMPLE_RATE))
    release = math.exp(-1.0 / ((0.070 + 0.090 * state.proximity) * SAMPLE_RATE))
    envelope = 0.0
    output = np.empty_like(samples)
    for index, sample in enumerate(samples):
        level = abs(sample)
        coeff = attack if level > envelope else release
        envelope = coeff * envelope + (1.0 - coeff) * level
        if envelope <= threshold:
            gain = 1.0
        else:
            over_db = amp_to_db(envelope / threshold)
            gain_db = -over_db * (1.0 - 1.0 / state.compressor_ratio)
            gain = db_to_amp(gain_db)
        output[index] = sample * gain * 1.35
    return output


def channel_filter(samples: FloatArray, state: CloseMicState) -> FloatArray:
    x = highpass_onepole(samples, 205.0)
    x = lowpass_onepole(x, state.channel_lowpass)
    low_mid = bandpass_cheap(x, 430.0, 850.0)
    presence = bandpass_cheap(x, 1450.0, 2500.0)
    presence_gain = db_to_amp(state.presence_db) - 1.0
    return x - 0.18 * low_mid + presence_gain * 0.42 * presence


def inject_radio_pop_residue(
    samples: FloatArray,
    onsets: list[Onset],
    state: CloseMicState,
    rng: random.Random,
) -> FloatArray:
    output = samples.copy()
    for onset in onsets:
        chance = state.pop_probability * (0.55 + 0.45 * state.bad_technique)
        if rng.random() > chance:
            continue
        start = max(0, onset.index - int(0.003 * SAMPLE_RATE))
        duration = int(rng.uniform(0.024, 0.070) * SAMPLE_RATE)
        frequency = rng.uniform(275.0, 365.0)
        decay = rng.uniform(0.010, 0.026)
        amplitude = (
            0.035
            + 0.16 * state.bad_technique
            + 0.11 * state.proximity
        ) * (0.65 + onset.strength)
        polarity = -1.0 if rng.random() < 0.5 else 1.0
        end = min(output.size, start + duration)
        if end <= start:
            continue
        t = np.arange(end - start, dtype=np.float64) / SAMPLE_RATE
        envelope = np.exp(-t / decay)
        output[start:end] += (
            polarity
            * amplitude
            * envelope
            * np.sin(2.0 * math.pi * frequency * t)
        )
    return output


def match_rms(samples: FloatArray, *, target_db: float) -> FloatArray:
    current = rms(samples)
    if current <= 1e-9:
        return samples
    gain = db_to_amp(target_db) / current
    return samples * gain


def limiter(samples: FloatArray, *, limit: float) -> FloatArray:
    output = np.empty_like(samples)
    release = math.exp(-1.0 / (0.025 * SAMPLE_RATE))
    gain = 1.0
    for index, sample in enumerate(samples):
        target = min(1.0, limit / max(abs(sample), 1e-9))
        if target < gain:
            gain = target
        else:
            gain = release * gain + (1.0 - release) * target
        output[index] = max(-limit, min(limit, sample * gain))
    return output


def lowpass_onepole(samples: FloatArray, cutoff: float) -> FloatArray:
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff / SAMPLE_RATE)
    return lfilter([alpha], [1.0, -(1.0 - alpha)], samples)


def highpass_onepole(samples: FloatArray, cutoff: float) -> FloatArray:
    low = lowpass_onepole(samples, cutoff)
    return samples - low


def bandpass_cheap(samples: FloatArray, low: float, high: float) -> FloatArray:
    return lowpass_onepole(highpass_onepole(samples, low), high)


def frame_rms(samples: FloatArray, frame: int, hop: int) -> FloatArray:
    if samples.size < frame:
        if not samples.size:
            return np.array([], dtype=np.float64)
        return np.asarray([rms(samples)], dtype=np.float64)
    squared = samples * samples
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    starts = np.arange(0, samples.size - frame + 1, hop)
    sums = cumulative[starts + frame] - cumulative[starts]
    return np.sqrt(sums / frame + 1e-15)


def zero_crossing_rate(samples: FloatArray) -> float:
    if samples.size < 2:
        return 0.0
    signs = samples >= 0.0
    crossings = np.count_nonzero(signs[1:] != signs[:-1])
    return crossings / (samples.size - 1)


def percentile(
    values: Sequence[float] | NDArray[np.floating], percentile_value: float
) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.percentile(array, percentile_value))


def rms(samples: Sequence[float] | NDArray[np.floating]) -> float:
    array = np.asarray(samples, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return math.sqrt(float(np.mean(array * array)) + 1e-15)


def db_to_amp(db_value: float) -> float:
    return 10.0 ** (db_value / 20.0)


def amp_to_db(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fmplay.close_mic",
        description="Render or stream close-mic ATC speech as raw s16le or WAV.",
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--intensity", choices=INTENSITIES, default="normal")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.output is None:
            stream_file(
                args.source,
                sys.stdout.buffer,
                seed=args.seed,
                intensity=args.intensity,
                ffmpeg_command=args.ffmpeg,
            )
        else:
            render_file(
                args.source,
                args.output,
                seed=args.seed,
                intensity=args.intensity,
                ffmpeg_command=args.ffmpeg,
            )
    except BrokenPipeError:
        return 0
    except CloseMicError as exc:
        print(f"fmplay-close-mic: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
