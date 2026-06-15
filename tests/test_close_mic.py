from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path

import numpy as np

from fmplay.close_mic import (
    SAMPLE_RATE,
    CloseMicStreamProcessor,
    process_array,
    process_samples,
    stream_file,
)


def test_close_mic_processing_is_seeded_and_deterministic() -> None:
    source = synthetic_voice_like_clip()

    first = process_samples(source, seed=1234, intensity="normal")
    second = process_samples(source, seed=1234, intensity="normal")

    assert first == second


def test_close_mic_process_array_keeps_numpy_path() -> None:
    source = np.asarray(synthetic_voice_like_clip(), dtype=np.float64)

    array_output = process_array(source, seed=1234, intensity="normal")
    list_output = process_samples(source, seed=1234, intensity="normal")

    assert isinstance(array_output, np.ndarray)
    assert array_output.dtype == np.float64
    assert array_output.tolist() == list_output


def test_close_mic_stream_processor_is_seeded_and_chunked() -> None:
    source = np.asarray(synthetic_voice_like_clip(), dtype=np.float64)

    def run_stream() -> np.ndarray:
        processor = CloseMicStreamProcessor(seed=1234, intensity="abusive")
        chunks = [
            processor.process_chunk(chunk)
            for chunk in np.array_split(source, [1300, 5000, 17000])
        ]
        chunks.append(processor.finish())
        return np.concatenate(chunks)

    first = run_stream()
    second = run_stream()

    assert first.shape == source.shape
    np.testing.assert_array_equal(first, second)
    assert not np.allclose(first, source)
    assert rms(first) > rms(source)


def test_close_mic_stream_file_uses_chunk_decoder(
    monkeypatch,
) -> None:
    chunks = [
        np.full(100, 0.02, dtype=np.float64),
        np.full(77, -0.03, dtype=np.float64),
    ]
    calls: list[tuple[Path, str, int]] = []

    def fake_decode_audio_chunks(
        source_path: Path,
        *,
        ffmpeg_command: str,
        chunk_samples: int,
    ):
        calls.append((source_path, ffmpeg_command, chunk_samples))
        yield from chunks

    monkeypatch.setattr(
        "fmplay.close_mic.decode_audio_chunks", fake_decode_audio_chunks
    )
    output = BytesIO()

    stream_file(
        Path("source.wav"),
        output,
        seed=1234,
        intensity="normal",
        ffmpeg_command="ffmpeg-test",
        chunk_samples=123,
    )

    assert calls == [(Path("source.wav"), "ffmpeg-test", 123)]
    assert len(output.getvalue()) == sum(chunk.size for chunk in chunks) * 2


def test_close_mic_intensity_changes_output_dynamics() -> None:
    source = synthetic_voice_like_clip()

    subtle = process_samples(source, seed=1234, intensity="subtle")
    abusive = process_samples(source, seed=1234, intensity="abusive")

    assert rms(abusive) > rms(subtle)
    assert near_rail_count(abusive) > near_rail_count(subtle)
    assert subtle != abusive


def synthetic_voice_like_clip() -> list[float]:
    samples = [0.0] * int(SAMPLE_RATE * 1.8)
    starts = [0.10, 0.34, 0.62, 0.95, 1.23, 1.48]
    for start_s in starts:
        start = int(start_s * SAMPLE_RATE)
        length = int(0.135 * SAMPLE_RATE)
        for n in range(length):
            t = n / SAMPLE_RATE
            envelope = min(1.0, n / 180) * max(0.0, 1.0 - n / length)
            samples[start + n] += (
                0.16
                * envelope
                * (
                    math.sin(2 * math.pi * 170 * t)
                    + 0.55 * math.sin(2 * math.pi * 680 * t)
                )
            )
    return samples


def rms(samples: list[float]) -> float:
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def near_rail_count(samples: list[float]) -> int:
    return sum(1 for sample in samples if abs(sample) > 0.80)
