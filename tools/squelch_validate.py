#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib>=3.9",
#   "numpy>=2.0",
#   "scipy>=1.14",
#   "soundfile>=0.12",
# ]
# ///
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal, stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fmplay.stages import RadioSquelchStage  # noqa: E402


@dataclass(frozen=True)
class AudioFeatures:
    path: str
    duration_s: float
    active_duration_s: float
    rms_db: float
    peak_db: float
    crest_db: float
    centroid_hz: float
    high_ratio: float
    flatness: float
    zcr: float
    attack_ms: float
    release_ms: float
    modulation_peak_hz: float
    band_250_500_db: float
    band_500_1000_db: float
    band_1000_2000_db: float
    band_2000_4000_db: float
    band_4000_8000_db: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate synthetic radio:squelch events against extracted reference "
            "snippets using ambience/event metrics."
        )
    )
    parser.add_argument(
        "--refs-dir", type=Path, default=Path("artifacts/squelch/references")
    )
    parser.add_argument(
        "--synthetic-dir", type=Path, default=Path("artifacts/squelch/synthetic")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/squelch/validation")
    )
    parser.add_argument("--generate", type=int, default=32)
    parser.add_argument("--duration", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=4100)
    args = parser.parse_args()

    args.synthetic_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.generate:
        generate_synthetic_batch(
            output_dir=args.synthetic_dir,
            count=args.generate,
            duration=args.duration,
            seed=args.seed,
        )

    ref_features = features_for_dir(args.refs_dir)
    syn_features = features_for_dir(args.synthetic_dir)
    if not ref_features:
        raise SystemExit(f"No reference WAVs found in {args.refs_dir}")
    if not syn_features:
        raise SystemExit(f"No synthetic WAVs found in {args.synthetic_dir}")

    summary = compare_feature_sets(ref_features, syn_features)
    summary["reference_count"] = len(ref_features)
    summary["synthetic_count"] = len(syn_features)
    summary["reference_dir"] = str(args.refs_dir)
    summary["synthetic_dir"] = str(args.synthetic_dir)

    write_jsonl(args.output_dir / "reference_features.jsonl", ref_features)
    write_jsonl(args.output_dir / "synthetic_features.jsonl", syn_features)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_feature_distributions(ref_features, syn_features, args.output_dir)
    plot_spectrogram_examples(ref_features, syn_features, args.output_dir)

    print(json.dumps(summary, indent=2))


def generate_synthetic_batch(
    *, output_dir: Path, count: int, duration: float, seed: int
) -> None:
    stage = RadioSquelchStage()
    for index in range(count):
        output = output_dir / f"squelch_{index:03d}.wav"
        stage.render(output, duration=duration, seed=seed + index)


def features_for_dir(path: Path) -> list[AudioFeatures]:
    return [features_for_file(wav_path) for wav_path in sorted(path.glob("*.wav"))]


def features_for_file(path: Path) -> AudioFeatures:
    y, sr = sf.read(path, always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = y.astype(np.float32, copy=False)
    active = active_region(y)
    duration_s = len(y) / sr
    active_duration_s = len(active) / sr
    spectrum, freqs = average_spectrum(active, sr)
    rms_db = dbfs(rms(active))
    peak_db = dbfs(float(np.max(np.abs(active))) if active.size else 0.0)
    band_levels = octave_band_levels(active, sr)
    attack_ms, release_ms = attack_release_ms(active, sr)
    return AudioFeatures(
        path=str(path),
        duration_s=duration_s,
        active_duration_s=active_duration_s,
        rms_db=rms_db,
        peak_db=peak_db,
        crest_db=peak_db - rms_db,
        centroid_hz=spectral_centroid(spectrum, freqs, low=180, high=9000),
        high_ratio=band_ratio(spectrum, freqs, low=2400, high=8500, total_high=9000),
        flatness=spectral_flatness(spectrum, freqs, low=180, high=9000),
        zcr=zero_crossing_rate(active),
        attack_ms=attack_ms,
        release_ms=release_ms,
        modulation_peak_hz=modulation_peak_hz(active, sr),
        band_250_500_db=band_levels["250_500"],
        band_500_1000_db=band_levels["500_1000"],
        band_1000_2000_db=band_levels["1000_2000"],
        band_2000_4000_db=band_levels["2000_4000"],
        band_4000_8000_db=band_levels["4000_8000"],
    )


def active_region(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y
    frame = 512
    hop = 128
    if y.size <= frame:
        return y
    energies = []
    starts = range(0, y.size - frame + 1, hop)
    for start in starts:
        energies.append(rms(y[start : start + frame]))
    energies_array = np.asarray(energies)
    threshold = max(float(np.percentile(energies_array, 70)) * 0.18, 10 ** (-62 / 20))
    active_frames = np.flatnonzero(energies_array >= threshold)
    if active_frames.size == 0:
        return y
    start = max(0, int(active_frames[0]) * hop - frame)
    end = min(y.size, int(active_frames[-1]) * hop + frame * 2)
    return y[start:end]


def compare_feature_sets(
    refs: list[AudioFeatures], syns: list[AudioFeatures]
) -> dict[str, object]:
    metrics = (
        "active_duration_s",
        "rms_db",
        "crest_db",
        "centroid_hz",
        "high_ratio",
        "flatness",
        "zcr",
        "attack_ms",
        "release_ms",
        "modulation_peak_hz",
        "band_250_500_db",
        "band_500_1000_db",
        "band_1000_2000_db",
        "band_2000_4000_db",
        "band_4000_8000_db",
    )
    distances: dict[str, float] = {}
    medians: dict[str, dict[str, float]] = {}
    for metric in metrics:
        ref_values = np.asarray([getattr(item, metric) for item in refs])
        syn_values = np.asarray([getattr(item, metric) for item in syns])
        distances[metric] = float(stats.wasserstein_distance(ref_values, syn_values))
        medians[metric] = {
            "reference": float(np.median(ref_values)),
            "synthetic": float(np.median(syn_values)),
        }

    band_distance = float(
        np.mean(
            [
                distances["band_250_500_db"],
                distances["band_500_1000_db"],
                distances["band_1000_2000_db"],
                distances["band_2000_4000_db"],
                distances["band_4000_8000_db"],
            ]
        )
    )
    timing_distance = float(
        np.mean(
            [
                distances["active_duration_s"] * 1000,
                distances["attack_ms"],
                distances["release_ms"],
            ]
        )
    )
    spectral_distance = float(
        np.mean(
            [
                distances["centroid_hz"] / 1000,
                distances["high_ratio"] * 10,
                distances["flatness"] * 10,
                band_distance,
            ]
        )
    )
    return {
        "distance": distances,
        "median": medians,
        "aggregate": {
            "band_lsd_like_db": band_distance,
            "timing_ms": timing_distance,
            "spectral_shape": spectral_distance,
        },
    }


def write_jsonl(path: Path, rows: list[AudioFeatures]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row)) + "\n")


def plot_feature_distributions(
    refs: list[AudioFeatures], syns: list[AudioFeatures], output_dir: Path
) -> None:
    metrics = (
        ("active_duration_s", "Active duration (s)"),
        ("rms_db", "RMS (dBFS)"),
        ("centroid_hz", "Centroid (Hz)"),
        ("high_ratio", "High-band ratio"),
        ("flatness", "Spectral flatness"),
        ("crest_db", "Crest (dB)"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
        axis.hist(
            [getattr(item, metric) for item in refs], bins=16, alpha=0.55, label="ref"
        )
        axis.hist(
            [getattr(item, metric) for item in syns],
            bins=16,
            alpha=0.55,
            label="synthetic",
        )
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "feature_distributions.png", dpi=160)
    plt.close(fig)


def plot_spectrogram_examples(
    refs: list[AudioFeatures], syns: list[AudioFeatures], output_dir: Path
) -> None:
    examples = (refs[0], syns[0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, item, title in zip(
        axes, examples, ("reference", "synthetic"), strict=True
    ):
        y, sr = sf.read(item.path, always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        f, t, zxx = signal.stft(y, fs=sr, nperseg=512, noverlap=384)
        mag = 20 * np.log10(np.abs(zxx) + 1e-8)
        axis.pcolormesh(t, f, mag, shading="auto", vmin=-90, vmax=-20)
        axis.set_ylim(0, 9000)
        axis.set_title(title)
        axis.set_xlabel("time (s)")
    axes[0].set_ylabel("Hz")
    fig.tight_layout()
    fig.savefig(output_dir / "spectrogram_examples.png", dpi=160)
    plt.close(fig)


def average_spectrum(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    if y.size == 0:
        return np.ones(1) * 1e-12, np.zeros(1)
    nperseg = min(1024, max(128, int(2 ** math.floor(math.log2(max(y.size, 128))))))
    freqs, _, zxx = signal.stft(y, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    spectrum = np.mean(np.abs(zxx), axis=1) + 1e-12
    return spectrum, freqs


def octave_band_levels(y: np.ndarray, sr: int) -> dict[str, float]:
    spectrum, freqs = average_spectrum(y, sr)
    return {
        "250_500": band_level_db(spectrum, freqs, 250, 500),
        "500_1000": band_level_db(spectrum, freqs, 500, 1000),
        "1000_2000": band_level_db(spectrum, freqs, 1000, 2000),
        "2000_4000": band_level_db(spectrum, freqs, 2000, 4000),
        "4000_8000": band_level_db(spectrum, freqs, 4000, 8000),
    }


def band_level_db(
    spectrum: np.ndarray, freqs: np.ndarray, low: float, high: float
) -> float:
    mask = (freqs >= low) & (freqs < high)
    return 20.0 * math.log10(float(np.mean(spectrum[mask])) + 1e-12)


def band_ratio(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    *,
    low: float,
    high: float,
    total_high: float,
) -> float:
    total_mask = (freqs >= 180) & (freqs <= total_high)
    band_mask = (freqs >= low) & (freqs <= high)
    return float(np.sum(spectrum[band_mask]) / max(np.sum(spectrum[total_mask]), 1e-12))


def spectral_centroid(
    spectrum: np.ndarray, freqs: np.ndarray, *, low: float, high: float
) -> float:
    mask = (freqs >= low) & (freqs <= high)
    band_spectrum = spectrum[mask]
    band_freqs = freqs[mask]
    return float(np.sum(band_freqs * band_spectrum) / max(np.sum(band_spectrum), 1e-12))


def spectral_flatness(
    spectrum: np.ndarray, freqs: np.ndarray, *, low: float, high: float
) -> float:
    mask = (freqs >= low) & (freqs <= high)
    band_spectrum = spectrum[mask]
    return float(
        np.exp(np.mean(np.log(band_spectrum))) / max(np.mean(band_spectrum), 1e-12)
    )


def attack_release_ms(y: np.ndarray, sr: int) -> tuple[float, float]:
    if y.size == 0:
        return 0.0, 0.0
    envelope = np.abs(signal.hilbert(y))
    peak_index = int(np.argmax(envelope))
    return peak_index / sr * 1000.0, (len(y) - peak_index) / sr * 1000.0


def modulation_peak_hz(y: np.ndarray, sr: int) -> float:
    if y.size < 256:
        return 0.0
    envelope = np.abs(signal.hilbert(y))
    envelope = signal.resample_poly(envelope, up=1, down=max(1, sr // 200))
    envelope = envelope - float(np.mean(envelope))
    spec = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(envelope.size, d=1 / 200)
    mask = (freqs >= 0.5) & (freqs <= 30)
    if not np.any(mask):
        return 0.0
    return float(freqs[mask][np.argmax(spec[mask])])


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y), dtype=np.float64) + 1e-12))


def dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def zero_crossing_rate(y: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(np.signbit(y)))))


if __name__ == "__main__":
    main()
