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
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fmplay.profiles import get_profile  # noqa: E402

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class ClipFeatures:
    path: str
    group: str
    duration_s: float
    rms_db: float
    peak_db: float
    crest_db: float
    active_rms_p95_db: float
    active_occupancy: float
    near_1db_pct: float
    near_03db_pct: float
    hot_cluster_count: int
    longest_flat_run_samples: int
    asymmetry: float
    low_20_180_ratio: float
    lowmid_180_520_ratio: float
    speech_300_3400_ratio: float
    high_3400_7600_ratio: float
    onset_count: int
    onset_pressure_p50: float
    onset_pressure_p90: float
    onset_low_p90_db: float
    onset_peak_p90_db: float
    onset_near_1db_pct: float


@dataclass(frozen=True)
class EventSlice:
    path: str
    group: str
    kind: str
    source_path: str
    start_s: float
    end_s: float
    score: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate close-mic ATC degradation using clipping, plosive/onset, "
            "level, and spectral metrics."
        )
    )
    parser.add_argument(
        "--refs-dir", type=Path, default=Path("artifacts/close-mic/references")
    )
    parser.add_argument(
        "--degraded-dir", type=Path, default=Path("artifacts/close-mic/degraded")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/close-mic/validation")
    )
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        default=[],
        help="Clean source clip to render through --profile, repeatable.",
    )
    parser.add_argument("--profile", default="atc-close-mic")
    parser.add_argument("--slice-count", type=int, default=10)
    args = parser.parse_args()

    args.degraded_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for source_path in args.source:
        render_profile(source_path, args.degraded_dir, args.profile)

    ref_paths = audio_paths(args.refs_dir)
    degraded_paths = audio_paths(args.degraded_dir)
    if not ref_paths:
        raise SystemExit(f"No reference audio found in {args.refs_dir}")
    if not degraded_paths:
        raise SystemExit(f"No degraded audio found in {args.degraded_dir}")

    ref_features = [features_for_file(path, "reference") for path in ref_paths]
    degraded_features = [
        features_for_file(path, "degraded") for path in degraded_paths
    ]
    all_features = [*ref_features, *degraded_features]

    summary = compare_sets(ref_features, degraded_features)
    summary["reference_count"] = len(ref_features)
    summary["degraded_count"] = len(degraded_features)
    summary["reference_dir"] = str(args.refs_dir)
    summary["degraded_dir"] = str(args.degraded_dir)

    write_jsonl(args.output_dir / "features.jsonl", all_features)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    slices = export_event_slices(
        [*ref_paths, *degraded_paths],
        output_dir=args.output_dir / "slices",
        count=args.slice_count,
    )
    write_jsonl(args.output_dir / "slices.jsonl", slices)
    plot_feature_distributions(ref_features, degraded_features, args.output_dir)

    print(json.dumps(summary, indent=2))


def render_profile(source_path: Path, output_dir: Path, profile_name: str) -> Path:
    profile = get_profile(profile_name)
    render = getattr(profile, "render", None)
    if render is None:
        raise SystemExit(f"Profile {profile_name!r} cannot render files")
    output = output_dir / f"{source_path.stem}.{profile_name}.wav"
    render(source_path, output)
    return output


def audio_paths(directory: Path) -> list[Path]:
    extensions = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.rglob("*") if path.suffix.lower() in extensions
    )


def features_for_file(path: Path, group: str) -> ClipFeatures:
    y, sr = decode_audio(path, SAMPLE_RATE)
    frame_length = int(0.025 * sr)
    hop = int(0.010 * sr)
    frames = frame_view(y, frame_length=frame_length, hop=hop)
    frame_rms = np.asarray([rms(frame) for frame in frames])
    active_threshold = max(float(np.percentile(frame_rms, 55)) * 0.8, 10 ** (-42 / 20))
    active_mask = frame_rms >= active_threshold
    active_frames = frames[active_mask]
    active = active_region_from_frames(y, active_mask, hop, frame_length)
    events = onset_events(y, sr, frame_rms, hop)
    onset_pressures = [event["pressure"] for event in events]
    onset_lows = [event["low_db"] for event in events]
    onset_peaks = [event["peak_db"] for event in events]
    onset_near = [event["near_1db_pct"] for event in events]
    band_ratios = average_band_ratios(active_frames, sr)
    peak_value = float(np.max(np.abs(y))) if y.size else 0.0
    rms_value = rms(y)
    return ClipFeatures(
        path=str(path),
        group=group,
        duration_s=len(y) / sr,
        rms_db=dbfs(rms_value),
        peak_db=dbfs(peak_value),
        crest_db=dbfs(peak_value) - dbfs(rms_value),
        active_rms_p95_db=percentile_db(frame_rms[active_mask], 95),
        active_occupancy=float(np.mean(active_mask)) if active_mask.size else 0.0,
        near_1db_pct=near_pct(y, -1.0),
        near_03db_pct=near_pct(y, -0.3),
        hot_cluster_count=hot_cluster_count(y),
        longest_flat_run_samples=longest_flat_run(y),
        asymmetry=waveform_asymmetry(active),
        low_20_180_ratio=band_ratios["low"],
        lowmid_180_520_ratio=band_ratios["lowmid"],
        speech_300_3400_ratio=band_ratios["speech"],
        high_3400_7600_ratio=band_ratios["high"],
        onset_count=len(events),
        onset_pressure_p50=percentile_or_zero(onset_pressures, 50),
        onset_pressure_p90=percentile_or_zero(onset_pressures, 90),
        onset_low_p90_db=percentile_or_zero(onset_lows, 90),
        onset_peak_p90_db=percentile_or_zero(onset_peaks, 90),
        onset_near_1db_pct=percentile_or_zero(onset_near, 90),
    )


def compare_sets(
    refs: list[ClipFeatures], degraded: list[ClipFeatures]
) -> dict[str, object]:
    metrics = (
        "rms_db",
        "peak_db",
        "crest_db",
        "active_rms_p95_db",
        "near_1db_pct",
        "near_03db_pct",
        "hot_cluster_count",
        "longest_flat_run_samples",
        "lowmid_180_520_ratio",
        "speech_300_3400_ratio",
        "onset_pressure_p90",
        "onset_low_p90_db",
        "onset_peak_p90_db",
        "onset_near_1db_pct",
    )
    distances = {}
    medians = {}
    for metric in metrics:
        ref_values = np.asarray([float(getattr(item, metric)) for item in refs])
        deg_values = np.asarray([float(getattr(item, metric)) for item in degraded])
        distances[metric] = float(stats.wasserstein_distance(ref_values, deg_values))
        medians[metric] = {
            "reference": float(np.median(ref_values)),
            "degraded": float(np.median(deg_values)),
            "delta": float(np.median(deg_values) - np.median(ref_values)),
        }

    missing = {
        "audible_clipping_risk": median(degraded, "near_1db_pct") < 0.03
        and median(degraded, "longest_flat_run_samples") < 8,
        "weak_plosive_pressure": (
            median(degraded, "onset_pressure_p90")
            < median(refs, "onset_pressure_p90") * 0.8
            and median(degraded, "onset_low_p90_db")
            < median(refs, "onset_low_p90_db") - 6.0
        ),
        "too_clean_dynamics": median(degraded, "crest_db") > median(refs, "crest_db")
        + 2.0,
    }
    return {
        "distance": distances,
        "median": medians,
        "diagnostics": missing,
    }


def export_event_slices(
    paths: list[Path], *, output_dir: Path, count: int
) -> list[EventSlice]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[tuple[float, Path, dict[str, float]]] = []
    for path in paths:
        y, sr = decode_audio(path, SAMPLE_RATE)
        frames = frame_view(y, frame_length=int(0.025 * sr), hop=int(0.010 * sr))
        frame_rms = np.asarray([rms(frame) for frame in frames])
        for event in onset_events(y, sr, frame_rms, int(0.010 * sr)):
            score = event["pressure"] + event["near_1db_pct"] * 0.05
            candidates.append((score, path, event))

    selected = sorted(candidates, key=lambda item: item[0], reverse=True)[:count]
    slices = []
    for index, (score, source_path, event) in enumerate(selected):
        y, sr = decode_audio(source_path, SAMPLE_RATE)
        start_s = max(0.0, event["time_s"] - 0.18)
        end_s = min(len(y) / sr, event["time_s"] + 0.46)
        clip = y[int(start_s * sr) : int(end_s * sr)]
        group = "degraded" if "degraded" in source_path.parts else "reference"
        output = output_dir / f"{index:03d}_{group}_{source_path.stem}.wav"
        sf.write(output, clip, sr)
        slices.append(
            EventSlice(
                path=str(output),
                group=group,
                kind="onset_pressure",
                source_path=str(source_path),
                start_s=start_s,
                end_s=end_s,
                score=float(score),
            )
        )
    return slices


def plot_feature_distributions(
    refs: list[ClipFeatures], degraded: list[ClipFeatures], output_dir: Path
) -> None:
    metrics = (
        ("rms_db", "RMS (dBFS)"),
        ("crest_db", "Crest (dB)"),
        ("near_1db_pct", "Near -1 dBFS (%)"),
        ("longest_flat_run_samples", "Longest flat run"),
        ("lowmid_180_520_ratio", "Low-mid ratio"),
        ("onset_pressure_p90", "Onset pressure p90"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
        axis.hist(
            [getattr(item, metric) for item in refs],
            bins=14,
            alpha=0.58,
            label="reference",
        )
        axis.hist(
            [getattr(item, metric) for item in degraded],
            bins=14,
            alpha=0.58,
            label="degraded",
        )
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "feature_distributions.png", dpi=160)
    plt.close(fig)


def decode_audio(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    raw = subprocess.check_output(command)
    return np.frombuffer(raw, dtype=np.float32), sample_rate


def onset_events(
    y: np.ndarray, sr: int, frame_rms: np.ndarray, hop: int
) -> list[dict[str, float]]:
    if frame_rms.size < 5:
        return []
    rises = np.diff(20.0 * np.log10(frame_rms + 1e-12))
    candidates = np.flatnonzero(
        (rises > 5.5) & (frame_rms[1:] > np.percentile(frame_rms, 68))
    ) + 1
    events = []
    window = int(0.060 * sr)
    for index in candidates:
        start = int(index) * hop
        event = y[start : start + window]
        if event.size < int(0.025 * sr):
            continue
        low_energy = band_energy(event, sr, 20, 380)
        events.append(
            {
                "time_s": start / sr,
                "pressure": low_energy / band_energy(event, sr, 300, 3000),
                "low_db": 10.0 * math.log10(max(low_energy, 1e-18)),
                "peak_db": dbfs(float(np.max(np.abs(event)))),
                "near_1db_pct": near_pct(event, -1.0),
            }
        )
    return events


def frame_view(y: np.ndarray, *, frame_length: int, hop: int) -> np.ndarray:
    if y.size < frame_length:
        return np.empty((0, frame_length), dtype=np.float32)
    starts = np.arange(0, y.size - frame_length + 1, hop)
    return np.stack([y[start : start + frame_length] for start in starts])


def active_region_from_frames(
    y: np.ndarray, active_mask: np.ndarray, hop: int, frame_length: int
) -> np.ndarray:
    active = np.flatnonzero(active_mask)
    if active.size == 0:
        return y
    start = max(0, int(active[0]) * hop - frame_length)
    end = min(y.size, int(active[-1]) * hop + frame_length * 2)
    return y[start:end]


def average_band_ratios(frames: np.ndarray, sr: int) -> dict[str, float]:
    ratios = {"low": [], "lowmid": [], "speech": [], "high": []}
    for frame in frames:
        total = band_energy(frame, sr, 20, min(7600, sr / 2 - 1))
        ratios["low"].append(band_energy(frame, sr, 20, 180) / total)
        ratios["lowmid"].append(band_energy(frame, sr, 180, 520) / total)
        ratios["speech"].append(band_energy(frame, sr, 300, 3400) / total)
        ratios["high"].append(
            band_energy(frame, sr, 3400, min(7600, sr / 2 - 1)) / total
        )
    return {
        key: float(np.mean(values)) if values else 0.0
        for key, values in ratios.items()
    }


def band_energy(y: np.ndarray, sr: int, low: float, high: float) -> float:
    if y.size < 16 or high <= low:
        return 1e-18
    spectrum = np.fft.rfft((y - np.mean(y)) * np.hanning(y.size))
    freqs = np.fft.rfftfreq(y.size, 1 / sr)
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(np.abs(spectrum[mask]) ** 2)) + 1e-18


def hot_cluster_count(y: np.ndarray) -> int:
    hot = np.abs(y) > 10 ** (-1 / 20)
    if not np.any(hot):
        return 0
    starts = np.flatnonzero(hot & np.concatenate(([True], ~hot[:-1])))
    return int(starts.size)


def longest_flat_run(y: np.ndarray) -> int:
    abs_y = np.abs(y)
    hot = abs_y > 10 ** (-0.4 / 20)
    longest = 0
    current = 0
    previous = 0.0
    for sample, is_hot in zip(abs_y, hot, strict=True):
        if is_hot and abs(float(sample) - previous) < 2.5e-4:
            current += 1
        elif is_hot:
            current = 1
        else:
            current = 0
        previous = float(sample)
        longest = max(longest, current)
    return longest


def waveform_asymmetry(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    positive = float(np.percentile(y, 99))
    negative = abs(float(np.percentile(y, 1)))
    return positive - negative


def near_pct(y: np.ndarray, db_threshold: float) -> float:
    threshold = 10 ** (db_threshold / 20)
    return float(np.mean(np.abs(y) >= threshold) * 100.0)


def percentile_or_zero(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, percentile))


def percentile_db(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return -120.0
    return dbfs(float(np.percentile(values, percentile)))


def median(items: list[ClipFeatures], metric: str) -> float:
    return float(np.median([float(getattr(item, metric)) for item in items]))


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y), dtype=np.float64) + 1e-15))


def dbfs(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-12))


def write_jsonl(path: Path, rows: list[ClipFeatures] | list[EventSlice]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row)) + "\n")


if __name__ == "__main__":
    main()
