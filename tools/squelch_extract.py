#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0",
#   "requests>=2.32",
#   "scipy>=1.14",
#   "soundfile>=0.12",
# ]
# ///
from __future__ import annotations

import argparse
import io
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from scipy import signal

DATASET = "jlvdoorn/atco2-asr-atcosim"
DATASET_SERVER = "https://datasets-server.huggingface.co"


@dataclass(frozen=True)
class Candidate:
    row_idx: int
    side: str
    start_s: float
    end_s: float
    score: float
    rms_db: float
    centroid_hz: float
    high_ratio: float
    flatness: float
    zcr: float
    source_text: str
    output_file: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract likely squelch/open-tail artifacts from the HF "
            "ATCO2+ATCOSIM ASR dataset."
        )
    )
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--rows", type=int, default=160)
    parser.add_argument("--page-size", type=int, default=40)
    parser.add_argument("--max-snippets", type=int, default=48)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/squelch/references")
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[Candidate] = []
    session = requests.Session()
    for row in iter_dataset_rows(
        session=session,
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        offset=args.offset,
        rows=args.rows,
        page_size=args.page_size,
    ):
        row_idx = int(row["row_idx"])
        data = row["row"]
        audio_url = data["audio"][0]["src"]
        y, sr = fetch_audio(session, audio_url)
        text = str(data.get("text", ""))
        candidates.extend(
            extract_candidates(
                y,
                sr,
                row_idx=row_idx,
                text=text,
                output_dir=args.output_dir,
            )
        )

    selected = sorted(candidates, key=lambda item: item.score, reverse=True)[
        : args.max_snippets
    ]
    selected_paths = {candidate.output_file for candidate in selected}
    for candidate in candidates:
        if candidate.output_file not in selected_paths:
            Path(candidate.output_file).unlink(missing_ok=True)

    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for candidate in selected:
            handle.write(json.dumps(asdict(candidate), ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "rows_scanned": args.rows,
                "candidates": len(candidates),
                "selected": len(selected),
                "output_dir": str(args.output_dir),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


def iter_dataset_rows(
    *,
    session: requests.Session,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    rows: int,
    page_size: int,
) -> Iterable[dict[str, object]]:
    remaining = rows
    cursor = offset
    while remaining > 0:
        length = min(page_size, remaining, 100)
        response = session.get(
            f"{DATASET_SERVER}/rows",
            params={
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": cursor,
                "length": length,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        page = payload.get("rows", [])
        if not page:
            break
        yield from page
        cursor += len(page)
        remaining -= len(page)


def fetch_audio(session: requests.Session, url: str) -> tuple[np.ndarray, int]:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    y, sr = sf.read(io.BytesIO(response.content), always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = y.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    return y, int(sr)


def extract_candidates(
    y: np.ndarray,
    sr: int,
    *,
    row_idx: int,
    text: str,
    output_dir: Path,
) -> list[Candidate]:
    if y.size < int(sr * 0.12):
        return []

    edge_duration = min(0.62, max(0.18, y.size / sr * 0.28))
    edges = (
        ("start", 0, int(edge_duration * sr)),
        ("end", max(0, y.size - int(edge_duration * sr)), y.size),
    )
    candidates: list[Candidate] = []
    for side, edge_start, edge_end in edges:
        edge = y[edge_start:edge_end]
        if edge.size < int(sr * 0.08):
            continue
        for start, end, features in noisy_regions(edge, sr):
            if end - start < int(0.035 * sr):
                continue
            global_start = edge_start + start
            global_end = edge_start + end
            score = squelch_score(features, side=side)
            if score < 1.05:
                continue

            filename = (
                f"{row_idx:06d}_{side}_{global_start / sr:.3f}_"
                f"{global_end / sr:.3f}.wav"
            )
            output_file = output_dir / filename
            snippet = pad_snippet(y[global_start:global_end], sr)
            sf.write(output_file, snippet, sr)
            candidates.append(
                Candidate(
                    row_idx=row_idx,
                    side=side,
                    start_s=global_start / sr,
                    end_s=global_end / sr,
                    score=score,
                    rms_db=features["rms_db"],
                    centroid_hz=features["centroid_hz"],
                    high_ratio=features["high_ratio"],
                    flatness=features["flatness"],
                    zcr=features["zcr"],
                    source_text=text,
                    output_file=str(output_file),
                )
            )
    return candidates


def noisy_regions(
    y: np.ndarray, sr: int
) -> Iterable[tuple[int, int, dict[str, float]]]:
    frame_length = max(256, int(0.024 * sr))
    hop = max(128, int(0.010 * sr))
    frames = frame_view(y, frame_length=frame_length, hop=hop)
    if frames.size == 0:
        return

    scores = []
    for frame in frames:
        features = frame_features(frame, sr)
        scores.append(
            features["high_ratio"] > 0.22
            and features["flatness"] > 0.18
            and features["centroid_hz"] > 1450
            and features["rms_db"] > -48
        )

    start_frame: int | None = None
    for index, active in enumerate(scores):
        if active and start_frame is None:
            start_frame = index
        elif not active and start_frame is not None:
            yield region_from_frames(y, sr, start_frame, index, frame_length, hop)
            start_frame = None
    if start_frame is not None:
        yield region_from_frames(y, sr, start_frame, len(scores), frame_length, hop)


def region_from_frames(
    y: np.ndarray,
    sr: int,
    start_frame: int,
    end_frame: int,
    frame_length: int,
    hop: int,
) -> tuple[int, int, dict[str, float]]:
    start = start_frame * hop
    end = min(len(y), (end_frame - 1) * hop + frame_length)
    segment = y[start:end]
    return start, end, segment_features(segment, sr)


def frame_view(y: np.ndarray, *, frame_length: int, hop: int) -> np.ndarray:
    if y.size < frame_length:
        return np.empty((0, frame_length), dtype=np.float32)
    count = 1 + (y.size - frame_length) // hop
    strides = (y.strides[0] * hop, y.strides[0])
    return np.lib.stride_tricks.as_strided(
        y, shape=(count, frame_length), strides=strides
    ).copy()


def frame_features(frame: np.ndarray, sr: int) -> dict[str, float]:
    windowed = frame * signal.windows.hann(frame.size, sym=False)
    spectrum = np.abs(np.fft.rfft(windowed)) + 1e-12
    freqs = np.fft.rfftfreq(frame.size, 1.0 / sr)
    total = float(np.sum(spectrum[(freqs >= 180) & (freqs <= 9000)]))
    high = float(np.sum(spectrum[(freqs >= 2400) & (freqs <= 8500)]))
    centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))
    geometric = float(np.exp(np.mean(np.log(spectrum))))
    arithmetic = float(np.mean(spectrum))
    return {
        "rms_db": dbfs(rms(frame)),
        "centroid_hz": centroid,
        "high_ratio": high / max(total, 1e-12),
        "flatness": geometric / max(arithmetic, 1e-12),
        "zcr": zero_crossing_rate(frame),
    }


def segment_features(segment: np.ndarray, sr: int) -> dict[str, float]:
    if segment.size == 0:
        return {
            "rms_db": -120.0,
            "centroid_hz": 0.0,
            "high_ratio": 0.0,
            "flatness": 0.0,
            "zcr": 0.0,
        }

    features = frame_features(segment, sr)
    features["duration_s"] = segment.size / sr
    features["crest_db"] = dbfs(float(np.max(np.abs(segment)))) - features["rms_db"]
    return features


def squelch_score(features: dict[str, float], *, side: str) -> float:
    edge_bonus = 0.12 if side == "end" else 0.04
    duration = features.get("duration_s", 0.0)
    duration_bonus = 0.24 if 0.045 <= duration <= 0.320 else -0.16
    return (
        edge_bonus
        + duration_bonus
        + 1.25 * features["high_ratio"]
        + 0.95 * features["flatness"]
        + min(features["centroid_hz"] / 5000.0, 1.15)
        + min(features["zcr"] * 6.0, 0.55)
        + max(0.0, (features["rms_db"] + 46.0) / 45.0)
    )


def pad_snippet(y: np.ndarray, sr: int) -> np.ndarray:
    pad = int(sr * 0.025)
    padded = np.pad(y, (pad, pad), mode="constant")
    fade = np.linspace(0.0, 1.0, pad, dtype=np.float32)
    padded[:pad] *= fade
    padded[-pad:] *= fade[::-1]
    return padded


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
