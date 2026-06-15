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
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
import soundfile as sf
from scipy import signal

DATASET = "jlvdoorn/atco2-asr-atcosim"
DATASET_SERVER = "https://datasets-server.huggingface.co"
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class ReferenceClip:
    source: str
    source_id: str
    source_url: str
    row_idx: int | None
    title: str
    text: str
    start_s: float
    end_s: float
    score: float
    rms_db: float
    peak_db: float
    crest_db: float
    near_1db_pct: float
    flat_run_samples: int
    onset_pressure_p90: float
    lowmid_ratio: float
    output_file: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect candidate close-mic / hot-mic ATC reference clips from "
            "Hugging Face and yt-dlp sources."
        )
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/close-mic/references")
    )
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--hf-offset", type=int, default=0)
    parser.add_argument("--hf-rows", type=int, default=80)
    parser.add_argument("--hf-page-size", type=int, default=20)
    parser.add_argument("--hf-max-clips", type=int, default=24)
    parser.add_argument("--request-timeout", type=float, default=18.0)
    parser.add_argument(
        "--yt-source",
        action="append",
        default=[],
        help="YouTube URL or yt-dlp search expression, repeatable.",
    )
    parser.add_argument(
        "--yt-search",
        action="append",
        default=[],
        help="Search query expanded as ytsearchN:QUERY, repeatable.",
    )
    parser.add_argument("--yt-search-results", type=int, default=3)
    parser.add_argument("--yt-section-start", type=float, default=0.0)
    parser.add_argument("--yt-section-duration", type=float, default=45.0)
    parser.add_argument("--yt-max-clips", type=int, default=8)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-youtube", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clips: list[ReferenceClip] = []
    session = requests.Session()

    if not args.skip_hf and args.hf_rows > 0:
        clips.extend(
            collect_hf_clips(
                session=session,
                output_dir=args.output_dir / "hf",
                dataset=args.dataset,
                config=args.config,
                split=args.split,
                offset=args.hf_offset,
                rows=args.hf_rows,
                page_size=args.hf_page_size,
                max_clips=args.hf_max_clips,
                request_timeout=args.request_timeout,
            )
        )

    yt_sources = list(args.yt_source)
    yt_sources.extend(
        f"ytsearch{args.yt_search_results}:{query}" for query in args.yt_search
    )
    if not args.skip_youtube and yt_sources:
        clips.extend(
            collect_youtube_clips(
                output_dir=args.output_dir / "youtube",
                sources=yt_sources,
                section_start=args.yt_section_start,
                section_duration=args.yt_section_duration,
                max_clips=args.yt_max_clips,
                timeout=args.request_timeout,
            )
        )

    selected = sorted(clips, key=lambda item: item.score, reverse=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for clip in selected:
            handle.write(json.dumps(asdict(clip), ensure_ascii=False) + "\n")

    summary = {
        "output_dir": str(args.output_dir),
        "manifest": str(manifest_path),
        "clips": len(selected),
        "hf_clips": sum(1 for item in selected if item.source == "hf"),
        "youtube_clips": sum(1 for item in selected if item.source == "youtube"),
        "top": [asdict(item) for item in selected[:8]],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def collect_hf_clips(
    *,
    session: requests.Session,
    output_dir: Path,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    rows: int,
    page_size: int,
    max_clips: int,
    request_timeout: float,
) -> list[ReferenceClip]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[ReferenceClip] = []
    for row in iter_dataset_rows(
        session=session,
        dataset=dataset,
        config=config,
        split=split,
        offset=offset,
        rows=rows,
        page_size=page_size,
        request_timeout=request_timeout,
    ):
        row_idx = int(row["row_idx"])
        data = row["row"]
        audio_url = data["audio"][0]["src"]
        text = str(data.get("text", ""))
        try:
            y, sr = fetch_hf_audio(session, audio_url, timeout=request_timeout)
        except requests.RequestException as exc:
            print(f"warning: failed HF row {row_idx}: {exc}")
            continue

        y = resample_mono(y, sr, SAMPLE_RATE)
        clip = score_and_write_clip(
            y,
            SAMPLE_RATE,
            output_dir=output_dir,
            source="hf",
            source_id=f"{split}-{row_idx}",
            source_url=audio_url,
            row_idx=row_idx,
            title=f"{dataset} {split}[{row_idx}]",
            text=text,
            start_s=0.0,
            end_s=len(y) / SAMPLE_RATE,
        )
        candidates.append(clip)

    selected = sorted(candidates, key=lambda item: item.score, reverse=True)[:max_clips]
    keep = {Path(item.output_file) for item in selected}
    for path in output_dir.glob("*.wav"):
        if path not in keep:
            path.unlink(missing_ok=True)
    return selected


def collect_youtube_clips(
    *,
    output_dir: Path,
    sources: list[str],
    section_start: float,
    section_duration: float,
    max_clips: int,
    timeout: float,
) -> list[ReferenceClip]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[ReferenceClip] = []
    for source in sources:
        for info in iter_yt_infos(source):
            if len(candidates) >= max_clips:
                return sorted(candidates, key=lambda item: item.score, reverse=True)
            url = str(info.get("webpage_url") or info.get("url") or source)
            title = str(info.get("title") or url)
            video_id = str(info.get("id") or slugify(title))
            try:
                y, sr = download_youtube_section(
                    url=url,
                    start=section_start,
                    duration=section_duration,
                    timeout=timeout,
                )
            except subprocess.CalledProcessError as exc:
                print(f"warning: yt-dlp failed for {url}: {last_process_line(exc)}")
                continue
            except subprocess.TimeoutExpired:
                print(f"warning: yt-dlp timed out for {url}")
                continue
            except FileNotFoundError:
                print("warning: yt-dlp was not found")
                return candidates

            y = resample_mono(y, sr, SAMPLE_RATE)
            clip = score_and_write_clip(
                y,
                SAMPLE_RATE,
                output_dir=output_dir,
                source="youtube",
                source_id=video_id,
                source_url=url,
                row_idx=None,
                title=title,
                text="",
                start_s=section_start,
                end_s=section_start + len(y) / SAMPLE_RATE,
            )
            candidates.append(clip)

    return sorted(candidates, key=lambda item: item.score, reverse=True)[:max_clips]


def iter_dataset_rows(
    *,
    session: requests.Session,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    rows: int,
    page_size: int,
    request_timeout: float,
) -> Iterable[dict[str, object]]:
    remaining = rows
    cursor = offset
    while remaining > 0:
        length = min(page_size, remaining, 100)
        params = {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": cursor,
            "length": length,
        }
        try:
            response = session.get(
                f"{DATASET_SERVER}/rows",
                params=params,
                timeout=request_timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            if cursor == 0:
                print(f"warning: /rows failed, trying /first-rows fallback: {exc}")
                response = session.get(
                    f"{DATASET_SERVER}/first-rows",
                    params={"dataset": dataset, "config": config, "split": split},
                    timeout=request_timeout,
                )
                response.raise_for_status()
                yield from response.json().get("rows", [])[:remaining]
            else:
                print(f"warning: stopping HF pagination at offset {cursor}: {exc}")
            return
        page = response.json().get("rows", [])
        if not page:
            return
        yield from page
        cursor += len(page)
        remaining -= len(page)


def fetch_hf_audio(
    session: requests.Session, url: str, *, timeout: float
) -> tuple[np.ndarray, int]:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    y, sr = sf.read(io.BytesIO(response.content), always_2d=False)
    return mono_float(y), int(sr)


def iter_yt_infos(source: str) -> Iterable[dict[str, object]]:
    command = ["yt-dlp", "--dump-json", "--flat-playlist", source]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"warning: yt-dlp info failed for {source}: {last_process_line(exc)}")
        return []

    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def download_youtube_section(
    *, url: str, start: float, duration: float, timeout: float
) -> tuple[np.ndarray, int]:
    if not shutil.which("yt-dlp"):
        raise FileNotFoundError("yt-dlp")
    with tempfile.TemporaryDirectory(prefix="fmplay-ytdlp-") as temp_dir:
        output_template = str(Path(temp_dir) / "clip.%(ext)s")
        command = [
            "yt-dlp",
            "--no-playlist",
            "--extract-audio",
            "--audio-format",
            "wav",
            "--audio-quality",
            "0",
            "--download-sections",
            f"*{start:.3f}-{start + duration:.3f}",
            "--force-keyframes-at-cuts",
            "-o",
            output_template,
            url,
        ]
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout * 4,
        )
        wavs = sorted(Path(temp_dir).glob("*.wav"))
        if not wavs:
            raise subprocess.CalledProcessError(1, command, stderr="no WAV output")
        y, sr = sf.read(wavs[0], always_2d=False)
        return mono_float(y), int(sr)


def score_and_write_clip(
    y: np.ndarray,
    sr: int,
    *,
    output_dir: Path,
    source: str,
    source_id: str,
    source_url: str,
    row_idx: int | None,
    title: str,
    text: str,
    start_s: float,
    end_s: float,
) -> ReferenceClip:
    output_dir.mkdir(parents=True, exist_ok=True)
    features = close_mic_features(y, sr)
    slug = slugify(f"{source_id}-{title}")[:80]
    output_file = output_dir / f"{slug}.wav"
    sf.write(output_file, y, sr)
    score = close_mic_score(features)
    return ReferenceClip(
        source=source,
        source_id=source_id,
        source_url=source_url,
        row_idx=row_idx,
        title=title,
        text=text,
        start_s=start_s,
        end_s=end_s,
        score=score,
        rms_db=features["rms_db"],
        peak_db=features["peak_db"],
        crest_db=features["crest_db"],
        near_1db_pct=features["near_1db_pct"],
        flat_run_samples=int(features["flat_run_samples"]),
        onset_pressure_p90=features["onset_pressure_p90"],
        lowmid_ratio=features["lowmid_ratio"],
        output_file=str(output_file),
    )


def close_mic_score(features: dict[str, float]) -> float:
    score = 0.0
    score += max(0.0, (features["rms_db"] + 18.0) / 5.0)
    score += max(0.0, (16.0 - features["crest_db"]) / 5.0)
    score += min(2.0, features["near_1db_pct"] / 0.10)
    score += min(2.0, features["onset_pressure_p90"] * 1.35)
    score += min(1.5, features["lowmid_ratio"] * 2.5)
    score += min(1.0, features["flat_run_samples"] / 80.0)
    return float(score)


def close_mic_features(y: np.ndarray, sr: int) -> dict[str, float]:
    if y.size == 0:
        return {
            "rms_db": -120.0,
            "peak_db": -120.0,
            "crest_db": 0.0,
            "near_1db_pct": 0.0,
            "flat_run_samples": 0.0,
            "onset_pressure_p90": 0.0,
            "lowmid_ratio": 0.0,
        }

    y = y.astype(np.float32, copy=False)
    rms_value = rms(y)
    peak_value = float(np.max(np.abs(y)))
    frame_length = max(256, int(0.025 * sr))
    hop = max(80, int(0.010 * sr))
    frames = frame_view(y, frame_length=frame_length, hop=hop)
    frame_rms = np.asarray([rms(frame) for frame in frames])
    active_threshold = max(float(np.percentile(frame_rms, 55)) * 0.8, 10 ** (-42 / 20))
    active_frames = frames[frame_rms >= active_threshold]
    lowmid_ratios = []
    for frame in active_frames:
        total = band_energy(frame, sr, 20, min(7600, sr / 2 - 1))
        lowmid_ratios.append(band_energy(frame, sr, 180, 520) / total)

    onset_pressures = onset_pressure_ratios(y, sr, frame_rms, hop)
    return {
        "rms_db": dbfs(rms_value),
        "peak_db": dbfs(peak_value),
        "crest_db": dbfs(peak_value) - dbfs(rms_value),
        "near_1db_pct": float(np.mean(np.abs(y) > 10 ** (-1 / 20)) * 100.0),
        "flat_run_samples": float(longest_flat_run(y)),
        "onset_pressure_p90": percentile_or_zero(onset_pressures, 90),
        "lowmid_ratio": float(np.mean(lowmid_ratios)) if lowmid_ratios else 0.0,
    }


def onset_pressure_ratios(
    y: np.ndarray, sr: int, frame_rms: np.ndarray, hop: int
) -> list[float]:
    if frame_rms.size < 5:
        return []
    db = 20 * np.log10(frame_rms + 1e-12)
    rises = np.diff(db)
    candidates = np.flatnonzero(
        (rises > 5.5) & (frame_rms[1:] > np.percentile(frame_rms, 68))
    ) + 1
    ratios = []
    window = int(0.060 * sr)
    for index in candidates:
        start = int(index) * hop
        event = y[start : start + window]
        if event.size < int(0.025 * sr):
            continue
        ratios.append(
            band_energy(event, sr, 20, 380) / band_energy(event, sr, 300, 3000)
        )
    return ratios


def longest_flat_run(y: np.ndarray) -> int:
    abs_y = np.abs(y)
    hot = abs_y > 10 ** (-0.4 / 20)
    if not np.any(hot):
        return 0
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


def resample_mono(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    y = mono_float(y)
    if sr == target_sr:
        return y
    gcd = np.gcd(sr, target_sr)
    return signal.resample_poly(y, target_sr // gcd, sr // gcd).astype(np.float32)


def mono_float(y: np.ndarray) -> np.ndarray:
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = y.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.5:
        y = y / peak
    return y


def frame_view(y: np.ndarray, *, frame_length: int, hop: int) -> np.ndarray:
    if y.size < frame_length:
        return np.empty((0, frame_length), dtype=np.float32)
    starts = np.arange(0, y.size - frame_length + 1, hop)
    return np.stack([y[start : start + frame_length] for start in starts])


def band_energy(y: np.ndarray, sr: int, low: float, high: float) -> float:
    if y.size < 16 or high <= low:
        return 1e-18
    spectrum = np.fft.rfft((y - np.mean(y)) * np.hanning(y.size))
    freqs = np.fft.rfftfreq(y.size, 1 / sr)
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(np.abs(spectrum[mask]) ** 2)) + 1e-18


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y), dtype=np.float64) + 1e-15))


def dbfs(value: float) -> float:
    return 20.0 * np.log10(max(float(value), 1e-12))


def percentile_or_zero(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, percentile))


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_.")
    return value or quote("clip")


def last_process_line(exc: subprocess.CalledProcessError) -> str:
    details = (exc.stderr or exc.stdout or "").strip()
    if not details:
        return f"exit code {exc.returncode}"
    return details.splitlines()[-1]


if __name__ == "__main__":
    main()
