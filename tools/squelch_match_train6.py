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
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from scipy import signal

DATASET = "jlvdoorn/atco2-asr-atcosim"
DATASET_SERVER = "https://datasets-server.huggingface.co"
WINDOWS = ((4.15, 4.27), (4.25, 4.37), (4.35, 4.47), (4.15, 4.45))
BANDS = ((80, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 7600))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a synthetic thin-gate squelch render against "
            "jlvdoorn/atco2-asr-atcosim train[6]."
        )
    )
    parser.add_argument("synthetic_wav", type=Path)
    parser.add_argument(
        "--reference-wav",
        type=Path,
        default=Path("artifacts/squelch/train6/train6.wav"),
    )
    args = parser.parse_args()

    reference_path = ensure_train6_reference(args.reference_wav)
    report = {
        "reference": str(reference_path),
        "synthetic": str(args.synthetic_wav),
        "reference_metrics": metrics(reference_path),
        "synthetic_metrics": metrics(args.synthetic_wav),
    }
    print(json.dumps(report, indent=2))


def ensure_train6_reference(path: Path) -> Path:
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        f"{DATASET_SERVER}/rows",
        params={
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": 6,
            "length": 1,
        },
        timeout=60,
    )
    response.raise_for_status()
    row = response.json()["rows"][0]["row"]
    audio_url = row["audio"][0]["src"]
    audio_response = requests.get(audio_url, timeout=60)
    audio_response.raise_for_status()
    y, sr = sf.read(io.BytesIO(audio_response.content), always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    sf.write(path, y, sr)
    return path


def metrics(path: Path) -> dict[str, object]:
    y, sr = sf.read(path, always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = y.astype(np.float32, copy=False)
    return {
        "sample_rate": sr,
        "duration_s": len(y) / sr,
        "windows": [window_metrics(y, sr, start, end) for start, end in WINDOWS],
    }


def window_metrics(
    y: np.ndarray, sr: int, start: float, end: float
) -> dict[str, object]:
    x = y[int(start * sr) : int(end * sr)]
    return {
        "window_s": [start, end],
        "rms_db": round(dbfs(rms(x)), 2),
        "zcr": round(zero_crossing_rate(x), 3),
        "bands_db": {
            f"{low}-{high}": round(dbfs(rms(bandpass(x, sr, low, high))), 2)
            for low, high in BANDS
        },
    }


def bandpass(y: np.ndarray, sr: int, low: int, high: int) -> np.ndarray:
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, y)


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
