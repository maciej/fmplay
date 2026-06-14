from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fmplay.libgsm import LibGsmError, NativeLibGsmCodec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fmplay.libgsm_stream",
        description="Stream native libgsm round-tripped audio as raw s16le.",
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--filter", required=True)
    args = parser.parse_args(argv)

    try:
        NativeLibGsmCodec().round_trip_stream(
            args.source,
            sys.stdout.buffer,
            ffmpeg_command=args.ffmpeg,
            filter_graph=args.filter,
        )
    except BrokenPipeError:
        return 0
    except LibGsmError as exc:
        print(f"fmplay-libgsm: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
