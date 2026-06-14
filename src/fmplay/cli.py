from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from fmplay.backends import PlaybackBackend, PlaybackError, default_backend
from fmplay.profiles import get_profile, list_profiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fmplay",
        description="Play audio through an fmplay profile.",
    )
    parser.add_argument(
        "--profile",
        default="passthrough",
        help="Playback/degradation profile to use. Default: passthrough.",
    )
    parser.add_argument("audio_file", type=Path, help="Audio file to play.")
    return parser


def run(
    argv: Sequence[str] | None = None, backend: PlaybackBackend | None = None
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    audio_file = args.audio_file.expanduser()
    if not audio_file.exists():
        parser.exit(1, f"fmplay: file not found: {audio_file}\n")
    if not audio_file.is_file():
        parser.exit(1, f"fmplay: not a file: {audio_file}\n")

    try:
        profile = get_profile(args.profile)
    except KeyError:
        available = ", ".join(list_profiles())
        parser.exit(
            2,
            f"fmplay: unknown profile '{args.profile}'. "
            f"Available profiles: {available}\n",
        )

    try:
        profile.play(audio_file, backend or default_backend())
    except PlaybackError as exc:
        parser.exit(1, f"fmplay: {exc}\n")

    return 0


def main(argv: Sequence[str] | None = None) -> None:
    sys.exit(run(argv))


if __name__ == "__main__":
    main()
