from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.text import Text

from fmplay.backends import PlaybackBackend, PlaybackError, default_backend
from fmplay.profiles import ProfileError, get_profile, list_profiles


def _error(message: str) -> None:
    console = Console(stderr=True)
    console.print("fmplay: ", Text(message), sep="")


def _play_audio(
    audio_file: Path, profile_name: str, backend: PlaybackBackend | None = None
) -> int:
    audio_file = audio_file.expanduser()
    if not audio_file.exists():
        _error(f"file not found: {audio_file}")
        raise SystemExit(1)
    if not audio_file.is_file():
        _error(f"not a file: {audio_file}")
        raise SystemExit(1)

    try:
        profile = get_profile(profile_name)
    except KeyError:
        available = ", ".join(list_profiles())
        _error(f"unknown profile '{profile_name}'. Available profiles: {available}")
        raise SystemExit(2) from None

    try:
        profile.play(audio_file, backend or default_backend())
    except KeyboardInterrupt:
        return 130
    except (PlaybackError, ProfileError) as exc:
        _error(str(exc))
        raise SystemExit(1) from exc

    return 0


def build_app(backend: PlaybackBackend | None = None) -> typer.Typer:
    app = typer.Typer(
        add_completion=False,
        help="Play audio through an fmplay profile.",
        rich_markup_mode="rich",
    )

    @app.command()
    def play(
        audio_file: Annotated[Path, typer.Argument(help="Audio file to play.")],
        profile: Annotated[
            str,
            typer.Option(
                "--profile",
                help="Playback/degradation profile to use.",
                show_default=True,
            ),
        ] = "passthrough",
    ) -> int:
        return _play_audio(audio_file, profile, backend)

    return app


def run(
    argv: Sequence[str] | None = None, backend: PlaybackBackend | None = None
) -> int:
    app = build_app(backend)
    try:
        result = app(
            args=list(argv) if argv is not None else None,
            prog_name="fmplay",
            standalone_mode=False,
        )
    except typer._click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from exc

    return int(result or 0)


def main(argv: Sequence[str] | None = None) -> None:
    sys.exit(run(argv))


if __name__ == "__main__":
    main()
