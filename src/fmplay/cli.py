from __future__ import annotations

import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.text import Text

from fmplay.backends import PlaybackBackend, PlaybackError, default_backend
from fmplay.profiles import ProfileError, get_profile, list_profiles
from fmplay.spectrogram import (
    SpectrogramError,
    print_kitty_image,
    render_spectrogram_image,
)


def _error(message: str) -> None:
    console = Console(stderr=True)
    console.print("fmplay: ", Text(message), sep="")


def _play_audio(
    audio_file: Path,
    profile_name: str,
    backend: PlaybackBackend | None = None,
    *,
    no_play: bool = False,
    spectrogram: bool = False,
    spectrogram_file: Path | None = None,
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
        with tempfile.TemporaryDirectory(prefix="fmplay-") as temp_dir:
            prepared_audio = _prepare_profile_audio(profile, audio_file, Path(temp_dir))
            if spectrogram or spectrogram_file is not None:
                spectrogram_path = (
                    spectrogram_file or Path(temp_dir) / "spectrogram.png"
                )
                spectrogram_path = spectrogram_path.expanduser()
                render_spectrogram_image(prepared_audio, spectrogram_path)
                if spectrogram_file is None:
                    print_kitty_image(spectrogram_path)
            if not no_play:
                (backend or default_backend()).play(prepared_audio)
    except KeyboardInterrupt:
        return 130
    except (PlaybackError, ProfileError, SpectrogramError) as exc:
        _error(str(exc))
        raise SystemExit(1) from exc

    return 0


def _prepare_profile_audio(profile: object, audio_file: Path, temp_dir: Path) -> Path:
    render = getattr(profile, "render", None)
    if callable(render):
        profile_name = getattr(profile, "name", "profile")
        transformed_path = temp_dir / f"{profile_name}.wav"
        render(audio_file, transformed_path)
        return transformed_path

    return audio_file


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
        no_play: Annotated[
            bool,
            typer.Option(
                "--no-play",
                help="Apply the profile without playing audio.",
            ),
        ] = False,
        spectrogram: Annotated[
            bool,
            typer.Option(
                "--spectrogram",
                help=(
                    "Draw a terminal spectrogram using the Kitty graphics "
                    "protocol. Use --spectrogram=FILE.png to write a PNG file."
                ),
            ),
        ] = False,
        spectrogram_file: Annotated[
            Path | None,
            typer.Option(
                "--spectrogram-file",
                hidden=True,
            ),
        ] = None,
    ) -> int:
        return _play_audio(
            audio_file,
            profile,
            backend,
            no_play=no_play,
            spectrogram=spectrogram,
            spectrogram_file=spectrogram_file,
        )

    return app


def run(
    argv: Sequence[str] | None = None, backend: PlaybackBackend | None = None
) -> int:
    app = build_app(backend)
    args = list(argv) if argv is not None else sys.argv[1:]
    try:
        result = app(
            args=_normalize_spectrogram_args(args),
            prog_name="fmplay",
            standalone_mode=False,
        )
    except typer._click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from exc

    return int(result or 0)


def _normalize_spectrogram_args(args: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for arg in args:
        if arg.startswith("--spectrogram="):
            value = arg.partition("=")[2]
            normalized.extend(["--spectrogram-file", value])
        else:
            normalized.append(arg)
    return normalized


def main(argv: Sequence[str] | None = None) -> None:
    sys.exit(run(argv))


if __name__ == "__main__":
    main()
