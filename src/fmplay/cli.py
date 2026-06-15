from __future__ import annotations

import random
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from fmplay.backends import PlaybackBackend, PlaybackError, default_backend
from fmplay.profiles import (
    ProfileError,
    ProfileInfo,
    get_profile,
    list_profile_summaries,
    list_profiles,
)
from fmplay.spectrogram import (
    SpectrogramError,
    print_kitty_image,
    render_spectrogram_image,
)
from fmplay.stages import (
    DEFAULT_PREVIEW_DURATION,
    GeneratedSource,
    get_stage,
    is_generated_source,
    list_stages,
    render_generated_source,
)

_COMMANDS = frozenset({"play", "preview", "profiles"})


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
            temp_path = Path(temp_dir)
            must_render = no_play or spectrogram or spectrogram_file is not None
            prepared_audio = (
                _prepare_profile_audio(profile, audio_file, temp_path)
                if must_render
                else None
            )
            if not _spectrogram_prints_to_console(spectrogram, spectrogram_file):
                _print_profile_info(profile, audio_file)
            if spectrogram or spectrogram_file is not None:
                if prepared_audio is None:
                    prepared_audio = _prepare_profile_audio(
                        profile, audio_file, temp_path
                    )
                spectrogram_path = (
                    spectrogram_file or Path(temp_dir) / "spectrogram.png"
                )
                spectrogram_path = spectrogram_path.expanduser()
                render_spectrogram_image(prepared_audio, spectrogram_path)
                if spectrogram_file is None:
                    print_kitty_image(spectrogram_path)
            if not no_play:
                playback_backend = backend or default_backend()
                if prepared_audio is None:
                    _play_or_stream_profile_audio(
                        profile, audio_file, temp_path, playback_backend
                    )
                else:
                    playback_backend.play(prepared_audio)
    except KeyboardInterrupt:
        return 130
    except (PlaybackError, ProfileError, SpectrogramError) as exc:
        _error(str(exc))
        raise SystemExit(1) from exc

    return 0


def _preview_audio(
    target_name: str,
    backend: PlaybackBackend | None = None,
    *,
    duration: float = DEFAULT_PREVIEW_DURATION,
    source: str = "silence",
    seed: int | None = None,
) -> int:
    if duration <= 0:
        _error("--duration must be greater than 0")
        raise SystemExit(2)
    if not is_generated_source(source):
        _error(_unknown_preview_source_message(source))
        raise SystemExit(2)

    try:
        stage = get_stage(target_name)
    except KeyError:
        stage = None

    preview_seed = (
        seed if seed is not None else random.SystemRandom().randrange(1, 2**31)
    )

    try:
        playback_backend = backend or default_backend()
        play_stream = getattr(playback_backend, "play_stream", None)
        if stage is not None:
            audio_stream = stage.stream(
                duration=duration,
                source=cast(GeneratedSource, source),
                seed=preview_seed,
            )
            _print_preview_info(stage, target_name, duration, source, preview_seed)
            if callable(play_stream):
                play_stream(audio_stream)
                return 0

            with tempfile.TemporaryDirectory(prefix="fmplay-preview-") as temp_dir:
                rendered_path = Path(temp_dir) / "cockpit-a320.wav"
                stage.render(
                    rendered_path,
                    duration=duration,
                    source=source,
                    seed=preview_seed,
                )
                playback_backend.play(rendered_path)
            return 0

        try:
            profile = get_profile(target_name)
        except KeyError:
            available = ", ".join((*list_profiles(), *list_stages()))
            _error(
                f"unknown preview target '{target_name}'. "
                f"Available preview targets: {available}"
            )
            raise SystemExit(2) from None

        with tempfile.TemporaryDirectory(prefix="fmplay-preview-") as temp_dir:
            source_path = Path(temp_dir) / "source.wav"
            render_generated_source(
                source_path,
                duration=duration,
                source=cast(GeneratedSource, source),
                seed=preview_seed,
            )
            _print_preview_info(profile, target_name, duration, source, preview_seed)
            _play_or_stream_profile_audio(
                profile, source_path, Path(temp_dir), playback_backend
            )
    except KeyboardInterrupt:
        return 130
    except (PlaybackError, ProfileError) as exc:
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


def _play_or_stream_profile_audio(
    profile: object,
    audio_file: Path,
    temp_dir: Path,
    backend: PlaybackBackend,
) -> None:
    stream = getattr(profile, "stream", None)
    play_stream = getattr(backend, "play_stream", None)
    if callable(stream) and callable(play_stream):
        audio_stream = stream(audio_file)
        if audio_stream is not None:
            play_stream(audio_stream)
            return

    backend.play(_prepare_profile_audio(profile, audio_file, temp_dir))


def _spectrogram_prints_to_console(
    spectrogram: bool, spectrogram_file: Path | None
) -> bool:
    return spectrogram and spectrogram_file is None


def _print_profile_info(profile: object, audio_file: Path) -> None:
    profile_info = _get_profile_info(profile)
    if profile_info is None or not profile_info.primitives:
        return

    console = Console()
    console.print(Text(f"{profile_info.name} profile", style="bold"))
    console.print(Text(profile_info.description))
    console.print(Text(f"Source: {audio_file}", style="dim"))

    table = Table(
        "Primitive",
        "ffmpeg graph",
        title="Applied transformations",
        show_lines=True,
    )
    for primitive in profile_info.primitives:
        table.add_row(
            Text(primitive.name, style="bold"),
            Text(primitive.graph, style="dim", overflow="fold"),
        )

    console.print(table)


def _get_profile_info(profile: object) -> ProfileInfo | None:
    describe = getattr(profile, "profile_info", None)
    if not callable(describe):
        return None

    return describe()


def _print_profiles() -> None:
    console = Console()
    console.print("Available profiles:")

    table = Table.grid(padding=(0, 2))
    for profile in list_profile_summaries():
        table.add_row(Text(profile.name, style="bold"), profile.description)

    console.print(table)


def _print_preview_info(
    target: object, target_name: str, duration: float, source: str, seed: int
) -> None:
    console = Console()
    console.print(Text(f"{target_name} preview", style="bold"))
    description = getattr(target, "description", None)
    if description:
        console.print(Text(str(description)))
    console.print(
        Text(
            f"Generated source: {source} | Duration: {duration:g}s | Seed: {seed}",
            style="dim",
        )
    )

    profile_info = _get_profile_info(target)
    if profile_info is None or not profile_info.primitives:
        return

    table = Table(
        "Primitive",
        "implementation",
        title="Preview components",
        show_lines=True,
    )
    for primitive in profile_info.primitives:
        table.add_row(
            Text(primitive.name, style="bold"),
            Text(primitive.graph, style="dim", overflow="fold"),
        )

    console.print(table)


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

    @app.command()
    def preview(
        target: Annotated[
            str,
            typer.Argument(
                help=(
                    "Profile or profile stage to preview, for example 'cockpit:a320'."
                )
            ),
        ],
        duration: Annotated[
            float,
            typer.Option(
                "--duration",
                "-d",
                help="Preview duration in seconds.",
                show_default=True,
            ),
        ] = DEFAULT_PREVIEW_DURATION,
        source: Annotated[
            str,
            typer.Option(
                "--source",
                help=(
                    "Generated source to preview against: silence, white, pink, brown."
                ),
                show_default=True,
            ),
        ] = "silence",
        seed: Annotated[
            int | None,
            typer.Option(
                "--seed",
                help=(
                    "RNG seed for repeatable generated noise and cockpit events. "
                    "Omit for a fresh variant."
                ),
                show_default=False,
            ),
        ] = None,
    ) -> int:
        """Preview a profile or reusable profile stage with generated audio."""

        return _preview_audio(
            target,
            backend,
            duration=duration,
            source=source,
            seed=seed,
        )

    @app.command()
    def profiles() -> int:
        """List available profiles."""

        _print_profiles()
        return 0

    return app


def _normalize_argv(argv: Sequence[str] | None) -> list[str] | None:
    if argv is None:
        return None

    args = _normalize_spectrogram_args(argv)
    if not args or args[0] in _COMMANDS or args[0] in {"--help", "-h"}:
        return args

    return ["play", *args]


def run(
    argv: Sequence[str] | None = None, backend: PlaybackBackend | None = None
) -> int:
    app = build_app(backend)
    args = list(argv) if argv is not None else sys.argv[1:]
    try:
        result = app(
            args=_normalize_argv(args),
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


def _unknown_preview_source_message(source: str) -> str:
    return (
        f"unknown generated source '{source}'. "
        "Available sources: brown, pink, silence, white"
    )


def main(argv: Sequence[str] | None = None) -> None:
    sys.exit(run(argv))


if __name__ == "__main__":
    main()
