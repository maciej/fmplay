from __future__ import annotations

import random
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

import typer
import typer.completion
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
    DEFAULT_SQUELCH_SAMPLE_RATE,
    GeneratedSource,
    get_stage,
    is_generated_source,
    list_generated_sources,
    list_squelch_event_kinds,
    list_stages,
    render_generated_source,
)

_COMMANDS = frozenset({"completion", "play", "preview", "profiles"})
_COMPLETION_OPTIONS = frozenset({"--install-completion", "--show-completion"})
_COMPLETION_SHELLS = ("bash", "fish", "zsh")


@dataclass(frozen=True)
class _SquelchPreviewOptions:
    event_type: str
    event_start: float
    event_duration: float
    event_level_db: float
    event_highpass: int
    event_lowpass: int
    sample_rate: int


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
    output: Path | None = None,
    no_play: bool = False,
    squelch_event: str | None = None,
    squelch_start: float | None = None,
    squelch_duration: float | None = None,
    squelch_level_db: float | None = None,
    squelch_highpass: int | None = None,
    squelch_lowpass: int | None = None,
    squelch_sample_rate: int = DEFAULT_SQUELCH_SAMPLE_RATE,
) -> int:
    if duration <= 0:
        _error("--duration must be greater than 0")
        raise SystemExit(2)
    if not is_generated_source(source):
        _error(_unknown_preview_source_message(source))
        raise SystemExit(2)
    if no_play and output is None:
        _error("--no-play requires --output for preview")
        raise SystemExit(2)

    try:
        stage = get_stage(target_name)
    except KeyError:
        stage = None

    if stage is None:
        try:
            profile = get_profile(target_name)
        except KeyError:
            available = ", ".join((*list_profiles(), *list_stages()))
            _error(
                f"unknown preview target '{target_name}'. "
                f"Available preview targets: {available}"
            )
            raise SystemExit(2) from None
    else:
        profile = None

    preview_seed = (
        seed if seed is not None else random.SystemRandom().randrange(1, 2**31)
    )
    squelch_options = _build_squelch_preview_options(
        target_name=target_name,
        squelch_event=squelch_event,
        squelch_start=squelch_start,
        squelch_duration=squelch_duration,
        squelch_level_db=squelch_level_db,
        squelch_highpass=squelch_highpass,
        squelch_lowpass=squelch_lowpass,
        squelch_sample_rate=squelch_sample_rate,
    )

    try:
        playback_backend = backend or default_backend()
        play_stream = getattr(playback_backend, "play_stream", None)
        if stage is not None:
            _print_preview_info(
                stage,
                target_name,
                duration,
                source,
                preview_seed,
                squelch_options=squelch_options,
            )
            if output is not None:
                output_path = output.expanduser()
                _render_stage_preview(
                    stage,
                    output_path,
                    duration=duration,
                    source=source,
                    seed=preview_seed,
                    squelch_options=squelch_options,
                )
                if not no_play:
                    playback_backend.play(output_path)
                return 0

            audio_stream = _stage_preview_stream(
                stage,
                duration=duration,
                source=source,
                seed=preview_seed,
                squelch_options=squelch_options,
            )
            if callable(play_stream):
                play_stream(audio_stream)
                return 0

            with tempfile.TemporaryDirectory(prefix="fmplay-preview-") as temp_dir:
                rendered_path = Path(temp_dir) / f"{target_name.replace(':', '-')}.wav"
                _render_stage_preview(
                    stage,
                    rendered_path,
                    duration=duration,
                    source=source,
                    seed=preview_seed,
                    squelch_options=squelch_options,
                )
                playback_backend.play(rendered_path)
            return 0

        with tempfile.TemporaryDirectory(prefix="fmplay-preview-") as temp_dir:
            source_path = Path(temp_dir) / "source.wav"
            render_generated_source(
                source_path,
                duration=duration,
                source=cast(GeneratedSource, source),
                seed=preview_seed,
            )
            _print_preview_info(profile, target_name, duration, source, preview_seed)
            if output is not None:
                output_path = output.expanduser()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                render = getattr(profile, "render", None)
                if callable(render):
                    render(source_path, output_path)
                else:
                    shutil.copyfile(source_path, output_path)
                if not no_play:
                    playback_backend.play(output_path)
                return 0

            _play_or_stream_profile_audio(
                profile, source_path, Path(temp_dir), playback_backend
            )
    except KeyboardInterrupt:
        return 130
    except (PlaybackError, ProfileError) as exc:
        _error(str(exc))
        raise SystemExit(1) from exc

    return 0


def _build_squelch_preview_options(
    *,
    target_name: str,
    squelch_event: str | None,
    squelch_start: float | None,
    squelch_duration: float | None,
    squelch_level_db: float | None,
    squelch_highpass: int | None,
    squelch_lowpass: int | None,
    squelch_sample_rate: int,
) -> _SquelchPreviewOptions | None:
    provided = (
        squelch_event is not None
        or squelch_start is not None
        or squelch_duration is not None
        or squelch_level_db is not None
        or squelch_highpass is not None
        or squelch_lowpass is not None
        or squelch_sample_rate != DEFAULT_SQUELCH_SAMPLE_RATE
    )
    if not provided:
        return None
    if target_name != "radio:squelch":
        _error("squelch preview options are only valid for target 'radio:squelch'")
        raise SystemExit(2)

    return _SquelchPreviewOptions(
        event_type=squelch_event or "thin_gate_flutter",
        event_start=0.0 if squelch_start is None else squelch_start,
        event_duration=0.30 if squelch_duration is None else squelch_duration,
        event_level_db=-40.0 if squelch_level_db is None else squelch_level_db,
        event_highpass=1900 if squelch_highpass is None else squelch_highpass,
        event_lowpass=7600 if squelch_lowpass is None else squelch_lowpass,
        sample_rate=squelch_sample_rate,
    )


def _stage_preview_stream(
    stage: object,
    *,
    duration: float,
    source: str,
    seed: int,
    squelch_options: _SquelchPreviewOptions | None,
) -> object:
    if squelch_options is not None:
        stream_custom = getattr(stage, "stream_custom", None)
        if not callable(stream_custom):
            raise ProfileError("this stage does not support custom squelch events")
        return stream_custom(
            duration=duration,
            source=cast(GeneratedSource, source),
            seed=seed,
            event_type=squelch_options.event_type,
            event_start=squelch_options.event_start,
            event_duration=squelch_options.event_duration,
            event_level_db=squelch_options.event_level_db,
            event_highpass=squelch_options.event_highpass,
            event_lowpass=squelch_options.event_lowpass,
            sample_rate=squelch_options.sample_rate,
        )

    return stage.stream(
        duration=duration,
        source=cast(GeneratedSource, source),
        seed=seed,
    )


def _render_stage_preview(
    stage: object,
    output_path: Path,
    *,
    duration: float,
    source: str,
    seed: int,
    squelch_options: _SquelchPreviewOptions | None,
) -> None:
    if squelch_options is not None:
        render_custom = getattr(stage, "render_custom", None)
        if not callable(render_custom):
            raise ProfileError("this stage does not support custom squelch events")
        render_custom(
            output_path,
            duration=duration,
            source=cast(GeneratedSource, source),
            seed=seed,
            event_type=squelch_options.event_type,
            event_start=squelch_options.event_start,
            event_duration=squelch_options.event_duration,
            event_level_db=squelch_options.event_level_db,
            event_highpass=squelch_options.event_highpass,
            event_lowpass=squelch_options.event_lowpass,
            sample_rate=squelch_options.sample_rate,
        )
        return

    stage.render(
        output_path,
        duration=duration,
        source=cast(GeneratedSource, source),
        seed=seed,
    )


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
        "implementation",
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
    target: object,
    target_name: str,
    duration: float,
    source: str,
    seed: int,
    *,
    squelch_options: _SquelchPreviewOptions | None = None,
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
    if squelch_options is not None:
        console.print(
            Text(
                "Squelch event: "
                f"{squelch_options.event_type} | "
                f"Start: {squelch_options.event_start:g}s | "
                f"Duration: {squelch_options.event_duration:g}s | "
                f"Level: {squelch_options.event_level_db:g} dBFS | "
                f"Band: {squelch_options.event_highpass}-"
                f"{squelch_options.event_lowpass} Hz | "
                f"Sample rate: {squelch_options.sample_rate} Hz",
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


def _filter_completion_values(values: Sequence[str], incomplete: str) -> list[str]:
    return [value for value in values if value.startswith(incomplete)]


def _complete_profile(incomplete: str) -> list[str]:
    return _filter_completion_values(list_profiles(), incomplete)


def _complete_preview_target(incomplete: str) -> list[str]:
    return _filter_completion_values((*list_profiles(), *list_stages()), incomplete)


def _complete_generated_source(incomplete: str) -> list[str]:
    return _filter_completion_values(list_generated_sources(), incomplete)


def _complete_squelch_event(incomplete: str) -> list[str]:
    return _filter_completion_values(list_squelch_event_kinds(), incomplete)


def _print_completion(shell: str) -> None:
    complete_var = "_FMPLAY_COMPLETE"
    typer.echo(
        typer.completion.get_completion_script(
            prog_name="fmplay",
            complete_var=complete_var,
            shell=shell,
        )
    )


def build_app(backend: PlaybackBackend | None = None) -> typer.Typer:
    app = typer.Typer(
        add_completion=True,
        help="Play audio through an fmplay profile.",
        rich_markup_mode="rich",
    )

    @app.callback()
    def root(
        profile: Annotated[
            str,
            typer.Option(
                "--profile",
                help="Playback/degradation profile to use with shorthand playback.",
                show_default=True,
                autocompletion=_complete_profile,
            ),
        ] = "passthrough",
    ) -> None:
        pass

    @app.command()
    def play(
        audio_file: Annotated[Path, typer.Argument(help="Audio file to play.")],
        profile: Annotated[
            str,
            typer.Option(
                "--profile",
                help="Playback/degradation profile to use.",
                show_default=True,
                autocompletion=_complete_profile,
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
                ),
                autocompletion=_complete_preview_target,
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
                autocompletion=_complete_generated_source,
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
        output: Annotated[
            Path | None,
            typer.Option(
                "--output",
                "-o",
                help="Render preview audio to this WAV file before optional playback.",
                show_default=False,
            ),
        ] = None,
        no_play: Annotated[
            bool,
            typer.Option(
                "--no-play",
                help="Render preview output without playing it. Requires --output.",
            ),
        ] = False,
        squelch_event: Annotated[
            str | None,
            typer.Option(
                "--squelch-event",
                help=(
                    "radio:squelch custom event: tail_crash, opening_spit, "
                    "threshold_chatter, carrier_snap, thin_gate_flutter."
                ),
                show_default=False,
                autocompletion=_complete_squelch_event,
            ),
        ] = None,
        squelch_start: Annotated[
            float | None,
            typer.Option(
                "--squelch-start",
                help="radio:squelch custom event start time in seconds.",
                show_default=False,
            ),
        ] = None,
        squelch_duration: Annotated[
            float | None,
            typer.Option(
                "--squelch-duration",
                help="radio:squelch custom event duration in seconds.",
                show_default=False,
            ),
        ] = None,
        squelch_level_db: Annotated[
            float | None,
            typer.Option(
                "--squelch-level-db",
                help="radio:squelch custom event approximate RMS level in dBFS.",
                show_default=False,
            ),
        ] = None,
        squelch_highpass: Annotated[
            int | None,
            typer.Option(
                "--squelch-highpass",
                help="radio:squelch custom event highpass cutoff in Hz.",
                show_default=False,
            ),
        ] = None,
        squelch_lowpass: Annotated[
            int | None,
            typer.Option(
                "--squelch-lowpass",
                help="radio:squelch custom event lowpass cutoff in Hz.",
                show_default=False,
            ),
        ] = None,
        squelch_sample_rate: Annotated[
            int,
            typer.Option(
                "--squelch-sample-rate",
                help="radio:squelch custom output sample rate.",
                show_default=True,
            ),
        ] = DEFAULT_SQUELCH_SAMPLE_RATE,
    ) -> int:
        """Preview a profile or reusable profile stage with generated audio."""

        return _preview_audio(
            target,
            backend,
            duration=duration,
            source=source,
            seed=seed,
            output=output,
            no_play=no_play,
            squelch_event=squelch_event,
            squelch_start=squelch_start,
            squelch_duration=squelch_duration,
            squelch_level_db=squelch_level_db,
            squelch_highpass=squelch_highpass,
            squelch_lowpass=squelch_lowpass,
            squelch_sample_rate=squelch_sample_rate,
        )

    @app.command()
    def profiles() -> int:
        """List available profiles."""

        _print_profiles()
        return 0

    @app.command()
    def completion(
        shell: Annotated[
            Literal["bash", "fish", "zsh"],
            typer.Argument(
                help="Shell completion script to generate.",
                autocompletion=lambda incomplete: _filter_completion_values(
                    _COMPLETION_SHELLS,
                    incomplete,
                ),
            ),
        ],
    ) -> int:
        """Generate a shell completion script."""

        _print_completion(shell)
        return 0

    return app


def _normalize_argv(argv: Sequence[str] | None) -> list[str] | None:
    if argv is None:
        return None

    args = _normalize_spectrogram_args(argv)
    if not args or args[0] in _COMMANDS or args[0] in {"--help", "-h"}:
        return args
    if args[0] in _COMPLETION_OPTIONS or any(
        args[0].startswith(f"{option}=") for option in _COMPLETION_OPTIONS
    ):
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
