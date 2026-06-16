#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx-whisper @ git+https://github.com/maciej/mlx-examples.git@maciej/whisper-beam-search-candidates#subdirectory=whisper",
#   "modal>=1.0",
#   "numpy>=2.0",
#   "rich>=15.0.0",
#   "scipy>=1.14",
#   "typer>=0.26.7",
# ]
# ///
from __future__ import annotations

import json as jsonlib
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any, Protocol

import typer
from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fmplay.profiles import AtcCloseMicProfile, ProfileError  # noqa: E402

DEFAULT_AUDIO = REPO_ROOT / "clips/audio/biebrza_broadcast.mp3"
DEFAULT_REFERENCE = REPO_ROOT / "clips/recipes/biebrza-broadcast.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/biebrza-whisper-bench"
DEFAULT_MODEL = "mlx-community/whisper-medium-mlx"
DEFAULT_MODAL_MODEL = "medium"
DEFAULT_MODAL_GPU = "T4"
DEFAULT_LANGUAGE = "pl"


class Backend(StrEnum):
    LOCAL = "local"
    MODAL = "modal"


class WhisperBackend(Protocol):
    name: str

    def __enter__(self) -> WhisperBackend: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        language: str,
        initial_prompt: str | None,
        verbose: bool,
    ) -> str: ...


@dataclass(frozen=True)
class WerResult:
    wer: float
    substitutions: int
    insertions: int
    deletions: int
    reference_words: int
    hypothesis_words: int


@dataclass(frozen=True)
class TranscriptionResult:
    label: str
    audio_path: str
    backend: str
    text: str
    seconds: float
    wer: WerResult


class LocalWhisperBackend:
    name = Backend.LOCAL.value

    def __enter__(self) -> LocalWhisperBackend:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        language: str,
        initial_prompt: str | None,
        verbose: bool,
    ) -> str:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=model,
            language=language,
            initial_prompt=initial_prompt,
            verbose=verbose,
        )
        return str(result["text"]).strip()


class ModalWhisperBackend:
    name = Backend.MODAL.value

    def __init__(self, *, gpu: str, show_output: bool) -> None:
        self.gpu = gpu
        self.show_output = show_output
        self._modal: Any | None = None
        self._transcribe: Any | None = None
        self._output_context: Any | None = None
        self._app_context: Any | None = None

    def __enter__(self) -> ModalWhisperBackend:
        self._modal, app, self._transcribe = build_modal_whisper_app(gpu=self.gpu)
        if self.show_output:
            self._output_context = self._modal.enable_output()
            self._output_context.__enter__()
        self._app_context = app.run()
        self._app_context.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        suppress = None
        if self._app_context is not None:
            suppress = self._app_context.__exit__(exc_type, exc_value, traceback)
        if self._output_context is not None:
            output_suppress = self._output_context.__exit__(
                exc_type, exc_value, traceback
            )
            suppress = bool(suppress) or bool(output_suppress)
        return suppress

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        language: str,
        initial_prompt: str | None,
        verbose: bool,
    ) -> str:
        if self._transcribe is None:
            raise RuntimeError("Modal backend must be entered before transcription")
        return self._transcribe.remote(
            audio_path.read_bytes(),
            audio_path.suffix,
            model=model,
            language=language,
            initial_prompt=initial_prompt,
            verbose=verbose,
        )


def build_modal_whisper_app(*, gpu: str) -> tuple[Any, Any, Any]:
    import modal

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    image = (
        modal.Image.debian_slim(python_version=python_version)
        .apt_install("ffmpeg")
        .uv_pip_install("openai-whisper")
    )
    app = modal.App(
        "fmplay-biebrza-whisper-bench",
        image=image,
        include_source=False,
    )

    @app.function(gpu=gpu, timeout=60 * 30, serialized=True, include_source=False)
    def transcribe(
        audio_bytes: bytes,
        suffix: str,
        *,
        model: str,
        language: str,
        initial_prompt: str | None,
        verbose: bool,
    ) -> str:
        import tempfile

        import whisper

        global _model_cache
        try:
            model_cache = _model_cache
        except NameError:
            model_cache = _model_cache = {}

        whisper_model = model_cache.get(model)
        if whisper_model is None:
            whisper_model = model_cache[model] = whisper.load_model(model)

        with tempfile.NamedTemporaryFile(suffix=suffix) as audio_file:
            audio_file.write(audio_bytes)
            audio_file.flush()
            result = whisper_model.transcribe(
                audio_file.name,
                language=language,
                initial_prompt=initial_prompt,
                verbose=verbose,
            )
        return str(result["text"]).strip()

    return modal, app, transcribe


def main(
    audio: Annotated[
        Path,
        typer.Option("--audio", help="Source audio clip to benchmark."),
    ] = DEFAULT_AUDIO,
    reference: Annotated[
        Path,
        typer.Option("--reference", help="Reference transcript for WER scoring."),
    ] = DEFAULT_REFERENCE,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", help="Directory for transcripts and summary JSON."
        ),
    ] = DEFAULT_OUTPUT_DIR,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help=(
                "MLX Whisper model repo/path. Default is multilingual medium; "
                "use mlx-community/whisper-large-v3-mlx for the v3 family."
            ),
        ),
    ] = DEFAULT_MODEL,
    backend: Annotated[
        Backend,
        typer.Option(
            "--backend",
            case_sensitive=False,
            help="Whisper execution backend.",
        ),
    ] = Backend.LOCAL,
    modal_model: Annotated[
        str,
        typer.Option(
            "--modal-model",
            help="OpenAI Whisper model name to use when --backend=modal.",
        ),
    ] = DEFAULT_MODAL_MODEL,
    modal_gpu: Annotated[
        str,
        typer.Option(
            "--modal-gpu",
            help="Modal GPU type to use when --backend=modal.",
        ),
    ] = DEFAULT_MODAL_GPU,
    language: Annotated[
        str,
        typer.Option("--language", help="Whisper language hint."),
    ] = DEFAULT_LANGUAGE,
    initial_prompt: Annotated[
        str | None,
        typer.Option("--initial-prompt", help="Optional Whisper initial prompt."),
    ] = None,
    close_mic_seed: Annotated[
        int | None,
        typer.Option(
            "--close-mic-seed",
            help="Seed for atc-close-mic:abusive; omit for a random seed.",
        ),
    ] = None,
    ffmpeg: Annotated[
        str,
        typer.Option("--ffmpeg", help="ffmpeg command path/name."),
    ] = "ffmpeg",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Show verbose Whisper output."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the benchmark report as JSON to stdout."),
    ] = False,
) -> None:
    """Benchmark Biebrza Whisper WER before and after atc-close-mic:abusive."""
    console = Console(stderr=json_output)
    audio_path = audio.expanduser().resolve()
    reference_path = reference.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    require_file(audio_path, "audio")
    require_file(reference_path, "reference transcript")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_text = reference_path.read_text(encoding="utf-8")
    degraded_path = output_dir / "biebrza_broadcast.atc-close-mic-abusive.wav"
    whisper_backend = build_whisper_backend(
        backend,
        modal_gpu=modal_gpu,
        show_output=not json_output,
    )
    backend_model = modal_model if backend == Backend.MODAL else model
    backend_gpu = modal_gpu if backend == Backend.MODAL else None

    with whisper_backend:
        if not json_output:
            print_config(
                console,
                audio_path=audio_path,
                reference_path=reference_path,
                backend=whisper_backend.name,
                model=backend_model,
                gpu=backend_gpu,
                language=language,
                initial_prompt=initial_prompt,
            )

        original = transcribe_and_score(
            whisper_backend=whisper_backend,
            console=console,
            label="original",
            audio_path=audio_path,
            reference_text=reference_text,
            model=backend_model,
            language=language,
            initial_prompt=initial_prompt,
            verbose=verbose,
        )

        close_mic_profile = (
            AtcCloseMicProfile(
                intensity="abusive",
                ffmpeg_command=ffmpeg,
            )
            if close_mic_seed is None
            else AtcCloseMicProfile(
                seed=close_mic_seed,
                intensity="abusive",
                ffmpeg_command=ffmpeg,
            )
        )
        console.print(
            "rendering [bold]atc-close-mic:abusive[/bold] "
            f"(seed={close_mic_profile.seed}) -> [cyan]{degraded_path}[/cyan]"
        )
        try:
            close_mic_profile.render(audio_path, degraded_path)
        except ProfileError as exc:
            typer.echo(f"failed to render atc-close-mic:abusive: {exc}", err=True)
            raise typer.Exit(1) from exc

        abusive = transcribe_and_score(
            whisper_backend=whisper_backend,
            console=console,
            label="atc-close-mic:abusive",
            audio_path=degraded_path,
            reference_text=reference_text,
            model=backend_model,
            language=language,
            initial_prompt=initial_prompt,
            verbose=verbose,
        )

        write_text(output_dir / "original.txt", original.text)
        write_text(output_dir / "atc-close-mic-abusive.txt", abusive.text)

        report = {
            "audio": str(audio_path),
            "reference": str(reference_path),
            "backend": whisper_backend.name,
            "model": backend_model,
            "gpu": backend_gpu,
            "language": language,
            "initial_prompt": initial_prompt,
            "close_mic_profile": "atc-close-mic:abusive",
            "close_mic_seed": close_mic_profile.seed,
            "results": [result_to_dict(original), result_to_dict(abusive)],
            "wer_delta": abusive.wer.wer - original.wer.wer,
        }
        report_path = output_dir / "summary.json"
        report_path.write_text(
            jsonlib.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if json_output:
        typer.echo(jsonlib.dumps(report, ensure_ascii=False))
    else:
        console.print()
        print_summary(console, original, abusive)
        console.print(f"\nwrote: [cyan]{report_path}[/cyan]")


def transcribe_and_score(
    *,
    whisper_backend: WhisperBackend,
    console: Console,
    label: str,
    audio_path: Path,
    reference_text: str,
    model: str,
    language: str,
    initial_prompt: str | None,
    verbose: bool,
) -> TranscriptionResult:
    console.print(f"transcribing [bold]{label}[/bold] via {whisper_backend.name}...")
    started = time.perf_counter()
    text = whisper_backend.transcribe(
        audio_path,
        model=model,
        language=language,
        initial_prompt=initial_prompt,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - started
    wer = word_error_rate(reference_text, text)
    console.print(f"{label}: WER [bold]{wer.wer:.2%}[/bold] in {elapsed:.1f}s")
    return TranscriptionResult(
        label=label,
        audio_path=str(audio_path),
        backend=whisper_backend.name,
        text=text,
        seconds=elapsed,
        wer=wer,
    )


def build_whisper_backend(
    backend: Backend,
    *,
    modal_gpu: str = DEFAULT_MODAL_GPU,
    show_output: bool,
) -> WhisperBackend:
    match backend:
        case Backend.LOCAL:
            return LocalWhisperBackend()
        case Backend.MODAL:
            return ModalWhisperBackend(gpu=modal_gpu, show_output=show_output)


def word_error_rate(reference: str, hypothesis: str) -> WerResult:
    ref_words = normalize_for_wer(reference)
    hyp_words = normalize_for_wer(hypothesis)
    substitutions, insertions, deletions = edit_counts(ref_words, hyp_words)
    ref_count = len(ref_words)
    errors = substitutions + insertions + deletions
    wer = errors / ref_count if ref_count else 0.0 if not hyp_words else 1.0
    return WerResult(
        wer=wer,
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        reference_words=ref_count,
        hypothesis_words=len(hyp_words),
    )


def normalize_for_wer(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).casefold()
    text = re.sub(r"[^\w\sąćęłńóśźż]", " ", text, flags=re.IGNORECASE)
    return text.split()


def edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    costs: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)
    ]

    for row in range(1, rows):
        costs[row][0] = (row, 0, 0, row)
    for col in range(1, cols):
        costs[0][col] = (col, 0, col, 0)

    for row in range(1, rows):
        for col in range(1, cols):
            if reference[row - 1] == hypothesis[col - 1]:
                candidates = (costs[row - 1][col - 1],)
            else:
                prev = costs[row - 1][col - 1]
                candidates = ((prev[0] + 1, prev[1] + 1, prev[2], prev[3]),)

            prev = costs[row][col - 1]
            candidates += ((prev[0] + 1, prev[1], prev[2] + 1, prev[3]),)
            prev = costs[row - 1][col]
            candidates += ((prev[0] + 1, prev[1], prev[2], prev[3] + 1),)
            costs[row][col] = min(candidates, key=lambda item: item[0])

    _, substitutions, insertions, deletions = costs[-1][-1]
    return substitutions, insertions, deletions


def result_to_dict(result: TranscriptionResult) -> dict[str, Any]:
    data = asdict(result)
    data["wer"] = asdict(result.wer)
    return data


def write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def print_config(
    console: Console,
    *,
    audio_path: Path,
    reference_path: Path,
    backend: str,
    model: str,
    gpu: str | None,
    language: str,
    initial_prompt: str | None,
) -> None:
    table = Table(title="Biebrza Whisper Bench", show_header=False)
    table.add_column("Setting", style="bold")
    table.add_column("Value", style="cyan")
    table.add_row("audio", str(audio_path))
    table.add_row("reference", str(reference_path))
    table.add_row("backend", backend)
    table.add_row("model", model)
    if gpu is not None:
        table.add_row("gpu", gpu)
    table.add_row("language", language)
    table.add_row("prompt", initial_prompt or "<none>")
    console.print(table)
    console.print()


def print_summary(
    console: Console, original: TranscriptionResult, abusive: TranscriptionResult
) -> None:
    table = Table(title="WER Summary")
    table.add_column("Input", style="bold")
    table.add_column("WER", justify="right")
    table.add_column("S", justify="right")
    table.add_column("I", justify="right")
    table.add_column("D", justify="right")
    table.add_column("Ref", justify="right")
    table.add_column("Hyp", justify="right")
    for result in (original, abusive):
        wer = result.wer
        table.add_row(
            result.label,
            f"{wer.wer:.2%}",
            str(wer.substitutions),
            str(wer.insertions),
            str(wer.deletions),
            str(wer.reference_words),
            str(wer.hypothesis_words),
        )
    console.print(table)
    delta = abusive.wer.wer - original.wer.wer
    sign = "+" if delta >= 0 else ""
    style = "red" if delta > 0 else "green"
    console.print(f"delta abusive-original: [{style}]{sign}{delta:.2%}[/{style}]")


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        typer.echo(f"{label} file not found: {path}", err=True)
        raise typer.Exit(1)
    if not path.is_file():
        typer.echo(f"{label} path is not a file: {path}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
