#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "elevenlabs>=2.15.0",
#   "python-dotenv>=1.0.1",
#   "rich>=15.0.0",
#   "typer>=0.26.0",
# ]
# ///

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from elevenlabs import ElevenLabs, VoiceSettings
from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "clips" / "audio"
TRANSCRIPT_PATH = Path(__file__).with_name("biebrza-broadcast.txt")


VOICE_ID = "hpp4J3VqNfWAUOO0d1Us"
VOICE_NAME = "Bella - Professional, Bright, Warm"
MODEL_ID = "eleven_multilingual_v2"
OUTPUT_FORMAT = "mp3_44100_128"
SEED = 20250614


@dataclass(frozen=True)
class Clip:
    text_path: Path
    output_path: Path
    explanation: str
    speed: float = 0.94


CLIP = Clip(
    text_path=TRANSCRIPT_PATH,
    output_path=OUTPUT_DIR / "biebrza_broadcast.mp3",
    explanation=(
        "A fake Polish radio-programme source take about Biebrza. The transcript "
        "is kept word-for-word identical to maciej/tr1 "
        "fixtures/biebrza-broadcast.txt; radio degradation belongs in fmplay "
        "profiles."
    ),
)


console = Console()
app = typer.Typer(
    add_completion=False,
    help="Generate the fake Biebrza radio-programme clip.",
    rich_markup_mode="rich",
)


def resolve_api_key() -> str:
    load_dotenv(REPO_ROOT / ".env")

    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if api_key:
        return api_key

    api_key_file = os.environ.get("ELEVENLABS_API_KEY_FILE", "").strip()
    if api_key_file:
        key_path = Path(api_key_file).expanduser()
        if key_path.exists():
            return key_path.read_text(encoding="utf-8").strip()

    raise RuntimeError(
        "Set ELEVENLABS_API_KEY in .env or ELEVENLABS_API_KEY_FILE in the environment."
    )


def render_clip(client: ElevenLabs, clip: Clip, output_path: Path) -> None:
    text = clip.text_path.read_text(encoding="utf-8")
    audio = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        model_id=MODEL_ID,
        language_code="pl",
        output_format=OUTPUT_FORMAT,
        seed=SEED,
        voice_settings=VoiceSettings(
            stability=0.58,
            similarity_boost=0.84,
            style=0.08,
            speed=clip.speed,
            use_speaker_boost=True,
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        for chunk in audio:
            handle.write(chunk)


@app.command()
def main(
    output: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            help="Override output audio path.",
        ),
    ] = None,
    explain: Annotated[
        bool, typer.Option("--explain", help="Print clip explanation and exit.")
    ] = False,
) -> None:
    if explain:
        console.print(f"biebrza_broadcast: {CLIP.explanation}")
        console.print(f"Voice: {VOICE_NAME} ({VOICE_ID})")
        return

    client = ElevenLabs(api_key=resolve_api_key())
    output_path = output if output else CLIP.output_path
    render_clip(client, CLIP, output_path)
    console.print(f"Wrote {output_path}")
    console.print(f"biebrza_broadcast: {CLIP.explanation}")
    console.print(f"Voice: {VOICE_NAME} ({VOICE_ID})")


if __name__ == "__main__":
    app()
