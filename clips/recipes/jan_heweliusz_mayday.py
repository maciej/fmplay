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
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from elevenlabs import ElevenLabs, VoiceSettings
from rich.console import Console
from rich.text import Text

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "clips" / "audio"


VOICE_ID = "pNInz6obpgDQGcFmaJgB"
VOICE_NAME = "Adam - Dominant, Firm"
MODEL_ID = "eleven_multilingual_v2"
OUTPUT_FORMAT = "mp3_44100_128"
SEED = 19930114


@dataclass(frozen=True)
class Clip:
    text: str
    output_path: Path
    explanation: str
    speed: float = 0.92


CLIPS = {
    "mayday": Clip(
        text=(
            "Mayday, Mayday, all ships, all ships. This is lorry ferry "
            "Jan Heweliusz. Heavy list to portside. I am in danger. "
            "My position shortly, near Kolliker. Sixteen miles west from Arkona."
        ),
        output_path=OUTPUT_DIR / "jan_heweliusz_mayday.mp3",
        explanation=(
            "At 04:36 during the 14 January 1993 MF Jan Heweliusz disaster, "
            "Captain Andrzej Ulasiewicz transmitted this radio distress call. "
            "This recipe intentionally renders a clean source take; radio, "
            "microphone, and receiver degradation belongs in fmplay profiles."
        ),
        speed=0.78,
    ),
    "abandon_ship": Clip(
        text=(
            "Uwaga załoga i pasażerowie! Ogłaszam alarm opuszczenia statku, "
            "alarm opuszczenia statku! Nagły przechył! Attention! Abandon ship! "
            "Sudden listing!"
        ),
        output_path=OUTPUT_DIR / "jan_heweliusz_abandon_ship.mp3",
        explanation=(
            "A bilingual abandon-ship alarm line for the Jan Heweliusz sudden-list "
            "distress sequence."
        ),
    ),
}


class ClipName(StrEnum):
    mayday = "mayday"
    abandon_ship = "abandon_ship"


console = Console()
err_console = Console(stderr=True)
app = typer.Typer(
    add_completion=False,
    help="Generate Jan Heweliusz distress-call clips.",
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
    audio = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=clip.text,
        model_id=MODEL_ID,
        language_code="pl",
        output_format=OUTPUT_FORMAT,
        seed=SEED,
        voice_settings=VoiceSettings(
            stability=0.42,
            similarity_boost=0.82,
            style=0.22,
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
    clip: Annotated[
        ClipName | None,
        typer.Option("--clip", help="Generate only one clip. Defaults to all clips."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            help="Override output audio path. Only valid with --clip.",
        ),
    ] = None,
    explain: Annotated[
        bool, typer.Option("--explain", help="Print clip explanations and exit.")
    ] = False,
) -> None:
    clip_name = clip.value if clip else None
    selected_clips = {clip_name: CLIPS[clip_name]} if clip_name else CLIPS

    if output and not clip:
        err_console.print(
            "recipe: ", Text("--output can only be used with --clip"), sep=""
        )
        raise typer.Exit(2)

    if explain:
        for name, clip in selected_clips.items():
            console.print(f"{name}: {clip.explanation}")
        return

    client = ElevenLabs(api_key=resolve_api_key())
    for name, clip in selected_clips.items():
        output_path = output if output else clip.output_path
        render_clip(client, clip, output_path)
        console.print(f"Wrote {output_path}")
        console.print(f"{name}: {clip.explanation}")
    console.print(f"Voice: {VOICE_NAME} ({VOICE_ID})")


if __name__ == "__main__":
    app()
