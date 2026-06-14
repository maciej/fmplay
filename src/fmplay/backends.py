from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Protocol


class PlaybackError(RuntimeError):
    """Raised when audio playback cannot be started or completed."""


class PlaybackBackend(Protocol):
    """Minimal playback interface used by profiles."""

    name: str

    def play(self, path: Path) -> None:
        """Play the audio file at path."""


class SubprocessBackend:
    """Playback backend implemented by invoking a system audio player."""

    name: str
    command: tuple[str, ...]

    def __init__(self, name: str, command: tuple[str, ...]) -> None:
        self.name = name
        self.command = command

    def play(self, path: Path) -> None:
        try:
            subprocess.run([*self.command, str(path)], check=True)
        except FileNotFoundError as exc:
            raise PlaybackError(
                f"Playback command not found: {self.command[0]}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise PlaybackError(
                f"{self.name} failed while playing {path}: exit code {exc.returncode}"
            ) from exc


def default_backend() -> PlaybackBackend:
    """Return the best available local playback backend."""

    if platform.system() == "Darwin" and shutil.which("afplay"):
        return SubprocessBackend("afplay", ("afplay",))

    if shutil.which("ffplay"):
        return SubprocessBackend("ffplay", ("ffplay", "-nodisp", "-autoexit"))

    raise PlaybackError(
        "No playback backend found. Install ffmpeg/ffplay, or run on macOS with afplay."
    )
