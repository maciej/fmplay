from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class PlaybackError(RuntimeError):
    """Raised when audio playback cannot be started or completed."""


@dataclass(frozen=True)
class AudioStream:
    """A command that writes playable audio bytes to stdout."""

    command: tuple[str, ...]
    input_format: str
    sample_rate: int | None = None
    channel_layout: str | None = None


class PlaybackBackend(Protocol):
    """Minimal playback interface used by profiles."""

    name: str

    def play(self, path: Path) -> None:
        """Play the audio file at path."""


class StreamingPlaybackBackend(PlaybackBackend, Protocol):
    """Playback backend that can consume an audio stream."""

    def play_stream(self, stream: AudioStream) -> None:
        """Play audio bytes produced by stream.command."""


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


class FfplayBackend(SubprocessBackend):
    """ffplay-backed file and stream playback."""

    def __init__(self) -> None:
        super().__init__("ffplay", ("ffplay", "-nodisp", "-autoexit"))

    def play_stream(self, stream: AudioStream) -> None:
        try:
            producer = subprocess.Popen(
                stream.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise PlaybackError(
                f"Stream command not found: {stream.command[0]}"
            ) from exc

        if producer.stdout is None:
            producer.kill()
            producer.wait()
            raise PlaybackError("stream command did not expose stdout")

        player_command = [*self.command, "-f", stream.input_format]
        if stream.sample_rate is not None:
            player_command.extend(["-sample_rate", str(stream.sample_rate)])
        if stream.channel_layout is not None:
            player_command.extend(["-ch_layout", stream.channel_layout])
        player_command.extend(["-i", "pipe:0"])
        try:
            player_result = subprocess.run(
                player_command,
                stdin=producer.stdout,
                check=False,
            )
        except FileNotFoundError as exc:
            producer.kill()
            producer.wait()
            raise PlaybackError("Playback command not found: ffplay") from exc
        except BaseException:
            producer.kill()
            producer.wait()
            raise
        finally:
            producer.stdout.close()

        stderr = producer.stderr.read().decode(errors="replace")
        returncode = producer.wait()
        if player_result.returncode != 0:
            raise PlaybackError(
                f"{self.name} failed while playing stream: "
                f"exit code {player_result.returncode}"
            )
        if returncode != 0:
            details = _format_subprocess_details(returncode, stderr)
            raise PlaybackError(
                f"{stream.command[0]} failed while producing audio stream{details}"
            )


def default_backend() -> PlaybackBackend:
    """Return the best available local playback backend."""

    if shutil.which("ffplay"):
        return FfplayBackend()

    if platform.system() == "Darwin" and shutil.which("afplay"):
        return SubprocessBackend("afplay", ("afplay",))

    raise PlaybackError(
        "No playback backend found. Install ffmpeg/ffplay, or run on macOS with afplay."
    )


def _format_subprocess_details(returncode: int, output: str) -> str:
    details = output.strip()
    if not details:
        return f": exit code {returncode}"

    return f": {details.splitlines()[-1]}"
