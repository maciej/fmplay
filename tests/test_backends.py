from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Any

from fmplay.backends import AudioStream, FfplayBackend


def test_ffplay_file_playback_suppresses_terminal_output(
    tmp_path: Path, monkeypatch
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"audio")
    calls: list[dict[str, Any]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("fmplay.backends.subprocess.run", fake_run)

    FfplayBackend().play(audio_file)

    assert calls[0]["stdout"] is subprocess.DEVNULL
    assert calls[0]["stderr"] is subprocess.DEVNULL


def test_ffplay_stream_playback_suppresses_terminal_output(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeProducer:
        def __init__(self) -> None:
            self.stdout = io.BytesIO(b"audio")
            self.stderr = io.BytesIO()

        def kill(self) -> None:
            pass

        def wait(self) -> int:
            return 0

    def fake_popen(command: tuple[str, ...], **kwargs: Any) -> FakeProducer:
        return FakeProducer()

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("fmplay.backends.subprocess.Popen", fake_popen)
    monkeypatch.setattr("fmplay.backends.subprocess.run", fake_run)

    FfplayBackend().play_stream(
        AudioStream(
            command=("ffmpeg", "-i", "audio.wav", "-f", "s16le", "pipe:1"),
            input_format="s16le",
            sample_rate=44100,
            channel_layout="stereo",
        )
    )

    assert calls[0]["stdout"] is subprocess.DEVNULL
    assert calls[0]["stderr"] is subprocess.DEVNULL
