from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fmplay.cli import run


class FakeBackend:
    name = "fake"

    def __init__(self) -> None:
        self.played: list[Path] = []

    def play(self, path: Path) -> None:
        self.played.append(path)


def test_default_profile_plays_file(tmp_path: Path) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"not a real wav; backend is mocked")
    backend = FakeBackend()

    assert run([str(audio_file)], backend=backend) == 0

    assert backend.played == [audio_file]


def test_passthrough_profile_plays_file(tmp_path: Path) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"not a real wav; backend is mocked")
    backend = FakeBackend()

    assert run(["--profile", "passthrough", str(audio_file)], backend=backend) == 0

    assert backend.played == [audio_file]


def test_missing_file_exits_with_message(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        run(["missing.wav"], backend=FakeBackend())

    assert exc_info.value.code == 1
    assert "file not found" in capsys.readouterr().err


def test_unknown_profile_exits_with_available_profiles(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"not a real wav; backend is mocked")

    with pytest.raises(SystemExit) as exc_info:
        run(["--profile", "fm-radio", str(audio_file)], backend=FakeBackend())

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "unknown profile 'fm-radio'" in err
    assert "gsm" in err
    assert "passthrough" in err


def test_profile_error_exits_with_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"not a real wav; ffmpeg is mocked")

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(command[0])

    monkeypatch.setattr("fmplay.profiles.subprocess.run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        run(["--profile", "gsm", str(audio_file)], backend=FakeBackend())

    assert exc_info.value.code == 1
    assert "ffmpeg was not found" in capsys.readouterr().err
