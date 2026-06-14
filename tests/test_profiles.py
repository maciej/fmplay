from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fmplay.profiles import GsmCodecProfile, ProfileError


class InspectingBackend:
    name = "inspect"

    def __init__(self) -> None:
        self.played: Path | None = None
        self.exists_while_playing = False
        self.contents = b""

    def play(self, path: Path) -> None:
        self.played = path
        self.exists_while_playing = path.exists()
        self.contents = path.read_bytes()


def test_gsm_profile_round_trips_through_libgsm_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"source audio")
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "-encoders" in command:
            return subprocess.CompletedProcess(
                command, 0, stdout=" A..... libgsm              libgsm GSM\n"
            )

        Path(command[-1]).write_bytes(b"gsm output")
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr("fmplay.profiles.subprocess.run", fake_run)
    backend = InspectingBackend()

    assert GsmCodecProfile().play(audio_file, backend) is None

    assert backend.played is not None
    assert backend.played.name == "gsm.wav"
    assert backend.exists_while_playing
    assert backend.contents == b"gsm output"
    assert len(calls) == 3
    assert "-encoders" in calls[0]
    assert "libgsm" in calls[1]
    assert calls[2][calls[2].index("-f") + 1] == "gsm"


def test_gsm_profile_uses_narrowband_fallback_without_libgsm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"source audio")
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "-encoders" in command:
            return subprocess.CompletedProcess(command, 0, stdout="")

        Path(command[-1]).write_bytes(b"fallback output")
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr("fmplay.profiles.subprocess.run", fake_run)
    backend = InspectingBackend()

    GsmCodecProfile().play(audio_file, backend)

    assert backend.exists_while_playing
    assert backend.contents == b"fallback output"
    assert len(calls) == 2
    fallback_command = calls[1]
    assert "libgsm" not in fallback_command
    assert fallback_command[fallback_command.index("-ar") + 1] == "8000"
    assert fallback_command[fallback_command.index("-ac") + 1] == "1"
    assert "acrusher=bits=8" in fallback_command[fallback_command.index("-af") + 1]


def test_gsm_profile_reports_ffmpeg_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"source audio")

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        if "-encoders" in command:
            return subprocess.CompletedProcess(command, 0, stdout="")

        raise subprocess.CalledProcessError(
            returncode=1, cmd=command, stderr="Unsupported input format"
        )

    monkeypatch.setattr("fmplay.profiles.subprocess.run", fake_run)

    with pytest.raises(ProfileError, match="Unsupported input format"):
        GsmCodecProfile().render(audio_file, tmp_path / "gsm.wav")
