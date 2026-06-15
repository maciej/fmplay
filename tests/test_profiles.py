from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pytest

import fmplay.profiles as profiles
from fmplay.libgsm import LibGsmError
from fmplay.profiles import (
    FmRadioProfile,
    GsmCodecProfile,
    LibGsmProfile,
    MarineVhf1993Profile,
    ProfileError,
)


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


def test_gsm_profile_streams_libgsm_when_available(
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
        assert "-encoders" in command
        return subprocess.CompletedProcess(
            command, 0, stdout=" A..... libgsm              libgsm GSM\n"
        )

    monkeypatch.setattr("fmplay.profiles.subprocess.run", fake_run)

    stream = GsmCodecProfile().stream(audio_file)

    assert stream.input_format == "gsm"
    assert stream.sample_rate == 8000
    assert stream.channel_layout is None
    assert "libgsm" in stream.command
    assert stream.command[stream.command.index("-f") + 1] == "gsm"
    assert stream.command[-1] == "pipe:1"


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


def test_libgsm_profile_round_trips_with_native_libgsm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"source audio")
    calls: list[tuple[Path, Path, str, str]] = []

    class FakeCodec:
        def round_trip_file(
            self,
            source_path: Path,
            output_path: Path,
            *,
            ffmpeg_command: str,
            filter_graph: str,
        ) -> None:
            calls.append((source_path, output_path, ffmpeg_command, filter_graph))
            output_path.write_bytes(b"native libgsm output")

    monkeypatch.setattr("fmplay.profiles.NativeLibGsmCodec", FakeCodec)
    backend = InspectingBackend()

    LibGsmProfile().play(audio_file, backend)

    assert backend.played is not None
    assert backend.played.name == "libgsm.wav"
    assert backend.exists_while_playing
    assert backend.contents == b"native libgsm output"
    assert calls[0][0] == audio_file
    assert calls[0][2] == "ffmpeg"
    assert "highpass=f=260" in calls[0][3]


def test_libgsm_profile_streams_with_python_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"source audio")
    availability_checks = 0

    class FakeCodec:
        @classmethod
        def ensure_available(cls) -> None:
            nonlocal availability_checks
            availability_checks += 1

    monkeypatch.setattr("fmplay.profiles.NativeLibGsmCodec", FakeCodec)

    stream = LibGsmProfile().stream(audio_file)

    assert availability_checks == 1
    assert stream.input_format == "s16le"
    assert stream.sample_rate == 8000
    assert stream.channel_layout == "mono"
    assert stream.command[:3] == (sys.executable, "-m", "fmplay.libgsm_stream")
    assert "--filter" in stream.command
    assert str(audio_file) == stream.command[-1]


def test_libgsm_profile_reports_missing_native_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"source audio")

    class FakeCodec:
        @classmethod
        def ensure_available(cls) -> None:
            raise LibGsmError("libgsm was not found")

    monkeypatch.setattr("fmplay.profiles.NativeLibGsmCodec", FakeCodec)

    with pytest.raises(ProfileError, match="libgsm was not found"):
        LibGsmProfile().stream(audio_file)


def test_marine_vhf_1993_profile_renders_pipeline(
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
        Path(command[-1]).write_bytes(b"marine vhf output")
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr("fmplay.profiles.subprocess.run", fake_run)
    backend = InspectingBackend()

    MarineVhf1993Profile().play(audio_file, backend)

    assert backend.played is not None
    assert backend.played.name == "marine-vhf-1993.wav"
    assert backend.exists_while_playing
    assert backend.contents == b"marine vhf output"
    assert len(calls) == 1

    command = calls[0]
    assert command[command.index("-ar") + 1] == "24000"
    assert command[command.index("-ac") + 1] == "1"

    filter_graph = command[command.index("-filter_complex") + 1]
    assert "highpass=f=260" in filter_graph
    assert "lowpass=f=3600" in filter_graph
    assert "acrusher=bits=11" in filter_graph
    assert "anoisesrc=r=48000:a=0.034:c=white:d=0.65:s=19930112" in filter_graph
    assert "anoisesrc=r=48000:a=0.01:c=pink:d=0.65:s=19930111" in filter_graph
    assert "random(2)" in filter_graph
    assert "anoisesrc=r=48000:a=0.018:c=white:s=19930114" in filter_graph
    assert "tremolo=f=5.1:d=0.025" in filter_graph
    assert "anoisesrc=r=48000:a=0.034:c=white:d=0.85:s=19930118" in filter_graph
    assert "random(4)" in filter_graph
    assert "[open_raw]highpass=f=55" in filter_graph
    assert "[tail_raw]highpass=f=55" in filter_graph
    assert "amix=inputs=5:duration=longest:weights=" in filter_graph
    assert "concat=n=5" in filter_graph
    assert "volume=0.9" in filter_graph


def test_marine_vhf_1993_reuses_radio_squelch_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, int, int]] = []
    expected_rng = random.Random(333)
    expected_opening_seed = expected_rng.randrange(1, 2**31)
    expected_tail_seed = expected_rng.randrange(1, 2**31)

    def fake_squelch_event_graph(
        *,
        event_type: str,
        output: str,
        seed: int,
        index: int = 0,
    ) -> str:
        calls.append((event_type, output, seed, index))
        return f"anullsrc=r=48000:cl=mono:d=0.1[{output}]"

    monkeypatch.setattr(
        "fmplay.stages.radio_squelch_event_graph",
        fake_squelch_event_graph,
    )

    filter_graph = profiles._marine_vhf_1993_filter_graph(333)

    assert calls == [
        ("opening_spit", "open", expected_opening_seed, 0),
        ("tail_crash", "tail", expected_tail_seed, 1),
    ]
    assert "[pre_static][open][body][tail][post_static]concat=n=5" in filter_graph


def test_marine_vhf_1993_profile_info_summarizes_reused_squelch_stage() -> None:
    audio_file = Path("source.wav")
    profile = MarineVhf1993Profile(squelch_seed=333)

    profile_info_graphs = {
        primitive.name: primitive.graph
        for primitive in profile.profile_info().primitives
    }
    render_args = profile._render_args(audio_file, "out.wav")
    render_graph = render_args[render_args.index("-filter_complex") + 1]

    assert (
        profile_info_graphs["squelch opening spit"]
        == "radio:squelch --squelch-event opening_spit --randomness normal"
    )
    assert (
        profile_info_graphs["squelch tail crash"]
        == "radio:squelch --squelch-event tail_crash --randomness normal"
    )
    assert "open_raw" in render_graph
    assert "tail_raw" in render_graph


def test_marine_vhf_1993_profile_reports_ffmpeg_failures(
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
        raise subprocess.CalledProcessError(
            returncode=1, cmd=command, stderr="Invalid filter graph"
        )

    monkeypatch.setattr("fmplay.profiles.subprocess.run", fake_run)

    with pytest.raises(ProfileError, match="Invalid filter graph"):
        MarineVhf1993Profile().render(audio_file, tmp_path / "marine.wav")


def test_fmradio_profile_renders_pipeline(
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
        Path(command[-1]).write_bytes(b"fm radio output")
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr("fmplay.profiles.subprocess.run", fake_run)
    backend = InspectingBackend()

    FmRadioProfile().play(audio_file, backend)

    assert backend.played is not None
    assert backend.played.name == "fmradio.wav"
    assert backend.exists_while_playing
    assert backend.contents == b"fm radio output"
    assert len(calls) == 1

    command = calls[0]
    assert command[command.index("-ar") + 1] == "44100"
    assert command[command.index("-ac") + 1] == "2"

    filter_graph = command[command.index("-filter_complex") + 1]
    assert "aformat=channel_layouts=stereo" in filter_graph
    assert "highpass=f=45" in filter_graph
    assert "lowpass=f=15000" in filter_graph
    assert "compand=attacks=0.006" in filter_graph
    assert "anoisesrc=r=48000:a=0.0045:c=white:s=98301" in filter_graph
    assert "anoisesrc=r=48000:a=0.0014:c=pink:s=98302" in filter_graph
    assert "sine=f=19000" in filter_graph
    assert "volume=0.92" in filter_graph


def test_fmradio_profile_reports_ffmpeg_failures(
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
        raise subprocess.CalledProcessError(
            returncode=1, cmd=command, stderr="Invalid FM filter graph"
        )

    monkeypatch.setattr("fmplay.profiles.subprocess.run", fake_run)

    with pytest.raises(ProfileError, match="Invalid FM filter graph"):
        FmRadioProfile().render(audio_file, tmp_path / "fmradio.wav")
