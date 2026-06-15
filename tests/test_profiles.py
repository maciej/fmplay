from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pytest

import fmplay.profiles as profiles
from fmplay.libgsm import LibGsmError
from fmplay.profiles import (
    AtcCloseMicProfile,
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


def test_atc_close_mic_profile_renders_voice_only_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"source audio")
    calls: list[tuple[Path, Path, int, str, str]] = []

    def fake_render_file(
        source_path: Path,
        output_path: Path,
        *,
        seed: int,
        intensity: str,
        ffmpeg_command: str,
    ) -> None:
        calls.append((source_path, output_path, seed, intensity, ffmpeg_command))
        output_path.write_bytes(b"atc close mic output")

    monkeypatch.setattr("fmplay.profiles.render_file", fake_render_file)
    backend = InspectingBackend()

    AtcCloseMicProfile(seed=42, intensity="hot").play(audio_file, backend)

    assert backend.played is not None
    assert backend.played.name == "atc-close-mic.wav"
    assert backend.exists_while_playing
    assert backend.contents == b"atc close mic output"
    assert calls == [
        (
            audio_file,
            backend.played,
            42,
            "hot",
            "ffmpeg",
        )
    ]


def test_atc_close_mic_profile_streams_processor_module() -> None:
    audio_file = Path("source.wav")

    stream = AtcCloseMicProfile(seed=99, intensity="abusive").stream(audio_file)

    assert stream.input_format == "s16le"
    assert stream.sample_rate == 16000
    assert stream.channel_layout == "mono"
    assert stream.command[:3] == (sys.executable, "-m", "fmplay.close_mic")
    assert stream.command[stream.command.index("--seed") + 1] == "99"
    assert stream.command[stream.command.index("--intensity") + 1] == "abusive"
    assert stream.command[-1] == str(audio_file)


def test_atc_close_mic_profile_info_exposes_voice_stage() -> None:
    profile_info = AtcCloseMicProfile().profile_info()
    primitives = {
        primitive.name: primitive.graph for primitive in profile_info.primitives
    }

    assert profile_info.name == "atc-close-mic"
    assert "seeded technique state" in primitives
    assert "onset detector" in primitives
    assert "synthetic plosives and close-mouth events" in primitives
    assert "event-conditioned overload" in primitives
    assert "ATC receiver passband" in primitives
    assert "anoisesrc" not in ";".join(primitives.values())


def test_atc_close_mic_profile_reports_processor_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"source audio")

    def fake_render_file(*args: object, **kwargs: object) -> None:
        from fmplay.close_mic import CloseMicError

        raise CloseMicError("decode failed")

    monkeypatch.setattr("fmplay.profiles.render_file", fake_render_file)

    with pytest.raises(ProfileError, match="decode failed"):
        AtcCloseMicProfile(seed=42).render(audio_file, tmp_path / "out.wav")


def test_atc_close_mic_profile_variants_are_registered() -> None:
    profile = profiles.get_profile("atc-close-mic:abusive")

    assert isinstance(profile, AtcCloseMicProfile)
    assert profile.name == "atc-close-mic:abusive"
    assert profile.intensity == "abusive"
    assert "atc-close-mic:hot" in profiles.list_profiles()


def test_marine_vhf_1993_profile_renders_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"source audio")
    calls: list[list[str]] = []
    close_mic_calls: list[tuple[Path, int, str]] = []

    def fake_render_file(
        source_path: Path,
        output_path: Path,
        *,
        seed: int,
        intensity: str,
        ffmpeg_command: str,
    ) -> None:
        close_mic_calls.append((source_path, seed, intensity))
        output_path.write_bytes(b"close mic voice")

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

    monkeypatch.setattr("fmplay.profiles.render_file", fake_render_file)
    monkeypatch.setattr("fmplay.profiles.subprocess.run", fake_run)
    backend = InspectingBackend()

    MarineVhf1993Profile(squelch_seed=333, close_mic_seed=444).play(
        audio_file, backend
    )

    assert backend.played is not None
    assert backend.played.name == "marine-vhf-1993.wav"
    assert backend.exists_while_playing
    assert backend.contents == b"marine vhf output"
    assert close_mic_calls == [(audio_file, 444, "abusive")]
    assert len(calls) == 1

    command = calls[0]
    assert "atc-close-mic-abusive.wav" in command[command.index("-i") + 1]
    assert command[command.index("-ar") + 1] == "24000"
    assert command[command.index("-ac") + 1] == "1"

    filter_graph = command[command.index("-filter_complex") + 1]
    assert filter_graph.startswith("[0:a]aresample=48000")
    assert "close_body" not in filter_graph
    assert "close_voice" not in filter_graph
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


def test_marine_vhf_1993_profile_streams_two_step_processor() -> None:
    audio_file = Path("source.wav")

    stream = MarineVhf1993Profile(squelch_seed=333, close_mic_seed=444).stream(
        audio_file
    )

    assert stream.input_format == "s16le"
    assert stream.sample_rate == 24000
    assert stream.channel_layout == "mono"
    assert stream.command[:3] == (sys.executable, "-m", "fmplay.marine_vhf_stream")
    assert stream.command[stream.command.index("--squelch-seed") + 1] == "333"
    assert stream.command[stream.command.index("--close-mic-seed") + 1] == "444"
    assert stream.command[-1] == str(audio_file)


def test_marine_vhf_1993_raw_input_args_accept_close_mic_pipe() -> None:
    args = MarineVhf1993Profile(squelch_seed=333)._render_raw_input_args(
        "pipe:0", "pipe:1"
    )

    input_index = args.index("-i")
    assert args[input_index - 6 : input_index + 2] == [
        "-f",
        "s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-i",
        "pipe:0",
    ]
    assert args[args.index("-filter_complex") + 1].startswith("[0:a]aresample=48000")
    assert args[args.index("-ar", input_index) + 1] == "24000"
    assert args[-3:] == ["-f", "s16le", "pipe:1"]


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
    assert "ATC receiver passband" not in profile_info_graphs
    assert (
        profile_info_graphs["abusive close-mic voice front end"]
        == "atc-close-mic:abusive"
    )
    assert "open_raw" in render_graph
    assert "tail_raw" in render_graph
    assert "close_voice" not in render_graph


def test_marine_vhf_1993_profile_reports_ffmpeg_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"source audio")

    def fake_render_file(
        source_path: Path,
        output_path: Path,
        *,
        seed: int,
        intensity: str,
        ffmpeg_command: str,
    ) -> None:
        output_path.write_bytes(b"close mic voice")

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

    monkeypatch.setattr("fmplay.profiles.render_file", fake_render_file)
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
