from __future__ import annotations

from pathlib import Path

import fmplay.marine_vhf_stream as marine_vhf_stream


class FakePipe:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False

    def write(self, data: bytes) -> int:
        self.data.extend(data)
        return len(data)

    def close(self) -> None:
        self.closed = True


class FakeStderr:
    def __init__(self, data: bytes = b"") -> None:
        self.data = data

    def read(self) -> bytes:
        return self.data


class FakeProcess:
    def __init__(self, return_code: int = 0, stderr: bytes = b"") -> None:
        self.stdin = FakePipe()
        self.stderr = FakeStderr(stderr)
        self.return_code = return_code
        self.killed = False

    def wait(self) -> int:
        return self.return_code

    def kill(self) -> None:
        self.killed = True


def test_marine_vhf_stream_pipes_close_mic_pcm_to_ffmpeg(monkeypatch) -> None:
    processes: list[FakeProcess] = []
    commands: list[list[str]] = []
    stream_calls: list[tuple[Path, int, str, str]] = []

    def fake_popen(command, *, stdin, stdout, stderr):
        process = FakeProcess()
        processes.append(process)
        commands.append(command)
        return process

    def fake_stream_file(
        source_path: Path,
        output,
        *,
        seed: int,
        intensity: str,
        ffmpeg_command: str,
    ) -> None:
        stream_calls.append((source_path, seed, intensity, ffmpeg_command))
        output.write(b"close-mic-pcm")

    monkeypatch.setattr(marine_vhf_stream.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(marine_vhf_stream, "stream_file", fake_stream_file)

    result = marine_vhf_stream.main(
        [
            "--ffmpeg",
            "ffmpeg-test",
            "--squelch-seed",
            "333",
            "--close-mic-seed",
            "444",
            "source.wav",
        ]
    )

    assert result == 0
    assert stream_calls == [(Path("source.wav"), 444, "abusive", "ffmpeg-test")]
    assert processes[0].stdin.data == b"close-mic-pcm"
    assert processes[0].stdin.closed
    command = commands[0]
    input_index = command.index("-i")
    assert command[:1] == ["ffmpeg-test"]
    assert command[input_index - 6 : input_index + 2] == [
        "-f",
        "s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-i",
        "pipe:0",
    ]
    assert command[-3:] == ["-f", "s16le", "pipe:1"]
