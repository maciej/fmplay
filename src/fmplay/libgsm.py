from __future__ import annotations

import ctypes
import ctypes.util
import os
import subprocess
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

SAMPLE_RATE = 8000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
FRAME_SAMPLES = 160
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH_BYTES
GSM_FRAME_BYTES = 33


class LibGsmError(RuntimeError):
    """Raised when native libgsm cannot encode or decode audio."""


class NativeLibGsmCodec:
    """Small ctypes adapter for the libgsm GSM 06.10 API."""

    def __init__(self, library_path: str | None = None) -> None:
        self._library = _load_library(library_path)

    @classmethod
    def ensure_available(cls) -> None:
        cls()

    def round_trip_file(
        self,
        source_path: Path,
        output_path: Path,
        *,
        ffmpeg_command: str,
        filter_graph: str,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(CHANNELS)
            wav.setsampwidth(SAMPLE_WIDTH_BYTES)
            wav.setframerate(SAMPLE_RATE)
            self.round_trip_stream(
                source_path,
                _WaveOutput(wav),
                ffmpeg_command=ffmpeg_command,
                filter_graph=filter_graph,
            )

    def round_trip_stream(
        self,
        source_path: Path,
        output: BinaryIO,
        *,
        ffmpeg_command: str,
        filter_graph: str,
    ) -> None:
        producer = _start_pcm_producer(source_path, ffmpeg_command, filter_graph)
        try:
            with _gsm_state(self._library) as encoder:
                with _gsm_state(self._library) as decoder:
                    for chunk in _iter_pcm_frames(producer.stdout):
                        output.write(self._round_trip_frame(encoder, decoder, chunk))
        except BrokenPipeError:
            return
        finally:
            if producer.stdout is not None:
                producer.stdout.close()

        stderr = (
            producer.stderr.read().decode(errors="replace") if producer.stderr else ""
        )
        returncode = producer.wait()
        if returncode != 0:
            raise LibGsmError(
                f"{ffmpeg_command} failed while preparing native libgsm input"
                f"{_format_process_details(returncode, stderr)}"
            )

    def _round_trip_frame(
        self, encoder: ctypes.c_void_p, decoder: ctypes.c_void_p, chunk: bytes
    ) -> bytes:
        original_length = len(chunk)
        if original_length < FRAME_BYTES:
            chunk = chunk + (b"\0" * (FRAME_BYTES - original_length))

        samples = (ctypes.c_short * FRAME_SAMPLES).from_buffer_copy(chunk)
        frame = (ctypes.c_ubyte * GSM_FRAME_BYTES)()
        decoded = (ctypes.c_short * FRAME_SAMPLES)()

        self._library.gsm_encode(encoder, samples, frame)
        if self._library.gsm_decode(decoder, frame, decoded) != 0:
            raise LibGsmError("libgsm failed to decode an encoded GSM frame")

        return bytes(decoded)[:original_length]


def _load_library(library_path: str | None) -> ctypes.CDLL:
    errors: list[str] = []
    for candidate in _library_candidates(library_path):
        try:
            library = ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
            continue

        _configure_library(library)
        return library

    suffix = f" Tried: {'; '.join(errors)}" if errors else ""
    raise LibGsmError(
        "libgsm was not found. Install libgsm or set FMPLAY_LIBGSM_PATH "
        "to the native library path before using the 'libgsm' profile."
        f"{suffix}"
    )


def _library_candidates(library_path: str | None) -> Iterator[str]:
    if library_path:
        yield library_path

    env_path = os.environ.get("FMPLAY_LIBGSM_PATH")
    if env_path:
        yield env_path

    found = ctypes.util.find_library("gsm")
    if found:
        yield found

    yield from (
        "/opt/homebrew/opt/libgsm/lib/libgsm.dylib",
        "/opt/homebrew/lib/libgsm.dylib",
        "/usr/local/opt/libgsm/lib/libgsm.dylib",
        "/usr/local/lib/libgsm.dylib",
        "/usr/lib/x86_64-linux-gnu/libgsm.so.1",
        "/usr/lib/aarch64-linux-gnu/libgsm.so.1",
        "/usr/lib64/libgsm.so.1",
        "/usr/lib/libgsm.so.1",
        "libgsm.so.1",
        "libgsm.so",
    )


def _configure_library(library: ctypes.CDLL) -> None:
    library.gsm_create.restype = ctypes.c_void_p
    library.gsm_destroy.argtypes = [ctypes.c_void_p]
    library.gsm_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_short),
        ctypes.POINTER(ctypes.c_ubyte),
    ]
    library.gsm_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.POINTER(ctypes.c_short),
    ]
    library.gsm_decode.restype = ctypes.c_int


@contextmanager
def _gsm_state(library: ctypes.CDLL) -> Iterator[ctypes.c_void_p]:
    state = library.gsm_create()
    if not state:
        raise LibGsmError("libgsm failed to create codec state")

    try:
        yield state
    finally:
        library.gsm_destroy(state)


def _start_pcm_producer(
    source_path: Path, ffmpeg_command: str, filter_graph: str
) -> subprocess.Popen[bytes]:
    try:
        return subprocess.Popen(
            [
                ffmpeg_command,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source_path),
                "-vn",
                "-map",
                "0:a:0",
                "-af",
                filter_graph,
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                str(CHANNELS),
                "-f",
                "s16le",
                "-c:a",
                "pcm_s16le",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise LibGsmError(
            f"{ffmpeg_command} was not found. Install ffmpeg to use "
            "the 'libgsm' profile."
        ) from exc


def _iter_pcm_frames(stream: BinaryIO | None) -> Iterator[bytes]:
    if stream is None:
        raise LibGsmError("ffmpeg did not expose stdout for native libgsm input")

    while True:
        chunk = stream.read(FRAME_BYTES)
        if not chunk:
            return
        yield chunk


def _format_process_details(returncode: int, output: str) -> str:
    details = output.strip()
    if not details:
        return f": exit code {returncode}"

    return f": {details.splitlines()[-1]}"


class _WaveOutput:
    def __init__(self, wav: wave.Wave_write) -> None:
        self._wav = wav

    def write(self, data: bytes) -> int:
        self._wav.writeframes(data)
        return len(data)
