from __future__ import annotations

import struct
import subprocess
import zlib
from pathlib import Path

import pytest

from fmplay.spectrogram import (
    SpectrogramError,
    print_kitty_image,
    render_spectrogram_image,
)


def test_render_spectrogram_adds_axis_frame(tmp_path: Path, monkeypatch) -> None:
    audio_file = tmp_path / "audio.wav"
    output_file = tmp_path / "spectrogram.png"
    audio_file.write_bytes(b"audio")

    def fake_run(command, *, check, capture_output, text):
        if "stream=sample_rate" in command:
            return subprocess.CompletedProcess(command, 0, stdout="24000\n")
        if "format=duration" in command:
            return subprocess.CompletedProcess(command, 0, stdout="14.0\n")

        assert text is False
        assert any("showspectrumpic=s=170x130" in part for part in command)
        return subprocess.CompletedProcess(command, 0, stdout=_rgb_bytes(170, 130))

    monkeypatch.setattr("fmplay.spectrogram.subprocess.run", fake_run)

    render_spectrogram_image(audio_file, output_file, size="240x200")

    png = output_file.read_bytes()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert _png_size(png) == (240, 200)
    pixels = _decode_png_rgb_rows(png)
    assert len({pixel for row in pixels for pixel in row}) > 2


def test_print_kitty_image_reports_unsupported_terminal(tmp_path: Path) -> None:
    image_file = tmp_path / "spectrogram.png"
    image_file.write_bytes(b"png")

    with pytest.raises(SpectrogramError, match="--spectrogram=file.png"):
        print_kitty_image(image_file)


def _rgb_bytes(width: int, height: int) -> bytes:
    rows = []
    for y in range(height):
        for x in range(width):
            rows.append(bytes((x % 256, y % 256, 128)))
    return b"".join(rows)


def _png_size(png: bytes) -> tuple[int, int]:
    assert png[12:16] == b"IHDR"
    return struct.unpack(">II", png[16:24])


def _decode_png_rgb_rows(png: bytes) -> list[list[bytes]]:
    width, height = _png_size(png)
    offset = 8
    idat = bytearray()
    while offset < len(png):
        length = struct.unpack(">I", png[offset : offset + 4])[0]
        chunk_type = png[offset + 4 : offset + 8]
        payload = png[offset + 8 : offset + 8 + length]
        if chunk_type == b"IDAT":
            idat.extend(payload)
        offset += 12 + length

    raw = zlib.decompress(idat)
    stride = width * 3 + 1
    rows = []
    for row in range(height):
        start = row * stride
        assert raw[start] == 0
        row_bytes = raw[start + 1 : start + stride]
        rows.append(
            [
                bytes(row_bytes[index : index + 3])
                for index in range(0, len(row_bytes), 3)
            ]
        )
    return rows
