from __future__ import annotations

import base64
import math
import struct
import subprocess
import sys
import zlib
from os import environ
from pathlib import Path
from shutil import which
from typing import BinaryIO


class SpectrogramError(RuntimeError):
    """Raised when a spectrogram cannot be rendered or displayed."""


def render_spectrogram_image(
    audio_path: Path,
    output_path: Path,
    *,
    ffmpeg_command: str = "ffmpeg",
    size: str = "1200x640",
) -> None:
    """Render a spectrogram PNG for audio_path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = _parse_size(size)
    layout = _SpectrogramLayout.from_size(width, height)
    sample_rate = _probe_sample_rate(audio_path, ffmpeg_command, "ffprobe")
    duration = _probe_duration(audio_path, ffmpeg_command, "ffprobe")
    filters = _terminal_spectrogram_filter(layout.plot_size)

    try:
        result = subprocess.run(
            [
                ffmpeg_command,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(audio_path),
                "-lavfi",
                filters,
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            check=True,
            capture_output=True,
            text=False,
        )
    except FileNotFoundError as exc:
        raise SpectrogramError(
            f"{ffmpeg_command} was not found. Install ffmpeg to render spectrograms."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SpectrogramError(
            f"{ffmpeg_command} failed while rendering spectrogram"
            f"{_format_subprocess_details(exc)}"
        ) from exc

    expected_bytes = layout.plot_width * layout.plot_height * 3
    if len(result.stdout) != expected_bytes:
        raise SpectrogramError(
            f"{ffmpeg_command} produced {len(result.stdout)} spectrogram bytes; "
            f"expected {expected_bytes}"
        )

    image = _compose_axis_image(
        result.stdout,
        layout,
        sample_rate=sample_rate,
        duration=duration,
    )
    output_path.write_bytes(_encode_rgb_png(width, height, image))


def print_kitty_image(image_path: Path, output: BinaryIO | None = None) -> None:
    """Write image_path to stdout using the Kitty graphics protocol."""

    if output is None and not terminal_supports_kitty_graphics():
        raise SpectrogramError(
            "This terminal does not appear to support the Kitty graphics protocol. "
            "Render to a file instead with --spectrogram=file.png."
        )

    stream = output or sys.stdout.buffer
    data = base64.b64encode(image_path.read_bytes())
    if not data:
        raise SpectrogramError(f"spectrogram image is empty: {image_path}")

    chunk_size = 4096
    chunks = [
        data[index : index + chunk_size] for index in range(0, len(data), chunk_size)
    ]

    for index, chunk in enumerate(chunks):
        more = 1 if index < len(chunks) - 1 else 0
        if index == 0:
            header = f"\033_Ga=T,f=100,t=d,q=2,m={more};".encode()
        else:
            header = f"\033_Gm={more};".encode()
        stream.write(header + chunk + b"\033\\")

    stream.write(b"\n")
    stream.flush()


def terminal_supports_kitty_graphics() -> bool:
    """Return whether stdout is likely to understand Kitty graphics escapes."""

    if not sys.stdout.isatty():
        return False

    term = environ.get("TERM", "").lower()
    term_program = environ.get("TERM_PROGRAM", "").lower()
    return any(
        [
            "kitty" in term,
            "ghostty" in term,
            "kitty" in term_program,
            "ghostty" in term_program,
            "wezterm" in term_program,
            bool(environ.get("KITTY_WINDOW_ID")),
            bool(environ.get("GHOSTTY_RESOURCES_DIR")),
            bool(environ.get("WEZTERM_EXECUTABLE")),
        ]
    )


def _format_subprocess_details(exc: subprocess.CalledProcessError) -> str:
    details = (exc.stderr or exc.stdout or "").strip()
    if isinstance(details, bytes):
        details = details.decode(errors="replace")
    if not details:
        return f": exit code {exc.returncode}"

    last_line = details.splitlines()[-1]
    return f": {last_line}"


def _terminal_spectrogram_filter(size: str) -> str:
    return (
        f"showspectrumpic=s={size}:legend=0:scale=log:fscale=lin:"
        "color=magma:saturation=1.75:gain=2.2:drange=85"
    )


class _SpectrogramLayout:
    def __init__(
        self,
        *,
        width: int,
        height: int,
        plot_x: int,
        plot_y: int,
        plot_width: int,
        plot_height: int,
    ) -> None:
        self.width = width
        self.height = height
        self.plot_x = plot_x
        self.plot_y = plot_y
        self.plot_width = plot_width
        self.plot_height = plot_height

    @classmethod
    def from_size(cls, width: int, height: int) -> _SpectrogramLayout:
        left_margin = 58
        right_margin = 12
        top_margin = 22
        bottom_margin = 48
        plot_width = max(100, width - left_margin - right_margin)
        plot_height = max(100, height - top_margin - bottom_margin)
        return cls(
            width=width,
            height=height,
            plot_x=left_margin,
            plot_y=top_margin,
            plot_width=plot_width,
            plot_height=plot_height,
        )

    @property
    def plot_size(self) -> str:
        return f"{self.plot_width}x{self.plot_height}"


def _parse_size(size: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in size.lower().split("x", 1))
    except ValueError as exc:
        raise SpectrogramError(f"invalid spectrogram size: {size}") from exc
    if width < 200 or height < 180:
        raise SpectrogramError("spectrogram size must be at least 200x180")
    return width, height


def _compose_axis_image(
    plot_rgb: bytes,
    layout: _SpectrogramLayout,
    *,
    sample_rate: int | None,
    duration: float | None,
) -> bytes:
    canvas = bytearray(_rgb(0, 0, 0) * layout.width * layout.height)
    _paste_rgb(
        canvas,
        layout.width,
        plot_rgb,
        layout.plot_width,
        layout.plot_height,
        layout.plot_x,
        layout.plot_y,
    )
    _draw_axes(canvas, layout, sample_rate=sample_rate, duration=duration)
    return bytes(canvas)


def _draw_axes(
    canvas: bytearray,
    layout: _SpectrogramLayout,
    *,
    sample_rate: int | None,
    duration: float | None,
) -> None:
    axis_color = _rgb(185, 194, 214)
    grid_color = _rgb(28, 34, 50)
    text_color = _rgb(222, 228, 238)
    muted_text = _rgb(154, 164, 186)
    plot_left = layout.plot_x
    plot_right = layout.plot_x + layout.plot_width - 1
    plot_top = layout.plot_y
    plot_bottom = layout.plot_y + layout.plot_height - 1

    _draw_line(
        canvas,
        layout.width,
        plot_left,
        plot_bottom,
        plot_right,
        plot_bottom,
        axis_color,
    )
    _draw_line(
        canvas, layout.width, plot_left, plot_top, plot_left, plot_bottom, axis_color
    )

    if sample_rate:
        nyquist_hz = sample_rate / 2
        for freq_hz in _frequency_labels(nyquist_hz):
            y = round(plot_bottom - (freq_hz / nyquist_hz * (layout.plot_height - 1)))
            _draw_line(
                canvas, layout.width, plot_left - 5, y, plot_left - 1, y, axis_color
            )
            _draw_line(canvas, layout.width, plot_left, y, plot_right, y, grid_color)
            label = "0" if freq_hz == 0 else f"{freq_hz // 1000}k"
            label_width = _text_width(label, scale=2)
            _draw_text(
                canvas,
                layout.width,
                plot_left - 10 - label_width,
                _clamp(y - 7, 0, layout.height - 14),
                label,
                text_color,
                scale=2,
            )
        axis_label = "freq_kHz"
        _draw_text_rotated_counterclockwise(
            canvas,
            layout.width,
            4,
            plot_top
            + (layout.plot_height // 2)
            + (_text_width(axis_label, scale=2) // 2),
            axis_label,
            muted_text,
            scale=2,
        )

    if duration and duration > 0:
        for second in _time_ticks(duration):
            x = round(plot_left + (second / duration * (layout.plot_width - 1)))
            _draw_line(
                canvas, layout.width, x, plot_bottom + 1, x, plot_bottom + 5, axis_color
            )
            _draw_line(canvas, layout.width, x, plot_top, x, plot_bottom, grid_color)
            label = _format_time_tick(second)
            label_width = _text_width(label, scale=2)
            _draw_text(
                canvas,
                layout.width,
                _clamp(x - (label_width // 2), 0, layout.width - label_width - 1),
                plot_bottom + 10,
                label,
                text_color,
                scale=2,
            )
        axis_label = "time"
        _draw_text(
            canvas,
            layout.width,
            plot_left
            + (layout.plot_width // 2)
            - (_text_width(axis_label, scale=2) // 2),
            layout.height - 17,
            axis_label,
            muted_text,
            scale=2,
        )


def _probe_sample_rate(
    audio_path: Path, ffmpeg_command: str, ffprobe_command: str
) -> int | None:
    command = _resolve_ffprobe_command(ffmpeg_command, ffprobe_command)
    try:
        result = subprocess.run(
            [
                command,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    try:
        return int(result.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None


def _probe_duration(
    audio_path: Path, ffmpeg_command: str, ffprobe_command: str
) -> float | None:
    command = _resolve_ffprobe_command(ffmpeg_command, ffprobe_command)
    try:
        result = subprocess.run(
            [
                command,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    try:
        duration = float(result.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


def _resolve_ffprobe_command(ffmpeg_command: str, ffprobe_command: str) -> str:
    ffmpeg_path = Path(ffmpeg_command)
    if ffmpeg_path.name == "ffmpeg":
        sibling = ffmpeg_path.with_name("ffprobe")
        if ffmpeg_path.parent != Path(".") and sibling.exists():
            return str(sibling)

    return ffprobe_command if which(ffprobe_command) else "ffprobe"


def _frequency_labels(nyquist_hz: float) -> tuple[int, ...]:
    candidates = (
        0,
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        8000,
        10000,
        12000,
        16000,
        20000,
    )
    return tuple(freq for freq in candidates if freq <= nyquist_hz)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _time_ticks(duration: float, target_ticks: int = 7) -> tuple[float, ...]:
    if duration <= 0:
        return ()

    step = _nice_step(duration / max(1, target_ticks - 1))
    ticks: list[float] = []
    value = 0.0
    while value < duration:
        ticks.append(value)
        value += step
    if not ticks or not math.isclose(ticks[-1], duration, rel_tol=0, abs_tol=0.01):
        ticks.append(duration)
    return tuple(ticks)


def _nice_step(raw_step: float) -> float:
    if raw_step <= 1:
        return 1

    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1:
        nice = 1
    elif normalized <= 2:
        nice = 2
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10
    return nice * magnitude


def _format_time_tick(seconds: float) -> str:
    rounded = int(round(seconds))
    minutes, second = divmod(rounded, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minute:02d}:{second:02d}"
    return f"{minute}:{second:02d}"


def _paste_rgb(
    canvas: bytearray,
    canvas_width: int,
    source: bytes,
    source_width: int,
    source_height: int,
    x: int,
    y: int,
) -> None:
    row_bytes = source_width * 3
    for row in range(source_height):
        source_start = row * row_bytes
        target_start = ((y + row) * canvas_width + x) * 3
        canvas[target_start : target_start + row_bytes] = source[
            source_start : source_start + row_bytes
        ]


def _draw_line(
    canvas: bytearray,
    width: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: bytes,
) -> None:
    if x1 == x2:
        start, end = sorted((y1, y2))
        for y in range(start, end + 1):
            _set_pixel(canvas, width, x1, y, color)
        return
    if y1 == y2:
        start, end = sorted((x1, x2))
        for x in range(start, end + 1):
            _set_pixel(canvas, width, x, y1, color)


def _draw_text(
    canvas: bytearray,
    width: int,
    x: int,
    y: int,
    text: str,
    color: bytes,
    *,
    scale: int = 1,
) -> None:
    cursor = x
    for char in text:
        glyph = _FONT_5X7.get(char, _FONT_5X7.get(char.lower(), _FONT_5X7[" "]))
        for row_index, row in enumerate(glyph):
            for col_index, pixel in enumerate(row):
                if pixel == " ":
                    continue
                for dy in range(scale):
                    for dx in range(scale):
                        _set_pixel(
                            canvas,
                            width,
                            cursor + col_index * scale + dx,
                            y + row_index * scale + dy,
                            color,
                        )
        cursor += 6 * scale


def _draw_text_rotated_counterclockwise(
    canvas: bytearray,
    width: int,
    x: int,
    baseline_y: int,
    text: str,
    color: bytes,
    *,
    scale: int = 1,
) -> None:
    cursor = 0
    for char in text:
        glyph = _FONT_5X7.get(char, _FONT_5X7.get(char.lower(), _FONT_5X7[" "]))
        for row_index, row in enumerate(glyph):
            for col_index, pixel in enumerate(row):
                if pixel == " ":
                    continue
                for dy in range(scale):
                    for dx in range(scale):
                        _set_pixel(
                            canvas,
                            width,
                            x + row_index * scale + dy,
                            baseline_y - cursor - col_index * scale - dx,
                            color,
                        )
        cursor += 6 * scale


def _text_width(text: str, *, scale: int = 1) -> int:
    if not text:
        return 0
    return (len(text) * 6 - 1) * scale


def _set_pixel(canvas: bytearray, width: int, x: int, y: int, color: bytes) -> None:
    if x < 0 or y < 0:
        return
    index = (y * width + x) * 3
    if index < 0 or index + 3 > len(canvas):
        return
    canvas[index : index + 3] = color


def _rgb(red: int, green: int, blue: int) -> bytes:
    return bytes((red, green, blue))


def _encode_rgb_png(width: int, height: int, pixels: bytes) -> bytes:
    row_bytes = width * 3
    raw = b"".join(
        b"\x00" + pixels[row * row_bytes : (row + 1) * row_bytes]
        for row in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, level=6))
        + _png_chunk(b"IEND", b"")
    )


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type)
    checksum = zlib.crc32(payload, checksum)
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", checksum & 0xFFFFFFFF)
    )


_FONT_5X7: dict[str, tuple[str, ...]] = {
    " ": ("     ", "     ", "     ", "     ", "     ", "     ", "     "),
    "(": ("  ## ", " #   ", "#    ", "#    ", "#    ", " #   ", "  ## "),
    ")": ("##   ", "  #  ", "   # ", "   # ", "   # ", "  #  ", "##   "),
    ":": ("     ", "  #  ", "     ", "     ", "     ", "  #  ", "     "),
    "0": (" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "),
    "1": ("  #  ", " ##  ", "# #  ", "  #  ", "  #  ", "  #  ", "#####"),
    "2": (" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"),
    "3": (" ### ", "#   #", "    #", "  ## ", "    #", "#   #", " ### "),
    "4": ("   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "),
    "5": ("#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "),
    "6": (" ### ", "#   #", "#    ", "#### ", "#   #", "#   #", " ### "),
    "7": ("#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "),
    "8": (" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "),
    "9": (" ### ", "#   #", "#   #", " ####", "    #", "#   #", " ### "),
    "H": ("#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"),
    "_": ("     ", "     ", "     ", "     ", "     ", "     ", "#####"),
    "e": ("     ", "     ", " ### ", "#   #", "#####", "#    ", " ### "),
    "f": ("  ## ", " #  #", " #   ", "###  ", " #   ", " #   ", " #   "),
    "h": ("#    ", "#    ", "# ## ", "##  #", "#   #", "#   #", "#   #"),
    "i": ("  #  ", "     ", " ##  ", "  #  ", "  #  ", "  #  ", " ### "),
    "k": ("#    ", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"),
    "m": ("     ", "     ", "## # ", "# # #", "# # #", "#   #", "#   #"),
    "q": ("     ", "     ", " ### ", "#   #", "# # #", " ####", "    #"),
    "r": ("     ", "     ", "# ## ", "##  #", "#    ", "#    ", "#    "),
    "s": ("     ", "     ", " ####", "#    ", " ### ", "    #", "#### "),
    "t": (" #   ", " #   ", "###  ", " #   ", " #   ", " #  #", "  ## "),
    "u": ("     ", "     ", "#   #", "#   #", "#   #", "#  ##", " ## #"),
    "y": ("     ", "     ", "#   #", "#   #", " ####", "    #", " ### "),
    "z": ("     ", "     ", "#####", "   # ", "  #  ", " #   ", "#####"),
}
