# fmplay

`fmplay` is an experimental Python utility library and CLI for playing and
transforming audio by applying degradation profiles.

The goal is to make it easy to preview how audio sounds after being passed
through familiar lossy, noisy, bandwidth-limited, or transmission-style effects.
Example profiles include:

- FM radio
- GSM voice call
- MP3 compression
- Walkie-talkie radio
- Other constrained or degraded playback chains

## Status

This project is early and exploratory.

The implementation details are intentionally unsettled for now:

- The set of underlying audio libraries has not been chosen yet.
- Some transformations may use existing tools or codecs.
- Some transformations may be implemented directly in this library.
- The Python API and CLI interface are not stable yet.

## Intended Shape

`fmplay` is expected to provide two primary interfaces:

- A Python library for applying degradation profiles programmatically.
- A CLI for transforming or previewing audio from the terminal.

## Development

The project will primarily be written in Python.

Use:

- `uv` for Python environment, dependency, and command management.
- `typer` and `rich` for CLI parsing, help, and terminal output.
- `ruff` for linting and formatting.

Set up the project:

```sh
uv sync
```

Run the CLI from the workspace:

```sh
uv run fmplay --profile passthrough audio.wav
```

Available profiles:

- `passthrough`: Plays the source file without applying any degradation.
- `gsm`: Plays a narrowband, mono GSM-phone-style degradation. It uses `ffmpeg`
  to render an 8 kHz speech-band file before playback. If your `ffmpeg` build
  supports the `libgsm` encoder, the profile round-trips through the actual GSM
  Full Rate codec; otherwise it falls back to narrowband filtering, compression,
  and bit-depth crushing.
- `marine-vhf-1993`: Plays a mono 1990s marine VHF Channel 16-style
  degradation. It renders clean speech through a staged approximation of a
  shipboard push-to-talk microphone, VHF-FM transmitter limiting, receiver
  hiss/threshold flutter, squelch open/close noise, and a small bridge speaker.

On macOS playback uses the system `afplay` command. On other platforms it will
use `ffplay` when available.

Example:

```sh
uv run fmplay --profile gsm audio.wav
```

Draw a terminal spectrogram in a Kitty graphics protocol terminal such as
Ghostty:

```sh
uv run fmplay --profile gsm --spectrogram audio.wav
```

Render the profiled spectrogram without playing audio:

```sh
uv run fmplay --profile marine-vhf-1993 --spectrogram --no-play audio.wav
```

Write the spectrogram to a PNG file instead of drawing it in the terminal:

```sh
uv run fmplay --profile marine-vhf-1993 --spectrogram=spectrogram.png --no-play audio.wav
```

Preview the Jan Heweliusz distress source through the historical VHF profile:

```sh
uv run fmplay --profile marine-vhf-1993 clips/audio/jan_heweliusz_mayday.mp3
```

Run checks:

```sh
uv run ruff format --check .
uv run ruff check .
uv run pytest
```
