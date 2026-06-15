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
- `fmradio`: Plays a public FM-radio-style degradation tuned near 98.3 MHz,
  roughly the middle of the public FM broadcast band. It uses `ffmpeg` to stream
  a stereo broadcast chain with FM-style bandwidth limiting, broadcast
  compression, mild receiver flutter, hiss, low tuner bed noise, and slight
  19 kHz pilot-tone leakage.
- `gsm`: Plays a narrowband, mono GSM-phone-style degradation. It uses `ffmpeg`
  to stream an 8 kHz speech-band signal by default. If your `ffmpeg` build
  supports the `libgsm` encoder, the profile round-trips through the actual GSM
  Full Rate codec; otherwise it falls back to narrowband filtering, compression,
  and bit-depth crushing.
- `libgsm`: Round-trips audio through the native `libgsm` library directly,
  using `ffmpeg` only to decode, prefilter, resample, and mix the source to
  8 kHz mono PCM. This profile requires `libgsm` at runtime; if it is missing,
  only this profile fails.
- `marine-vhf-1993`: Plays a mono 1990s marine VHF Channel 16-style
  degradation. It streams clean speech through a staged approximation of a
  shipboard push-to-talk microphone, VHF-FM transmitter limiting, receiver
  hiss/threshold flutter, squelch open/close noise, and a small bridge speaker.

Playback uses `ffplay` when available so profiled audio can be streamed directly
to the player instead of being rendered to a temporary file first. If `ffplay`
is unavailable on macOS, playback falls back to the system `afplay` command.

Example:

```sh
uv run fmplay --profile fmradio audio.wav
```

Preview a profile or reusable profile stage using generated audio:

```sh
uv run fmplay preview fmradio --source white --duration 10
```

Reusable profile stages are findable by `preview` but are not listed as normal
profiles. The `cockpit:a320` stage synthesizes an Airbus A320 cockpit ambient
bed with seeded ECS/pack roar, windshield boundary-layer flow, avionics fan
texture, detuned engine tonal leakage, and sparse cockpit ticks/creaks. Omit
`--seed` for a fresh variant, or pass one to repeat the exact same texture:

```sh
uv run fmplay preview cockpit:a320
uv run fmplay preview cockpit:a320 --seed 42 --duration 30
```

See [docs/cockpit-audio-simulation.md](docs/cockpit-audio-simulation.md) for
the model notes, tuning caveats, and references to keep for future cockpit
audio work.

The `radio:squelch` stage synthesizes a randomized library of receiver squelch
opens, tail crashes, carrier snaps, and weak-signal chatter for future ATC or
radio scenes:

```sh
uv run fmplay preview radio:squelch --duration 20
uv run fmplay preview radio:squelch --seed 42 --duration 20
```

See [docs/squelch-sound-library.md](docs/squelch-sound-library.md) for the
model notes, Hugging Face reference workflow, and validation tooling.

Draw a terminal spectrogram in a Kitty graphics protocol terminal such as
Ghostty. Spectrograms are rendered from a profiled temporary file before
playback:

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
