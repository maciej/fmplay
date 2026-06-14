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
- `ruff` for linting and formatting.

Set up the project:

```sh
uv sync
```

Run the CLI from the workspace:

```sh
uv run fmplay --profile passthrough audio.wav
```

The initial `passthrough` profile plays the source file without applying any
degradation. On macOS it uses the system `afplay` command. On other platforms it
will use `ffplay` when available.

Run checks:

```sh
uv run ruff format --check .
uv run ruff check .
uv run pytest
```
