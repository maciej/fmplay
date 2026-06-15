# Clips

This directory stores reproducible audio clips.

- `recipes/` contains executable Python scripts that generate clips.
- `audio/` contains generated audio files and is intentionally ignored by git.
- `audio/.gitkeep` is tracked so the output directory exists in fresh clones.

Copy `.env.example` to `.env`, set `ELEVENLABS_API_KEY`, then run every
recipe:

```sh
for recipe in ./clips/recipes/*.py; do
  "$recipe"
done
```

The Jan Heweliusz recipe currently writes:

- `clips/audio/jan_heweliusz_mayday.mp3`
- `clips/audio/jan_heweliusz_abandon_ship.mp3`

The Biebrza fake radio-programme recipe writes:

- `clips/audio/biebrza_broadcast.mp3`

These files are clean source takes. Use `fmplay` profiles for historical radio,
microphone, receiver, or codec degradation.

Regenerate only one clip with:

```sh
./clips/recipes/jan_heweliusz_mayday.py --clip abandon_ship
```

Preview the mayday clip through a 1993 marine VHF Channel 16 profile:

```sh
uv run fmplay --profile marine-vhf-1993 clips/audio/jan_heweliusz_mayday.mp3
```

Preview the Biebrza fake radio programme through a radio profile:

```sh
uv run fmplay --profile marine-vhf-1993 clips/audio/biebrza_broadcast.mp3
```
