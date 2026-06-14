# Clips

This directory stores reproducible audio clips.

- `recipes/` contains executable Python scripts that generate clips.
- `audio/` contains generated audio files and is intentionally ignored by git.
- `audio/.gitkeep` is tracked so the output directory exists in fresh clones.

Copy `.env.example` to `.env`, set `ELEVENLABS_API_KEY`, then run a recipe:

```sh
./clips/recipes/*.py
```

The Jan Heweliusz recipe currently writes:

- `clips/audio/jan_heweliusz_mayday.mp3`
- `clips/audio/jan_heweliusz_abandon_ship.mp3`

Regenerate only one clip with:

```sh
./clips/recipes/jan_heweliusz_mayday.py --clip abandon_ship
```
