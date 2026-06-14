# Codex

## Worktrees

Codex worktrees are normal Git worktrees, so ignored files such as `.env` and
`clips/audio/*` are not present by default. The checked-in Codex local
environment at `.codex/environments/environment.toml` bootstraps new Codex
worktrees by hard-linking `.env` and files from `clips/audio/` from the main
checkout, then running `uv sync`.

The setup script resolves the main checkout through Git's common directory, so
it does not depend on a hard-coded local path. If hard-linking is not available,
it falls back to copying the file.
