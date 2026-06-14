from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fmplay.backends import PlaybackBackend


class Profile(Protocol):
    """A playback or degradation profile."""

    name: str

    def play(self, path: Path, backend: PlaybackBackend) -> None:
        """Play path through this profile."""


@dataclass(frozen=True)
class PassthroughProfile:
    """Play the original audio file without transformation."""

    name: str = "passthrough"

    def play(self, path: Path, backend: PlaybackBackend) -> None:
        backend.play(path)


_PROFILES: dict[str, Profile] = {
    PassthroughProfile.name: PassthroughProfile(),
}


def get_profile(name: str) -> Profile:
    return _PROFILES[name]


def list_profiles() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))
