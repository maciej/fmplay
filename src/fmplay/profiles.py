from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fmplay.backends import PlaybackBackend


class ProfileError(RuntimeError):
    """Raised when a profile cannot transform or prepare audio."""


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


@dataclass(frozen=True)
class GsmCodecProfile:
    """Play audio as if it passed through an old narrowband GSM phone path."""

    name: str = "gsm"
    ffmpeg_command: str = "ffmpeg"

    def play(self, path: Path, backend: PlaybackBackend) -> None:
        with tempfile.TemporaryDirectory(prefix="fmplay-gsm-") as temp_dir:
            transformed_path = Path(temp_dir) / "gsm.wav"
            self.render(path, transformed_path)
            backend.play(transformed_path)

    def render(self, source_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._encoder_available("libgsm"):
            self._render_with_libgsm(source_path, output_path)
            return

        self._render_narrowband_fallback(source_path, output_path)

    def _render_with_libgsm(self, source_path: Path, output_path: Path) -> None:
        gsm_path = output_path.with_suffix(".gsm")
        self._run_ffmpeg(
            [
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-vn",
                "-map",
                "0:a:0",
                "-af",
                _GSM_PREFILTER,
                "-ar",
                "8000",
                "-ac",
                "1",
                "-c:a",
                "libgsm",
                "-f",
                "gsm",
                str(gsm_path),
            ],
            "encoding audio with libgsm",
        )
        self._run_ffmpeg(
            [
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "gsm",
                "-i",
                str(gsm_path),
                "-vn",
                "-ar",
                "8000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            "decoding libgsm audio",
        )

    def _render_narrowband_fallback(self, source_path: Path, output_path: Path) -> None:
        self._run_ffmpeg(
            [
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-vn",
                "-map",
                "0:a:0",
                "-af",
                _GSM_FALLBACK_FILTER,
                "-ar",
                "8000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            "rendering narrowband GSM-style audio",
        )

    def _encoder_available(self, encoder_name: str) -> bool:
        try:
            result = subprocess.run(
                [self.ffmpeg_command, "-hide_banner", "-encoders"],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise ProfileError(
                f"{self.ffmpeg_command} was not found. Install ffmpeg to use "
                f"the '{self.name}' profile."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise ProfileError(
                f"{self.ffmpeg_command} failed while listing encoders"
                f"{_format_subprocess_details(exc)}"
            ) from exc

        return any(
            len(parts) >= 2 and parts[1] == encoder_name
            for parts in (line.split() for line in result.stdout.splitlines())
        )

    def _run_ffmpeg(self, args: list[str], action: str) -> None:
        try:
            subprocess.run(
                [self.ffmpeg_command, *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise ProfileError(
                f"{self.ffmpeg_command} was not found. Install ffmpeg to use "
                f"the '{self.name}' profile."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise ProfileError(
                f"{self.ffmpeg_command} failed while {action}"
                f"{_format_subprocess_details(exc)}"
            ) from exc


_GSM_PREFILTER = ",".join(
    [
        "highpass=f=260",
        "lowpass=f=3400",
        "compand=attacks=0.02:decays=0.20:points=-90/-90|-50/-42|-24/-18|-8/-8|0/-3",
    ]
)

_GSM_FALLBACK_FILTER = ",".join(
    [
        _GSM_PREFILTER,
        "acrusher=bits=8:mode=log:aa=1",
    ]
)


def _format_subprocess_details(exc: subprocess.CalledProcessError) -> str:
    details = (exc.stderr or exc.stdout or "").strip()
    if not details:
        return f": exit code {exc.returncode}"

    last_line = details.splitlines()[-1]
    return f": {last_line}"


_PROFILES: dict[str, Profile] = {
    GsmCodecProfile.name: GsmCodecProfile(),
    PassthroughProfile.name: PassthroughProfile(),
}


def get_profile(name: str) -> Profile:
    return _PROFILES[name]


def list_profiles() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))
