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
        _run_ffmpeg(self.ffmpeg_command, self.name, args, action)


@dataclass(frozen=True)
class MarineVhf1993Profile:
    """Play audio as a nearby ship may have heard 1993 VHF Channel 16."""

    name: str = "marine-vhf-1993"
    ffmpeg_command: str = "ffmpeg"

    def play(self, path: Path, backend: PlaybackBackend) -> None:
        with tempfile.TemporaryDirectory(prefix="fmplay-marine-vhf-1993-") as temp_dir:
            transformed_path = Path(temp_dir) / "marine-vhf-1993.wav"
            self.render(path, transformed_path)
            backend.play(transformed_path)

    def render(self, source_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _run_ffmpeg(
            self.ffmpeg_command,
            self.name,
            [
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-vn",
                "-filter_complex",
                _marine_vhf_1993_filter_graph(),
                "-map",
                "[out]",
                "-ar",
                "24000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            "rendering 1993 marine VHF Channel 16 audio",
        )


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


@dataclass(frozen=True)
class _FilterStage:
    name: str
    graph: str


_MARINE_VHF_1993_STAGES = (
    _FilterStage(
        "transmitter microphone and limiter",
        ",".join(
            [
                "[0:a]aresample=48000",
                "aformat=channel_layouts=mono",
                "highpass=f=260",
                "lowpass=f=3600",
                (
                    "compand=attacks=0.004:decays=0.08:"
                    "points=-80/-70|-45/-30|-24/-14|-12/-7|-3/-2|0/-1:"
                    "soft-knee=2:gain=5"
                ),
                "alimiter=limit=0.88",
                "acrusher=bits=11:mode=log:aa=1",
                "tremolo=f=3.2:d=0.035",
                r"volume='if(lt(mod(t\,2.4)\,0.035)\,0.62\,1)':eval=frame",
            ]
        )
        + "[tx]",
    ),
    _FilterStage(
        "vhf receiver hiss",
        ",".join(
            [
                "anoisesrc=r=48000:a=0.018:c=white:s=19930114",
                "highpass=f=2400",
                "lowpass=f=6200",
            ]
        )
        + "[hiss]",
    ),
    _FilterStage(
        "shipboard power rumble",
        ",".join(
            [
                "anoisesrc=r=48000:a=0.006:c=pink:s=19930115",
                "lowpass=f=260",
            ]
        )
        + "[rumble]",
    ),
    _FilterStage(
        "nearby ship receiver speaker",
        ",".join(
            [
                "[tx][hiss][rumble]amix=inputs=3:duration=first:"
                "weights='1 0.35 0.12':normalize=0",
                "highpass=f=330",
                "lowpass=f=3100",
                "equalizer=f=850:t=q:w=1.3:g=4",
                "equalizer=f=2300:t=q:w=1.4:g=-3",
                "alimiter=limit=0.92",
            ]
        )
        + "[body]",
    ),
    _FilterStage(
        "squelch open noise",
        ",".join(
            [
                "anoisesrc=r=48000:a=0.075:c=white:d=0.11:s=19930116",
                "highpass=f=2200",
                "lowpass=f=6200",
                "afade=t=out:st=0.04:d=0.07",
            ]
        )
        + "[open_noise]",
    ),
    _FilterStage(
        "push-to-talk open click",
        ",".join(
            [
                "sine=f=950:r=48000:d=0.018",
                "afade=t=out:st=0:d=0.018",
                "volume=2.8",
            ]
        )
        + "[open_click]",
    ),
    _FilterStage(
        "squelch open mix",
        ",".join(
            [
                "[open_noise][open_click]amix=inputs=2:duration=longest:normalize=0",
                "alimiter=limit=0.9",
            ]
        )
        + "[open]",
    ),
    _FilterStage(
        "squelch tail noise",
        ",".join(
            [
                "anoisesrc=r=48000:a=0.08:c=white:d=0.22:s=19930117",
                "highpass=f=2200",
                "lowpass=f=6200",
                "afade=t=out:st=0.13:d=0.09",
            ]
        )
        + "[tail_noise]",
    ),
    _FilterStage(
        "push-to-talk release click",
        ",".join(
            [
                "sine=f=520:r=48000:d=0.022",
                "afade=t=out:st=0:d=0.022",
                "volume=1.8",
            ]
        )
        + "[tail_click]",
    ),
    _FilterStage(
        "squelch tail mix",
        ",".join(
            [
                "[tail_click][tail_noise]amix=inputs=2:duration=longest:normalize=0",
                "alimiter=limit=0.9",
            ]
        )
        + "[tail]",
    ),
    _FilterStage(
        "final concatenation",
        ",".join(
            [
                "[open][body][tail]concat=n=3:v=0:a=1",
                "aresample=24000",
                "aformat=channel_layouts=mono",
                "alimiter=limit=0.95",
            ]
        )
        + "[out]",
    ),
)


def _marine_vhf_1993_filter_graph() -> str:
    graphs: list[str] = []
    for stage in _MARINE_VHF_1993_STAGES:
        if not stage.name:
            raise ProfileError("Marine VHF profile contains an unnamed filter stage.")
        graphs.append(stage.graph)
    return ";".join(graphs)


def _run_ffmpeg(
    ffmpeg_command: str, profile_name: str, args: list[str], action: str
) -> None:
    try:
        subprocess.run(
            [ffmpeg_command, *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ProfileError(
            f"{ffmpeg_command} was not found. Install ffmpeg to use "
            f"the '{profile_name}' profile."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ProfileError(
            f"{ffmpeg_command} failed while {action}{_format_subprocess_details(exc)}"
        ) from exc


def _format_subprocess_details(exc: subprocess.CalledProcessError) -> str:
    details = (exc.stderr or exc.stdout or "").strip()
    if not details:
        return f": exit code {exc.returncode}"

    last_line = details.splitlines()[-1]
    return f": {last_line}"


_PROFILES: dict[str, Profile] = {
    GsmCodecProfile.name: GsmCodecProfile(),
    MarineVhf1993Profile.name: MarineVhf1993Profile(),
    PassthroughProfile.name: PassthroughProfile(),
}


def get_profile(name: str) -> Profile:
    return _PROFILES[name]


def list_profiles() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))
