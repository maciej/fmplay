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
    description: str

    def play(self, path: Path, backend: PlaybackBackend) -> None:
        """Play path through this profile."""


@dataclass(frozen=True)
class ProfileSummary:
    """Display metadata for an available profile."""

    name: str
    description: str


@dataclass(frozen=True)
class PassthroughProfile:
    """Play the original audio file without transformation."""

    name: str = "passthrough"
    description: str = "Play the source file without applying degradation."

    def play(self, path: Path, backend: PlaybackBackend) -> None:
        backend.play(path)


@dataclass(frozen=True)
class GsmCodecProfile:
    """Play audio as if it passed through an old narrowband GSM phone path."""

    name: str = "gsm"
    description: str = "Narrowband mono GSM-phone-style degradation."
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
    description: str = "1990s marine VHF Channel 16 radio degradation."
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


@dataclass(frozen=True)
class FmRadioProfile:
    """Play audio through a plain public FM radio broadcast path."""

    name: str = "fmradio"
    description: str = "Public FM radio degradation tuned near 98.3 MHz."
    frequency_mhz: float = 98.3
    ffmpeg_command: str = "ffmpeg"

    def play(self, path: Path, backend: PlaybackBackend) -> None:
        with tempfile.TemporaryDirectory(prefix="fmplay-fmradio-") as temp_dir:
            transformed_path = Path(temp_dir) / "fmradio.wav"
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
                _fmradio_filter_graph(self.frequency_mhz),
                "-map",
                "[out]",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            f"rendering public FM radio audio at {self.frequency_mhz:.1f} MHz",
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


_MARINE_VHF_STATIC_FLUTTER = (
    r"volume='0.78+0.07*sin(13.7*t)+0.04*sin(51*t)+"
    r"if(lt(mod(t+0.07\,0.37)\,0.026)\,0.18\,0)':eval=frame"
)


def _fmradio_filter_graph(frequency_mhz: float) -> str:
    seed_base = int(round(frequency_mhz * 1000))
    stages = (
        _FilterStage(
            "broadcast processor",
            ",".join(
                [
                    "[0:a]aresample=48000",
                    "aformat=channel_layouts=stereo",
                    "highpass=f=45",
                    "lowpass=f=15000",
                    "equalizer=f=180:t=q:w=0.9:g=-1.5",
                    "equalizer=f=3300:t=q:w=1.1:g=1.8",
                    (
                        "compand=attacks=0.006:decays=0.18:"
                        "points=-90/-85|-45/-36|-18/-14|-6/-5|0/-2:"
                        "soft-knee=3:gain=1.5"
                    ),
                    "alimiter=limit=0.93",
                    r"volume='0.98+0.018*sin(1.7*t)+0.01*sin(6.1*t)':eval=frame",
                    "tremolo=f=0.32:d=0.012",
                ]
            )
            + "[program]",
        ),
        _FilterStage(
            f"{frequency_mhz:.1f} MHz receiver hiss",
            ",".join(
                [
                    f"anoisesrc=r=48000:a=0.0045:c=white:s={seed_base + 1}",
                    "highpass=f=6800",
                    "lowpass=f=17000",
                ]
            )
            + "[hiss]",
        ),
        _FilterStage(
            "quiet tuner bed",
            ",".join(
                [
                    f"anoisesrc=r=48000:a=0.0014:c=pink:s={seed_base + 2}",
                    "lowpass=f=180",
                ]
            )
            + "[bed]",
        ),
        _FilterStage(
            "stereo pilot leakage",
            ",".join(["sine=f=19000:r=48000", "volume=0.0016"]) + "[pilot]",
        ),
        _FilterStage(
            "receiver speaker output",
            ",".join(
                [
                    (
                        "[program][hiss][bed][pilot]amix=inputs=4:duration=first:"
                        "weights='1 0.9 0.45 0.35':normalize=0"
                    ),
                    "lowpass=f=16500",
                    "alimiter=limit=0.95",
                    "volume=0.92",
                ]
            )
            + "[out]",
        ),
    )

    return ";".join(stage.graph for stage in stages)


def _marine_vhf_receiver_static_graph(
    *,
    prefix: str,
    duration: str,
    hiss_seed: int,
    low_seed: int,
    crackle_gate_state: int,
    crackle_level_state: int,
    fade_filters: tuple[str, ...],
    output: str,
) -> str:
    return ";".join(
        [
            ",".join(
                [
                    f"anoisesrc=r=48000:a=0.034:c=white:d={duration}:s={hiss_seed}",
                    "highpass=f=1600",
                    "lowpass=f=6400",
                    "equalizer=f=3100:t=q:w=1.4:g=2.5",
                    "tremolo=f=7.3:d=0.06",
                    _MARINE_VHF_STATIC_FLUTTER,
                ]
            )
            + f"[{prefix}_hiss]",
            ",".join(
                [
                    (
                        "aevalsrc='if(lt(random("
                        f"{crackle_gate_state}"
                        r")\,0.00032)\,random("
                        f"{crackle_level_state}"
                        r")*2-1\,0)':s=48000:d="
                        f"{duration}"
                    ),
                    "highpass=f=2600",
                    "lowpass=f=9000",
                    "volume=0.55",
                ]
            )
            + f"[{prefix}_crackle]",
            ",".join(
                [
                    f"anoisesrc=r=48000:a=0.01:c=pink:d={duration}:s={low_seed}",
                    "highpass=f=90",
                    "lowpass=f=520",
                ]
            )
            + f"[{prefix}_low]",
            ",".join(
                [
                    (
                        f"[{prefix}_hiss][{prefix}_crackle][{prefix}_low]"
                        "amix=inputs=3:duration=longest:"
                        "weights='1 0.28 0.12':normalize=0"
                    ),
                    "highpass=f=260",
                    "lowpass=f=5600",
                    "equalizer=f=900:t=q:w=1.1:g=1.5",
                    "equalizer=f=2800:t=q:w=1.2:g=-1.8",
                    *fade_filters,
                    "alimiter=limit=0.88",
                ]
            )
            + f"[{output}]",
        ]
    )


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
                "tremolo=f=5.1:d=0.025",
                r"volume='0.93+0.04*sin(17*t)':eval=frame",
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
        "pre-transmission receiver static",
        _marine_vhf_receiver_static_graph(
            prefix="pre",
            duration="0.65",
            hiss_seed=19930112,
            low_seed=19930111,
            crackle_gate_state=2,
            crackle_level_state=3,
            fade_filters=(
                "afade=t=in:st=0:d=0.015",
                "afade=t=out:st=0.56:d=0.09",
            ),
            output="pre_static",
        ),
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
        "post-transmission receiver static",
        _marine_vhf_receiver_static_graph(
            prefix="post",
            duration="0.85",
            hiss_seed=19930118,
            low_seed=19930119,
            crackle_gate_state=4,
            crackle_level_state=5,
            fade_filters=("afade=t=out:st=0.72:d=0.13",),
            output="post_static",
        ),
    ),
    _FilterStage(
        "final concatenation",
        ",".join(
            [
                "[pre_static][open][body][tail][post_static]concat=n=5:v=0:a=1",
                "aresample=24000",
                "aformat=channel_layouts=mono",
                "alimiter=limit=0.95",
                "volume=0.9",
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
    FmRadioProfile.name: FmRadioProfile(),
    GsmCodecProfile.name: GsmCodecProfile(),
    MarineVhf1993Profile.name: MarineVhf1993Profile(),
    PassthroughProfile.name: PassthroughProfile(),
}


def get_profile(name: str) -> Profile:
    return _PROFILES[name]


def list_profiles() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))


def list_profile_summaries() -> tuple[ProfileSummary, ...]:
    return tuple(
        ProfileSummary(name=name, description=_PROFILES[name].description)
        for name in sorted(_PROFILES)
    )
