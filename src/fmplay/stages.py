from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from fmplay.backends import AudioStream
from fmplay.profiles import ProfileError, ProfileInfo, ProfilePrimitive

GeneratedSource = Literal["silence", "white", "pink", "brown"]

SAMPLE_RATE = 48000
CHANNEL_LAYOUT = "stereo"
DEFAULT_PREVIEW_DURATION = 45.0
DEFAULT_SEED = 320_232


class ProfileStage(Protocol):
    """A reusable profile stage addressable outside the profile list."""

    name: str
    description: str

    def stream(
        self,
        *,
        duration: float = DEFAULT_PREVIEW_DURATION,
        source: GeneratedSource = "silence",
        seed: int = DEFAULT_SEED,
    ) -> AudioStream:
        """Return a playable preview stream for this stage."""

    def render(
        self,
        output_path: Path,
        *,
        duration: float = DEFAULT_PREVIEW_DURATION,
        source: GeneratedSource = "silence",
        seed: int = DEFAULT_SEED,
    ) -> None:
        """Render a finite preview for non-streaming playback backends."""


@dataclass(frozen=True)
class A320CockpitStage:
    """Synthetic ambient cockpit bed for an Airbus A320 in flight."""

    name: str = "cockpit:a320"
    description: str = "Airbus A320 cockpit ambient noise profile stage."
    ffmpeg_command: str = "ffmpeg"

    def stream(
        self,
        *,
        duration: float = DEFAULT_PREVIEW_DURATION,
        source: GeneratedSource = "silence",
        seed: int = DEFAULT_SEED,
    ) -> AudioStream:
        return AudioStream(
            command=tuple(
                [
                    self.ffmpeg_command,
                    *self._render_args("pipe:1", duration, source, seed),
                ]
            ),
            input_format="s16le",
            sample_rate=SAMPLE_RATE,
            channel_layout=CHANNEL_LAYOUT,
        )

    def render(
        self,
        output_path: Path,
        *,
        duration: float = DEFAULT_PREVIEW_DURATION,
        source: GeneratedSource = "silence",
        seed: int = DEFAULT_SEED,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [
                    self.ffmpeg_command,
                    *self._render_args(str(output_path), duration, source, seed),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise ProfileError(
                f"{self.ffmpeg_command} was not found. Install ffmpeg to preview "
                f"the '{self.name}' stage."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise ProfileError(
                f"{self.ffmpeg_command} failed while rendering {self.name}"
                f"{_format_subprocess_details(exc)}"
            ) from exc

    def profile_info(self) -> ProfileInfo:
        return ProfileInfo(
            name=self.name,
            description=self.description,
            primitives=(
                ProfilePrimitive(
                    "ECS and avionics bed",
                    "low-frequency pack roar, fans, and cockpit ventilation",
                ),
                ProfilePrimitive(
                    "windshield boundary layer",
                    "broadband side-window flow with slow pressure modulation",
                ),
                ProfilePrimitive(
                    "engine leakage",
                    "detuned CFM56/V2500-like tonal bed and low fuselage rumble",
                ),
                ProfilePrimitive(
                    "cockpit texture",
                    "seeded relay ticks, panel creaks, and trim-like flecks",
                ),
            ),
        )

    def _render_args(
        self, output: str, duration: float, source: GeneratedSource, seed: int
    ) -> list[str]:
        if duration <= 0:
            raise ProfileError("--duration must be greater than 0")
        if source not in _SOURCE_KINDS:
            raise ProfileError(_unknown_source_message(source))

        args = [
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-filter_complex",
            _a320_cockpit_filter_graph(duration=duration, source=source, seed=seed),
            "-map",
            "[out]",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
        ]
        if output == "pipe:1":
            args.extend(["-f", "s16le"])
        args.append(output)
        return args


def render_generated_source(
    output_path: Path,
    *,
    duration: float = DEFAULT_PREVIEW_DURATION,
    source: GeneratedSource = "silence",
    seed: int = DEFAULT_SEED,
    ffmpeg_command: str = "ffmpeg",
) -> None:
    if duration <= 0:
        raise ProfileError("--duration must be greater than 0")
    if source not in _SOURCE_KINDS:
        raise ProfileError(_unknown_source_message(source))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = _generated_source_filter_graph(
        duration=duration,
        source=source,
        seed=seed,
        level=0.09 if source == "silence" else 0.16,
    )
    try:
        subprocess.run(
            [
                ffmpeg_command,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-filter_complex",
                filter_graph,
                "-map",
                "[out]",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ProfileError(
            f"{ffmpeg_command} was not found. Install ffmpeg to preview profiles."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ProfileError(
            f"{ffmpeg_command} failed while rendering preview source"
            f"{_format_subprocess_details(exc)}"
        ) from exc


def get_stage(name: str) -> ProfileStage:
    return _STAGES[name]


def list_stages() -> tuple[str, ...]:
    return tuple(sorted(_STAGES))


def is_generated_source(value: str) -> bool:
    return value in _SOURCE_KINDS


def _a320_cockpit_filter_graph(
    *, duration: float, source: GeneratedSource, seed: int
) -> str:
    rng = random.Random(seed)
    duration_text = f"{duration:.3f}"
    seeds = [rng.randrange(1, 2_147_483_647) for _ in range(12)]
    engine_base = rng.uniform(116.0, 123.5)
    engine_split = rng.uniform(1.6, 3.1)
    whine_base = rng.uniform(470.0, 535.0)
    slow_mod = rng.uniform(0.13, 0.23)

    stages = [
        ",".join(
            [
                f"anoisesrc=r={SAMPLE_RATE}:a=0.052:c=brown:d={duration_text}:s={seeds[0]}",
                "highpass=f=24",
                "lowpass=f=620",
                "equalizer=f=82:t=q:w=0.8:g=3.5",
                "equalizer=f=178:t=q:w=1.0:g=2.2",
                "equalizer=f=360:t=q:w=1.2:g=-1.4",
                _volume_expr(f"0.86+0.035*sin(2*PI*{slow_mod:.4f}*t)"),
                "pan=stereo|c0=0.96*c0|c1=0.91*c0",
            ]
        )
        + "[packs]",
        ",".join(
            [
                f"anoisesrc=r={SAMPLE_RATE}:a=0.031:c=pink:d={duration_text}:s={seeds[1]}",
                "highpass=f=145",
                "lowpass=f=3300",
                "equalizer=f=710:t=q:w=0.9:g=2.6",
                "equalizer=f=1750:t=q:w=1.3:g=2.0",
                _volume_expr("0.78+0.050*sin(2*PI*0.071*t)+0.020*sin(2*PI*0.37*t)"),
                "pan=stereo|c0=0.82*c0|c1=1.00*c0",
            ]
        )
        + "[wind_body]",
        ",".join(
            [
                f"anoisesrc=r={SAMPLE_RATE}:a=0.010:c=white:d={duration_text}:s={seeds[2]}",
                "highpass=f=2400",
                "lowpass=f=9600",
                "equalizer=f=5100:t=q:w=1.5:g=-2.0",
                _volume_expr("0.50+0.030*sin(2*PI*0.19*t)"),
                "pan=stereo|c0=0.74*c0|c1=0.88*c0",
            ]
        )
        + "[wind_hiss]",
        ",".join(
            [
                f"anoisesrc=r={SAMPLE_RATE}:a=0.012:c=white:d={duration_text}:s={seeds[3]}",
                "highpass=f=520",
                "lowpass=f=4600",
                "equalizer=f=1150:t=q:w=1.0:g=3.2",
                "equalizer=f=2850:t=q:w=1.2:g=-2.2",
                _volume_expr("0.62+0.018*sin(2*PI*1.7*t)"),
                "pan=stereo|c0=0.93*c0|c1=0.78*c0",
            ]
        )
        + "[fans]",
        _engine_tone_graph(
            base=engine_base,
            split=engine_split,
            whine=whine_base,
            duration=duration_text,
        ),
        ",".join(
            [
                f"anoisesrc=r={SAMPLE_RATE}:a=0.022:c=brown:d={duration_text}:s={seeds[4]}",
                "highpass=f=32",
                "lowpass=f=260",
                "equalizer=f=118:t=q:w=1.1:g=2.5",
                _volume_expr("0.55+0.025*sin(2*PI*0.29*t)"),
                "pan=stereo|c0=0.88*c0|c1=0.92*c0",
            ]
        )
        + "[fuselage]",
        _event_graph(
            label="ticks",
            duration=duration_text,
            gate_state=seeds[5],
            level_state=seeds[6],
            probability=0.0000048,
            highpass=750,
            lowpass=9000,
            volume=0.30,
            left=0.88,
            right=0.72,
        ),
        _event_graph(
            label="creaks",
            duration=duration_text,
            gate_state=seeds[7],
            level_state=seeds[8],
            probability=0.0000022,
            highpass=65,
            lowpass=650,
            volume=0.18,
            left=0.72,
            right=0.86,
        ),
    ]

    mix_labels = [
        "packs",
        "wind_body",
        "wind_hiss",
        "fans",
        "engines",
        "fuselage",
        "ticks",
        "creaks",
    ]
    if source != "silence":
        stages.append(
            _generated_source_filter_graph(
                duration=duration,
                source=source,
                seed=seeds[9],
                level=0.035,
                output_label="preview_source",
            )
        )
        mix_labels.append("preview_source")

    stages.append(
        "".join(f"[{label}]" for label in mix_labels)
        + f"amix=inputs={len(mix_labels)}:duration=longest:normalize=0,"
        + ",".join(
            [
                "highpass=f=20",
                "lowpass=f=16000",
                "compand=attacks=0.030:decays=0.35:points=-90/-90|-42/-38|-20/-18|-7/-7|0/-2:soft-knee=4",
                "alimiter=limit=0.91",
                "volume=2.8",
                "aformat=sample_fmts=s16:channel_layouts=stereo",
            ]
        )
        + "[out]"
    )
    return ";".join(stages)


def _engine_tone_graph(
    *, base: float, split: float, whine: float, duration: str
) -> str:
    freqs = (
        base,
        base + split,
        base * 2.0,
        (base + split) * 2.0,
        whine,
        whine + 34.0,
        whine * 1.52,
        whine * 2.03,
    )
    levels = (0.0086, 0.0078, 0.0046, 0.0040, 0.0058, 0.0048, 0.0030, 0.0022)
    parts = []
    labels = []
    for index, (freq, level) in enumerate(zip(freqs, levels, strict=True)):
        label = f"engine_tone_{index}"
        labels.append(label)
        parts.append(
            ",".join(
                [
                    f"sine=f={freq:.3f}:r={SAMPLE_RATE}:d={duration}",
                    _volume_expr(
                        f"{level:.6f}*(1+0.060*sin(2*PI*{0.21 + index * 0.017:.4f}*t))"
                    ),
                ]
            )
            + f"[{label}]"
        )

    parts.append(
        "".join(f"[{label}]" for label in labels)
        + f"amix=inputs={len(labels)}:duration=longest:normalize=0,"
        + ",".join(
            [
                "lowpass=f=2500",
                "aecho=0.18:0.20:21|37:0.10|0.07",
                "pan=stereo|c0=0.84*c0|c1=0.91*c0",
            ]
        )
        + "[engines]"
    )
    return ";".join(parts)


def _event_graph(
    *,
    label: str,
    duration: str,
    gate_state: int,
    level_state: int,
    probability: float,
    highpass: int,
    lowpass: int,
    volume: float,
    left: float,
    right: float,
) -> str:
    return (
        ",".join(
            [
                (
                    "aevalsrc='if(lt(random("
                    f"{gate_state}"
                    rf")\,{probability:.8f})\,(random("
                    f"{level_state}"
                    r")*2-1)\,0)':"
                    f"s={SAMPLE_RATE}:d={duration}"
                ),
                f"highpass=f={highpass}",
                f"lowpass=f={lowpass}",
                f"volume={volume:.4f}",
                "aecho=0.35:0.26:8|23:0.28|0.16",
                f"pan=stereo|c0={left:.3f}*c0|c1={right:.3f}*c0",
            ]
        )
        + f"[{label}]"
    )


def _generated_source_filter_graph(
    *,
    duration: float,
    source: GeneratedSource,
    seed: int,
    level: float,
    output_label: str = "out",
) -> str:
    duration_text = f"{duration:.3f}"
    if source == "silence":
        source_graph = f"anullsrc=r={SAMPLE_RATE}:cl={CHANNEL_LAYOUT}:d={duration_text}"
    else:
        source_graph = (
            f"anoisesrc=r={SAMPLE_RATE}:a={level:.4f}:c={source}:"
            f"d={duration_text}:s={seed}"
        )

    return (
        ",".join(
            [
                source_graph,
                "aformat=channel_layouts=stereo",
                "highpass=f=20",
                "lowpass=f=18000",
                "alimiter=limit=0.90",
            ]
        )
        + f"[{output_label}]"
    )


def _volume_expr(expr: str) -> str:
    return f"volume='{expr}':eval=frame"


def _unknown_source_message(source: str) -> str:
    return (
        f"unknown generated source '{source}'. "
        f"Available sources: {', '.join(sorted(_SOURCE_KINDS))}"
    )


def _format_subprocess_details(exc: subprocess.CalledProcessError) -> str:
    details = (exc.stderr or exc.stdout or "").strip()
    if not details:
        return f": exit code {exc.returncode}"
    return f": {details.splitlines()[-1]}"


_SOURCE_KINDS = frozenset({"silence", "white", "pink", "brown"})

_STAGES: dict[str, ProfileStage] = {
    A320CockpitStage.name: A320CockpitStage(),
}
