from __future__ import annotations

import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from fmplay.backends import AudioStream
from fmplay.profiles import ProfileError, ProfileInfo, ProfilePrimitive

GeneratedSource = Literal["silence", "white", "pink", "brown"]
SquelchEventKind = Literal[
    "tail_crash",
    "opening_spit",
    "threshold_chatter",
    "carrier_snap",
    "thin_gate_flutter",
]

SAMPLE_RATE = 48000
CHANNEL_LAYOUT = "stereo"
DEFAULT_PREVIEW_DURATION = 45.0
DEFAULT_SEED = 320_232
DEFAULT_SQUELCH_SAMPLE_RATE = SAMPLE_RATE
_SQUELCH_EVENT_KINDS = (
    "tail_crash",
    "opening_spit",
    "threshold_chatter",
    "carrier_snap",
    "thin_gate_flutter",
)


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


@dataclass(frozen=True)
class RadioSquelchStage:
    """Synthetic receiver squelch artifacts for VHF/ATC-style radio previews."""

    name: str = "radio:squelch"
    description: str = "Synthetic VHF receiver squelch clicks, tails, and chatter."
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
            channel_layout="mono",
        )

    def stream_custom(
        self,
        *,
        duration: float = DEFAULT_PREVIEW_DURATION,
        source: GeneratedSource = "silence",
        seed: int = DEFAULT_SEED,
        event_type: str,
        event_start: float,
        event_duration: float,
        event_level_db: float,
        event_highpass: int,
        event_lowpass: int,
        sample_rate: int = DEFAULT_SQUELCH_SAMPLE_RATE,
    ) -> AudioStream:
        return AudioStream(
            command=tuple(
                [
                    self.ffmpeg_command,
                    *self._render_args(
                        "pipe:1",
                        duration,
                        source,
                        seed,
                        event_type=event_type,
                        event_start=event_start,
                        event_duration=event_duration,
                        event_level_db=event_level_db,
                        event_highpass=event_highpass,
                        event_lowpass=event_lowpass,
                        sample_rate=sample_rate,
                    ),
                ]
            ),
            input_format="s16le",
            sample_rate=sample_rate,
            channel_layout="mono",
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

    def render_custom(
        self,
        output_path: Path,
        *,
        duration: float = DEFAULT_PREVIEW_DURATION,
        source: GeneratedSource = "silence",
        seed: int = DEFAULT_SEED,
        event_type: str,
        event_start: float,
        event_duration: float,
        event_level_db: float,
        event_highpass: int,
        event_lowpass: int,
        sample_rate: int = DEFAULT_SQUELCH_SAMPLE_RATE,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [
                    self.ffmpeg_command,
                    *self._render_args(
                        str(output_path),
                        duration,
                        source,
                        seed,
                        event_type=event_type,
                        event_start=event_start,
                        event_duration=event_duration,
                        event_level_db=event_level_db,
                        event_highpass=event_highpass,
                        event_lowpass=event_lowpass,
                        sample_rate=sample_rate,
                    ),
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
                    "tail crash",
                    "carrier-drop release burst with band-limited receiver hiss",
                ),
                ProfilePrimitive(
                    "opening spit",
                    "short gate-open static plus high click before program audio",
                ),
                ProfilePrimitive(
                    "threshold chatter",
                    "weak-signal squelch flutter split into repeated short bursts",
                ),
                ProfilePrimitive(
                    "carrier snap",
                    "low pop and relay-like click from the receiver audio gate",
                ),
                ProfilePrimitive(
                    "thin gate flutter",
                    "low-level high-band receiver noise inside a speech chunk",
                ),
            ),
        )

    def _render_args(
        self,
        output: str,
        duration: float,
        source: GeneratedSource,
        seed: int,
        *,
        event_type: str | None = None,
        event_start: float | None = None,
        event_duration: float | None = None,
        event_level_db: float | None = None,
        event_highpass: int | None = None,
        event_lowpass: int | None = None,
        sample_rate: int = DEFAULT_SQUELCH_SAMPLE_RATE,
    ) -> list[str]:
        if duration <= 0:
            raise ProfileError("--duration must be greater than 0")
        if source not in _SOURCE_KINDS:
            raise ProfileError(_unknown_source_message(source))
        _validate_squelch_sample_rate(sample_rate)

        args = [
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-filter_complex",
            _radio_squelch_filter_graph(
                duration=duration,
                source=source,
                seed=seed,
                explicit_event=_explicit_squelch_event(
                    event_type=event_type,
                    event_start=event_start,
                    event_duration=event_duration,
                    event_level_db=event_level_db,
                    event_highpass=event_highpass,
                    event_lowpass=event_lowpass,
                    seed=seed,
                    total_duration=duration,
                ),
            ),
            "-map",
            "[out]",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
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


@dataclass(frozen=True)
class _SquelchEvent:
    kind: str
    start: float
    duration: float
    hiss_level: float
    highpass: int
    lowpass: int
    click_freq: float
    click_level: float
    pop_freq: float
    pop_level: float
    crackle_probability: float
    tremolo_rate: float
    tremolo_depth: float
    fade_in: float
    fade_out: float
    hiss_weight: float = 0.86
    body_weight: float = 0.86
    crackle_weight: float = 0.20
    click_weight: float = 0.05
    pop_weight: float = 0.08
    body_level_factor: float = 0.62


def _radio_squelch_filter_graph(
    *,
    duration: float,
    source: GeneratedSource,
    seed: int,
    explicit_event: _SquelchEvent | None = None,
) -> str:
    rng = random.Random(seed)
    duration_text = f"{duration:.3f}"
    events = (
        [explicit_event]
        if explicit_event is not None
        else _squelch_events(duration=duration, rng=rng)
    )
    stages: list[str] = []
    mix_labels: list[str] = []
    thin_mode = (
        explicit_event is not None and explicit_event.kind == "thin_gate_flutter"
    )

    for index, event in enumerate(events):
        label = f"squelch_{index}"
        stages.append(_squelch_event_graph(index=index, event=event, output=label))
        mix_labels.append(label)

    if source != "silence":
        source_seed = rng.randrange(1, 2_147_483_647)
        stages.append(
            _radio_preview_source_filter_graph(
                duration=duration,
                source=source,
                seed=source_seed,
                level=0.010,
                output_label="preview_source",
            )
        )
        mix_labels.append("preview_source")

    stages.append(f"anullsrc=r={SAMPLE_RATE}:cl=mono:d={duration_text}[quiet]")
    mix_labels.append("quiet")

    stages.append(
        "".join(f"[{label}]" for label in mix_labels)
        + f"amix=inputs={len(mix_labels)}:duration=longest:normalize=0,"
        + ",".join(_radio_squelch_output_filters(thin_mode=thin_mode))
        + "[out]"
    )
    return ";".join(stages)


def _radio_squelch_output_filters(*, thin_mode: bool) -> list[str]:
    if thin_mode:
        return [
            "highpass=f=55",
            "lowpass=f=7800",
            "alimiter=limit=0.94",
            "aformat=sample_fmts=s16:channel_layouts=mono",
        ]

    return [
        "highpass=f=55",
        "lowpass=f=7600",
        "equalizer=f=2600:t=q:w=1.1:g=-0.8",
        "equalizer=f=6100:t=q:w=1.2:g=-2.4",
        "compand=attacks=0.001:decays=0.060:points=-90/-90|-42/-34|-18/-14|-5/-4|0/-2:soft-knee=2",
        "alimiter=limit=0.94",
        "volume=2.35",
        "aformat=sample_fmts=s16:channel_layouts=mono",
    ]


def _squelch_events(*, duration: float, rng: random.Random) -> list[_SquelchEvent]:
    if duration <= 0.18:
        return [_make_squelch_event("carrier_snap", 0.0, rng)]

    events: list[_SquelchEvent] = []
    cursor = rng.uniform(0.04, min(0.45, max(duration * 0.35, 0.05)))
    while cursor < duration - 0.05:
        kind = rng.choices(
            ("tail_crash", "opening_spit", "threshold_chatter", "carrier_snap"),
            weights=(0.46, 0.27, 0.17, 0.10),
            k=1,
        )[0]
        events.append(_make_squelch_event(kind, cursor, rng))
        cursor += rng.uniform(0.52, 2.25)

    if not events:
        kind = rng.choice(("tail_crash", "opening_spit", "carrier_snap"))
        events.append(_make_squelch_event(kind, max(0.0, duration * 0.18), rng))

    return [
        event for event in events if event.start + min(event.duration, 0.05) <= duration
    ]


def _make_squelch_event(kind: str, start: float, rng: random.Random) -> _SquelchEvent:
    if kind == "tail_crash":
        duration = rng.uniform(0.155, 0.380)
        return _SquelchEvent(
            kind=kind,
            start=start,
            duration=duration,
            hiss_level=rng.uniform(0.095, 0.180),
            highpass=rng.randrange(520, 1450),
            lowpass=rng.randrange(4600, 7800),
            click_freq=rng.uniform(420.0, 880.0),
            click_level=rng.uniform(0.34, 0.78),
            pop_freq=rng.uniform(75.0, 160.0),
            pop_level=rng.uniform(0.18, 0.42),
            crackle_probability=rng.uniform(0.00023, 0.00058),
            tremolo_rate=rng.uniform(6.0, 14.5),
            tremolo_depth=rng.uniform(0.025, 0.100),
            fade_in=rng.uniform(0.001, 0.006),
            fade_out=min(duration * rng.uniform(0.42, 0.70), 0.18),
        )
    if kind == "opening_spit":
        duration = rng.uniform(0.070, 0.180)
        return _SquelchEvent(
            kind=kind,
            start=start,
            duration=duration,
            hiss_level=rng.uniform(0.060, 0.125),
            highpass=rng.randrange(840, 2100),
            lowpass=rng.randrange(5000, 8300),
            click_freq=rng.uniform(900.0, 2150.0),
            click_level=rng.uniform(0.32, 0.82),
            pop_freq=rng.uniform(120.0, 260.0),
            pop_level=rng.uniform(0.05, 0.18),
            crackle_probability=rng.uniform(0.00018, 0.00046),
            tremolo_rate=rng.uniform(10.0, 22.0),
            tremolo_depth=rng.uniform(0.020, 0.085),
            fade_in=rng.uniform(0.0005, 0.004),
            fade_out=min(duration * rng.uniform(0.45, 0.75), 0.095),
        )
    if kind == "threshold_chatter":
        duration = rng.uniform(0.450, 1.050)
        return _SquelchEvent(
            kind=kind,
            start=start,
            duration=duration,
            hiss_level=rng.uniform(0.055, 0.125),
            highpass=rng.randrange(620, 1750),
            lowpass=rng.randrange(4300, 7600),
            click_freq=rng.uniform(650.0, 1500.0),
            click_level=rng.uniform(0.22, 0.60),
            pop_freq=rng.uniform(90.0, 190.0),
            pop_level=rng.uniform(0.05, 0.18),
            crackle_probability=rng.uniform(0.00030, 0.00076),
            tremolo_rate=rng.uniform(12.0, 24.0),
            tremolo_depth=rng.uniform(0.35, 0.65),
            fade_in=rng.uniform(0.002, 0.010),
            fade_out=min(duration * rng.uniform(0.22, 0.40), 0.17),
        )
    if kind == "thin_gate_flutter":
        return _manual_squelch_event(
            kind="thin_gate_flutter",
            start=start,
            duration=0.30,
            level_db=-40.0,
            highpass=1900,
            lowpass=7600,
            rng=rng,
        )

    duration = rng.uniform(0.040, 0.095)
    return _SquelchEvent(
        kind="carrier_snap",
        start=start,
        duration=duration,
        hiss_level=rng.uniform(0.020, 0.055),
        highpass=rng.randrange(1000, 2600),
        lowpass=rng.randrange(4400, 7400),
        click_freq=rng.uniform(580.0, 1800.0),
        click_level=rng.uniform(0.36, 0.98),
        pop_freq=rng.uniform(70.0, 180.0),
        pop_level=rng.uniform(0.12, 0.38),
        crackle_probability=rng.uniform(0.00008, 0.00024),
        tremolo_rate=rng.uniform(8.0, 18.0),
        tremolo_depth=rng.uniform(0.010, 0.045),
        fade_in=rng.uniform(0.0005, 0.003),
        fade_out=min(duration * rng.uniform(0.38, 0.70), 0.052),
    )


def _explicit_squelch_event(
    *,
    event_type: str | None,
    event_start: float | None,
    event_duration: float | None,
    event_level_db: float | None,
    event_highpass: int | None,
    event_lowpass: int | None,
    seed: int,
    total_duration: float,
) -> _SquelchEvent | None:
    if event_type is None:
        return None
    if event_type not in _SQUELCH_EVENT_KINDS:
        raise ProfileError(
            f"unknown squelch event '{event_type}'. "
            f"Available squelch events: {', '.join(_SQUELCH_EVENT_KINDS)}"
        )

    start = 0.0 if event_start is None else event_start
    duration = 0.30 if event_duration is None else event_duration
    level_db = -40.0 if event_level_db is None else event_level_db
    highpass = 1900 if event_highpass is None else event_highpass
    lowpass = 7600 if event_lowpass is None else event_lowpass

    if start < 0:
        raise ProfileError("--squelch-start must be greater than or equal to 0")
    if duration <= 0:
        raise ProfileError("--squelch-duration must be greater than 0")
    if start + duration > total_duration:
        raise ProfileError("--squelch-start + --squelch-duration exceeds --duration")
    if not -90.0 <= level_db <= -3.0:
        raise ProfileError("--squelch-level-db must be between -90 and -3")
    if highpass < 20:
        raise ProfileError("--squelch-highpass must be at least 20 Hz")
    if lowpass <= highpass:
        raise ProfileError("--squelch-lowpass must be greater than --squelch-highpass")
    if lowpass > 20_000:
        raise ProfileError("--squelch-lowpass must be at most 20000 Hz")

    rng = random.Random(seed)
    return _manual_squelch_event(
        kind=event_type,
        start=start,
        duration=duration,
        level_db=level_db,
        highpass=highpass,
        lowpass=lowpass,
        rng=rng,
    )


def _manual_squelch_event(
    *,
    kind: str,
    start: float,
    duration: float,
    level_db: float,
    highpass: int,
    lowpass: int,
    rng: random.Random,
) -> _SquelchEvent:
    if kind == "thin_gate_flutter":
        hiss_level = _thin_gate_hiss_level(level_db, highpass=highpass, lowpass=lowpass)
        return _SquelchEvent(
            kind=kind,
            start=start,
            duration=duration,
            hiss_level=hiss_level,
            highpass=highpass,
            lowpass=lowpass,
            click_freq=1000.0,
            click_level=0.0,
            pop_freq=120.0,
            pop_level=0.0,
            crackle_probability=0.00010,
            tremolo_rate=rng.uniform(15.0, 26.0),
            tremolo_depth=rng.uniform(0.10, 0.20),
            fade_in=0.020,
            fade_out=0.045,
            hiss_weight=1.0,
            body_weight=0.60,
            crackle_weight=0.05,
            click_weight=0.0,
            pop_weight=0.0,
            body_level_factor=0.35,
        )

    event = _make_squelch_event(kind, start, rng)
    return _SquelchEvent(
        kind=event.kind,
        start=start,
        duration=duration,
        hiss_level=max(event.hiss_level * 0.6, _db_to_amplitude(level_db) * 1.8),
        highpass=highpass,
        lowpass=lowpass,
        click_freq=event.click_freq,
        click_level=event.click_level,
        pop_freq=event.pop_freq,
        pop_level=event.pop_level,
        crackle_probability=event.crackle_probability,
        tremolo_rate=event.tremolo_rate,
        tremolo_depth=event.tremolo_depth,
        fade_in=event.fade_in,
        fade_out=min(event.fade_out, duration * 0.7),
        hiss_weight=event.hiss_weight,
        body_weight=event.body_weight,
        crackle_weight=event.crackle_weight,
        click_weight=event.click_weight,
        pop_weight=event.pop_weight,
        body_level_factor=event.body_level_factor,
    )


def _thin_gate_hiss_level(level_db: float, *, highpass: int, lowpass: int) -> float:
    bandwidth_fraction = max(0.08, (lowpass - highpass) / (SAMPLE_RATE / 2))
    return _db_to_amplitude(level_db) * math.sqrt(3.0 / bandwidth_fraction) * 1.40


def _db_to_amplitude(level_db: float) -> float:
    return 10 ** (level_db / 20.0)


def _validate_squelch_sample_rate(sample_rate: int) -> None:
    if sample_rate not in {8000, 12000, 16000, 24000, 48000}:
        raise ProfileError(
            "--squelch-sample-rate must be one of 8000, 12000, 16000, 24000, 48000"
        )


def _squelch_event_graph(index: int, event: _SquelchEvent, output: str) -> str:
    start_ms = max(0, int(round(event.start * 1000)))
    duration = f"{event.duration:.4f}"
    fade_out_start = max(0.0, event.duration - event.fade_out)
    seeds = [index * 8 + offset + 11 for offset in range(4)]
    envelope = (
        f"{1.0 - event.tremolo_depth:.4f}+"
        f"{event.tremolo_depth:.4f}*sin(2*PI*{event.tremolo_rate:.4f}*t)"
    )

    parts = [
        ",".join(
            [
                f"anoisesrc=r={SAMPLE_RATE}:a={event.hiss_level:.5f}:c=white:d={duration}:s={seeds[0]}",
                f"highpass=f={event.highpass}",
                f"lowpass=f={event.lowpass}",
                "equalizer=f=2200:t=q:w=1.3:g=0.3",
                "equalizer=f=5600:t=q:w=1.4:g=-1.8",
                _volume_expr(envelope),
                f"afade=t=in:st=0:d={event.fade_in:.4f}",
                f"afade=t=out:st={fade_out_start:.4f}:d={event.fade_out:.4f}",
                f"adelay={start_ms}:all=1",
            ]
        )
        + f"[sq_{index}_hiss]",
        ",".join(
            [
                _noise_source(
                    level=event.hiss_level * event.body_level_factor,
                    color="pink",
                    duration=duration,
                    seed=seeds[3],
                ),
                "highpass=f=110",
                "lowpass=f=1700",
                "equalizer=f=360:t=q:w=1.0:g=3.4",
                "equalizer=f=820:t=q:w=1.2:g=1.2",
                f"afade=t=in:st=0:d={event.fade_in:.4f}",
                f"afade=t=out:st={fade_out_start:.4f}:d={event.fade_out:.4f}",
                f"adelay={start_ms}:all=1",
            ]
        )
        + f"[sq_{index}_body]",
        ",".join(
            [
                (
                    "aevalsrc='if(lt(random("
                    f"{seeds[1]}"
                    rf")\,{event.crackle_probability:.8f})\,(random("
                    f"{seeds[2]}"
                    r")*2-1)\,0)':"
                    f"s={SAMPLE_RATE}:d={duration}"
                ),
                "highpass=f=2300",
                "lowpass=f=9800",
                "volume=0.58",
                f"adelay={start_ms}:all=1",
            ]
        )
        + f"[sq_{index}_crackle]",
        ",".join(
            [
                f"sine=f={event.click_freq:.2f}:r={SAMPLE_RATE}:d=0.014",
                "afade=t=out:st=0:d=0.014",
                f"volume={event.click_level:.4f}",
                "highpass=f=260",
                "lowpass=f=6200",
                f"adelay={start_ms}:all=1",
            ]
        )
        + f"[sq_{index}_click]",
        ",".join(
            [
                f"sine=f={event.pop_freq:.2f}:r={SAMPLE_RATE}:d=0.042",
                "afade=t=out:st=0:d=0.042",
                f"volume={event.pop_level:.4f}",
                "lowpass=f=420",
                f"adelay={start_ms}:all=1",
            ]
        )
        + f"[sq_{index}_pop]",
        "".join(
            f"[sq_{index}_{label}]"
            for label in ("hiss", "body", "crackle", "click", "pop")
        )
        + (
            "amix=inputs=5:duration=longest:"
            f"weights='{event.hiss_weight:.3f} {event.body_weight:.3f} "
            f"{event.crackle_weight:.3f} {event.click_weight:.3f} "
            f"{event.pop_weight:.3f}':normalize=0,"
        )
        + "alimiter=limit=0.92"
        + f"[{output}]",
    ]
    return ";".join(parts)


def _noise_source(*, level: float, color: str, duration: str, seed: int) -> str:
    level = max(level, 0.000001)
    return f"anoisesrc=r={SAMPLE_RATE}:a={level:.5f}:c={color}:d={duration}:s={seed}"


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


def _radio_preview_source_filter_graph(
    *,
    duration: float,
    source: GeneratedSource,
    seed: int,
    level: float,
    output_label: str,
) -> str:
    duration_text = f"{duration:.3f}"
    if source == "silence":
        source_graph = f"anullsrc=r={SAMPLE_RATE}:cl=mono:d={duration_text}"
    else:
        source_graph = (
            f"anoisesrc=r={SAMPLE_RATE}:a={level:.4f}:c={source}:"
            f"d={duration_text}:s={seed}"
        )

    return (
        ",".join(
            [
                source_graph,
                "aformat=channel_layouts=mono",
                "highpass=f=120",
                "lowpass=f=6200",
                "volume=0.55",
                "alimiter=limit=0.80",
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
    RadioSquelchStage.name: RadioSquelchStage(),
}
