# Close-Mic ATC Profile

## Purpose

`atc-close-mic` simulates clean speech spoken too close to an ATC-style
microphone and radio front end. It is intended to affect the voice signal only:
proximity coloration, plosive pressure, mouth/breath events, overload,
compression, and narrow ATC passband behavior are applied before any separate
scene bed, receiver noise, squelch, or background layer.

`marine-vhf-1993` reuses this as a voice front end with the
`atc-close-mic:abusive` preset, then sends the stressed voice through its
marine VHF chain.

## Report Findings

The initial research report split the reference problem into two acoustic
domains:

- `ATCO2-ASR`: receiver-captured real ATC radio, best treated as the target
  distribution for end-state radio sound.
- `ATCOSIM`: clean close-talk ATC simulation speech, useful as a close-talk and
  phraseology prior, but not as the final receiver-side acoustic target.

The report's key modeling recommendation was that the close-mic effect should
not be a static telephone EQ. The convincing cues are correlated, front-end
events:

- directional-mic proximity below roughly 1 kHz, especially lower-mid pressure
  and body;
- transient plosive airflow that drives later nonlinear stages;
- soft, slightly asymmetric mic/preamp overload with rare hard clipping;
- speech compression and limiting that react to hot syllables and pops;
- final ATC channel filtering that removes most true LF content while leaving
  compressed pressure residue and 1.5-3 kHz presence;
- randomized utterance-level technique states rather than independent random
  knobs.

The recommended ordering was:

```text
clean speech
-> active-speech normalization
-> close-mic geometry coloration
-> onset-conditioned plosive / breath / mouth-noise enhancement
-> mic / preamp saturation or partial overload
-> speech compression / AGC
-> ATC radio bandpass and mild channel coloration
-> optional low receiver noise / AM-ish grit
-> final peak safety limiter
-> output loudness match
```

This ordering is important because pops and proximity are acoustic/front-end
phenomena. They need to happen before saturation and compression so downstream
stages react to them. The radio passband belongs downstream; otherwise the
result tends to sound like post-filter clipping or a generic megaphone.

## Current Implementation

The implementation lives in `src/fmplay/close_mic.py`. The profile entry point
is `AtcCloseMicProfile` in `src/fmplay/profiles.py`.

Implemented primitives:

- Seeded correlated state: proximity, bad technique, drive, pop probability,
  pop gain, burst gain, breath/click probability, hard-clip probability,
  compressor ratio, channel low-pass, and presence gain are sampled together.
- Active-level staging: the offline processor normalizes active-ish speech
  energy before driving the close-mic stages.
- Proximity coloration: low and low-mid body are boosted before onset/event
  synthesis.
- Signal-only onset detection: candidate stop-like events are detected from
  short-time energy rises and local peaks, without transcript or phoneme input.
- Event synthesis: candidate onsets can receive low-frequency air pops,
  release bursts, breath noise, and sparse clicks.
- Event drive: selected onsets get short local gain and asymmetric partial hard
  clipping.
- Saturation and compression: the chain applies soft asymmetric saturation,
  speech compression, limiter stages, and RMS matching.
- ATC channel shaping: final high-pass, low-pass, low-mid reduction, and
  presence lift keep the output in a narrow communications band.
- Presets: `subtle`, `normal`, `hot`, and `abusive`.
- NumPy/SciPy path: continuous signal operations use arrays and `scipy.signal`
  filters for batch-oriented synthetic data generation.

The offline render path uses whole-clip processing:

```text
decode source with ffmpeg
-> process_array(...)
-> write 16 kHz mono WAV or raw s16le stream
```

The streaming path uses `CloseMicStreamProcessor`, which is a bounded-state
approximation:

```text
ffmpeg decode chunks
-> running input gain
-> persistent proximity/channel filters
-> local onset detection with recent context
-> event synthesis and overload
-> compressor / limiter state
-> raw 16 kHz mono s16le chunks
```

It does not attempt to be bit-identical to offline rendering. Offline has
whole-file statistics and full event visibility; streaming uses running levels,
local context, and persistent IIR/dynamics state.

## Presets

Use the CLI variants:

```sh
uv run fmplay --profile atc-close-mic:subtle audio.wav
uv run fmplay --profile atc-close-mic:normal audio.wav
uv run fmplay --profile atc-close-mic:hot audio.wav
uv run fmplay --profile atc-close-mic:abusive audio.wav
```

Or from Python:

```python
from pathlib import Path
from fmplay.profiles import AtcCloseMicProfile

AtcCloseMicProfile(seed=42, intensity="hot").render(
    Path("clean.wav"),
    Path("degraded.wav"),
)
```

`subtle` is restrained close-talk coloration. `normal` is the default hot-mic
behavior. `hot` increases proximity, event drive, and clipping clusters.
`abusive` intentionally models poor technique with dense limiting and stronger
event artifacts.

## Marine VHF Integration

`marine-vhf-1993` applies `atc-close-mic:abusive` to the input voice before the
marine radio chain.

Offline render:

```text
source
-> offline close-mic abusive temp WAV
-> ffmpeg marine VHF graph
-> output WAV
```

Streaming playback:

```text
source
-> Python close-mic streaming chunks
-> raw 16 kHz mono s16le over ffmpeg stdin pipe
-> ffmpeg marine VHF graph
-> raw 24 kHz mono s16le on stdout
```

The streaming path is implemented by `python -m fmplay.marine_vhf_stream` and
uses classic OS pipes. It no longer writes an intermediate close-mic WAV.

## Reference Collection

The validation reference set intentionally mixes:

- Hugging Face ATC rows from `jlvdoorn/atco2-asr-atcosim`.
- Short `yt-dlp` excerpts from public aviation/ATC communication videos.

Hugging Face rows are better for receiver-side ATC statistics. Video excerpts
are useful contrast material and can expose hot-mic examples that do not appear
in a small dataset slice.

Acquire references:

```sh
uv run tools/close_mic_collect.py \
  --output-dir artifacts/close-mic/references \
  --hf-rows 80 \
  --hf-page-size 20 \
  --hf-max-clips 24 \
  --yt-search "ATC radio communication pilot microphone" \
  --yt-search-results 3 \
  --yt-section-duration 35 \
  --yt-max-clips 4
```

The collector writes:

- `manifest.jsonl`: one row per selected reference clip.
- `summary.json`: selected counts and top-scoring candidates.
- `hf/*.wav` and `youtube/*.wav`: local reference audio.

The Hugging Face Dataset Viewer `/rows` endpoint can occasionally stall. The
collector uses short request timeouts and falls back to `/first-rows` when the
first paginated request fails.

## Validation Workflow

Run:

```sh
uv run tools/close_mic_validate.py \
  --refs-dir artifacts/close-mic/references \
  --degraded-dir artifacts/close-mic/degraded \
  --output-dir artifacts/close-mic/validation \
  --source /Users/maciej/code/fmplay/clips/audio/jan_heweliusz_abandon_ship.mp3 \
  --profile atc-close-mic \
  --slice-count 12
```

The validator renders each `--source` through `--profile`, compares generated
clips to the reference set, and writes:

- `summary.json`: distribution distances, medians, and failure diagnostics.
- `features.jsonl`: per-clip metrics.
- `slices.jsonl`: exported listening-slice metadata.
- `slices/*.wav`: short windows around strongest onset-pressure candidates.
- `feature_distributions.png`: reference-vs-generated histograms.

Important diagnostics:

- `audible_clipping_risk`: true when degraded clips lack both near-rail samples
  and flat/hot runs.
- `weak_plosive_pressure`: true when onset windows lack the low-frequency /
  low-mid pressure seen in references.
- `too_clean_dynamics`: true when crest factor remains much higher than the
  reference median.

Important metrics:

- `near_1db_pct`, `near_03db_pct`, `hot_cluster_count`, and
  `longest_flat_run_samples`: clipping or near-clipping behavior.
- `onset_pressure_p90`: 20-380 Hz onset energy against 300-3000 Hz speech
  energy in the first 60 ms after detected onsets.
- `onset_low_p90_db`: absolute 20-380 Hz onset energy, useful when the ratio is
  numerically unstable.
- `onset_near_1db_pct`: whether onsets are actually driving the front end.
- `lowmid_180_520_ratio`: close-talk body that can survive downstream radio
  filtering.

## Coverage Against The Report

Covered:

- Close-mic processing is a voice-only profile stage.
- The chain follows the recommended order: level staging, proximity, events,
  saturation, compression, channel filtering, limiting, output match.
- Plosive/breath/click synthesis is signal-only and transcript-free.
- Presets are seeded and randomized through correlated technique state.
- `marine-vhf-1993` uses `atc-close-mic:abusive` before adding radio/squelch
  behavior.
- Validation checks clipping, onset pressure, dynamics, and spectral shape
  rather than relying on one binary score.
- NumPy/SciPy are available for larger synthetic data generation.
- Streaming playback now composes Python DSP and ffmpeg through pipes.

Partially covered:

- Active speech level is approximated with short-time RMS heuristics. It is not
  ITU-T P.56.
- ATC passband and presence behavior are implemented, but channel variants are
  coarse presets rather than calibrated 8.33 kHz / 25 kHz channel families.
- The validator gathers Hugging Face and video references, but it does not yet
  condition target distributions by ATCO2 metadata such as airport, role, SNR,
  or channel context.
- Streaming close-mic processing has bounded local onset context, but not full
  lookahead or whole-file loudness matching.

Not covered yet:

- Internal oversampling around saturation/clipping. Nonlinear stages currently
  run at the profile sample rate, so aliasing is still possible.
- A true two-stage AGC model with separately tuned compressor and limiter
  acceptance bands.
- Objective intelligibility metrics such as STOI/ESTOI.
- ITU speech quality metrics such as P.863 or no-reference P.563.
- ATC ASR guardrails, e.g. Whisper-ATC WER/CER and callsign/command token
  stability.
- Fréchet Audio Distance or speech-embedding distribution matching.
- Subjective listening protocol such as a compact MUSHRA/P.800-style panel.
- Full subset-conditioned calibration from a chosen ATCO2 reference pool.

## Known Gaps And Next Steps

Highest-value next steps:

1. Add an ITU-T P.56 active speech level implementation or dependency and use
   it in both validation and offline rendering.
2. Add local oversampling for the saturator, event clipping, and limiter stages.
3. Extend the reference manifest with role/SNR/source-domain fields and compute
   separate target bands for ATCO2 receiver audio versus ATCOSIM clean headset
   audio.
4. Add an ASR intelligibility guardrail for ATC phraseology and callsign
   preservation.
5. Add a streaming/offline comparison report so users understand where the
   bounded streaming approximation differs perceptually from the whole-clip
   render.

Failure modes to watch:

- If output sounds like telephone EQ, the proximity/event/nonlinear layers are
  too weak or happening in the wrong order.
- If output sounds like a megaphone, reduce narrow upper-mid resonance and use
  broader 1.5-2.4 kHz presence instead.
- If every phrase sounds clipped, reduce hard-clip probability and keep true
  flat tops limited to hot events.
- If plosives are visible but not audible, increase onset-conditioned LF/low-mid
  energy before saturation and compression rather than boosting static bass.
