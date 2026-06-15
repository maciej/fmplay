# Synthetic Radio Squelch Sound Library

## Purpose

`radio:squelch` is a reusable preview stage for short radio receiver artifacts.
It is not a full ATC profile. It generates synthetic squelch opens, closes, and
weak-signal chatter that can be layered around future radio/cockpit scenarios.

Run:

```sh
fmplay preview radio:squelch --duration 20 --seed 42
```

Omit `--seed` for a fresh randomized library pass.

## Reference Basis

The primary data target is
[`jlvdoorn/atco2-asr-atcosim`](https://huggingface.co/datasets/jlvdoorn/atco2-asr-atcosim),
an ASR dataset combining ATCO2-ASR and ATCOSIM with `audio`, `text`, and `info`
columns. ATCO2 material is relevant because the ATCO2 papers describe ATC speech
as VHF-radio voice communication, with the larger ATCO2 project collecting audio
from public radio-frequency channels using VHF receivers.

The circuit model is a receiver audio gate. A squelch circuit suppresses receiver
noise when no usable signal is present. Noise-operated squelch commonly measures
high-frequency discriminator noise above the speech band; when that noise falls,
the audio opens, and when it rises, the audio closes. The audible artifacts are
therefore best modeled as short receiver-noise bursts shaped by audio-gate timing,
not as speech, ambience, or codec noise.

## Synthetic Types

The stage uses four stochastic event families:

| Type | Modeled event | Audible shape |
| --- | --- | --- |
| `tail_crash` | Carrier drop and delayed squelch close | 115-310 ms bright hiss, crackle, low pop, decaying release |
| `opening_spit` | Receiver audio gate opens before stable speech | 45-145 ms narrow static and click |
| `threshold_chatter` | Weak signal near squelch threshold | Repeated fluttery bursts with strong envelope modulation |
| `carrier_snap` | Tight receiver gate or relay-like transition | 26-78 ms click/pop with little hiss |
| `thin_gate_flutter` | In-speech receiver gate/noise patch like ATCO2+ATCOSIM `train[6]` near 4.25 s | Low-level 16 kHz-compatible high-band noise, no tail crash or relay click |

The generator randomizes timing, duration, highpass/lowpass cutoffs, click/pop
frequencies, crackle density, tremolo rate, and fade shape. The output is mono
48 kHz PCM, because the artifact is receiver-local and normally enters later
scene composition as a mono radio element.

## Deterministic Thin-Gate Match

The `train[6]` reference in `jlvdoorn/atco2-asr-atcosim` contains a quieter
squelch-like patch around `4.15-4.45s`. It is not a classic end-of-transmission
tail. It is closer to thin receiver noise or gate flutter inside the utterance:
low RMS, high zero-crossing rate, and energy biased toward the upper
communications band.

Render a deliberate approximation:

```sh
uv run fmplay preview radio:squelch \
  --duration 7.744 \
  --seed 6 \
  --squelch-event thin_gate_flutter \
  --squelch-start 4.15 \
  --squelch-duration 0.30 \
  --squelch-level-db -40 \
  --squelch-highpass 1900 \
  --squelch-lowpass 7600 \
  --squelch-sample-rate 16000 \
  --output artifacts/squelch/train6/thin_gate_flutter.wav \
  --no-play
```

Compare it against the exact HF row:

```sh
uv run tools/squelch_match_train6.py \
  artifacts/squelch/train6/thin_gate_flutter.wav
```

Current tuning target for the full `4.15-4.45s` window is approximately
`-40 dBFS` RMS with ZCR near `0.48`, high-band energy around `4-7.6 kHz`, and
weak low/mid leakage. The isolated synthetic artifact intentionally does not
copy all low/mid energy in the real clip because that part contains surrounding
speech and channel residue, not just the gate flutter itself.

## Extraction Tool

Use the extraction helper to pull likely reference artifacts from the HF dataset.
It downloads row audio through the Hugging Face Dataset Viewer URLs, analyzes the
first and last edge windows, and stores high-scoring noise-like snippets.

```sh
uv run tools/squelch_extract.py \
  --rows 240 \
  --max-snippets 64 \
  --output-dir artifacts/squelch/references
```

The heuristic favors short edge events with high spectral flatness,
high-frequency energy, high zero-crossing rate, and plausible duration. It is
deliberately conservative about what gets used: snippets are references for
statistics and listening checks, not source material for the synthetic library.

## Validation Tool

The validation helper follows the same philosophy as the cockpit-audio research
notes: compare event-native features first, then inspect plots and audio.

```sh
uv run tools/squelch_validate.py \
  --refs-dir artifacts/squelch/references \
  --synthetic-dir artifacts/squelch/synthetic \
  --output-dir artifacts/squelch/validation \
  --generate 48
```

It writes:

- `summary.json`: Wasserstein distances and aggregate spectral/timing scores.
- `reference_features.jsonl` and `synthetic_features.jsonl`: event features.
- `feature_distributions.png`: distribution overlays.
- `spectrogram_examples.png`: quick visual sanity check.

Tracked metrics include active duration, RMS, crest factor, centroid,
high-band ratio, spectral flatness, zero-crossing rate, attack/release timing,
modulation peak, and octave-like band levels from 250 Hz to 8 kHz.

## Tuning Rules

Use reference snippets only to tune distributions:

- Match duration and release timing before tone color.
- Match high-band ratio and flatness before RMS.
- Keep clicks/pops sparse; real squelch tails are primarily noise bursts.
- Treat `threshold_chatter` as a minority case, not the default.
- Inspect spectrograms when scalar distances improve; low scalar error can still
  hide an obviously artificial click or steady noise bed.

## References

- [jlvdoorn/atco2-asr-atcosim](https://huggingface.co/datasets/jlvdoorn/atco2-asr-atcosim)
- [ATCO2 corpus paper](https://arxiv.org/abs/2211.04054)
- [Lessons Learned in ATCO2](https://arxiv.org/abs/2305.01155)
- [ATCO2 data description](https://www.atco2.org/data)
- [Bi-level/noise-operated squelch explanation](https://www.repeater-builder.com/micor/micor-bi-level-squelch-circuit.html)
- [Carrier and tone squelch patent discussing squelch tail](https://patents.google.com/patent/US3654555A/en)
- [Reverse Burst / Squelch Tail Elimination](https://www.repeater-builder.com/micor/reverseburst.html)
