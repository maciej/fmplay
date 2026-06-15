# Cockpit Audio Simulation

## Purpose

`cockpit:a320` is a reusable profile stage, not a normal degradation profile.
It synthesizes Airbus A320 cockpit ambience so later profiles and scenarios can
compose around it instead of treating cockpit noise as a one-off effect.

The stage is currently surfaced through:

```sh
fmplay preview cockpit:a320
```

It intentionally does not appear in the regular `fmplay profiles` list.

## Current A320 Cruise/Generic Cockpit Model

The current model is layered procedural audio. It approximates a steady
in-flight A320 cockpit bed using these components:

| Real component | Synthetic primitive | ffmpeg technique | Knobs to tune |
| --- | --- | --- | --- |
| ECS / packs | Low cockpit pressure roar | Brown noise, lowpass filtering, low-frequency EQ, slow volume modulation | Noise level, lowpass cutoff, 80-200 Hz EQ, modulation depth |
| Avionics and cockpit ventilation | Mid-band fan texture | White/pink noise, bandpass filtering, cockpit-colored EQ | Fan level, 500-3000 Hz EQ, stereo placement |
| Windshield and side-window airflow | Boundary-layer airflow bed | Pink/white noise, highpass/lowpass shaping, slow pressure modulation | Airflow level, upper cutoff, high-frequency darkness, modulation rate |
| Distant engines | Tonal leakage and low fuselage rumble | Detuned sine banks, lowpass filtering, subtle echo, brown-noise rumble | Tonal base frequency, detune amount, harmonic levels, rumble level |
| Panels, relays, fittings | Sparse cockpit flecks | Seeded `aevalsrc` impulses, filtering, short echo | Event probability, event level, high/low cutoffs, stereo placement |
| Generated preview source | Optional source bed | `anullsrc` or `anoisesrc` | `--source silence\|white\|pink\|brown`, source level, seed |

The stage supports seeded randomness. Passing `--seed` repeats the same tonal
choices, noise seeds, and sparse event texture. Omitting `--seed` gives a fresh
variant.

## Why This Approach

The stage avoids sampled ambience loops. It uses streaming ffmpeg sources and
filters so the preview can start immediately, stay cheap to run, and remain
deterministic when a seed is supplied.

This also keeps the model easy to morph later. Taxi, takeoff, climb, descent,
approach, and landing can become parameter sets over the same core primitives
rather than separate loop libraries.

## Tuning Notes

Public cockpit videos are useful for color, density, rough event timing, and
relative source balance. They are weak references for absolute level because
they often include speech, ATC, camera AGC, clipping, lossy platform
compression, and microphone placement artifacts.

For the current stage, public A320 cockpit videos were treated as secondary
references. Their speech-heavy vertical spectrogram bands and camera clipping
were not copied into the ambience model. The useful target was the underlying
bed: steady low pressure, mid fan/vent texture, restrained high airflow, distant
engine leakage, and occasional cockpit mechanical flecks.

## Future Scenario Parameters

| Scenario | Expected changes |
| --- | --- |
| `taxi` | More packs/APU presence, ground rumble, brake/floor rattles, low engine idle |
| `takeoff` | Rising engine tones, stronger low-frequency body, runway vibration, acceleration-linked airflow |
| `climb` | High engine load, airflow increasing, fewer ground transients |
| `cruise` | Steady ECS, wind, fan texture, and distant engine leakage |
| `descent` | Steady airflow, lower engine contribution, occasional configuration events |
| `approach` | Flap/gear airflow, changing wind color, optional alerts and cockpit clicks |
| `landing` | Touchdown impulse, runway roll, spoilers, reversers, brake rumble |

Future implementations should prefer explicit parameters for phase, speed,
engine load, configuration, and weather over hard-coded one-off graphs.

## References To Keep

### Primary / Engineering

- [Airbus A320 Aircraft Characteristics](https://www.aircraft.airbus.com/sites/g/files/jlcbta126/files/2025-01/AC_A320_0624.pdf)
- [DLR Hu et al., Contributions of Different Aeroacoustic Sources to Aircraft Cabin Noise](https://elib.dlr.de/82852/1/AIAA2013_Hu.pdf)
- [DLR Klabes et al., Measurement and Prediction of Pressure Point Spectra](https://elib.dlr.de/97218/1/Klabes.AIAA.2015.pdf)
- [EASA.A.064 Airbus A318/A319/A320/A321](https://www.easa.europa.eu/en/document-library/type-certificates/noise/easaa064-airbus-a318-a319-a320-a321-single-aisle)
- [CFM56 overview, CFM International](https://www.cfmaeroengines.com/cfm56)
- [IAE / V2500, Pratt & Whitney](https://links.prattwhitney.com/i-a-e/index.html)

### Secondary Audio References

- [A320 cockpit Prague to Dusseldorf](https://www.youtube.com/watch?v=8GwEzS78e4E)
- [A320 cockpit Warsaw RWY 33 landing](https://www.youtube.com/watch?v=dirFZaZ-ZbM&vl=en)

