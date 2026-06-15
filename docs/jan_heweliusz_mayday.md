# Jan Heweliusz Mayday Source Notes

## Purpose

This note records the public source trail for the MF Jan Heweliusz distress
audio and separates historical radio/logging artifacts from later distribution
damage. Use it to revise the source TTS recipe and to tune a more faithful
marine VHF/logging-chain profile.

## Source Material Location

Primary public source:

- [Jan Heweliusz - wzywanie pomocy (część pierwsza)](https://www.youtube.com/watch?v=NBizVLd5cL4)

The uploader describes this as a recording made at Rügen Radio on 14 January
1993. They also state that the presented version was shortened by removing
several minutes of silence between messages.

Practical listening window for the opening distress call in part 1:

- `~1:55`: "MAYDAY, MAYDAY, all ships..."
- `~2:00-2:05`: "Polish ferry Jan Heweliusz..."
- `~2:15-2:16`: "...I give you my position shortly"
- `~2:16 onward`: Rønne Radio replies.

The uploader's hand transcript for 04:37 has the opening call as a repeated
Mayday to all ships from "Polish ferry Jan Heweliusz", reporting a heavy list
to portside, an unclear phrase around "in... position", then "I give you my
position shortly." It then has Rønne Radio asking for position and Jan
Heweliusz replying that it is not far from Kollicker and will give distance and
bearing shortly.

That hand transcript is more useful than noisy automatic speech recognition for
the opening call. ASR on the same fragments was unreliable and repeatedly
hallucinated, especially over the faint Mayday audio.

Secondary public source:

- [Jan Heweliusz - wzywanie pomocy (część druga)](https://www.youtube.com/watch?v=ed3No42pdi4)

Part 2 appears to be later distress and coordination traffic, including the
Frank Michael material. It is not the opening Mayday call, but it is useful for
the broader event soundscape and later traffic references.

The current recipe text in
[`clips/recipes/jan_heweliusz_mayday.py`](/Users/maciej/.codex/worktrees/ced3/fmplay/clips/recipes/jan_heweliusz_mayday.py)
appears to blend or compress the initial Mayday with later position traffic,
especially "Sixteen miles west from Arkona." That Arkona position exchange
appears later in the Rønne Radio dialogue, not in the opening distress call.

## Historical Context

MF Jan Heweliusz was a Polish roll-on/roll-off railway and vehicle ferry. It
sank in the Baltic Sea near Rügen on 14 January 1993 after developing a severe
list during extreme weather. Public summaries describe this as the largest
peacetime maritime disaster in Polish history.

The distress traffic involved the Jan Heweliusz and nearby/coastal radio
participants including Mikołaj Kopernik, Rønne Radio, Arkona/Rügen Radio,
Silesia, Nieborów, and later references to Frank Michael. The public transcript
trail shows early Polish traffic with Mikołaj Kopernik, the 04:37 English
Mayday call, Rønne Radio's position requests, and later coordination.

Useful context links:

- [DlaPilota account with 04:37 Mayday transcript and later traffic](https://dlapilota.pl/wiadomosci/morskie-skrzydla-darlowa/nie-dane-nam-bylo-ratowac-heweliusza-relacja-z-pokladu-mi-14pl)
- [Onet excerpt with Rønne Radio position/passenger exchange](https://podroze.onet.pl/ciekawe/mayday-mayday-jan-heweliusz-pasazerowie-skacza-do-lodowatego-baltyku/l94lygd)
- [Polish Wikipedia overview and bibliography](https://pl.wikipedia.org/wiki/Katastrofa_promu_Jan_Heweliusz)

Polish Radio's recent podcast is a secondary/current archival-publication lead:

- [Heweliusz. Prawdziwa historia](https://www.polskieradio.pl/podcasty/heweliusz-prawdziwa-historia)
- [Episode 5: Tajemnica Heweliusza. Co go zatopiło?](https://www.polskieradio.pl/podcasty/heweliusz-prawdziwa-historia/tajemnica-heweliusza-co-go-zatopilo-i-heweliusz-prawdziwa-historia-5)
- [Polish Radio / Trójka article on archival recordings](https://trojka.polskieradio.pl/artykul/3596356%2Cnieznane-nagrania-nowe-watki-podcast-polskiego-radia-o-zatonieciu-heweliusza)
- [Episode 7 on YouTube, with "Odnalezione nagrania" and Frank Michael sections](https://www.youtube.com/watch?v=pQxtmJHTsmo)

Polish Radio says the podcast uses previously unpublished archival recordings,
court documents, and unique radio-communication audio from the sinking ship.
Episode 7's chapter list includes "Odnalezione nagrania" and later Frank
Michael discussion.

## Audio Deformation / Provenance Analysis

The public YouTube audio should not be treated as a single target. It is a stack
of at least four likely layers.

### Original Radio / Acquisition Layer

This is the layer a historical fmplay profile should model most directly:

- Marine VHF Channel 16 FM receiver characteristics.
- Narrow communications voice passband rather than full-band speech.
- Squelch opens/closes and short noise bursts around transmissions.
- Weak-signal fading, capture effects, front-end noise, and intermittent
  readability.
- Receiver/operator monitoring chain at Rügen Radio or associated coastal-radio
  logging equipment.
- Possible line-monitoring coloration if the logging recorder received a
  console feed rather than raw discriminator/receiver audio.

Speculation: Rügen Radio's logging receiver may have had AGC, limiter, or
console-level processing before the recorder. The public recording alone does
not prove the exact receiver or console path.

### Original Recording / Storage Layer

This is likely present in the 1993 evidence/logging chain and may be worth
modeling if the profile aims for "received and logged at a coastal station":

- Analog magnetic logging tape or similar continuous radio logging system.
- Tape hiss and limited high-frequency extension.
- Wow/flutter, especially if copied or played back on imperfect machinery.
- Azimuth or head-alignment loss causing dullness or phase smear.
- Dropouts, overload/saturation, and level pumping.
- Generation loss if the public copy descends from evidence copies rather than
  the original logging medium.
- Possible print-through or low-level pre/post echoes on tape.

Speculation: The exact recorder format is not confirmed here. "Analog logging
tape" is the most plausible working model for a 1993 coastal-radio recording,
but it should remain a parameterized layer, not a hard-coded claim.

### Later Archival / Digitization Layer

This layer may be present in the public files, but it should be treated as
provenance damage unless intentionally recreating a specific archive copy:

- Playback deck quality and head alignment during digitization.
- ADC sample rate, bit depth, and clocking.
- Level normalization or gain staging after playback.
- Noise reduction or cleanup.
- Edits removing long silences between messages.
- Channel folding, stereo duplication, or mono-to-stereo handling.
- Unknown copy generations before upload.

The uploader explicitly notes that silence was removed, so timing between
messages in the YouTube file is not evidence of real-time gaps.

### Web / YouTube Layer

This layer should generally not be baked into a historical degradation profile:

- YouTube upload transcoding and lossy Opus/AAC delivery.
- Resampling and platform loudness normalization.
- Possible mono/stereo conversion.
- Codec pre-echo, warbling, bandwidth decisions, and noise-shaping artifacts.
- Any additional artifacts introduced by downloading/re-encoding snippets for
  analysis.

Use the YouTube upload as evidence for transcript timing and broad texture, not
as a clean spectral target. A profile that copies YouTube compression will sound
"web archived", not "1993 VHF/logged".

## Practical Modeling Guidance

Keep the source TTS clean and historically faithful. The TTS recipe should
render the best available transcript of the actual speech, not bake in radio
phrasing from later exchanges unless the clip intentionally spans those
exchanges.

Apply degradation via fmplay profiles:

- First model the VHF/channel/receiver constraints.
- Add coastal-radio logging-chain coloration as a separate historical layer.
- Avoid modeling YouTube codec artifacts in the historical profile.
- Treat the public upload as a timing/transcript and broad-texture reference.
- When comparing spectra, prefer short, clearly voiced windows and ignore codec
  edge behavior, silence edits, and platform loudness.

For a more faithful Mayday source text, start from the uploader's 04:37 hand
transcript rather than the current compressed text. Keep later lines such as
"16 miles west from Arkona" in a separate follow-up clip or dialogue sequence.

## Verification Notes

ASR was attempted on downloaded fragments using `uv run --with mlx-whisper`.
It helped confirm some later exchanges, but it was unreliable on the opening
Mayday because the radio audio is faint, noisy, and heavily masked. The uploader
hand transcript, source description, and manual timing alignment were more
useful for the opening call.

Helpful command shape:

```sh
yt-dlp -f 251/140/bestaudio --download-sections '*0:01:50-0:02:25' \
  -o '/tmp/fmplay-heweliusz/rugen-part1-open-call.%(ext)s' \
  'https://www.youtube.com/watch?v=NBizVLd5cL4'

uv run --with mlx-whisper python -c 'import mlx_whisper; print(mlx_whisper.transcribe("/tmp/fmplay-heweliusz/rugen-part1-open-call.webm"))'
```

Treat machine transcripts from this material as clues, not authoritative text.
