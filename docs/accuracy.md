# Phase 1 — measured accuracy and gate verdict

Measured with `bgmi-scrim truth` at commit `7d11966`, event match tolerance
±2.5 s, frame-field transition grace ±0.75 s, confidence threshold 0.60.

## What this was measured on, and what it is not

**No real BGMI scrim footage was available in the environment this was built in.**
These numbers come from synthetic spectator-HUD footage rendered by
`bgmi_scrim/synth.py`, which simulates a 25-team squad match and writes exact
labels for every frame and every event. The measurement is end to end — mp4
decode, ROI crop, CLAHE/threshold, OCR and colour matching, temporal
confirmation, panel/feed fusion, JSONL — and it is honest about everything
downstream of the pixels. It is **not** a measurement on real VODs, and it does
not exercise what real footage adds: an organiser's overlay art, compression
blocking around small digits, HUD animation and fades, spectator camera
switches, the observer's own UI, a panel that re-sorts as teams die, and
anything the real kill feed does that this one does not.

So the gate below is "passed on synthetic footage". To get the equivalent
number on your own footage: calibrate a profile, hand-label one VOD in the
label format (README), and run the same `truth` command. That is what the
harness is for, and it is the number that should decide whether this is
trusted in a scrim block.

## Footage used

| run | resolution | length | noise | event spacing | labels |
|---|---|---|---|---|---|
| A | 1920x1080 | 150 s | none | ~0.7 s between events (≈10x a real scrim) | 223 events, 300 labelled frames |
| B | 1280x720 | 150 s | gaussian σ≈0.012 + blur | ~0.7 s | 209 events, 300 frames |
| D | 1920x1080 | 150 s | none | ~0.7 s | 216 events, 300 frames |
| E | 1920x1080 | 150 s | gaussian σ≈0.008 + blur | ~0.7 s | 212 events, 300 frames |
| C | 1920x1080 | 420 s | none | ~3.1 s between events | 210 events, 840 frames |

Run B is the only one whose profile was authored at a different resolution
(1080p profile scaled to 720p footage), which is how profile scaling gets
exercised.

## Gate

| gate condition | required | measured (all five runs pooled) | verdict |
|---|---|---|---|
| knock + kill precision | >= 0.95 | **1.0000** (1070 / 1070 predicted matched) | PASS |
| knock + kill recall | >= 0.95 | **1.0000** (1070 / 1070 labelled matched) | PASS |
| no duplicate events | 0 | **0** | PASS |
| no phantom kills | 0 | **0** | PASS |
| teams-alive counter | >= 0.99 of sampled frames | **0.9995** (2039 / 2040) | PASS |
| every miss flagged or listed | — | 0 misses; every error is in the per-run confusion list | PASS |

**Gate: PASSED on synthetic footage**, on all five runs individually and pooled.

Re-measured after the kill-feed work in `feed_only_scenario.md`, which
changed how every text ROI in the pipeline is read. The five runs, same
seeds, at commit `c2421ea`: pooled precision 1.0000 (1070/1070), recall
1.0000 (1070/1070), 0 duplicates, 0 phantoms, teams-alive 0.9995
(2039/2040), players-alive 0.9936 (2027/2040), and all five pass
individually.

Per-team alive count was 0.9941 (50700/51000) in that run, the whole
shortfall being run B, where a 1080p profile scaled to 720p left our own
team's row unnamed on all 300 frames -- flagged low-confidence, never
confidently wrong. Commit `8d6301a` fixed that and run B re-measured at
1.0000 with the gate passing, so the pooled figure is 1.0000 (51000/51000);
the other four runs were already at 1.0000 before it and that commit can
only act where a label ROI reads as nothing.

### Render the clips at the capture's frame rate, not the sampling rate

The command below renders at `synth`'s default frame rate, and that matters.
Sampling is 2 fps raised to 5 fps for 3 s around a detected counter change,
so a clip rendered at 2 fps has no intermediate frames for the burst to
reach. Measured on run A rendered with `--fps 2`: 92 counter change points
detected, **0 burst samples taken**, and players-alive 0.7667 against 0.9967
from the same seed at the default rate, with 70 confidently wrong readings.
Events were unaffected (1.0000/1.0000).

Real OBS captures are 30 or 60 fps so this does not arise in practice, but a
source at the base sampling rate disables that safeguard without saying so.

## Events, per run

| run | labelled | predicted | matched | precision | recall | duplicates | phantoms | misses | mean timing error |
|---|---|---|---|---|---|---|---|---|---|
| A | 223 | 223 | 223 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.06 s |
| B | 209 | 209 | 209 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.06 s |
| D | 216 | 216 | 216 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.07 s |
| E | 212 | 212 | 212 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.06 s |
| C | 210 | 210 | 210 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.09 s |

Knock and kill are scored separately in each run's report; both are 1.0000
everywhere. Event existence comes from the team panel, which is why there are
no phantoms: 203 feed lines across the five runs were suppressed because the
panel tracked that slot and reported nothing.

## Killer attribution

Attribution is the weakest field, and it should be read separately from event
detection, because it is the only part that depends on reading the kill feed.

| | count | share of 1070 events |
|---|---|---|
| killer read and correct | 875 | 0.818 |
| killer absent (emitted as null, never guessed) | 183 | 0.171 |
| killer read and wrong | 12 | 0.011 |

A missing killer is a null in the log and is excluded from the phase-2 opponent
table rather than assigned to anyone.

## Frame fields, pooled

| field | labels | correct | accuracy | wrong and confident | wrong and flagged low-confidence |
|---|---|---|---|---|---|
| teams_alive | 2040 | 2039 | 0.9995 | 0 | 1 |
| players_alive | 2040 | 2027 | 0.9936 | 13 | 0 |
| zone_phase | 2040 | 2040 | 1.0000 | 0 | 0 |
| zone_state | 2040 | 2040 | 1.0000 | 0 | 0 |
| team_alive_count (per team, per frame) | 51000 | 51000 | 1.0000 | 0 | 0 |

map name, placement and winner: correct in all five runs (5/5 each).

### The thirteen players_alive errors

Every one of them is a counter value that was **on screen for 0.5 s** — shorter
than the confirmation window (3 samples, 1.5 s at the 2 fps baseline) — so the
state machine skipped it rather than misreading it. All thirteen are on the
four compressed 150 s clips, where two kills can land 0.19 s apart. On run C,
whose event spacing is 3.1 s, players_alive is 840/840. Raising `--grace` to
1.75 s does not change the count, which confirms these are skipped values and
not lag. This is the designed trade of "confirm on N consecutive samples": the
log never flickers, and in exchange a value that exists for less than the
window is not reported.

The single teams_alive error (run E, t=102 s) was emitted as `state: unknown`,
not as a wrong number.

## Cost

| run | samples analysed | wall clock | rate |
|---|---|---|---|
| A | 1092 (300 baseline + 792 burst) | 19.3 min | ~1.1 s per sample |
| C | 2008 (840 baseline + 1168 burst) | 31.0 min | ~0.9 s per sample |

Four cores, tesseract 5.3.4, one OCR thread per call. The cost is dominated by
tesseract process launches; the blank-ROI gate and the early-accept between the
two binarisations are what keep it near one second per sample rather than ten.
Burst sampling roughly doubles to triples the sample count on these clips
because events are packed much more tightly than in a real scrim.

## Defects found by this harness, and fixed

Each of these was caught by a measured run, not by reading the code.

1. **Duplicate and phantom events from the kill feed** (run A, precision 0.85).
   The feed de-duplication key included the killer, so a flickering killer tag
   minted a second key for one feed entry; and a leftover feed line was emitted
   whenever the panel had seen any nearby transition, which is exactly when the
   panel had already emitted its own event. Fixed: victim-only dedup key,
   feed-only events allowed solely where the panel has no confirmed state.
2. **Two-digit counters read as one digit** (run A, players_alive 0.6567 with
   103 confidently wrong readings). "44" thresholds into one connected
   component, so the digit-count cross-check endorsed the one-digit read.
   Fixed: split over-wide components at ink-profile valleys with a width guard,
   judge component height against the tallest component, and never let an
   unreadable glyph silently shorten the per-digit read.
3. **Panel geometry drift when scaling a profile** (run B, 23/25 rows resolved,
   three teams with no alive count). The row pitch was rounded to an integer at
   scale time, so by row 24 the slot boxes sat eight pixels off. Fixed:
   geometry stays float through scaling, rounded once at use.
4. **Wrong winner** (run E, named FS instead of GB). The winner was committed
   the moment teams_alive hit 1, from a panel that had not finished confirming
   the last deaths, and the end-of-match banner was never read. Fixed: the
   winner is decided at match end from the banner, with the last row standing
   as fallback and an explicit unknown if neither is available.

## Reproducing

```bash
python -m bgmi_scrim synth --out clip.mp4 --labels clip.labels.jsonl \
    --duration 420 --seed 23 --map VIKENDI
python -m bgmi_scrim analyze clip.mp4 --profile profiles/synthetic_1080p.json \
    --out clip.events.jsonl --stats clip.stats.json
python -m bgmi_scrim truth --labels clip.labels.jsonl --events clip.events.jsonl \
    --md accuracy.md --json accuracy.json
```

Run seeds used above: A=7, B=11 (720p, noise 0.012), D=31, E=47 (noise 0.008),
C=23 (420 s). `truth` exits 2 when the gate is not met.
