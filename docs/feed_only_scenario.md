# A map stream with a kill feed and no HUD

What the worker produces when the only things in frame are the map, one
labelled dot per living player, and the kill feed: no team-list panel, no
lobby counters, no placement card, no winner banner.

Measured at commit `c2421ea` on a 420 s 1080p clip rendered by
`bgmi_scrim/map_synth.py`: 25 teams, both zone circles, labels in the
`teamnamexplayername` form a real overlay writes, and a feed naming players
rather than slots. Ground truth for every event and every player position is
written alongside. As everywhere else in this repository, this measures the
method on synthetic footage, not a guarantee on your VODs.

## Profile

Only the feed is calibrated, so only the feed can be read. The profile holds
one ROI:

```json
"rois": { "kill_feed": { "box": [1410, 40, 500, 26], "kind": "feed",
                         "rows": 5, "row_pitch": 34, "row_h": 26 } }
```

plus `team_tags` and a `roster` mapping each tag to its player names. The
roster is what turns a named feed line into a slot: a feed that writes
`SOULxVIPER` carries no slot number, and the roster's order supplies one.
Without a `team_panel` ROI the fuser has no authority to defer to, so feed
events are emitted directly.

## Events

| type | truth | predicted | matched | precision | recall | dup | phantom | miss | mean dt |
|---|---|---|---|---|---|---|---|---|---|
| knock | 96 | 96 | 96 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.26 s |
| kill | 96 | 95 | 95 | 1.0000 | 0.9896 | 0 | 0 | 1 | 0.33 s |

Killer attribution correct on 190 of 191. Two entries in the confusion list
for the whole clip: one kill at 83.5 s whose killer was read as the wrong
team, and one kill at 154.68 s never read. 0 OCR engine failures.

## Positions and behaviour, from the same clip

| measurement | value |
|---|---|
| named player position error | median 7.32 m, p90 26.09 m |
| team identity precision | 0.9858 |
| player identity precision | 0.9777 |
| zone phases | 8 of 8, announce within ±8.0 s |
| team landings | 24 of 25 |
| rotations | 19 of 23 teams |

These match the phase 3 baseline in `map_accuracy.md` (7.42 m, 0.9849, 8 of 8,
24 of 25) within noise. Putting a kill feed beside the map costs the map
layer nothing.

## What the report cannot say

The gate verdict on this input is **NOT MET**, and that is the correct
answer rather than a failure. Every event condition passes: precision,
recall, no duplicates, no phantom kills, every miss accounted. It fails
`teams_alive >= 0.99`, because this input contains no teams-alive counter.
A gate condition about a HUD element cannot be satisfied by footage that
does not show it.

Four things come back `not in log`:

* **placement and placement points** -- no placement card. Section 1 reports
  kill points only and says it claims no placement points.
* **match winner** -- no banner.
* **match start and end** -- both are gated on the counters, so the match
  never formally opens.
* **kills and deaths by zone phase** in section 3 -- the HUD log carries no
  phase. The map's own phases are read, and section 5 uses them.

Everything else populates: survival and per-slot time alive, fights, trade
rate, knocks converted and conceded, opponents, the full positioning and
rotation section, and three ranked practice focuses.

## One thing to know about sampling

Burst sampling raises 2 fps to 5 fps for 3 s around a detected counter
change. A feed-only profile has no counters, so nothing triggers it, and a
source rendered at 2 fps has no intermediate frames to sample even when
something does. On this clip the analyzer reported 840 samples and 0 burst
samples. Event timing is therefore limited by the base rate, which the mean
offsets above (0.26 s and 0.33 s) already reflect.

The same constraint bites harder on a HUD clip: at a 2 fps source, gate run A
found 92 counter change points, took 0 burst samples, and `players_alive`
fell from 0.9967 to 0.7667. Real OBS captures are 30 or 60 fps, so this does
not arise in practice, but a source at the base sampling rate silently
disables the safeguard.

## Reproducing

```bash
python -m bgmi_scrim mapsynth --out mapfeed.mp4 --labels mapfeed.labels.jsonl \
    --duration 420 --seed 5 --sample-fps 2 --map-area 422 43 972 972
python -m bgmi_scrim analyze mapfeed.mp4 --profile feed_only.json \
    --out events.jsonl --team TX
python -m bgmi_scrim truth --labels mapfeed.labels.jsonl --events events.jsonl \
    --md events.accuracy.md
python -m bgmi_scrim mapanalyze mapfeed.mp4 --profile map.json \
    --tracks tracks.jsonl --map-events mapevents.jsonl --team TX
python -m bgmi_scrim report --events events.jsonl --tracks tracks.jsonl \
    --map-events mapevents.jsonl --team TX --out report.md
```
