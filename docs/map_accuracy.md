# Phase 3 — measured accuracy (map stream)

Measured with `bgmi-scrim maptruth` at commit `e056a0d`, position tolerance
±0.35 s, an identity counted correct when the dot is within 80 m of a player it
claims to be.

## What this was measured on, and what it is not

**No real BGMI map-stream footage was available where this was built.** These
numbers come from synthetic observer-map footage rendered by
`bgmi_scrim/map_synth.py`: a 25-team lobby, terrain, named places, both zone
circles, and one labelled dot per living player, with the true world position of
every player written out at every sampled moment. The measurement runs the whole
pipeline — decode, dot detection, label OCR, colour book, tracking, identity
settling, RANSAC circle fitting, behaviour derivation — and is honest about
everything downstream of the pixels.

It does **not** test what real footage adds: an organiser's map art and icons,
compression blocking on 4-pixel dots, dots passing under HUD furniture, the map
panning or zooming, spectator UI drawn over the map, label fonts other than this
one, and teams whose colours an organiser reuses. Treat it as a measurement of
the method, not a guarantee on your VODs. To get your own numbers, label one map
stream in the format in `bgmi_scrim/map_truth.py` and run the same command.

The two 420 s clips were rendered at the 2 fps the analyzer samples at, so the
files stay small; `m30fps` is rendered at a real 30 fps and sampled down, which
is what exercises the sampling path.

## Footage

| run | footage | labelled positions |
|---|---|---|
| m1 | 1920x1080, 420 s, clean, ERANGEL | 46585 player-samples |
| m2 | 1920x1080, 420 s, gaussian noise σ≈0.01, MIRAMAR | 46869 player-samples |
| m30fps | 1920x1080, 60 s rendered at 30 fps (sampled at 2 fps), drop phase only | 10689 player-samples |

## Positions

| run | named-player error (median) | p90 | ≤25 m | team-dot error (median) | p90 |
|---|---|---|---|---|---|
| m1 | 7.42 m | 29.45 m | 0.891 | 7.67 m | 27.94 m |
| m2 | 7.27 m | 19.55 m | 0.9122 | 7.68 m | 30.45 m |
| m30fps | 10.79 m | 38.7 m | 0.7368 | 10.29 m | 32.49 m |

At 8.23 m per pixel on these clips, a median error of about 7 m is under one pixel: the positions are as good as the picture allows.

## Identity

| run | samples | with a team | with a player name | team precision | player precision |
|---|---|---|---|---|---|
| m1 | 27983 | 23229 | 17918 | 0.9849 | 0.9777 |
| m2 | 26273 | 23230 | 15894 | 0.9421 | 0.9822 |
| m30fps | 2734 | 1024 | 285 | 0.999 | 0.9965 |

## Coverage

| run | player-slots covered by a dot | team-slots identified | samples carrying a player name |
|---|---|---|---|
| m1 | 0.8654 | 0.8061 | 0.3846 |
| m2 | 0.8616 | 0.7715 | 0.3391 |
| m30fps | 0.6495 | 0.3547 | 0.0267 |

Coverage is bounded by the medium, not by the code: four team-mates inside one building are one dot, and that dot is counted once. `m30fps` covers only the drop phase, where every squad is stacked, which is why its coverage is the lowest of the three.

## Zone circles

| run | ring | fitted | held back | centre error median | centre p90 | radius error median |
|---|---|---|---|---|---|---|
| m1 | blue | 132 | 78 | 7.07 m | 19.29 m | 4.3 m |
| m1 | safe | 137 | 73 | 6.32 m | 8.99 m | 4.8 m |
| m2 | blue | 144 | 66 | 6.66 m | 22.33 m | 4.45 m |
| m2 | safe | 189 | 21 | 5.33 m | 7.91 m | 4.8 m |
| m30fps | blue | 27 | 3 | 13.33 m | 19.18 m | 6.2 m |
| m30fps | safe | 26 | 4 | 0.92 m | 6.66 m | 5.35 m |

## Behaviour layer

The same derivation code is run twice: once on the extracted tracks and once on the labelled truth. What is compared is therefore the extraction, not the definitions.

| run | phases (truth → extracted) | announce error | team landings matched | same place | landing position error | rotations (teams) | distance ratio |
|---|---|---|---|---|---|---|---|
| m1 | 8 → 8 | ±8.0 s | 24/25 | 0.7917 | 84.22 m | 19/23 | 1.126 |
| m2 | 8 → 8 | ±2.25 s | 23/25 | 0.6957 | 105.28 m | 18/22 | 0.947 |
| m30fps | 2 → 2 | ±4.0 s | 16/25 | 0.75 | 41.73 m | 1/1 | 3.576 |

## Cost

| run | samples | label reads | wall clock |
|---|---|---|---|
| m1 | 840 | 1775 | 94 s |
| m2 | 840 | 1690 | 89 s |
| m30fps | 120 | 553 | 28 s |

Four cores. Label OCR dominates, which is why names are read on a schedule and only next to dots whose track does not already know who it is.

## Defects this harness caught, and fixed

1. **A name read off a neighbouring dot.** Crowded label boxes put whole teams on the wrong track. Fixed with a crowding check, a strict score for player names, and a colour book that vetoes a name whose dot colour belongs to another team.
2. **Colour used as identification.** With 25 teams sharing the hue circle, nearest-hue is not an identification; it now names a team only when the runner-up is clearly further away, and otherwise only corroborates or vetoes.
3. **Identity decided at the wrong moment.** A squad stacked on its drop is unreadable, so identity is now settled at the end of the run and applied to the track's whole history.
4. **One bad circle fit moving the zone.** Blue-circle centre error had a p90 of 1157 m on m1 and 1897 m on m2. A stabiliser now holds back a fit that disagrees with the recent ones until it repeats.
5. **Phases invented by fitting jitter.** m1 reported 16 phases where the match had 8. The safe-circle series is median-filtered and a phase must be confirmed on consecutive samples.
6. **Jitter counted as rotation.** Path length alone called 480 m of shuffling a rotation; a rotation now needs net displacement too.

## Known limits, measured

* **Landing time lags the truth by a median of 24-28 s on the long clips**,
  while the landing *place* agrees 76-83% of the time and the position to a
  median of about 85-105 m. The place is right; the moment is late, because a
  squad's dots keep moving and merging after touchdown, so the end of the
  descent is seen later on the extracted path than on a clean one. On the 60 s
  drop-phase clip, where nothing else is happening, the same code lands within
  12 s and 14 m.
* **A team whose identity only resolves after the drop has no descent to
  measure.** Those landings carry `state: no_descent_seen` and are reported
  as "the first tracked position", not as a drop. One to three teams per run.
* **Rotation distance agrees to within about 10%** of the truth path
  (`median_distance_ratio` 0.95-1.13 on the two full matches). On the 60 s
  clip there are almost no rotations, so its ratio is noise.
* **Zone fits are held back, not guessed.** The stabiliser refuses a fit that
  disagrees with the recent ones until it repeats, which is why a third of
  samples on m1 report the ring as unknown rather than reporting a ring that
  jumped a kilometre.

## Reproducing

```bash
python -m bgmi_scrim mapsynth --out map.mp4 --labels map.labels.jsonl \
    --duration 420 --seed 5 --sample-fps 2 --map-area 422 43 972 972
python -m bgmi_scrim mapanalyze map.mp4 --profile profiles/synthetic_map_1080p.json \
    --tracks map.tracks.jsonl --map-events map.mapevents.jsonl --team TX
python -m bgmi_scrim maptruth --labels map.labels.jsonl --tracks map.tracks.jsonl \
    --profile profiles/synthetic_map_1080p.json --md map_accuracy.md
```
