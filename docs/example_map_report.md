# TX — synthetic map block (2 matches) map report

Matches: 2 (M1=m1.tracks.jsonl; M2=m2.tracks.jsonl). Positions read at confidence >= 0.60. Places are named POIs where a POI table was calibrated, and map grid references otherwise.

## 1. Drop and loot

| match | map | landed | place | first move | looted there | drop to first circle centre |
|---|---|---|---|---|---|---|
| M1 | ERANGEL | 00:19 | B2 | 01:05 | 0 s | 4190 m |
| M2 | ERANGEL | 00:19 | B1 | 00:55 | 0 s | 4235 m |

## 2. Rotations, by zone phase

| match | phase | announced | we started | delay | mode | from → to | distance | arrived | margin to close |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 1 | 00:00 | no rotation in log | - | - | - | - | - | - |
| M1 | 2 | 00:46 | 01:05 | 20 s | vehicle | B2 → B3 | 4242 m | 06:24 | -306 s |
| M1 | 3 | 01:34 | no rotation in log | - | - | - | - | - | - |
| M1 | 4 | 02:08 | no rotation in log | - | - | - | - | - | - |
| M1 | 5 | 02:56 | no rotation in log | - | - | - | - | - | - |
| M1 | 6 | 03:42 | no rotation in log | - | - | - | - | - | - |
| M1 | 7 | 04:40 | no rotation in log | - | - | - | - | - | - |
| M1 | 8 | 05:34 | no rotation in log | - | - | - | - | - | - |
| M2 | 1 | 00:00 | no rotation in log | - | - | - | - | - | - |
| M2 | 2 | 00:32 | 00:55 | 24 s | vehicle | B1 → C3 | 4074 m | 04:01 | -163 s |
| M2 | 3 | 01:20 | no rotation in log | - | - | - | - | - | - |
| M2 | 4 | 02:08 | no rotation in log | - | - | - | - | - | - |
| M2 | 5 | 02:58 | no rotation in log | - | - | - | - | - | - |
| M2 | 6 | 03:42 | 04:14 | 32 s | vehicle | D3 → E3 | 1159 m | 06:14 | -104 s |
| M2 | 7 | 04:42 | no rotation in log | - | - | - | - | - | - |
| M2 | 8 | 05:16 | no rotation in log | - | - | - | - | - | - |

Late arrivals (under 30 s to spare, or after the close): M1 06:24 B3 phase 2, arrived 306s AFTER the circle closed; M2 04:01 C3 phase 2, arrived 163s AFTER the circle closed; M2 06:14 E3 phase 6, arrived 104s AFTER the circle closed.

## 3. Zone discipline

| match | time outside the blue circle | share of time on the circle edge | phases never entered | tracked window |
|---|---|---|---|---|
| M1 | 380 s | 1.0 | 2, 3, 4, 5, 6, 7, 8 | 00:00–06:24 |
| M2 | 374 s | - | 5, 6, 7, 8 | 00:00–06:14 |

## 4. Fights, placed on the map

No HUD event log was supplied, so no kill or knock can be placed. Pass `--hud-events` with the phase-1 events.jsonl for the same match.

## 5. Pressure at phase changes

| match | phase | our place | players within 200 m | teams identified nearby |
|---|---|---|---|---|
| M1 | 1 | G1 | 0 | none identified |
| M1 | 3 | B1 | 0 | none identified |
| M1 | 4 | B2 | 0 | none identified |
| M1 | 5 | B2 | 0 | none identified |
| M1 | 6 | B2 | 0 | none identified |
| M1 | 7 | B2 | 0 | none identified |
| M1 | 8 | B3 | 0 | none identified |
| M2 | 1 | A1 | 0 | none identified |
| M2 | 3 | B2 | 0 | none identified |
| M2 | 4 | B2 | 0 | none identified |
| M2 | 5 | C2 | 0 | none identified |
| M2 | 6 | C3 | 0 | none identified |
| M2 | 7 | D3 | 0 | none identified |
| M2 | 8 | D3 | 0 | none identified |

## 6. Squad cohesion

| match | median spread | widest | share of samples over 250 m |
|---|---|---|---|
| M1 | 1692 m | 3319 m | 0.94 |
| M2 | 1860 m | 2629 m | 0.98 |

Spread is measured between named team-mates only. While their dots overlap the stream cannot separate them, which is itself the measurement that the squad was together.

## 7. Practice focuses

1. **Stop arriving on the closing edge**
   - Evidence: 3 of 3 rotations finished under 30s before the circle closed; tightest -306s at M1 06:24 B3
   - Target next block: no rotation finishing under 30s before close
2. **Less time in the blue**
   - Evidence: 754s outside the circle across 2 matches; worst M1 with 380s
   - Target next block: under 188s per match
3. **Drop closer to the first circle**
   - Evidence: median 4213 m from the first circle centre; furthest 4235 m at M2 00:19 B1
   - Target next block: median under 2949 m

## 8. Other teams seen on the map

Every team the stream named, not just ours: where they landed, how often they moved, and when the stream last had them. This is scouting material; a team the stream never named does not appear.

**M1** (ERANGEL)

| team | landed | place | rotations | first rotation | last seen |
|---|---|---|---|---|---|
| AUT | 00:23 | F5 | 0 | none in log | 01:24 |
| BLIND | 01:06 | C3 | 1 | 01:14 C3 → C3 (vehicle) | 02:12 |
| ENT | 01:43 | C3 | 0 | none in log | 01:43 |
| FS | 01:40 | B7 | 0 | none in log | 01:47 |
| GB | - | - | 0 | none in log | 00:02 |
| GODL | 00:48 | F1 | 2 | 00:48 F1 → F1 (vehicle) | 03:02 |
| HYD | 01:02 | D3 | 1 | 01:37 D3 → D5 (vehicle) | 05:20 |
| IQOO | 00:55 | E6 | 1 | 01:35 D6 → D5 (vehicle) | 02:15 |
| K9 | 00:28 | F8 | 2 | 00:49 F8 → D5 (vehicle) | 06:59 |
| MARC | 00:27 | E4 | 1 | 01:26 E4 → D5 (vehicle) | 04:04 |
| MDY | 01:30 | D7 | 1 | 01:30 D7 → C6 (vehicle) | 03:22 |
| NGE | 01:05 | C4 | 1 | 01:39 C5 → D4 (vehicle) | 04:40 |
| NOVA | 00:11 | H7 | 2 | 00:11 H7 → H6 (vehicle) | 05:59 |
| OR | 00:29 | G8 | 1 | 01:48 G7 → F6 (vehicle) | 03:49 |
| ORCA | 00:43 | B3 | 0 | none in log | 02:03 |
| REV | 01:24 | F3 | 1 | 01:24 F3 → E5 (vehicle) | 03:41 |
| SKY | 00:10 | G3 | 2 | 00:10 G3 → G3 (vehicle) | 05:41 |
| SOUL | 00:49 | D2 | 1 | 01:16 D2 → E3 (vehicle) | 03:29 |
| TSM | 01:03 | A6 | 1 | 01:38 B6 → E4 (vehicle) | 06:02 |
| VLT | 01:20 | B4 | 1 | 01:43 B4 → B4 (foot) | 04:28 |
| VST | 00:26 | G4 | 2 | 00:38 E1 → G4 (foot) | 04:47 |
| XSP | 00:56 | C6 | 1 | 01:15 C6 → C6 (vehicle) | 02:34 |
| ZEUS | 00:29 | F6 | 1 | 00:52 F6 → F6 (vehicle) | 02:50 |

**M2** (ERANGEL)

| team | landed | place | rotations | first rotation | last seen |
|---|---|---|---|---|---|
| AUT | 00:43 | F5 | 1 | 00:43 F5 → E4 (vehicle) | 03:52 |
| BLIND | 01:32 | F3 | 0 | none in log | 01:33 |
| ENT | 00:48 | C3 | 2 | 01:13 C3 → C4 (vehicle) | 04:56 |
| GB | 01:57 | C2 | 1 | 02:04 C2 → E4 (vehicle) | 04:47 |
| GODL | 00:51 | F1 | 1 | 01:05 F1 → F3 (vehicle) | 06:59 |
| GXR | 00:30 | C7 | 1 | 01:06 C7 → C7 (vehicle) | 04:31 |
| HYD | 01:02 | E3 | 2 | 01:17 E3 → E3 (vehicle) | 04:26 |
| IQOO | 00:50 | D6 | 1 | 01:25 D6 → D4 (vehicle) | 03:05 |
| K9 | 00:29 | F8 | 1 | 00:38 F8 → F6 (vehicle) | 06:59 |
| MARC | 00:24 | D5 | 1 | 01:20 D5 → E4 (vehicle) | 04:03 |
| MDY | - | - | 2 | 01:19 E7 → D4 (vehicle) | 06:12 |
| NGE | 00:44 | C4 | 1 | 00:51 C4 → D4 (vehicle) | 05:05 |
| NOVA | 00:49 | H6 | 1 | 02:06 G6 → G6 (foot) | 03:39 |
| OR | 00:28 | H7 | 1 | 01:43 G7 → G7 (vehicle) | 02:56 |
| ORCA | 02:08 | B3 | 0 | none in log | 02:08 |
| REV | 00:26 | F3 | 0 | none in log | 01:36 |
| SKY | 00:57 | G3 | 1 | 00:57 G3 → G3 (vehicle) | 03:35 |
| SOUL | 01:23 | E2 | 0 | none in log | 01:32 |
| TSM | 00:26 | B6 | 1 | 01:57 B6 → B5 (vehicle) | 05:29 |
| VST | - | - | 0 | none in log | 00:07 |
| XSP | 00:47 | C6 | 1 | 01:43 C6 → D5 (vehicle) | 05:48 |
| ZEUS | 01:28 | F6 | 1 | 01:43 F6 → E4 (vehicle) | 04:07 |

## 9. Not measurable from a map stream

- Elevation, buildings and whether a team is inside or outside: the map stream is a flat top-down view.
- Which player of a squad is which while their dots overlap: at the map's scale team-mates standing together are one dot, so positions are reported per team with the player named only where the stream showed a readable name.
- Vehicle type, fuel or whether a team is on foot in a vehicle's shadow: only speed is measurable, and the foot/vehicle split is a speed threshold.
- The zone timer itself: phase announcements and closes are read from the circles moving, not from the countdown.
- Damage, knocks and kills: those come from the HUD stream (phase 1). Pass an events log to place them on the map; without one this report has no fights.
- Loot quality and what a team picked up: looting here means a stretch of standing still, nothing more.

