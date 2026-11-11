# BGMI scrim match-analysis worker

Reads Indian-scrim BGMI spectator-view footage (squads TPP, 4-man teams, up to 25
teams, 720p/1080p OBS mp4) and extracts a timestamped event log from the HUD
pixels. No game API, no observer client, no network calls — everything comes off
the frames. Phase 2 turns those event logs into a coach-facing preparation
report and derives nothing the log does not contain.

```
phase 1   HUD video   --->  events.jsonl                  (analyze)
phase 2   events.jsonl ---> report.md                     (report)
phase 3   map video   --->  tracks.jsonl + map_events.jsonl (mapanalyze)
                       ---> map_report.md                 (mapreport)
all       the logs    ---> dashboard.html                 (dashboard)
```

## Install

```bash
pip install -r requirements.txt      # opencv, numpy, pytesseract
apt-get install -y tesseract-ocr     # the OCR binary itself
pip install -e .                     # optional: installs the `bgmi-scrim` command
```

Everything runs offline. No web server, no database, no overlay, no UI beyond the
calibrator.

## Phase 1 — event extraction

```bash
python -m bgmi_scrim analyze match1.mp4 --profile profiles/myorg_1080p.json \
    --out match1.events.jsonl --team TX --stats match1.stats.json
```

What is extracted, each record with a timestamp and a confidence in 0..1:

| record | fields |
|---|---|
| `teams_alive`, `players_alive` | lobby-wide counters, as confirmed transitions |
| `team_alive_count` | per-team alive count from the team list panel (`team`, `value`) |
| `knock`, `kill` | `victim_team`, `victim_slot`, `killer_team`/`killer_slot` when the feed shows them, else absent (null) |
| `zone_phase`, `zone_state` | phase number; blue-zone `shrinking` vs `waiting` |
| `placement` | our team's rank when eliminated |
| `winner` | winning team tag |
| `map_name`, `match_start`, `match_end` | one per match; several matches in one file are separated by counter resets and carry `match_index` |

Every record carries `state`: `ok`, or `unknown` with no value when the reading
sits below the profile's confidence threshold. A low-confidence reading is never
turned into a guess.

### How it works

1. **Sampling** (`sampler.py`) — a cheap first pass reads only the lobby counters
   at 2 fps and notes every timestamp where one changed. The full pass then runs
   on the 2 fps baseline plus a 5 fps burst inside a 3 s window around each of
   those change points. Decoding is sequential (`grab` for skipped frames) because
   seeking in OBS-captured mp4 is unreliable.
2. **Profiles** (`profile.py`) — a profile is JSON data: one box per HUD field for
   one resolution and one organiser overlay, plus HSV bands, confirmation counts
   and thresholds. Boxes are stored at the profile's reference resolution and
   scaled to whatever the video actually is, so a 1080p profile runs 720p footage.
   No HUD geometry exists in the code.
3. **Preprocessing** (`preprocess.py`) — per ROI: crop, 2x (or more, up to
   `min_height`) upscale, CLAHE, adaptive threshold. A second Otsu binarisation is
   produced as well: adaptive threshold is right for gradient overlays but turns a
   small flat HUD plate into speckle, and the OCR layer keeps whichever reads with
   more confidence. Blank ROIs (an empty feed row, no banner) are gated out by
   contrast and ink fraction before OCR, which is both faster and safer.
4. **Reading** (`ocr.py`, `hud.py`) — numerals: a whole-ROI read (psm 7), a
   per-digit read of segmented connected components (psm 10), and the component
   count itself, which is a font-independent count of how many digits are on
   screen. A value whose digit count disagrees with the component count is capped
   below any sane threshold, so it surfaces as `unknown` rather than as a
   confident wrong number. Icons and states (alive / knocked / dead, zone
   shrinking / waiting) are HSV colour matching, not OCR. Team tags are matched
   against the lobby roster with confusable-glyph variants (`O`/`0`, `I`/`1`/`L`).
5. **Temporal state machine** (`statemachine.py`) — a field's value changes only
   after the same reading is seen on N consecutive samples, and what gets emitted
   is the transition (timestamped at the first sample of the confirming run), not
   the frame reading. Lobby counters are additionally gated monotonically: inside
   a match they only fall, and a jump back up past `reset_at` is read as a new
   match. This is what removes OCR flicker from the log.
6. **Panel row identity** (`analyze.py`) — which team sits on which panel row is
   resolved once in a pre-pass over frames spread across the video, by majority
   vote with a roster-constrained unique assignment (two rows cannot be the same
   team). Re-reads every `tag_refresh_s` can move a row if they disagree
   decisively on two consecutive epochs, which is how a panel that re-sorts gets
   picked up. A row that never resolves is reported as `ROWn`, not guessed.
7. **Fusion** (`fuse.py`) — the panel is the authority on *whether* a player went
   down (a colour read, confirmed over N samples); the feed is the only witness to
   *who* did it (OCR). A feed line is believed only after two sightings and is then
   suppressed for `dedup_window_s`, so one feed entry cannot emit twice. A
   feed-only event is emitted only where the panel has no confirmed state for that
   victim at all (panel ROI missing, row never resolved, indicator occluded);
   where the panel does track the slot and said nothing, the feed line is dropped
   and counted in `suppressed_feed`. That rule is what keeps phantom and duplicate
   kills out, at the cost of leaning on the panel for event existence.

### Calibration

```bash
# 1. scale a template to this footage's resolution
python -m bgmi_scrim calibrate vod.mp4 --template profiles/bgmi_template_1080p.json \
    --out profiles/myorg_1080p.json --team TX

# 2. pick the boxes by hand (needs an OpenCV display; writes the profile)
python -m bgmi_scrim calibrate vod.mp4 --out profiles/myorg_1080p.json \
    --base profiles/bgmi_template_1080p.json --at 120

# 3. verify: draws every box on a frame and prints what each ROI reads
python -m bgmi_scrim calibrate vod.mp4 --preview profiles/myorg_1080p.json \
    --preview-png check.png --at 120
```

Step 3 is the one that matters: it renders the panel's per-slot and per-tag boxes
too, and prints the readings with their confidences plus a `low_confidence_rois`
list. Iterate on the JSON until that list is empty. `profiles/bgmi_template_*.json`
ship with `"_calibrated": false` and placeholder boxes — `analyze` warns loudly
when a profile is still a template, because **the shipped templates are not
measured BGMI positions**, they are a documented starting shape.

### Ground-truth harness

Hand-label frames and events in one JSONL file:

```json
{"kind":"frame","t":62.0,"teams_alive":18,"players_alive":51,"zone_phase":4,"zone_state":"shrinking","team_alive":{"TX":3}}
{"kind":"event","t":63.2,"type":"knock","victim_team":"TX","victim_slot":2,"killer_team":"GB","killer_slot":1}
{"kind":"scalar","t":1.0,"type":"map_name","value":"ERANGEL"}
```

Only the keys present on a line are scored, so partial labelling is fine.

```bash
python -m bgmi_scrim truth --labels labels.jsonl --events match1.events.jsonl \
    --md accuracy.md --json accuracy.json
```

It reports, per field, precision/recall (events) or accuracy (frame fields),
duplicate and phantom counts, killer-attribution accuracy, a **confusion list
naming every single miss, phantom, duplicate and wrong read**, and a pass/fail
verdict on the phase-1 gate. Exit code is 2 when the gate is not met.

Two knobs matter when reading the numbers. `--tolerance` (default 2.5 s) is how
far a predicted event may sit from its label and still count as the same event.
`--grace` (default 0.75 s) is the transition tolerance on frame fields: a
confirmed transition is emitted at the first sample of its confirming run, so near
a change the log lags the label by up to one sampling interval. The report prints
strict (no grace) accuracy alongside.

### Measured accuracy

Pooled over five measured runs (1,070 labelled knock/kill events, 2,040 labelled
frames): knock and kill precision **1.0000**, recall **1.0000**, 0 duplicates,
0 phantoms, 0 misses; teams-alive 0.9995; zone phase and state 1.0000; per-team
alive count 1.0000; map, placement and winner correct in all five.

**Those numbers are from synthetic spectator-HUD footage, not from real scrim
VODs** — no real footage was available where this was built. See
[`docs/accuracy.md`](docs/accuracy.md) for exactly what was measured, what it
does not cover, the four real defects the harness caught, and how to get the
same numbers for your own footage.

## Phase 3 — map stream (positions)

```bash
# calibrate the map box once per overlay, then check it
python -m bgmi_scrim mapcalibrate map.mp4 --template profiles/mapstream_template_1080p.json \
    --out profiles/myorg_map.json --map ERANGEL
python -m bgmi_scrim mapcalibrate map.mp4 --preview profiles/myorg_map.json \
    --preview-png check.png --at 120

# extract positions and behaviour
python -m bgmi_scrim mapanalyze map.mp4 --profile profiles/myorg_map.json \
    --tracks m1.tracks.jsonl --map-events m1.mapevents.jsonl --team TX

# the coach-facing report (add --hud-events to place kills on the map)
python -m bgmi_scrim mapreport --tracks m1.tracks.jsonl --map-events m1.mapevents.jsonl \
    --team TX --hud-events m1.events.jsonl --out map_report.md
```

Input is the full-screen observer map, where every player is a coloured dot
labelled `TEAMxPLAYER`. Output is `tracks.jsonl` (a position per dot per sample,
in metres) and `map_events.jsonl` (landings, loot stretches, rotations, zone
phases and entries, squad spread), for **every team the stream names**, not just
yours.

### What the medium can and cannot give you

At a full-map view the scale is about **8 m per pixel**, and a player dot is a
few pixels across, so **two team-mates standing together are one dot**. That is
not a bug to fix, it is the resolution of the picture. Everything is therefore
reported at two levels:

* **team** — the median position of that team's dots, which is what rotations,
  zone timing and drops are derived from. This is reliable.
* **player** — only where the label beside a dot was read cleanly. A player's
  name is never carried over to a dot whose colour says it belongs to someone
  else, and never invented when four names are piled on top of each other.

`mapcalibrate --preview` prints `metres_per_pixel` and `two_dots_merge_below_m`
for your own footage so you know which of the two you are working with.

### How identity survives a crowded map

1. **Labels** are matched whole against the roster, because `TXxALFA`
   uppercases to `TXXALFA`, where no regex can tell the separator from the tag.
2. **Team tags** are read even out of a crowded blob (`TXXALFATXXBRAVO`): the
   tag followed by an `X` appears several times over, which is enough for the
   team even when no player can be named.
3. **Colour** is a second witness, read from different pixels than the text: a
   colour book learns each team's hue, vetoes names that disagree, and refuses
   to name a team from colour alone when 25 teams share the hue circle.
4. **Identity is settled once, at the end of the run**, and applied to a
   track's whole history — a squad stacked on its drop is named from the label
   that only becomes legible two minutes later.

Zone circles are fitted with RANSAC (map text is the same white as the safe
circle, and one blob of it drags a least-squares fit off the map), then
stabilised: a fit that disagrees with the recent ones is held back until it
repeats, and a phase is declared only after the new circle is confirmed.

Phase 3 measured accuracy: [`docs/map_accuracy.md`](docs/map_accuracy.md).

## The dashboard

```bash
python -m bgmi_scrim dashboard \
    --events m1.events.jsonl m2.events.jsonl m3.events.jsonl \
    --tracks m1.tracks.jsonl m2.tracks.jsonl \
    --map-events m1.mapevents.jsonl m2.mapevents.jsonl \
    --team TX --block "Tuesday block A" --out dashboard.html
```

One HTML file, around 230 KB for a five-match block. No server, no database and no
build step, because the rest of this toolchain has none: it opens from a USB stick
in the practice room, offline. `--tracks` and `--map-events` are optional and are
matched to `--events` by position; a match with no map stream still gets its
result, timeline and tables, and says so where the map would be.

What is on it:

* **The map, scrubbable.** Every tracked team's trail from where it landed, the
  safe and blue circles at that moment, drop markers, and a nearest-teams list
  that re-sorts as you scrub. Space plays, arrow keys step five seconds.
* **The phase ribbon.** Eight segments whose widths are the real durations of the
  phases, each carrying a glyph sized to that circle's radius — phases are not
  equal lengths and a plain progress bar would say they were.
* **A kill timeline.** Our kills above the axis, our losses below in the one
  colour reserved for them, the rest of the lobby in grey behind, over the
  teams-alive curve. Click to seek.
* **The block tables**: result against the block average, survival, fights per
  phase, opponents, the three ranked focuses, and what is not measurable.

The page never prints a number the log does not have. Anything marked unknown or
below the confidence threshold is drawn as a dashed `not in log` chip, and the
header states the threshold every figure was read at.

## Phase 2 — team preparation report

```bash
python -m bgmi_scrim report --events m1.events.jsonl m2.events.jsonl m3.events.jsonl \
    --team TX --block "Tuesday block A" --out prep.md \
    --asked-for "rotation timings" "utility usage"
```

Sections: result (placement points, kill points, total, per match and cumulative,
against the block average), survival, fights (kills/deaths per phase, trade rate,
knock conversion, knocks conceded then lost), opponents (who took our lives, in
which phase), three ranked practice focuses each with its single strongest piece
of log evidence and a measurable target, and **not measurable yet**.

An example built from the five measured runs is in
[`docs/example_report.md`](docs/example_report.md).

Rules the report keeps:

* Nothing is derived that the event log does not contain. Metrics needing minimap
  or positional data are listed under "not measurable yet", never estimated.
* Every figure cites match number and video timestamp (`M2 08:14`).
* Records with `state: unknown` or confidence below the threshold are excluded,
  and the exclusion count is printed at the top.
* Placement points are an organiser rule, not footage: the table is data with a
  documented default and is printed in the report.
* A placement the log does not contain is printed as "not in log" and scores no
  points. A death with no killer in the log is excluded from the opponent table
  rather than assigned.

## Testing

```bash
python -m pytest -q                         # logic tests, no video needed
python -m bgmi_scrim synth --out clip.mp4 --labels clip.labels.jsonl --duration 150
```

`synth` renders synthetic spectator-HUD footage with exact labels. It is test
scaffolding: it measures the extractor end to end without hand-labelling, but
numbers measured on it are numbers on synthetic footage and are not a substitute
for measuring on real scrim VODs.

## Layout

```
bgmi_scrim/
  profile.py        calibration profiles (data: ROI boxes, HSV bands, thresholds)
  sampler.py        2 fps baseline + 5 fps change bursts, sequential decode
  preprocess.py     crop / upscale / CLAHE / adaptive + Otsu, blank-ROI gate
  ocr.py            numerals and text with calibrated confidence
  hud.py            per-field detectors (counters, panel, feed, zone, banners)
  statemachine.py   N-consecutive confirmation, monotonic gating, transitions
  killfeed.py       feed parsing, sighting confirmation, de-duplication
  fuse.py           panel authority + feed attribution
  analyze.py        the phase-1 pipeline
  calibrate.py      interactive / template / preview calibrator
  truth.py          ground-truth harness and gate evaluation
  report.py         phase 2
  synth.py          synthetic HUD footage with labels (testing only)
  map_profile.py    map-stream profile (map box, dot/label geometry, bands)
  map_world.py      pixels to metres, POI tables, grid references
  map_dots.py       dot detection, label matching, the team colour book
  map_track.py      tracking and identity votes
  map_zones.py      RANSAC circle fitting and the zone stabiliser
  map_movement.py   landings, loot, rotations, zone relationship
  map_analyze.py    the phase-3 pipeline
  map_calibrate.py  map calibrator, preview and POI picker
  map_truth.py      positional ground-truth harness
  map_synth.py      synthetic map footage with exact positions
  map_report.py     phase-3 report
  dashboard.py      builds the dashboard's data model from the logs
  dashboard_template.html  the page itself (one file, no build step)
  cli.py            analyze | calibrate | truth | synth | report | dashboard
                    mapanalyze | mapcalibrate | maptruth | mapsynth | mapreport
profiles/           bgmi_template_1080p/720p, mapstream_template_1080p
                    (all uncalibrated), synthetic_1080p, synthetic_map_1080p
maps/               POI tables (calibrated per map; grid refs without one)
docs/accuracy.md    phase-1 measured numbers and the gate verdict
docs/map_accuracy.md phase-3 measured numbers
```
