# BGMI scrim match-analysis worker

Reads Indian-scrim BGMI spectator-view footage (squads TPP, 4-man teams, up to 25
teams, 720p/1080p OBS mp4) and extracts a timestamped event log from the HUD
pixels. No game API, no observer client, no network calls — everything comes off
the frames. Phase 2 turns those event logs into a coach-facing preparation
report and derives nothing the log does not contain.

```
phase 1   video  --->  events.jsonl        (analyze)
phase 2   events.jsonl ---> report.md      (report)
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

See [`docs/accuracy.md`](docs/accuracy.md) for the real numbers, the gate verdict,
and exactly what footage they were measured on.

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
  cli.py            analyze | calibrate | truth | synth | report
profiles/           bgmi_template_1080p/720p (uncalibrated), synthetic_1080p
docs/accuracy.md    measured numbers and the gate verdict
```
