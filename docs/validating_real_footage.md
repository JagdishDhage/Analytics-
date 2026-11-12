# Putting this on a real VOD

Every number this toolchain has reported so far was measured on synthetic
footage. That measures the method; it does not measure your overlay, your
encoder or your lobby. This is the sequence that does, and what to do when a
step fails.

## What you need

| thing | why |
|---|---|
| one HUD recording (spectator view) | phase 1: counters, team list, kill feed |
| the map recording for the same match, if you have it | phase 3: positions, rotations, zones |
| the lobby roster: team tags, and player names as the overlay writes them | labels are matched against this list, so a name that is missing is never read |
| the organiser's points table | placement points are a rule, not something in the footage |
| the map name | it fixes the world size, and so every distance in metres |

A single match is enough to calibrate. Measuring needs labels, and the section
on labelling says how many.

## 1. Let it look at the capture

```bash
python -m bgmi_scrim calibrate vod.mp4 --auto \
    --base profiles/bgmi_template_1080p.json --out profiles/myorg.json --team TX
```

This reads a handful of frames spread through the file and prints what it
found: the picture's position inside the frame (letterboxing and pillarboxing
are common and silently shift every box), the team list and its row pitch, the
two lobby counters, the map name, and the kill feed. It writes a profile with
those boxes and lists the fields it could not propose.

It tells the truth about its own confidence: the profile is written with
`"_calibrated": false`, and `analyze` warns while that flag is false.

For the map stream:

```bash
python -m bgmi_scrim mapcalibrate map.mp4 --auto \
    --base profiles/mapstream_template_1080p.json --out profiles/myorg_map.json \
    --map ERANGEL
```

## 2. Look at what it proposed

```bash
python -m bgmi_scrim calibrate vod.mp4 --preview profiles/myorg.json \
    --preview-png check.png --at 120
```

The preview draws every box on a real frame and prints what each one reads. A
green box must sit on each player pip, an orange box on each team tag, magenta
boxes on the feed rows. `low_confidence_rois` in the output names anything that
read badly.

Fix by hand in the JSON what the proposal missed — typically the zone phase
number, the zone state indicator, the elimination card and the winner banner,
which only appear at certain moments and so are rarely in the sampled frames.
Re-run the preview until the list is empty, then set `"_calibrated": true`.

The map preview also prints `metres_per_pixel` and `two_dots_merge_below_m`,
which tells you the distance below which two team-mates are one dot in your
capture.

## 3. Label a sample

The harness scores whatever you label, so label a window rather than a match:

* **events**: every knock and kill in one continuous 5-minute stretch, which on
  a real scrim is 15–30 events. Fewer than ~20 and a single miss moves recall by
  five points, so the number means little.
* **frames**: the lobby counters at a fixed interval through the same stretch,
  say every 10 seconds, which is 30 labels.
* **scalars**: the map name, your placement, the winner.

```json
{"kind":"event","t":613.4,"type":"knock","victim_team":"TX","victim_slot":2,"killer_team":"GB","killer_slot":1}
{"kind":"frame","t":620.0,"teams_alive":14,"players_alive":47,"zone_phase":4}
{"kind":"scalar","t":5.0,"type":"map_name","value":"ERANGEL"}
```

Timestamps are seconds from the start of the file. Label from the video itself,
not from the tool's output, or the measurement is circular.

## 4. Measure

```bash
python -m bgmi_scrim analyze vod.mp4 --profile profiles/myorg.json \
    --out vod.events.jsonl --content-box auto --team TX
python -m bgmi_scrim truth --labels vod.labels.jsonl --events vod.events.jsonl \
    --md accuracy.md --json accuracy.json
```

`truth` exits 2 when the phase-1 gate is not met. Read the confusion list before
the summary: it names every miss, phantom and duplicate, and one systematic
cause usually explains most of them.

For the map stream, `mapanalyze` then `maptruth` the same way, against position
labels.

## 5. When a step fails

| what you see | what it usually is | what to do |
|---|---|---|
| `analyze` warns the profile is a template | `_calibrated` is still false | finish the preview pass, then set it true |
| every counter reads `unknown` | the box is on the label, not the digits, or the picture is letterboxed | tighten the box in the preview; run with `--content-box auto` |
| counters read, nothing else | the panel proposal missed | check the orange and green boxes in the preview; set `row_pitch` from two rows' pixel distance |
| team tags read as `ROW3` | the roster does not contain that tag, or the tag box clips it | add the tag to `team_tags`; widen `tag_w` |
| knocks found, kills missing | `slot_states` colour bands do not match the overlay's dead colour | sample the HSV of a dead pip and widen that band |
| phantom events in the feed | the feed box overlaps another HUD element | narrow `kill_feed.box`; raise `feed_sightings` to 3 |
| map: no dots | `sat_min`/`val_min` too high for a desaturated overlay, or `map_area` wrong | check the preview; lower the thresholds |
| map: positions jump between team-mates | the squad is inside one dot at your capture's scale | expected; read the team path, not the player path |
| map: zone rings never fit | the ring colour is outside the band, or the ring is dashed | widen the band; raise `tol_px` to the ring's thickness |

## 6. What the numbers will and will not tell you

A good result on one labelled match says the profile fits that overlay and that
encoder. It does not transfer to another organiser's overlay, and it does not
cover what the sampled window never contained: if nobody was knocked in a
vehicle, or no circle closed on a building, those cases are untested.

Re-measure when the overlay changes, when the encoder changes, and when the
lobby size changes. The harness is cheap to re-run; that is the point of it.
