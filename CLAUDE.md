# Working on this repository

Context for a Claude Code session picking this up. The README covers what the
tool does; this covers where the work stands, what is actually measured, and
the rules this codebase is built to.

## What this is

A BGMI scrim analysis worker. It reads HUD pixels and map pixels out of
recorded spectator footage and produces event logs, coach-facing reports, and
a per-team scouting corpus. No game API, no observer client, no network calls.

```
phase 1   HUD video     -> events.jsonl                     (analyze)
phase 2   events.jsonl  -> report.md                         (report)
phase 3   map video     -> tracks.jsonl + mapevents.jsonl    (mapanalyze)
                        -> map_report.md                     (mapreport)
corpus    map video     -> corpus/matches/<id>.json          (ingest)
                        -> scouting.md, profiles.json        (scout)
all       the logs      -> dashboard.html                    (dashboard)
```

## The rules this code is built to

These came from the original brief and every part of the codebase follows
them. Breaking one is a bug even where the output looks better.

1. **Never a guess.** A reading the pipeline is unsure of is emitted as
   `unknown` or flagged below the confidence threshold. A confidently wrong
   value is the one outcome that is never acceptable. Several commits exist
   purely to turn a confident wrong value into a flagged one.
2. **Profiles are data, never hardcoded.** All HUD and map geometry lives in
   `profiles/*.json`. No ROI coordinate belongs in a `.py` file.
3. **Derive nothing the log does not contain.** The reports may not estimate.
   If a metric needs data the extraction does not produce, it goes under "not
   measurable yet" with a reason.
4. **Report the real numbers.** Accuracy claims in `docs/` must come from a
   measurement run, and the commit that changes behaviour records the figure
   it measured. Say when something is unmeasured rather than implying it is.
5. **Every claim cites a match and a timestamp.** The reports are read by a
   coach between scrim blocks, so a figure that cannot be traced back to the
   footage is not useful.

## Measured state

Run with `python -m pytest -q`: 119 tests.

**Phase 1 gate, five synthetic clips pooled** (`docs/accuracy.md`):
precision 1.0000 (1070/1070), recall 1.0000, 0 duplicates, 0 phantoms,
teams-alive 0.9995 (2039/2040). All five clips pass individually.

**Phase 3 map extraction** (`docs/map_accuracy.md`): 7.4 m median position
error, 0.985 team identity, 8 of 8 zone phases, 24 of 25 landings.

**Feed-only map stream** (`docs/feed_only_scenario.md`): a map stream with a
kill feed and no HUD panel reads knocks 1.0000/1.0000 and kills 1.0000/0.9896,
0 duplicates, 0 phantoms, killer correct on 190 of 191.

**All of it is on synthetic footage.** `bgmi_scrim/synth.py` and
`map_synth.py` render the clips and write exact labels. That measures the
method end to end and says nothing about a real VOD: real footage adds the
organiser's overlay art, compression on small digits, camera switches, and a
panel that re-sorts as teams die. `docs/validating_real_footage.md` is the
runbook for getting real numbers.

## Known gaps, honestly

* **Contact detection is unmeasured on footage.** `contacts.py` derives fights
  from positions because a map stream has no kill feed. The synthetic clips
  never bring teams close: across one 420 s clip the nearest two teams came
  was 146 m by centroid and 523 m by nearest player pair, and a four-match
  corpus produced two contacts in total. The logic is driven by constructed
  paths in `tests/test_contacts.py`. **Making `map_synth` produce
  close-quarters fights is the next thing needed to put a number on this.**
* **The synthetic corpus values are not play.** The aggregation is correct but
  the simulated behaviour is not realistic: one team read as a median squad
  spread of 1277 m and 4.5 zone phases never entered. Real squads hold within
  a couple of hundred metres and rotate into the circle. Do not treat any
  synthetic scouting figure as a pattern.
* **No trained model exists.** The corpus deliberately stores facts so a model
  can be trained later without re-reading video. That step is not built.
* **Burst sampling needs a source above the base rate.** Sampling is 2 fps
  raised to 5 fps around a counter change, so a clip rendered at 2 fps has no
  intermediate frames to reach. Measured: 92 change points, 0 burst samples,
  players-alive 0.7667 against 0.9967 at the normal rate. Real captures are
  30/60 fps so it does not arise in practice, but it fails silently.

## Hard-won lessons, do not re-break these

Each of these was measured, and the obvious-looking alternative was worse.

* **`tessedit_char_whitelist` is mishandled by the LSTM engine (`--oem 1`).**
  Passing it welds words into one token and reports confidence 0. Text ROIs
  constrain the alphabet in Python instead. **Numerals keep the engine-side
  whitelist**: filtering digits in Python took players-alive from 0.99 to
  0.7667 with 70 confidently wrong reads, because the engine offers a letter
  where a digit belongs and the filter drops it, turning "44" into "4".
* **A short label and a long line want opposite read rules.** `read_text`
  takes `label_roi`: a panel tag prefers the longer read and retries an empty
  one with the engine constrained; a kill-feed line prefers confidence.
  Ranking by confidence everywhere cost the panel our own team's row;
  ranking by length everywhere cost the feed four phantom events.
* **Only a caller that knows the region holds text may ask for the retry.** A
  feed row that reads as nothing usually *is* nothing, and squeezing
  characters out of it invents events.
* **A trailing `I` or `L` is not a slot number.** OCR renders "1" as either,
  but those letters also end real tags: `SOUL`, `GODL`. Taking the letter as a
  slot invented "slot 1" for every one of those teams' feed lines.
* **Match a feed name as a whole token.** A sliding window finds `ORxBRAVO`
  inside `BLINDxBRAVO` at 0.80 and reported `BLIND#2` as `OR#2`. Whole-token
  matching scores it 0.63 and refuses.
* **A share needs its denominator.** A zone-edge reading of "100%" came from
  4.5 s inside the circle out of 384.5 s tracked, and the report had promoted
  it to a ranked practice focus with a target. `edge_time` returns the time
  inside with the fraction, and under 30 s the share is not reported at all.
* **Elimination-based naming of the last panel row measured worse** and is off
  by default: it put seven of another team's events on our tag.
* **`tag_min_score` stays at 0.60.** Raising it to 0.72 helped compressed
  footage (0.944 to 0.9955) and hurt clean footage (1.0 to 0.9867).

## Before changing extraction behaviour

Re-measure, do not reason. Twice in this project a change that was obviously
correct measured worse, and once a reported regression turned out to be a
faulty reproduction (gate clips rendered at `--fps 2` instead of the
documented default). Freeze the code while a measurement runs: results from a
run whose working tree changed underneath it are worthless.

```bash
# the gate, five clips, seeds from docs/accuracy.md
python -m bgmi_scrim synth --out A.mp4 --labels A.labels.jsonl --duration 150 --seed 7
python -m bgmi_scrim analyze A.mp4 --profile profiles/synthetic_1080p.json --out A.events.jsonl
python -m bgmi_scrim truth --labels A.labels.jsonl --events A.events.jsonl --md A.accuracy.md
```

Seeds: A=7, B=11 (720p, `--noise 0.012`), D=31, E=47 (`--noise 0.008`),
C=23 (420 s, VIKENDI). `truth` exits 2 when the gate is not met. Do **not**
pass `--fps`; the default rate is what the documented numbers used.

## Where the user is

They have a real Erangel map stream and intend to ingest 200 to 400 matches,
one at a time, from one organiser, to build per-team playing patterns. Their
feeds show labelled dots (`teamnamexplayername`) and **no kill feed**.

The immediate task is calibration, which is interactive and has not been done
on their footage yet:

```bash
python -m bgmi_scrim mapcalibrate <video> --out mymap.json --map ERANGEL --auto
# fill in roster: team tag -> player names exactly as the overlay writes them,
# because a name missing from the roster is never read
python -m bgmi_scrim mapcalibrate <video> --preview mymap.json --preview-png check.png
# green ring on every dot, magenta box on each name, red circle on both rings,
# then set "_calibrated": true
python -m bgmi_scrim ingest <video> --corpus ./corpus --match-id m1 --profile mymap.json
python -m bgmi_scrim scout --corpus ./corpus --out scouting.md
```

If ingest reports few teams, the labels or roster are wrong. Zero contacts
means teams never closed or identity was too weak to pair them. Zero zone
phases means the circle colour bands are wrong. Check on match three, not
match three hundred.

Cost measured here: a 420 s stream takes about 98 s to extract on four cores,
so a 25 to 30 minute match is six to seven minutes and 300 matches is 30 to 35
hours of processing, once.

## Conventions

* Commit messages explain why, and quote the figure that justified the change.
  No model identifiers in commits or code.
* `docs/` holds measured results; update the relevant file in the same commit
  that changes a number, and name the commit the figure came from.
* Branch: `claude/bgmi-scrim-analysis-worker-8kyvh7`.
