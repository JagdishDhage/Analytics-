# The scouting corpus: many matches, every team

The preparation report answers "how did my team play in this match". This
answers "how does every team play", across as many matches as you feed it,
from map streams alone.

One match goes in at a time. `ingest` reads the stream, derives what every
team in it did, and appends that to the corpus as facts. The video is never
needed again. `scout` aggregates the whole corpus into one profile per team.

## Using it

```bash
# once per organiser overlay: build a map profile
bgmi-scrim mapcalibrate match1.mp4

# then one match at a time, 200-400 times
bgmi-scrim ingest match1.mp4 --corpus ./corpus --match-id 2026-04-12-m1 \
    --profile mymap.json --played-at 2026-04-12
bgmi-scrim ingest match2.mp4 --corpus ./corpus --match-id 2026-04-12-m2 \
    --profile mymap.json --played-at 2026-04-12

# whenever you want the picture so far
bgmi-scrim scout --corpus ./corpus --out scouting.md
bgmi-scrim scout --corpus ./corpus --team SOUL GODL --out rivals.md
bgmi-scrim scout --corpus ./corpus --json profiles.json   # for a model later
```

`ingest` without a video re-ingests tracks you already extracted:

```bash
bgmi-scrim ingest --corpus ./corpus --match-id m1 \
    --tracks m1.tracks.jsonl --map-events m1.mapevents.jsonl
```

## Corpus layout

```
corpus/
  matches/<match-id>.json     per-match facts for every team  (a few hundred KB)
  extracted/<match-id>.*      tracks and map events from ingest (~2 MB per match)
```

Only `matches/` is read by `scout`. `extracted/` can be deleted once a match
is ingested; keep it if you may want to re-derive facts without re-reading the
video, which is worth it while the derivation is still changing.

## What a profile holds

Per team, aggregated over every match it appears in:

| group | figures |
|---|---|
| drop | landing places with their frequency, median landing time, looting time where they landed |
| rotation | rotations per match, median delay after each circle is announced, share already moving before it, vehicle share, distance per match, arrivals after the circle closed |
| zone | time outside the blue, share of time inside spent on the edge with the time it is a share of, phases never entered, squad spread |
| fights | contacts per match, engagement distance, share initiated, share received, vehicle approach, won/lost/even record, losses taken and dealt |
| survival | median and best placement, elimination time, players lost |

A team seen in fewer than three matches is marked thin, and its figures are
not to be read as tendencies.

## Fights without a kill feed

A map stream showing dots and labels never says who killed whom. What it says
is where every team was and when a dot stopped existing, so the vocabulary
here is deliberately weaker than a kill feed's:

* **a player lost** is a dot that stopped appearing well before the stream
  ended. Usually that is death; sometimes it is the tracker losing a dot. It
  carries a confidence and is never called a kill.
* **a contact** is two teams within 120 m, with what each side lost during it
  and for 25 s after. It is not an announced event.
* **the initiator** is whichever side closed more ground in the 30 s before
  contact. That is not who shot first, which a map cannot show.
* **placement** is the order in which teams stopped appearing. A team still on
  the map when the stream stops is recorded as surviving the stream, which is
  not the same as placing.

## What this has and has not been measured on

The extraction underneath is measured in `map_accuracy.md`: 7.4 m median
position error, 0.985 team identity, 8 of 8 zone phases on synthetic footage.

**Contact detection is not measured on footage.** In the rendered map clips
teams barely come near each other: across one 420 s clip the closest two teams
ever came was 146 m by centroid and 523 m by nearest player pair, so no
contact opened at all. Over a four-match corpus, two contacts were derived in
total. The logic is instead driven by constructed paths in
`tests/test_contacts.py`, which proves the derivation and says nothing about
real footage.

**The synthetic footage is a poor model of how teams actually play**, so the
aggregate values it produces are not meaningful as patterns, even though the
aggregation of them is correct. On a four-match synthetic corpus one team read
as a median squad spread of 1277 m, 4.5 zone phases never entered and 214 s
outside the blue. Real squads hold together within a couple of hundred metres
and rotate into the circle. What the synthetic corpus demonstrates is that the
machinery runs and the arithmetic is right, nothing about play.

To get figures that mean something, ingest your own matches. To know how
accurate they are, label one map stream in the format in
`bgmi_scrim/map_truth.py` and run `maptruth` on it.

## Before committing to hundreds of matches

Measured here: a 420 s map stream takes about 98 s to extract on four cores.
A 25 to 30 minute match is therefore roughly six to seven minutes, so 300
matches is 30 to 35 hours of single-stream processing, or eight to twelve
hours wall clock if several run in parallel on a desktop. It is a one-off:
the corpus keeps the facts and the video is never read again.

Two things to check on the first two or three matches rather than the three
hundredth:

1. **The profile still fits.** An organiser changing their overlay mid-season
   makes every later match read badly and quietly. `mapcalibrate --preview` on
   a frame from each new batch is the cheap guard, and an ingest reporting
   zero contacts and few identified teams is the symptom.
2. **Team tags are consistent.** Aggregation keys on the tag read from the dot
   labels, so `SOUL` in match 7 and match 312 only combine if the overlay
   writes them identically. Decide too whether a roster change should split a
   team's history, because this does not: it keeps one profile per tag.
