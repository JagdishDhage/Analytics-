# Run A - 1080p synthetic, 150 s, dense

Event match tolerance: ±2.50s; frame-field transition grace: ±0.75s

## Gate

| check | result |
|---|---|
| kill_knock_precision>=0.95 | PASS |
| kill_knock_recall>=0.95 | PASS |
| no_duplicate_events | PASS |
| no_phantom_kills | PASS |
| teams_alive_frames>=0.99 | PASS |
| every_miss_accounted | PASS |

**Gate: PASSED**

## Events

| type | truth | predicted | matched | precision | recall | F1 | dup | phantom | miss | mean dt |
|---|---|---|---|---|---|---|---|---|---|---|
| knock | 124 | 124 | 124 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.07s |
| kill | 99 | 99 | 99 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.06s |

| type | killer checked | killer correct | killer null | killer wrong |
|---|---|---|---|---|
| knock | 124 | 100 | 23 | 1 |
| kill | 99 | 85 | 13 | 1 |

## Frame fields

| field | labels | correct | accuracy | wrong (confident) | wrong (flagged low-confidence) |
|---|---|---|---|---|---|
| teams_alive | 300 | 300 | 1.0000 | 0 | 0 |
| players_alive | 300 | 299 | 0.9967 | 1 | 0 |
| zone_phase | 300 | 300 | 1.0000 | 0 | 0 |
| zone_state | 300 | 300 | 1.0000 | 0 | 0 |
| team_alive_count | 7500 | 7500 | 1.0000 | 0 | 0 |

Strict (no transition grace):

| field | labels | correct | accuracy |
|---|---|---|---|
| teams_alive | 300 | 300 | 1.0000 |
| players_alive | 300 | 299 | 0.9967 |
| zone_phase | 300 | 300 | 1.0000 |
| zone_state | 300 | 300 | 1.0000 |

## Scalars

| type | labels | correct | values read |
|---|---|---|---|
| map_name | 1 | 1 | ['ERANGEL'] |
| placement | 1 | 1 | [5] |
| winner | 1 | 1 | ['VST'] |

## Confusion list (3 entries)

| t | kind | detail |
|---|---|---|
| 16.50 | frame_field | field=players_alive, truth=95, read=96, confidence=0.961, state=ok, flagged=False |
| 119.90 | killer_attribution | type=kill, victim=VLT#1, truth_killer=BLIND#4, read_killer=REV#4, confidence=0.9214 |
| 122.50 | killer_attribution | type=knock, victim=VST#2, truth_killer=VLT#4, read_killer=IQOO#1, confidence=0.95 |

