# Run C - 1080p synthetic, 420 s, realistic event spacing

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
| knock | 113 | 113 | 113 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.12s |
| kill | 97 | 97 | 97 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.07s |

| type | killer checked | killer correct | killer null | killer wrong |
|---|---|---|---|---|
| knock | 113 | 98 | 15 | 0 |
| kill | 97 | 80 | 16 | 1 |

## Frame fields

| field | labels | correct | accuracy | wrong (confident) | wrong (flagged low-confidence) |
|---|---|---|---|---|---|
| teams_alive | 840 | 840 | 1.0000 | 0 | 0 |
| players_alive | 840 | 840 | 1.0000 | 0 | 0 |
| zone_phase | 840 | 840 | 1.0000 | 0 | 0 |
| zone_state | 840 | 840 | 1.0000 | 0 | 0 |
| team_alive_count | 21000 | 21000 | 1.0000 | 0 | 0 |

Strict (no transition grace):

| field | labels | correct | accuracy |
|---|---|---|---|
| teams_alive | 840 | 840 | 1.0000 |
| players_alive | 840 | 840 | 1.0000 |
| zone_phase | 840 | 840 | 1.0000 |
| zone_state | 840 | 840 | 1.0000 |

## Scalars

| type | labels | correct | values read |
|---|---|---|---|
| map_name | 1 | 1 | ['VIKENDI'] |
| placement | 1 | 1 | [6] |
| winner | 1 | 1 | ['GODL'] |

## Confusion list (1 entries)

| t | kind | detail |
|---|---|---|
| 298.40 | killer_attribution | type=kill, victim=HYD#4, truth_killer=GODL#3, read_killer=GODL#1, confidence=0.95 |

