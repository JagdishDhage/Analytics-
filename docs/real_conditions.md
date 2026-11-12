# Real-capture conditions

Measured at commit `663b6ae`. Every earlier number in this repository came
from footage rendered exactly as the profile expected it. A real VOD is
compressed, often rescaled, and often letterboxed, so these runs take the same
clips and the same labels through an encoder first. They are still synthetic
footage: what they test is the pipeline's tolerance of what a capture does to
it, not an organiser's overlay.

The fixtures come from one 150 s HUD clip and one 420 s map clip:

| variant | ffmpeg |
|---|---|
| heavy compression | `-crf 34 -preset veryfast` |
| rescaled | `scale=1600:900`, `scale=1280:720` |
| letterboxed | `scale=1824:1026,pad=1920:1080:48:27` |

The letterboxed runs use `--content-box auto`, which finds the picture inside
the bars; without it a profile reads the wrong pixels and reports nothing.

## Phase 1, the HUD stream

| capture | events | precision | recall | dup | phantom | miss | teams-alive | players-alive |
|---|---|---|---|---|---|---|---|---|
| clean 1080p (reference) | 1070 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.9995 | 0.9936 |
| heavy compression, crf 34 | 223 | 0.9563 | 0.9821 | 0 | 10 | 4 | 0.9667 | 0.9400 |
| rescaled to 1600x900 | 223 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.9900 | 0.9967 |
| rescaled to 1280x720 | 223 | 0.9955 | 1.0000 | 0 | 1 | 0 | 0.9900 | 0.9767 |
| letterboxed into 1920x1080 | 223 | 1.0000 | 0.9955 | 0 | 0 | 1 | 1.0000 | 0.9967 |

## Phase 3, the map stream

| capture | position median | p90 | team identity | dot coverage | zone phases |
|---|---|---|---|---|---|
| heavy compression, crf 34 | 7.72 m | 20.87 m | 0.9702 | 0.88 | 8 → 8 |
| rescaled to 1280x720 | 11.79 m | 17.5 m | 0.9785 | 0.8818 | 8 → 8 |
| letterboxed into 1920x1080 | 8.55 m | 15.39 m | 0.9798 | 0.8733 | 8 → 8 |
| clean 1080p (regression check) | 7.32 m | 26.09 m | 0.9858 | 0.8654 | 8 → 8 |

## What broke, and why

Five defects, every one of them the same shape: something named a thing it
could not actually identify, and nothing downstream questioned it.

1. **One unreadable team tag cost a whole team's events.** Under compression a
   panel row failed to resolve and another row took the wrong tag, so one
   team's events were filed under a row number while its real events counted
   as missing: 18 of 19 phantoms and all 10 misses in one run. A row now needs
   a minimum vote weight and a majority share before it can claim a tag, and
   the share is judged against the tags still unclaimed.
2. **The panel and the feed each reported the same events separately.** Where a
   row had no name, the panel emitted the event under a row number and the feed
   emitted it again under the real name. The feed now supplies the victim's
   name to the panel's event instead, and the pair is emitted once.
3. **A two-letter team tag absorbed longer ones.** At 720p the map stream put
   1564 samples on `OR` and 300 on `AUT` that belonged to other teams. Both the
   label match and the tag match now need a margin over the runner-up.
4. **Zone-circle thresholds were absolute pixel counts.** A ring at 720p holds
   about two thirds of the points it holds at 1080p, so fixed thresholds
   rejected good fits and lost two of eight phases. They scale with the
   capture now, and the point floor follows the smallest circle a band allows,
   which is what keeps the final 66 m circle.
5. **A feed line named an event about a different player.** On the compressed
   capture, three events the panel could not name took the name of whichever
   feed line was nearest in time: a knock on slot 4 became NOVA#2, a kill on
   slot 4 became XSP#2, a knock on slot 3 became MARC#4. The panel knows which
   pip went down even when it cannot read the tag, and that slot number was
   being thrown away. A feed line may now name an anonymous panel event only
   when it is about the same slot.
6. **A garbled tag can claim a team on a weak similarity.** One row on the
   compressed capture matched TX at 0.6 and took the tag, filing ten of
   another team's events under it. Raising the panel's `tag_min_score` to 0.72
   takes that capture from 0.9440 precision to 0.9955 — and costs the clean
   capture, which falls from 1.0000 to 0.9867 because a row that reads
   perfectly well is refused. One threshold cannot serve both, so the shipped
   default stays at 0.60, which is right for a capture you control, and the
   knob is documented for sources that are heavily compressed.
7. **A circle that grew was believed because it repeated.** A fit jumped to
   2822 m between the real 115 m and 66 m circles and invented a ninth phase.
   The zone only shrinks inside a match; both the stabiliser and the phase
   detector enforce that now.

## A knob worth knowing about

`team_panel.tag_min_score` is how sure the panel has to be of a team tag before
a row claims it. Measured on the same clip at both settings:

| capture | 0.60 (default) | 0.72 |
|---|---|---|
| clean 1080p | precision 1.0000, recall 1.0000 | precision 0.9867, recall 0.9955 |
| heavy compression | precision 0.9563, recall 0.9821 | precision 0.9955, recall 0.9910 |

Raise it for a source that has been through a hard encoder; leave it alone for
a capture you control.

## What these runs still do not test

An encoder is not an overlay. These fixtures have the same HUD art, the same
fonts, the same colours and the same lobby as the clean runs — only the pixels
that reach the decoder differ. A real organiser's overlay will differ in all of
those, which is what `docs/validating_real_footage.md` is for, and what a
labelled VOD would settle.

