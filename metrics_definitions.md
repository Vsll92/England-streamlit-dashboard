
# Metrics definitions

## Data model used

The dashboard works from the real uploaded England event subset built from these files:

- passes
- ball receipts
- carries
- shots
- dribbles
- ball recoveries
- dispossessions
- miscontrols
- fouls won
- match details

## Core field usage

### Possession
The uploaded data already contains a `possession` field, so possessions are grouped by:

- `match_id`
- `possession`

### Offensive event table
A unified England event table is built from the source files with these common fields:

- `match_id`
- `period`
- `minute`
- `second`
- `index`
- `possession`
- `play_pattern`
- `team`
- `player`
- `position`
- start location
- end location when available
- event subtype
- success flag
- xG for shots

## Pitch zones

### Final third (Z3)
A touch or action is tagged as final-third if:

- `x >= 80`

### Box
A touch or action is tagged as box involvement if:

- `x >= 102`
- `18 <= y <= 62`

### Zone 14
A touch or action is tagged as zone 14 if:

- `80 <= x < 102`
- `30 <= y <= 50`

### Lanes
The pitch width is split into three equal channels:

- Right
- Center
- Left

## Build-up logic

A possession is tagged as a build-up possession when:

- the first England action starts in the defensive third (`x < 40`), or
- the first England action is by a goalkeeper / centre-back in England’s own half

### Short vs long build-up
Build-up is classified as:

- **Long** if the first three passes contain a long pass (`pass_length >= 30`) or reach `x >= 70` immediately
- **Short** otherwise

### Build-up direction
The direction split uses the **first progressive pass or carry** in the build-up possession, classified by its destination lane.

## Final-third entry logic

A final-third entry is counted when:

- a pass starts outside the final third and ends inside it, or
- a carry starts outside the final third and ends inside it

### Entry types
- **Short Pass**: pass entry with no cross flag and no long-pass flag
- **Direct/Deep Pass**: pass entry with `pass_length >= 30`
- **Cross-type Entry**: pass entry flagged as a cross
- **Carry**: carry entry

### Post-entry outcomes
After the **first Z3 entry** in a possession, the dashboard checks whether England:

- retained possession
- entered the box
- created a shot
- moved into zone 14
- moved wide
- lost possession

## Chance creation logic

### Shot-ending possessions
A possession is shot-ending when it contains at least one shot event.

### xG
Uses `shot_statsbomb_xg` directly from the shot data.

### xGOT
Unavailable in the uploaded source files.

### Chance source classification
Shot source is classified in this order:

1. Set Piece (`From Corner`, `From Free Kick`, `From Throw In`)
2. Cross
3. Cutback
4. Through Ball
5. Frontal Pass
6. Carry/Dribble
7. High Regain
8. Second Ball / Rebound
9. Other

## Player metrics

### xA
Derived by linking `shot_key_pass_id` back to pass `id` and summing the shot xG created by that pass.

### xG Chain
Not present as a source field. Derived at **player level** by assigning each player the total xG generated in possessions they were involved in.

### Minutes
Displayed as `minutes_est`, estimated from the first and last event minute for each player in each match in the operational extract.

### Retention rate after reception
For each receipt, the next same-player action in the same possession is inspected.
Retention is counted when that next action is **not** an immediate turnover.

## Unavailable fields
These were requested but are not natively available in the uploaded source data:

- xGOT
- native xG Chain field
- native xT field

Where possible, a transparent derived replacement was used. Otherwise the metric is marked unavailable.
