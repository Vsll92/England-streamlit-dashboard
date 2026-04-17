
# Data audit summary

## 1. File names

The uploaded package contains:

- `SB_Euro24_matchdetails.xlsx`
- `SB_Euro24_allevents.xlsx`
- event-specific workbooks under `Events/`, including:
  - passes
  - ball receipts
  - carries
  - shots
  - dribbles
  - recoveries
  - dispossessions
  - miscontrols
  - fouls won
  - many additional defensive / contextual event files

The dashboard operational extract uses the England-only CSV slices generated from the uploaded source files.

## 2. Detected tables / files

The main operational tables used are:

- match details
- passes
- ball receipts
- carries
- shots
- dribbles
- recoveries
- dispossessions
- miscontrols
- fouls won

## 3. Key columns

### Match details
- `match_id`
- `match_date`
- `competition`
- `competition_stage`
- `home_team`
- `away_team`
- scores
- venue / referee metadata

### Passes
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
- `location`
- `pass_end_location`
- `pass_length`
- `pass_recipient`
- `pass_cross`
- `pass_cut_back`
- `pass_through_ball`
- `pass_switch`
- `pass_outcome`

### Ball receipts
- `match_id`
- `index`
- `possession`
- `player`
- `position`
- `location`
- `ball_receipt_outcome`

### Carries
- `match_id`
- `index`
- `possession`
- `player`
- `position`
- `location`
- `carry_end_location`

### Shots
- `match_id`
- `index`
- `possession`
- `player`
- `position`
- `location`
- `shot_end_location`
- `shot_outcome`
- `shot_statsbomb_xg`
- `shot_key_pass_id`

## 4. Missing values

Key practical findings from the England subset:

- pass start and end coordinates are complete for the extracted pass table
- shot coordinates and shot xG are complete for the extracted shot table
- `pass_recipient` has a small missing share
- `pass_outcome` is missing for most passes because successful StatsBomb passes are recorded with null outcome
- cross / cutback / through-ball flags are mostly null because they are sparse event flags
- `shot_key_pass_id` is missing for non-assisted shots

## 5. Coordinate system

The event locations are in a **120 x 80** StatsBomb-style coordinate system.

For the dashboard:
- x runs from own goal to opponent goal
- y is standardized to a football-analysis left-to-right attacking view

## 6. Possessions

Possession IDs already exist in the uploaded source data, so possessions do **not** need to be fully reconstructed from scratch.

## 7. xG, xGOT, xG Chain

- `shot_statsbomb_xg` exists in the shot data
- `xGOT` is not present in the uploaded source files
- `xG Chain` is not present as a native field and is derived only at player level in this dashboard
