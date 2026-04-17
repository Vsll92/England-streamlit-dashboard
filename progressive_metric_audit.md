# Progressive-metric audit for England Euro 2024 dashboard

## Root cause of the `match_date` error
The chance-chain view reduced a possession chain to event-only columns and then later UI/report code expected `match_date` to still be present. Once the event slice dropped metadata, any downstream `chain["match_date"]` access could fail.

### Fix applied
- Reattached safe metadata (`match_date`, `match_date_label`, `opponent`, `competition_stage`, `match_label`) to each stored chance chain.
- Added safe date labels so display code can fall back to `Unknown Date` instead of crashing.
- Updated the chance page to render best-chance tables from the new metadata-safe table rather than assuming raw chains still carry every display field.

## Why the old progressive-pass total looked suspicious
The earlier rule counted a completed pass as progressive whenever `end_x - start_x >= 10`.
That is too loose for this sample because it:
- counted many routine forward passes in early phases,
- did not separate attempts from completed progressive passes,
- did not expose set-piece/open-play handling clearly,
- did not provide any match-level validation view.

## Corrected progressive-pass definition
The updated dashboard now uses **open-play pass attempts only** and a more conservative progressive rule on the 120x80 pitch:
- start and end both in own half: gain **30m or more**
- start in own half and end in attacking half: gain **15m or more**
- start and end in attacking half: gain **10m or more**

Headline KPI = **successful completed progressive passes**.
Validation tables also show attempts and unsuccessful attempts.

## Validation results on the current England sample
- Distinct England pass events used: **4540**
- Distinct event ids after deduplication: **4540**
- Selected matches: **7**
- Progressive-pass attempts: **684**
- Successful progressive passes: **472**
- Unsuccessful progressive passes: **212**
- Average progressive-pass attempts per match: **97.7**
- Team leader by progressive-pass attempts: **Kyle Walker**

Other validated progression metrics:
- Progressive carries: **174**
- Passes into Z3: **346 attempts**, **243 successful**
- Carries into Z3: **105**
- Carries into box: **27**

## Match-level progressive-pass breakdown
- Serbia: **87 attempts / 53 completed**
- Denmark: **71 attempts / 45 completed**
- Slovenia: **123 attempts / 95 completed**
- Slovakia: **134 attempts / 87 completed**
- Switzerland: **135 attempts / 103 completed**
- Netherlands: **81 attempts / 61 completed**
- Spain: **53 attempts / 28 completed**

## Technical changes applied
- Added period filtering to exclude shootout period 5
- Added event-id deduplication before aggregation
- Added progression validation summary tables
- Added match-level and player-level progressive-pass breakdowns
- Updated overview chart to show successful vs unsuccessful actions with explicit definitions in hover text
