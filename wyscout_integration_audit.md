# Dashboard + Wyscout audit summary

## Project structure
- Entry point: `app.py`
- Event loader: `data_loader.py`
- Core metric layer: `metrics.py`
- Plot layer: `plots.py`
- Report/export layer: `report_generator.py`
- New Wyscout loader: `wyscout_loader.py`
- New dashboard helpers: `dashboard_helpers.py`

## Real event sources used
- matchdetails_eng.csv
- passes_eng.csv
- ball_receipts_eng.csv
- carrys_eng.csv
- shots_eng.csv
- dribbles_eng.csv
- ball_recoverys_eng.csv
- dispossesseds_eng.csv
- miscontrols_eng.csv
- foul_wons_eng.csv

## Uploaded Wyscout Excel files used
- Team Stats England attacking.xlsx
- Team Stats England general.xlsx
- Team Stats England indexes.xlsx
- Team Stats England passing.xlsx

## Wyscout tables added to the dashboard
- Build-up context: long pass %, average passes per possession, average pass length, goal kicks, accurate progressive passes
- Final Third context: passes to final third, smart passes, crosses, and their accurate totals
- Chance Creation context: team xG, shots, shots on target, positional attacks, counterattacks, corners, free kicks, and cross accuracy
- Final Report context: match-level Wyscout references are merged by match date into the report bundle

## Current dependency status
- openpyxl: available
- xlsxwriter: available
- Streamlit runtime in this container: unavailable

## Key page fixes
- Duplicate Best Chances table removed
- Best Chances now has one analyst-facing explanation block and one table
- Final Third language uses Z14 instead of central pocket
- Shot map now colours by shot result
- Shot assist map redesigned with assist-type arrows and shot endpoints
- Build-up passing lanes redesigned around route families instead of raw-line clutter
- Player tables now render from the event-derived bundle and export safely
