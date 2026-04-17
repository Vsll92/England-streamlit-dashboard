# England Euro 2024 Dashboard — Fixed & Upgraded

## Bug Fixes Applied

### 1. KeyError: 'match_id' in best-chance chain selection (FIXED)
**Root cause:** `metrics.py` line 675 built chain DataFrames from `grp[["minute", "type", "player", ...]]` — the column list never included `match_id`. Meanwhile, `app.py` line 672 called `int(c["match_id"].iloc[0])`, causing the crash.

**Fix:** Added `chain["match_id"] = int(row["match_id"])` in `metrics.py` when constructing each chain. Also added `safe_chain_lookup()` in `app.py` that handles missing columns gracefully with a try/except fallback.

### 2. Short build-up and Long/direct build-up showing 0.0% (FIXED)
**Root cause:** The `build_split` Series was computed at line 618 of `metrics.py` correctly (e.g. Short=51%, Long=49%), but **was never inserted into the `overview` dict**. So `bundle['overview'].get('build_short_pct', 0)` always returned 0.

**Fix:** Moved the `build_split` computation *before* the overview dict construction, then added `"build_short_pct"` and `"build_long_pct"` keys directly into the overview dict.

**Verified values:** Short build-up = 51.0%, Long/direct = 49.0% (361 build-up possessions out of 660 total).

### 3. Build-up passing lanes map redesigned
**Before:** Zone-to-zone arrows with cluttered mid-point white dots, uniform blue coloring, and no visual hierarchy.

**After:** Redesigned `grouped_route_map()` in `plots.py`:
- Arrow width proportional to attempt volume (1.5–5.5px)
- Three-tier colour ramp: blue (≥70% completion), green (45–70%), amber (<45%)
- Volume counts shown as subtle text at midpoints instead of white dots
- Legend entries explaining the colour coding
- Top 10 routes only (reduced from 12)

### 4. Overlapping filters and titles fixed
**Changes across `app.py`:**
- All filter controls placed in dedicated `st.columns()` rows *above* charts
- Added `st.markdown("")` spacers between KPI rows and chart rows
- Increased chart `margin.t` from 58→68px for title breathing room
- Tactical summary notes repositioned to bottom of pitch (y=-2 descending) instead of overlapping the top
- Expanded pitch y-axis range to `[-22, 84]` to accommodate bottom notes
- Plotly modebar set to vertical orientation to avoid title collision
- Legend positioned at `y=1.02` to sit above chart area

## UI/UX Improvements

- **CSS overhaul:** Tighter metric cards (14px border-radius, 100px min-height), better typography hierarchy (0.82rem labels, 1.55rem values)
- **Title formatting:** All chart titles use `<b>` bold with `<span>` subtitles in muted `#94a3b8`
- **Section notes:** Added explicit build-up classification definition visible to the analyst
- **Build-up tab:** Clear explanation of Short vs Long/Direct rules in the section header
- **Audit tab:** Added build-up metric validation box showing the exact classification rules and counts
- **Tab labels shortened:** "Build-up Phase" → cleaner tab bar
- **Error handling:** `safe_chain_lookup()` prevents raw tracebacks; metric card values handle None/NaN

## Metric Definitions (re-validated)

| Metric | Definition | Verified Value |
|--------|-----------|----------------|
| Build-up possession | Starts at x<40 OR by GK/CB below x=60 | 361 of 660 |
| Short build-up | First 3 passes all <30m AND none reaches x≥70 | 51.0% |
| Long/direct build-up | Any of first 3 passes ≥30m OR reaches x≥70 | 49.0% |
| Progressive pass | Zone-based x-gain thresholds on 120×80 pitch | Validated |
| Z3 entry | Completed pass/carry from x<80 to x≥80 | Validated |

## Files Changed

| File | Changes |
|------|---------|
| `metrics.py` | Added build_short_pct/build_long_pct to overview dict; added match_id to chains |
| `app.py` | Complete rewrite: layout fixes, safe_chain_lookup, CSS overhaul, spacing, error handling |
| `plots.py` | Redesigned grouped_route_map, improved all chart margins/titles/legends |
| `config.py` | Minor title improvement |

## How to Run

```bash
pip install streamlit pandas numpy plotly reportlab openpyxl xlsxwriter
cd england_offensive_dashboard
streamlit run app.py
```
