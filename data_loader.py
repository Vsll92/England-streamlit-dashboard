from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    AWAY_SCORE_CANDIDATES,
    AWAY_TEAM_CANDIDATES,
    COMPETITION_CANDIDATES,
    DATA_DIR,
    DATE_CANDIDATES,
    HOME_SCORE_CANDIDATES,
    HOME_TEAM_CANDIDATES,
    PITCH_WIDTH,
    STAGE_CANDIDATES,
    TEAM_NAME,
)
from schema_utils import get_series, parse_location, safe_get_column

FILE_MAP = {
    "matchdetails": "matchdetails_eng.csv",
    "passes": "passes_eng.csv",
    "receipts": "ball_receipts_eng.csv",
    "carries": "carrys_eng.csv",
    "shots": "shots_eng.csv",
    "dribbles": "dribbles_eng.csv",
    "recoveries": "ball_recoverys_eng.csv",
    "dispossessed": "dispossesseds_eng.csv",
    "miscontrols": "miscontrols_eng.csv",
    "foul_wons": "foul_wons_eng.csv",
}


def load_raw_tables(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    data_dir = data_dir or DATA_DIR
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in FILE_MAP.items():
        path = data_dir / filename
        if path.exists():
            tables[name] = pd.read_csv(path)
        else:
            tables[name] = pd.DataFrame()
    return tables


def prepare_match_metadata(match_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    if match_df.empty:
        out = pd.DataFrame(columns=["match_id", "match_date", "competition", "competition_stage", "home_team", "away_team", "home_score", "away_score", "opponent", "match_label"])
        meta_map = {"date_col": None, "competition_col": None, "stage_col": None}
        return out, meta_map

    out = match_df.copy()
    if "Unnamed: 0" in out.columns:
        out = out.drop(columns=["Unnamed: 0"])

    date_col = safe_get_column(out, DATE_CANDIDATES)
    comp_col = safe_get_column(out, COMPETITION_CANDIDATES)
    stage_col = safe_get_column(out, STAGE_CANDIDATES)
    home_col = safe_get_column(out, HOME_TEAM_CANDIDATES, "home_team")
    away_col = safe_get_column(out, AWAY_TEAM_CANDIDATES, "away_team")
    home_score_col = safe_get_column(out, HOME_SCORE_CANDIDATES)
    away_score_col = safe_get_column(out, AWAY_SCORE_CANDIDATES)

    out["match_date"] = get_series(out, [date_col] if date_col else [], default=pd.NaT, dtype="date")
    out["competition"] = get_series(out, [comp_col] if comp_col else [], default="Unknown Competition").fillna("Unknown Competition")
    out["competition_stage"] = get_series(out, [stage_col] if stage_col else [], default="Unknown Stage").fillna("Unknown Stage")
    out["home_team"] = get_series(out, [home_col] if home_col else [], default="Unknown Home").fillna("Unknown Home")
    out["away_team"] = get_series(out, [away_col] if away_col else [], default="Unknown Away").fillna("Unknown Away")
    out["home_score"] = get_series(out, [home_score_col] if home_score_col else [], default=np.nan, dtype="numeric")
    out["away_score"] = get_series(out, [away_score_col] if away_score_col else [], default=np.nan, dtype="numeric")

    if "match_id" not in out.columns:
        out["match_id"] = np.arange(1, len(out) + 1)

    out["opponent"] = np.where(out["home_team"].eq(TEAM_NAME), out["away_team"], out["home_team"])
    out["match_label"] = (
        out["match_date"].dt.strftime("%Y-%m-%d").fillna("Unknown Date")
        + " | "
        + TEAM_NAME
        + " vs "
        + out["opponent"].fillna("Unknown Opponent")
        + " | "
        + out["competition_stage"].fillna("Unknown Stage")
    )

    meta_map = {"date_col": date_col, "competition_col": comp_col, "stage_col": stage_col}
    keep = ["match_id", "match_date", "competition", "competition_stage", "home_team", "away_team", "home_score", "away_score", "opponent", "match_label"]
    return out[keep].drop_duplicates("match_id"), meta_map


def attach_coordinates(df: pd.DataFrame, loc_col: str | None, end_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if "Unnamed: 0" in out.columns:
        out = out.drop(columns=["Unnamed: 0"])

    if loc_col and loc_col in out.columns:
        coords = out[loc_col].map(parse_location)
        out["location_x"] = [x for x, _ in coords]
        out["location_y"] = [y for _, y in coords]
        out["y_std"] = PITCH_WIDTH - out["location_y"]
    else:
        out["location_x"] = np.nan
        out["location_y"] = np.nan
        out["y_std"] = np.nan

    if end_col and end_col in out.columns:
        end_coords = out[end_col].map(parse_location)
        out[f"{end_col}_x"] = [x for x, _ in end_coords]
        out[f"{end_col}_y"] = [y for _, y in end_coords]
        out["end_y_std"] = PITCH_WIDTH - out[f"{end_col}_y"]
    else:
        if end_col:
            out[f"{end_col}_x"] = np.nan
            out[f"{end_col}_y"] = np.nan
        out["end_y_std"] = np.nan
    return out


def enrich_event_tables(raw: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, str | None]]:
    match_meta, meta_map = prepare_match_metadata(raw.get("matchdetails", pd.DataFrame()))
    event_tables: dict[str, pd.DataFrame] = {}

    coord_map = {
        "passes": ("location", "pass_end_location"),
        "receipts": ("location", None),
        "carries": ("location", "carry_end_location"),
        "shots": ("location", "shot_end_location"),
        "dribbles": ("location", None),
        "recoveries": ("location", None),
        "dispossessed": ("location", None),
        "miscontrols": ("location", None),
        "foul_wons": ("location", None),
    }

    for name, df in raw.items():
        if name == "matchdetails":
            continue
        loc_col, end_col = coord_map.get(name, (None, None))
        out = attach_coordinates(df, loc_col, end_col)
        if not match_meta.empty and "match_id" in out.columns:
            out = out.merge(match_meta, on="match_id", how="left")
        event_tables[name] = out

    return event_tables, match_meta, meta_map


def load_project_data(data_dir: Path | None = None) -> dict[str, Any]:
    raw = load_raw_tables(data_dir=data_dir)
    tables, match_meta, meta_map = enrich_event_tables(raw)
    return {
        "raw": raw,
        "tables": tables,
        "match_meta": match_meta,
        "meta_map": meta_map,
    }
