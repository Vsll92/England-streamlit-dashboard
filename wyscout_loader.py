from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import TEAM_NAME
from schema_utils import safe_date_parse


WYSCOUT_FILES = {
    "attacking": "Team Stats England attacking.xlsx",
    "general": "Team Stats England general.xlsx",
    "indexes": "Team Stats England indexes.xlsx",
    "passing": "Team Stats England passing.xlsx",
}


def _find_file(filename: str, search_roots: list[Path]) -> Path | None:
    for root in search_roots:
        candidate = root / filename
        if candidate.exists():
            return candidate
    return None


def _clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out = out[out.get("Team").eq(TEAM_NAME)].copy()
    out["Date"] = safe_date_parse(out.get("Date"))
    out = out[out["Date"].notna()].copy()
    out = out.reset_index(drop=True)
    return out


def _extract_match_suffix(match_text: str) -> str:
    text = str(match_text or "")
    if " - England" in text:
        return text.split(" - England", 1)[0].strip()
    if text.startswith("England - "):
        right = text.split("England - ", 1)[1]
        pieces = right.split()
        keep: list[str] = []
        for piece in pieces:
            if any(ch.isdigit() for ch in piece) and ":" in piece:
                break
            keep.append(piece)
        return " ".join(keep).replace("(P)", "").replace("(E)", "").strip()
    return text


def _prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    out = _clean_rows(df)
    if out.empty:
        return out
    out = out.rename(columns={"Date": "match_date", "Competition": "competition_wyscout", "Duration": "duration_wyscout", "Scheme": "scheme_wyscout", "Match": "match_wyscout"})
    out["opponent_from_match"] = out["match_wyscout"].map(_extract_match_suffix)
    return out[[c for c in out.columns if c in {"match_date", "match_wyscout", "competition_wyscout", "duration_wyscout", "scheme_wyscout", "opponent_from_match"}]].drop_duplicates()


def _attacking(df: pd.DataFrame) -> pd.DataFrame:
    out = _clean_rows(df)
    if out.empty:
        return out
    base = _prepare_base(out)
    add = pd.DataFrame({
        "match_date": out["Date"],
        "ws_xg": pd.to_numeric(out.get("xG"), errors="coerce"),
        "ws_shots": pd.to_numeric(out.get("Shots / on target"), errors="coerce"),
        "ws_shots_on_target": pd.to_numeric(out.get("Unnamed: 8"), errors="coerce"),
        "ws_shots_on_target_pct": pd.to_numeric(out.get("Unnamed: 9"), errors="coerce"),
        "ws_positional_attacks": pd.to_numeric(out.get("Positional attacks / with shots"), errors="coerce"),
        "ws_positional_attacks_with_shots": pd.to_numeric(out.get("Unnamed: 11"), errors="coerce"),
        "ws_positional_attacks_shot_pct": pd.to_numeric(out.get("Unnamed: 12"), errors="coerce"),
        "ws_counterattacks": pd.to_numeric(out.get("Counterattacks / with shots"), errors="coerce"),
        "ws_counterattacks_with_shots": pd.to_numeric(out.get("Unnamed: 14"), errors="coerce"),
        "ws_counterattacks_shot_pct": pd.to_numeric(out.get("Unnamed: 15"), errors="coerce"),
        "ws_corners": pd.to_numeric(out.get("Corners / with shots"), errors="coerce"),
        "ws_corners_with_shots": pd.to_numeric(out.get("Unnamed: 17"), errors="coerce"),
        "ws_corners_shot_pct": pd.to_numeric(out.get("Unnamed: 18"), errors="coerce"),
        "ws_free_kicks": pd.to_numeric(out.get("Free kicks / with shots"), errors="coerce"),
        "ws_free_kicks_with_shots": pd.to_numeric(out.get("Unnamed: 20"), errors="coerce"),
        "ws_free_kicks_shot_pct": pd.to_numeric(out.get("Unnamed: 21"), errors="coerce"),
        "ws_penalties": pd.to_numeric(out.get("Penalties / converted"), errors="coerce"),
        "ws_penalties_converted": pd.to_numeric(out.get("Unnamed: 23"), errors="coerce"),
        "ws_penalty_conversion_pct": pd.to_numeric(out.get("Unnamed: 24"), errors="coerce"),
        "ws_crosses": pd.to_numeric(out.get("Crosses / accurate"), errors="coerce"),
        "ws_crosses_accurate": pd.to_numeric(out.get("Unnamed: 26"), errors="coerce"),
        "ws_cross_accuracy_pct": pd.to_numeric(out.get("Unnamed: 27"), errors="coerce"),
        "ws_offensive_duels": pd.to_numeric(out.get("Offensive duels / won"), errors="coerce"),
        "ws_offensive_duels_won": pd.to_numeric(out.get("Unnamed: 29"), errors="coerce"),
        "ws_offensive_duels_win_pct": pd.to_numeric(out.get("Unnamed: 30"), errors="coerce"),
        "ws_offsides": pd.to_numeric(out.get("Offsides"), errors="coerce"),
    })
    return base.merge(add, on="match_date", how="left")


def _general(df: pd.DataFrame) -> pd.DataFrame:
    out = _clean_rows(df)
    if out.empty:
        return out
    base = _prepare_base(out)
    add = pd.DataFrame({
        "match_date": out["Date"],
        "ws_goals": pd.to_numeric(out.get("Goals"), errors="coerce"),
        "ws_general_xg": pd.to_numeric(out.get("xG"), errors="coerce"),
        "ws_general_shots": pd.to_numeric(out.get("Shots / on target"), errors="coerce"),
        "ws_general_shots_on_target": pd.to_numeric(out.get("Unnamed: 9"), errors="coerce"),
        "ws_general_shots_on_target_pct": pd.to_numeric(out.get("Unnamed: 10"), errors="coerce"),
        "ws_passes": pd.to_numeric(out.get("Passes / accurate"), errors="coerce"),
        "ws_passes_accurate": pd.to_numeric(out.get("Unnamed: 12"), errors="coerce"),
        "ws_pass_accuracy_pct": pd.to_numeric(out.get("Unnamed: 13"), errors="coerce"),
        "ws_possession_pct": pd.to_numeric(out.get("Possession, %"), errors="coerce"),
        "ws_losses": pd.to_numeric(out.get("Losses / Low / Medium / High"), errors="coerce"),
        "ws_losses_low": pd.to_numeric(out.get("Unnamed: 16"), errors="coerce"),
        "ws_losses_medium": pd.to_numeric(out.get("Unnamed: 17"), errors="coerce"),
        "ws_losses_high": pd.to_numeric(out.get("Unnamed: 18"), errors="coerce"),
        "ws_recoveries": pd.to_numeric(out.get("Recoveries / Low / Medium / High"), errors="coerce"),
        "ws_recoveries_low": pd.to_numeric(out.get("Unnamed: 20"), errors="coerce"),
        "ws_recoveries_medium": pd.to_numeric(out.get("Unnamed: 21"), errors="coerce"),
        "ws_recoveries_high": pd.to_numeric(out.get("Unnamed: 22"), errors="coerce"),
        "ws_duels": pd.to_numeric(out.get("Duels / won"), errors="coerce"),
        "ws_duels_won": pd.to_numeric(out.get("Unnamed: 24"), errors="coerce"),
        "ws_duels_win_pct": pd.to_numeric(out.get("Unnamed: 25"), errors="coerce"),
    })
    return base.merge(add, on="match_date", how="left")


def _indexes(df: pd.DataFrame) -> pd.DataFrame:
    out = _clean_rows(df)
    if out.empty:
        return out
    base = _prepare_base(out)
    add = pd.DataFrame({
        "match_date": out["Date"],
        "ws_match_tempo": pd.to_numeric(out.get("Match tempo"), errors="coerce"),
        "ws_avg_passes_per_possession": pd.to_numeric(out.get("Average passes per possession"), errors="coerce"),
        "ws_long_pass_pct": pd.to_numeric(out.get("Long pass %"), errors="coerce"),
        "ws_ppda": pd.to_numeric(out.get("PPDA"), errors="coerce"),
        "ws_avg_shot_distance": pd.to_numeric(out.get("Average shot distance"), errors="coerce"),
        "ws_avg_pass_length": pd.to_numeric(out.get("Average pass length"), errors="coerce"),
    })
    return base.merge(add, on="match_date", how="left")


def _passing(df: pd.DataFrame) -> pd.DataFrame:
    out = _clean_rows(df)
    if out.empty:
        return out
    base = _prepare_base(out)
    add = pd.DataFrame({
        "match_date": out["Date"],
        "ws_passing_total": pd.to_numeric(out.get("Passes / accurate"), errors="coerce"),
        "ws_passing_accurate": pd.to_numeric(out.get("Unnamed: 7"), errors="coerce"),
        "ws_passing_accuracy_pct": pd.to_numeric(out.get("Unnamed: 8"), errors="coerce"),
        "ws_forward_passes": pd.to_numeric(out.get("Forward passes / accurate"), errors="coerce"),
        "ws_forward_passes_accurate": pd.to_numeric(out.get("Unnamed: 10"), errors="coerce"),
        "ws_forward_passes_accuracy_pct": pd.to_numeric(out.get("Unnamed: 11"), errors="coerce"),
        "ws_back_passes": pd.to_numeric(out.get("Back passes / accurate"), errors="coerce"),
        "ws_back_passes_accurate": pd.to_numeric(out.get("Unnamed: 13"), errors="coerce"),
        "ws_back_passes_accuracy_pct": pd.to_numeric(out.get("Unnamed: 14"), errors="coerce"),
        "ws_lateral_passes": pd.to_numeric(out.get("Lateral passes / accurate"), errors="coerce"),
        "ws_lateral_passes_accurate": pd.to_numeric(out.get("Unnamed: 16"), errors="coerce"),
        "ws_lateral_passes_accuracy_pct": pd.to_numeric(out.get("Unnamed: 17"), errors="coerce"),
        "ws_long_passes": pd.to_numeric(out.get("Long passes / accurate"), errors="coerce"),
        "ws_long_passes_accurate": pd.to_numeric(out.get("Unnamed: 19"), errors="coerce"),
        "ws_long_passes_accuracy_pct": pd.to_numeric(out.get("Unnamed: 20"), errors="coerce"),
        "ws_passes_to_final_third": pd.to_numeric(out.get("Passes to final third / accurate"), errors="coerce"),
        "ws_passes_to_final_third_accurate": pd.to_numeric(out.get("Unnamed: 22"), errors="coerce"),
        "ws_passes_to_final_third_accuracy_pct": pd.to_numeric(out.get("Unnamed: 23"), errors="coerce"),
        "ws_progressive_passes": pd.to_numeric(out.get("Progressive passes / accurate"), errors="coerce"),
        "ws_progressive_passes_accurate": pd.to_numeric(out.get("Unnamed: 25"), errors="coerce"),
        "ws_progressive_passes_accuracy_pct": pd.to_numeric(out.get("Unnamed: 26"), errors="coerce"),
        "ws_smart_passes": pd.to_numeric(out.get("Smart passes / accurate"), errors="coerce"),
        "ws_smart_passes_accurate": pd.to_numeric(out.get("Unnamed: 28"), errors="coerce"),
        "ws_smart_passes_accuracy_pct": pd.to_numeric(out.get("Unnamed: 29"), errors="coerce"),
        "ws_throw_ins": pd.to_numeric(out.get("Throw ins / accurate"), errors="coerce"),
        "ws_throw_ins_accurate": pd.to_numeric(out.get("Unnamed: 31"), errors="coerce"),
        "ws_throw_ins_accuracy_pct": pd.to_numeric(out.get("Unnamed: 32"), errors="coerce"),
        "ws_goal_kicks": pd.to_numeric(out.get("Goal kicks"), errors="coerce"),
    })
    return base.merge(add, on="match_date", how="left")


def load_wyscout_team_stats(search_root: Path | None = None) -> dict[str, Any]:
    project_dir = Path(__file__).resolve().parent
    roots = [project_dir, project_dir.parent]
    if search_root is not None:
        roots.insert(0, search_root)

    sheet_audit: dict[str, dict[str, Any]] = {}
    merged: pd.DataFrame | None = None
    loaders = {
        "attacking": _attacking,
        "general": _general,
        "indexes": _indexes,
        "passing": _passing,
    }

    for key, filename in WYSCOUT_FILES.items():
        path = _find_file(filename, roots)
        if path is None:
            sheet_audit[key] = {"file": filename, "status": "missing", "sheets": [], "columns": []}
            continue
        xls = pd.ExcelFile(path)
        sheet_name = xls.sheet_names[0]
        raw = pd.read_excel(path, sheet_name=sheet_name)
        sheet_audit[key] = {
            "file": str(path),
            "status": "loaded",
            "sheets": xls.sheet_names,
            "columns": list(raw.columns),
            "rows": int(len(raw)),
        }
        tidy = loaders[key](raw)
        if tidy.empty:
            continue
        if merged is None:
            merged = tidy.copy()
        else:
            merged = merged.merge(tidy, on=[c for c in ["match_date", "match_wyscout", "competition_wyscout", "duration_wyscout", "scheme_wyscout", "opponent_from_match"] if c in merged.columns and c in tidy.columns], how="outer")

    if merged is None:
        merged = pd.DataFrame()
    if not merged.empty:
        merged = merged.sort_values("match_date").drop_duplicates("match_date")

    return {"team_stats": merged, "audit": sheet_audit}
