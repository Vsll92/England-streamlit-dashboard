from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

from schema_utils import first_nonempty_text, module_available


def summarize_best_chances(bundle: dict) -> pd.DataFrame:
    pos_df = bundle["pos_df"].copy()
    events = bundle["events"].copy()
    shots = bundle.get("shots", pd.DataFrame()).copy()
    match = bundle["match"].copy()
    if pos_df.empty or events.empty:
        return pd.DataFrame(columns=["match_date", "opponent", "competition_stage", "possession", "possession_xg", "key_players", "end_action", "shot_result"])
    shot_pos = pos_df[pos_df["has_shot"]].sort_values(["xg", "match_id", "possession"], ascending=[False, True, True]).head(20)
    meta = match[["match_id", "match_date", "opponent", "competition_stage"]].drop_duplicates()
    rows = []
    for _, prow in shot_pos.iterrows():
        g = events[(events["match_id"] == prow["match_id"]) & (events["possession"] == prow["possession"])]
        if g.empty:
            continue
        gshots = g[g["type"] == "Shot"]
        key_players = ", ".join(pd.Series(g["player"]).dropna().astype(str).drop_duplicates().head(4).tolist())
        end_action = first_nonempty_text(g["type"].tail(1), default="Unknown")
        shot_meta = shots[(shots["match_id"] == prow["match_id"]) & (shots["possession"] == prow["possession"])].sort_values(["period", "index", "minute", "second"]) if not shots.empty else pd.DataFrame()
        shot_result = first_nonempty_text(shot_meta.get("shot_outcome", pd.Series(dtype=object)).tail(1), default="No shot") if not shot_meta.empty else ("Shot" if not gshots.empty else "No shot")
        rows.append({
            "match_id": int(prow["match_id"]),
            "possession": int(prow["possession"]),
            "possession_xg": float(prow["xg"]),
            "key_players": key_players,
            "end_action": end_action,
            "shot_result": shot_result,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.merge(meta, on="match_id", how="left")
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out[["match_date", "opponent", "competition_stage", "possession", "possession_xg", "key_players", "end_action", "shot_result", "match_id"]]


def build_route_families(bundle: dict) -> pd.DataFrame:
    build_passes = bundle["build_passes"].copy()
    if build_passes.empty:
        return pd.DataFrame(columns=["route", "attempts", "completed", "completion_pct", "start_x", "start_y", "end_x", "end_y", "main_players", "match_count"])
    if "third" not in build_passes.columns:
        build_passes["third"] = pd.cut(build_passes["location_x"], bins=[-1, 40, 80, 121], labels=["Defensive Third", "Middle Third", "Final Third"]).astype(object).fillna("Unknown")
    if "end_third" not in build_passes.columns:
        build_passes["end_third"] = pd.cut(build_passes["pass_end_location_x"], bins=[-1, 40, 80, 121], labels=["Defensive Third", "Middle Third", "Final Third"]).astype(object).fillna("Unknown")
    if "lane" not in build_passes.columns:
        build_passes["lane"] = pd.cut(build_passes["y_std"], bins=[-1, 80/3, 160/3, 81], labels=["Right", "Center", "Left"]).astype(object).fillna("Unknown")
    if "end_lane" not in build_passes.columns:
        build_passes["end_lane"] = pd.cut(build_passes["end_y_std"], bins=[-1, 80/3, 160/3, 81], labels=["Right", "Center", "Left"]).astype(object).fillna("Unknown")
    build_passes["start_zone"] = build_passes["third"].fillna("Unknown") + " | " + build_passes["lane"].fillna("Unknown")
    build_passes["end_zone"] = build_passes["end_third"].fillna("Unknown") + " | " + build_passes["end_lane"].fillna("Unknown")
    grp = build_passes.groupby(["start_zone", "end_zone"])
    out = grp.agg(
        attempts=("id", "count"),
        completed=("successful", "sum"),
        start_x=("location_x", "mean"),
        start_y=("y_std", "mean"),
        end_x=("pass_end_location_x", "mean"),
        end_y=("end_y_std", "mean"),
        match_count=("match_id", "nunique"),
    ).reset_index()
    out["completion_pct"] = np.where(out["attempts"] > 0, out["completed"] / out["attempts"] * 100, 0)
    main_players = (
        build_passes.groupby(["start_zone", "end_zone", "player"]).size().reset_index(name="n")
        .sort_values(["start_zone", "end_zone", "n"], ascending=[True, True, False])
        .groupby(["start_zone", "end_zone"])
        .head(2)
        .groupby(["start_zone", "end_zone"])["player"].apply(lambda s: ", ".join(s.astype(str))).reset_index(name="main_players")
    )
    out = out.merge(main_players, on=["start_zone", "end_zone"], how="left")
    out["route"] = out["start_zone"] + " → " + out["end_zone"]
    return out.sort_values(["attempts", "completion_pct"], ascending=[False, False]).reset_index(drop=True)


def build_build_up_outcomes(bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos_df = bundle["pos_df"].copy()
    events = bundle["events"].copy()
    shots = bundle.get("shots", pd.DataFrame()).copy()
    match = bundle["match"].copy()
    if pos_df.empty or events.empty:
        empty = pd.DataFrame(columns=["label", "count", "pct"])
        return empty, empty
    build = pos_df[pos_df["is_build_up"]].copy()
    rows = []
    for _, prow in build.iterrows():
        g = events[(events["match_id"] == prow["match_id"]) & (events["possession"] == prow["possession"])]
        g = g.sort_values(["period", "index", "minute", "second"])
        first_four = g.head(4)
        turnover_early = bool(((first_four["type"].isin(["Dispossessed", "Miscontrol"])) | ((first_four["type"] == "Pass") & (~first_four["success"].fillna(False)))).any())
        progressed_middle = bool(((g["x"] >= 40) | (g["end_x"] >= 40)).fillna(False).any())
        progressed_final = bool(((g["x"] >= 80) | (g["end_x"] >= 80)).fillna(False).any())
        shot_later = bool((g["type"] == "Shot").any())
        max_x = pd.to_numeric(g["end_x"].fillna(g["x"]), errors="coerce").max()
        rows.append({
            "match_id": prow["match_id"],
            "possession": prow["possession"],
            "keep_possession": not turnover_early,
            "lose_possession": turnover_early,
            "progress_middle": progressed_middle,
            "progress_final": progressed_final,
            "forced_backward": bool(max_x < 40) if pd.notna(max_x) else False,
            "shot_later": shot_later,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        empty = pd.DataFrame(columns=["label", "count", "pct"])
        return empty, empty
    summary = pd.DataFrame({
        "label": ["Keep possession", "Lose possession early", "Progress to middle third", "Progress to final third", "Forced backward", "Shot later in possession"],
        "count": [out["keep_possession"].sum(), out["lose_possession"].sum(), out["progress_middle"].sum(), out["progress_final"].sum(), out["forced_backward"].sum(), out["shot_later"].sum()],
    })
    summary["pct"] = np.where(len(out) > 0, summary["count"] / len(out) * 100, 0)

    meta = match[["match_id", "opponent", "competition_stage", "match_date"]].drop_duplicates()
    by_match = out.merge(meta, on="match_id", how="left").groupby(["match_date", "opponent", "competition_stage"], dropna=False).agg(
        build_up_possessions=("possession", "count"),
        keep_possession=("keep_possession", "sum"),
        lose_possession=("lose_possession", "sum"),
        progress_middle=("progress_middle", "sum"),
        progress_final=("progress_final", "sum"),
        shot_later=("shot_later", "sum"),
    ).reset_index()
    by_match["keep_possession_pct"] = np.where(by_match["build_up_possessions"] > 0, by_match["keep_possession"] / by_match["build_up_possessions"] * 100, 0)
    return summary, by_match


def build_pattern_table(bundle: dict) -> pd.DataFrame:
    events = bundle["events"].copy()
    build_pos = bundle["pos_df"].copy()
    if events.empty or build_pos.empty:
        return pd.DataFrame(columns=["pattern", "frequency", "match_count", "sample_match_id", "sample_possession", "steps"])
    valid = build_pos[build_pos["is_build_up"]][["match_id", "possession"]].drop_duplicates()
    seq = events.merge(valid, on=["match_id", "possession"], how="inner")
    seq = seq[seq["type"].isin(["Pass", "Carry"])].copy()
    rows = []
    for (match_id, possession), g in seq.groupby(["match_id", "possession"]):
        g = g.sort_values(["period", "index", "minute", "second"]).head(4)
        if g.empty:
            continue
        steps = []
        for _, row in g.iterrows():
            label = f"{row['player']} ({'P' if row['type']=='Pass' else 'C'})"
            steps.append(label)
        rows.append({
            "match_id": match_id,
            "possession": possession,
            "pattern": " → ".join(steps),
            "steps": g[["type", "player", "x", "y", "end_x", "end_y"]].to_dict("records"),
        })
    if not rows:
        return pd.DataFrame(columns=["pattern", "frequency", "match_count", "sample_match_id", "sample_possession", "steps"])
    temp = pd.DataFrame(rows)
    grouped = temp.groupby("pattern").agg(
        frequency=("possession", "count"),
        match_count=("match_id", "nunique"),
        sample_match_id=("match_id", "first"),
        sample_possession=("possession", "first"),
        steps=("steps", "first"),
    ).reset_index().sort_values(["frequency", "match_count"], ascending=[False, False])
    return grouped.head(12).reset_index(drop=True)


def make_export_bytes(tables: dict[str, pd.DataFrame]) -> tuple[str, bytes, str]:
    if module_available("xlsxwriter") or module_available("openpyxl"):
        engine = "xlsxwriter" if module_available("xlsxwriter") else "openpyxl"
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine=engine) as writer:
            for name, df in tables.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
        return "xlsx", bio.getvalue(), f"Excel export created with {engine}."

    bio = BytesIO()
    with ZipFile(bio, "w", compression=ZIP_DEFLATED) as zf:
        for name, df in tables.items():
            zf.writestr(f"{name}.csv", df.to_csv(index=False))
    return "zip", bio.getvalue(), "Excel engines are unavailable, so the export falls back to CSV files in a ZIP archive."
