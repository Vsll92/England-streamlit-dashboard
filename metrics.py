from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd

from config import PITCH_LENGTH, PITCH_WIDTH, TEAM_NAME
from schema_utils import safe_bool


def lane_from_y(y: float) -> str:
    if pd.isna(y):
        return "Unknown"
    if y < PITCH_WIDTH / 3:
        return "Right"
    if y < 2 * PITCH_WIDTH / 3:
        return "Center"
    return "Left"


def third_from_x(x: float) -> str:
    if pd.isna(x):
        return "Unknown"
    if x < PITCH_LENGTH / 3:
        return "Defensive Third"
    if x < 2 * PITCH_LENGTH / 3:
        return "Middle Third"
    return "Final Third"


def in_box(x: float, y: float) -> bool:
    return pd.notna(x) and pd.notna(y) and x >= 102 and 18 <= y <= 62


def in_z14(x: float, y: float) -> bool:
    return pd.notna(x) and pd.notna(y) and 80 <= x < 102 and 30 <= y <= 50


def role_group(position: object) -> str:
    s = str(position or "")
    if "Goalkeeper" in s:
        return "Goalkeeper"
    if "Center Back" in s:
        return "Centre Backs"
    if "Back" in s or "Wing Back" in s:
        return "Full Backs / Wing Backs"
    if "Midfield" in s:
        return "Midfielders"
    if "Forward" in s or "Wing" in s or "Striker" in s:
        return "Forwards"
    return "Others"


def safe_div(num: float, den: float) -> float:
    if den in (0, None) or pd.isna(den):
        return 0.0
    return float(num) / float(den)


def is_progressive(start_x: pd.Series | float, end_x: pd.Series | float) -> pd.Series | bool:
    """StatsBomb 120x80 pitch progressive progression using x-gain thresholds by field zone."""
    if isinstance(start_x, pd.Series) or isinstance(end_x, pd.Series):
        sx = pd.to_numeric(start_x, errors="coerce")
        ex = pd.to_numeric(end_x, errors="coerce")
        gain = ex - sx
        return ((sx < 60) & (ex < 60) & (gain >= 30)) | ((sx < 60) & (ex >= 60) & (gain >= 15)) | ((sx >= 60) & (ex >= 60) & (gain >= 10))
    if pd.isna(start_x) or pd.isna(end_x):
        return False
    gain = float(end_x) - float(start_x)
    return ((start_x < 60 and end_x < 60 and gain >= 30) or (start_x < 60 and end_x >= 60 and gain >= 15) or (start_x >= 60 and end_x >= 60 and gain >= 10))


def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def make_common_events(tables: dict[str, pd.DataFrame], match_ids: tuple[int, ...]) -> dict[str, pd.DataFrame]:
    out = {}
    for name, df in tables.items():
        if df.empty:
            out[name] = df.copy()
            continue
        cur = df[df["match_id"].isin(match_ids)].copy()
        if "period" in cur.columns:
            cur = cur[pd.to_numeric(cur["period"], errors="coerce").fillna(0).le(4)].copy()
        if "id" in cur.columns:
            cur = cur.drop_duplicates(subset=["id"]).copy()
        if "team" in cur.columns:
            cur = cur[cur["team"].eq(TEAM_NAME)].copy()
        out[name] = cur
    return out


def build_events_frame(filtered: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []

    def add_common(df: pd.DataFrame, typ: str, end_x_col: str | None = None, end_y_col: str | None = None) -> pd.DataFrame:
        if df.empty:
            return _empty_df(["match_id", "possession", "index", "period", "minute", "second", "type", "player", "position", "play_pattern", "x", "y", "end_x", "end_y", "success", "subtype", "event_id"])
        base = pd.DataFrame({
            "match_id": df.get("match_id"),
            "possession": df.get("possession"),
            "index": df.get("index"),
            "period": df.get("period"),
            "minute": df.get("minute"),
            "second": df.get("second"),
            "timestamp": df.get("timestamp"),
            "type": typ,
            "player": df.get("player"),
            "position": df.get("position"),
            "play_pattern": df.get("play_pattern"),
            "x": df.get("location_x"),
            "y": df.get("y_std"),
            "event_id": df.get("id"),
        })
        base["team"] = TEAM_NAME
        base["role_group"] = base["position"].map(role_group)
        base["success"] = True
        base["subtype"] = typ
        base["end_x"] = df[end_x_col] if end_x_col and end_x_col in df.columns else np.nan
        base["end_y"] = df[end_y_col] if end_y_col and end_y_col in df.columns else np.nan
        return base

    passes = filtered.get("passes", pd.DataFrame()).copy()
    if not passes.empty:
        base = add_common(passes, "Pass", "pass_end_location_x", "end_y_std")
        base["success"] = passes["pass_outcome"].isna() if "pass_outcome" in passes.columns else True
        base["subtype"] = np.select(
            [
                safe_bool(passes, "pass_cross"),
                safe_bool(passes, "pass_cut_back"),
                safe_bool(passes, "pass_through_ball"),
                pd.to_numeric(passes.get("pass_length"), errors="coerce").fillna(0).ge(30),
            ],
            ["Cross", "Cutback", "Through Ball", "Long Pass"],
            default="Pass",
        )
        base["recipient"] = passes.get("pass_recipient")
        frames.append(base)

    carries = filtered.get("carries", pd.DataFrame()).copy()
    if not carries.empty:
        base = add_common(carries, "Carry", "carry_end_location_x", "end_y_std")
        base["subtype"] = "Carry"
        frames.append(base)

    shots = filtered.get("shots", pd.DataFrame()).copy()
    if not shots.empty:
        base = add_common(shots, "Shot", "shot_end_location_x", "end_y_std")
        base["success"] = shots["shot_outcome"].eq("Goal") if "shot_outcome" in shots.columns else False
        base["subtype"] = shots.get("shot_type", pd.Series("Shot", index=shots.index)).fillna("Shot")
        base["xg"] = pd.to_numeric(shots.get("shot_statsbomb_xg"), errors="coerce")
        frames.append(base)

    for name, typ in [("dribbles", "Dribble"), ("recoveries", "Recovery"), ("receipts", "Receipt"), ("dispossessed", "Dispossessed"), ("miscontrols", "Miscontrol"), ("foul_wons", "Foul Won")]:
        df = filtered.get(name, pd.DataFrame())
        if df.empty:
            continue
        base = add_common(df, typ, None, None)
        if name == "dribbles" and "dribble_outcome" in df.columns:
            base["success"] = df["dribble_outcome"].eq("Complete")
        elif name in {"dispossessed", "miscontrols"}:
            base["success"] = False
        base["subtype"] = typ
        frames.append(base)

    if not frames:
        return _empty_df(["match_id", "possession", "index", "period", "minute", "second", "type", "player", "position", "play_pattern", "x", "y", "end_x", "end_y", "success", "subtype", "role_group", "team", "xg", "recipient"])
    events = pd.concat(frames, ignore_index=True, sort=False)
    events["xg"] = pd.to_numeric(events.get("xg"), errors="coerce")
    events["lane"] = events["y"].map(lane_from_y)
    events["end_lane"] = events["end_y"].map(lane_from_y)
    events["third"] = events["x"].map(third_from_x)
    events["end_third"] = events["end_x"].map(third_from_x)
    events["in_box_start"] = [in_box(x, y) for x, y in zip(events["x"], events["y"])]
    events["in_box_end"] = [in_box(x, y) for x, y in zip(events["end_x"], events["end_y"])]
    events["in_z14_start"] = [in_z14(x, y) for x, y in zip(events["x"], events["y"])]
    events["in_z14_end"] = [in_z14(x, y) for x, y in zip(events["end_x"], events["end_y"])]
    events = events.sort_values(["match_id", "period", "index", "minute", "second"], kind="stable")
    return events


def compute_analysis_bundle(project_data: dict[str, Any], selected_match_ids: tuple[int, ...]) -> dict[str, Any]:
    match = project_data["match_meta"]
    tables = project_data["tables"]
    selected_matches = match[match["match_id"].isin(selected_match_ids)].copy()

    filtered = make_common_events(tables, selected_match_ids)
    passes = filtered.get("passes", pd.DataFrame()).copy()
    receipts = filtered.get("receipts", pd.DataFrame()).copy()
    carries = filtered.get("carries", pd.DataFrame()).copy()
    shots = filtered.get("shots", pd.DataFrame()).copy()
    dribbles = filtered.get("dribbles", pd.DataFrame()).copy()
    recoveries = filtered.get("recoveries", pd.DataFrame()).copy()
    dispossessed = filtered.get("dispossessed", pd.DataFrame()).copy()
    miscontrols = filtered.get("miscontrols", pd.DataFrame()).copy()
    foul_wons = filtered.get("foul_wons", pd.DataFrame()).copy()

    for df in [passes, carries, receipts, shots, dribbles, recoveries, dispossessed, miscontrols, foul_wons]:
        if not df.empty:
            df["role_group"] = df["position"].map(role_group) if "position" in df.columns else "Others"
            if "y_std" in df.columns:
                df["lane"] = df["y_std"].map(lane_from_y)
            if "location_x" in df.columns:
                df["third"] = df["location_x"].map(third_from_x)

    if not passes.empty:
        passes["successful"] = passes["pass_outcome"].isna() if "pass_outcome" in passes.columns else True
        passes["cross"] = safe_bool(passes, "pass_cross")
        passes["cutback"] = safe_bool(passes, "pass_cut_back")
        passes["through_ball"] = safe_bool(passes, "pass_through_ball")
        passes["switch"] = safe_bool(passes, "pass_switch")
        passes["pass_length_num"] = pd.to_numeric(passes.get("pass_length"), errors="coerce")
        passes["progressive_pass"] = passes["successful"] & is_progressive(passes.get("location_x", pd.Series(np.nan, index=passes.index)), passes.get("pass_end_location_x", pd.Series(np.nan, index=passes.index)))
        passes["entry_to_z3"] = passes["successful"] & (passes.get("location_x", pd.Series(np.nan, index=passes.index)) < 80) & (passes.get("pass_end_location_x", pd.Series(np.nan, index=passes.index)) >= 80)
        passes["box_entry"] = passes["successful"] & (passes.get("location_x", pd.Series(np.nan, index=passes.index)) < 102) & (passes.get("pass_end_location_x", pd.Series(np.nan, index=passes.index)) >= 102) & (passes.get("pass_end_location_y", pd.Series(np.nan, index=passes.index)).between(18, 62))
        passes["to_z14"] = [in_z14(x, y) for x, y in zip(passes.get("pass_end_location_x", pd.Series(np.nan, index=passes.index)), passes.get("end_y_std", pd.Series(np.nan, index=passes.index)))]
        passes["entry_type"] = np.select([passes["cross"], passes["pass_length_num"].fillna(0).ge(30)], ["Cross-type Entry", "Direct/Deep Pass"], default="Short Pass")
        passes["subtype"] = np.select([passes["cross"], passes["cutback"], passes["through_ball"], passes["pass_length_num"].fillna(0).ge(30)], ["Cross", "Cutback", "Through Ball", "Long Pass"], default="Pass")
    else:
        passes["successful"] = []

    if not carries.empty:
        carries["progressive_carry"] = is_progressive(carries.get("location_x", pd.Series(np.nan, index=carries.index)), carries.get("carry_end_location_x", pd.Series(np.nan, index=carries.index)))
        carries["entry_to_z3"] = (carries.get("location_x", pd.Series(np.nan, index=carries.index)) < 80) & (carries.get("carry_end_location_x", pd.Series(np.nan, index=carries.index)) >= 80)
        carries["box_entry"] = (carries.get("location_x", pd.Series(np.nan, index=carries.index)) < 102) & (carries.get("carry_end_location_x", pd.Series(np.nan, index=carries.index)) >= 102) & (carries.get("carry_end_location_y", pd.Series(np.nan, index=carries.index)).between(18, 62))
        carries["to_z14"] = [in_z14(x, y) for x, y in zip(carries.get("carry_end_location_x", pd.Series(np.nan, index=carries.index)), carries.get("end_y_std", pd.Series(np.nan, index=carries.index)))]
        carries["entry_type"] = "Carry"

    if not shots.empty:
        shots["xg"] = pd.to_numeric(shots.get("shot_statsbomb_xg"), errors="coerce").fillna(0)
        shots["is_sot"] = shots.get("shot_outcome", pd.Series("", index=shots.index)).isin(["Goal", "Saved", "Saved to Post"])
        shots["is_goal"] = shots.get("shot_outcome", pd.Series("", index=shots.index)).eq("Goal")
        shots["in_box"] = [in_box(x, y) for x, y in zip(shots.get("location_x", pd.Series(np.nan, index=shots.index)), shots.get("y_std", pd.Series(np.nan, index=shots.index)))]
        shots["in_z14"] = [in_z14(x, y) for x, y in zip(shots.get("location_x", pd.Series(np.nan, index=shots.index)), shots.get("y_std", pd.Series(np.nan, index=shots.index)))]
        shots["shot_distance"] = np.sqrt((PITCH_LENGTH - shots.get("location_x", pd.Series(np.nan, index=shots.index))) ** 2 + (40 - shots.get("y_std", pd.Series(np.nan, index=shots.index))) ** 2)
        shots["zone"] = np.where(shots["in_box"], "Inside Box", "Outside Box")
        shots["lane"] = shots.get("y_std", pd.Series(np.nan, index=shots.index)).map(lane_from_y)

    if not receipts.empty:
        receipts["in_z14"] = [in_z14(x, y) for x, y in zip(receipts.get("location_x", pd.Series(np.nan, index=receipts.index)), receipts.get("y_std", pd.Series(np.nan, index=receipts.index)))]
        receipts["in_box"] = [in_box(x, y) for x, y in zip(receipts.get("location_x", pd.Series(np.nan, index=receipts.index)), receipts.get("y_std", pd.Series(np.nan, index=receipts.index)))]

    events = build_events_frame(filtered)

    pos_rows = []
    grouped_pos: dict[tuple[int, int], pd.DataFrame] = {}
    if not events.empty:
        for (match_id, possession), grp in events.groupby(["match_id", "possession"], dropna=True):
            grp = grp.sort_values(["period", "index", "minute", "second"], kind="stable").reset_index(drop=True)
            grouped_pos[(int(match_id), int(possession))] = grp
            first = grp.iloc[0]
            shot_grp = grp[grp["type"].eq("Shot")]
            pos_rows.append({
                "match_id": int(match_id),
                "possession": int(possession),
                "period": first.get("period", np.nan),
                "play_pattern": first.get("play_pattern", "Unknown"),
                "start_x": first.get("x", np.nan),
                "start_y": first.get("y", np.nan),
                "start_role_group": first.get("role_group", "Others"),
                "n_events": len(grp),
                "has_shot": bool(grp["type"].eq("Shot").any()),
                "xg": float(shot_grp["xg"].fillna(0).sum()) if "xg" in shot_grp.columns else 0.0,
                "has_final_third": bool(((grp["x"] >= 80) | (grp["end_x"] >= 80)).fillna(False).any()),
                "has_box": bool((grp["in_box_start"] | grp["in_box_end"]).any()),
                "has_z14": bool((grp["in_z14_start"] | grp["in_z14_end"]).any()),
            })
    pos_df = pd.DataFrame(pos_rows)

    def classify_build_up(row: pd.Series) -> tuple[bool, str]:
        key = (int(row["match_id"]), int(row["possession"]))
        grp = grouped_pos.get(key)
        if grp is None or grp.empty:
            return (False, "N/A")
        start_x = row.get("start_x", np.nan)
        role = row.get("start_role_group", "Others")
        is_build = (pd.notna(start_x) and start_x < 40) or (role in {"Goalkeeper", "Centre Backs"} and pd.notna(start_x) and start_x < 60)
        if not is_build:
            return (False, "N/A")
        first_passes = grp[grp["type"].eq("Pass")].head(3)
        long_trigger = False
        if not first_passes.empty:
            long_trigger = bool(first_passes["subtype"].eq("Long Pass").any() or (first_passes["end_x"] >= 70).any())
        return (True, "Long" if long_trigger else "Short")

    if not pos_df.empty:
        build_res = pos_df.apply(classify_build_up, axis=1)
        pos_df["is_build_up"] = [x[0] for x in build_res]
        pos_df["build_up_type"] = [x[1] for x in build_res]
        pos_df["is_offensive_possession"] = pos_df["has_final_third"] | pos_df["has_box"] | pos_df["has_shot"] | pos_df["n_events"].ge(3)
    else:
        pos_df["is_build_up"] = pd.Series(dtype=bool)
        pos_df["build_up_type"] = pd.Series(dtype=object)
        pos_df["is_offensive_possession"] = pd.Series(dtype=bool)

    build_poss = pos_df[pos_df.get("is_build_up", pd.Series(dtype=bool))][["match_id", "possession", "build_up_type"]].copy() if not pos_df.empty else _empty_df(["match_id", "possession", "build_up_type"])

    def merge_build(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or build_poss.empty:
            return df.iloc[0:0].copy()
        return df.merge(build_poss, on=["match_id", "possession"], how="inner")

    build_passes = merge_build(passes)
    build_carries = merge_build(carries)
    build_receipts = merge_build(receipts)

    progress_actions = []
    if not build_passes.empty:
        bp = build_passes.copy()
        bp["action"] = "Pass"
        bp["start_x"] = bp["location_x"]
        bp["end_x"] = bp["pass_end_location_x"]
        bp["source_zone"] = bp["lane"].fillna("Unknown") + " | " + bp["third"].fillna("Unknown")
        bp["end_lane"] = bp["end_y_std"].map(lane_from_y)
        bp["end_third"] = bp["pass_end_location_x"].map(third_from_x)
        bp["target_zone"] = bp["end_lane"].fillna("Unknown") + " | " + bp["end_third"].fillna("Unknown")
        progress_actions.append(bp[["match_id", "possession", "index", "action", "start_x", "end_x", "source_zone", "target_zone", "end_lane"]])
    if not build_carries.empty:
        bc = build_carries.copy()
        bc["action"] = "Carry"
        bc["start_x"] = bc["location_x"]
        bc["end_x"] = bc["carry_end_location_x"]
        bc["source_zone"] = bc["lane"].fillna("Unknown") + " | " + bc["third"].fillna("Unknown")
        bc["end_lane"] = bc["end_y_std"].map(lane_from_y)
        bc["end_third"] = bc["carry_end_location_x"].map(third_from_x)
        bc["target_zone"] = bc["end_lane"].fillna("Unknown") + " | " + bc["end_third"].fillna("Unknown")
        progress_actions.append(bc[["match_id", "possession", "index", "action", "start_x", "end_x", "source_zone", "target_zone", "end_lane"]])
    if progress_actions:
        prog = pd.concat(progress_actions, ignore_index=True)
        prog = prog[prog["end_x"] > prog["start_x"]].sort_values(["match_id", "possession", "index"])
        first_progression = prog.groupby(["match_id", "possession"], as_index=False).head(1)
    else:
        first_progression = _empty_df(["match_id", "possession", "source_zone", "target_zone", "end_lane"])

    grouped_player_events = {
        key: grp[["index", "type", "x", "end_x", "success", "subtype", "player"]].sort_values("index").reset_index(drop=True)
        for key, grp in events.groupby(["match_id", "possession", "player"], dropna=True)
    }

    def post_receipt_metrics(row: pd.Series) -> pd.Series:
        key = (row["match_id"], row["possession"], row["player"])
        grp = grouped_player_events.get(key)
        if grp is None or grp.empty:
            return pd.Series([False, False, False, False])
        nxt = grp[grp["index"] > row["index"]].head(1)
        if nxt.empty:
            return pd.Series([False, False, False, False])
        nxt = nxt.iloc[0]
        target_x = nxt["end_x"] if pd.notna(nxt["end_x"]) else nxt["x"]
        progressive = pd.notna(row.get("location_x")) and pd.notna(target_x) and (target_x - row["location_x"] >= 10)
        carry_after = nxt["type"] == "Carry"
        turnover = (nxt["type"] in {"Dispossessed", "Miscontrol"}) or (nxt["type"] == "Pass" and not bool(nxt["success"]))
        retained = not turnover
        return pd.Series([progressive, carry_after, turnover, retained])

    if not build_receipts.empty:
        build_receipts[["progressive_reception", "carry_after_reception", "turnover_after_reception", "retained_after_reception"]] = build_receipts.apply(post_receipt_metrics, axis=1)
    else:
        build_receipts = build_receipts.assign(progressive_reception=pd.Series(dtype=bool), carry_after_reception=pd.Series(dtype=bool), turnover_after_reception=pd.Series(dtype=bool), retained_after_reception=pd.Series(dtype=bool))

    all_receipts = receipts.copy()
    if not all_receipts.empty:
        all_receipts[["progressive_reception", "carry_after_reception", "turnover_after_reception", "retained_after_reception"]] = all_receipts.apply(post_receipt_metrics, axis=1)
    else:
        all_receipts = all_receipts.assign(progressive_reception=pd.Series(dtype=bool), carry_after_reception=pd.Series(dtype=bool), turnover_after_reception=pd.Series(dtype=bool), retained_after_reception=pd.Series(dtype=bool))

    if not build_receipts.empty:
        build_receivers = (
            build_receipts.groupby("player")
            .agg(receptions=("id", "count"), progressive_receptions=("progressive_reception", "sum"), carries_after_reception=("carry_after_reception", "sum"), turnovers_after_reception=("turnover_after_reception", "sum"), avg_x=("location_x", "mean"), avg_y=("y_std", "mean"))
            .reset_index()
            .sort_values("receptions", ascending=False)
        )
        build_receivers["forward_passes_received"] = build_receivers["receptions"]
        build_receivers["xT_added_after_reception"] = np.nan
    else:
        build_receivers = _empty_df(["player", "receptions", "progressive_receptions", "forward_passes_received", "carries_after_reception", "turnovers_after_reception", "xT_added_after_reception"])

    if not build_passes.empty and "pass_recipient" in build_passes.columns:
        bp_success = build_passes[build_passes["successful"]].copy().sort_values(["match_id", "possession", "index"])
        bp_success["combo"] = bp_success["player"].fillna("?") + " → " + bp_success["pass_recipient"].fillna("?")
        combo_rows = []
        for _, grp in bp_success.groupby(["match_id", "possession"]):
            combo_rows.append(" | ".join(grp["combo"].head(2).tolist()))
        build_combo_table = pd.Series(combo_rows, name="pattern").value_counts().reset_index()
        build_combo_table.columns = ["pattern", "count"]
        build_combo_table = build_combo_table.head(12)
    else:
        build_combo_table = _empty_df(["pattern", "count"])

    entry_frames = []
    if not passes.empty:
        ep = passes[passes["entry_to_z3"]].copy()
        if not ep.empty:
            ep["entry_x"] = ep["pass_end_location_x"]
            ep["entry_y"] = ep["end_y_std"]
            ep["entry_lane"] = ep["entry_y"].map(lane_from_y)
            entry_frames.append(ep[["id", "match_id", "possession", "index", "player", "position", "entry_type", "entry_x", "entry_y", "entry_lane"]])
    if not carries.empty:
        ec = carries[carries["entry_to_z3"]].copy()
        if not ec.empty:
            ec["entry_x"] = ec["carry_end_location_x"]
            ec["entry_y"] = ec["end_y_std"]
            ec["entry_lane"] = ec["entry_y"].map(lane_from_y)
            entry_frames.append(ec[["id", "match_id", "possession", "index", "player", "position", "entry_type", "entry_x", "entry_y", "entry_lane"]])

    entries = pd.concat(entry_frames, ignore_index=True).sort_values(["match_id", "possession", "index"]) if entry_frames else _empty_df(["id", "match_id", "possession", "index", "player", "position", "entry_type", "entry_x", "entry_y", "entry_lane"])
    first_entries = entries.groupby(["match_id", "possession"], as_index=False).head(1).copy() if not entries.empty else entries.copy()

    def entry_outcomes(row: pd.Series) -> pd.Series:
        grp = grouped_pos.get((int(row["match_id"]), int(row["possession"])))
        if grp is None or grp.empty:
            return pd.Series([False, False, False, False, False, False, False, False, False, 0.0, 0.0])
        post = grp[grp["index"] > row["index"]].copy()
        if post.empty:
            return pd.Series([False, False, False, False, False, False, False, False, False, 0.0, 0.0])
        first2 = post.head(2)
        turnover_mask = post["type"].isin(["Dispossessed", "Miscontrol"]) | ((post["type"] == "Pass") & (~post["success"].fillna(True)))
        immediate_turnover = (first2["type"].isin(["Dispossessed", "Miscontrol"]) | ((first2["type"] == "Pass") & (~first2["success"].fillna(True)))).any()
        return pd.Series([
            not bool(immediate_turnover),
            bool((post["in_box_start"] | post["in_box_end"]).any()),
            bool(post["type"].eq("Shot").any()),
            bool((post["in_z14_start"] | post["in_z14_end"]).any()),
            bool((((post["x"] >= 80) & post["lane"].isin(["Left", "Right"])) | ((post["end_x"] >= 80) & post["end_lane"].isin(["Left", "Right"]))).fillna(False).any()),
            bool(turnover_mask.any()),
            bool(((post["type"] == "Pass") & (post["subtype"] == "Cutback")).any()),
            bool(((post["type"] == "Pass") & (post["subtype"] == "Long Pass") & (post["lane"] != post["end_lane"])).any()),
            bool(((post["type"] == "Pass") & (post["subtype"] == "Cross")).any()),
            float((post["in_box_start"] | post["in_box_end"]).sum()),
            float(post.loc[post["type"] == "Shot", "xg"].fillna(0).sum()),
        ])

    if not first_entries.empty:
        first_entry_metrics = first_entries.apply(entry_outcomes, axis=1)
        first_entry_metrics.columns = ["retain_possession", "enter_box", "create_shot", "move_z14", "move_wide", "lose_possession", "cutback_after", "switch_after", "cross_after", "box_touches_after", "xg_after"]
        first_entries = pd.concat([first_entries.reset_index(drop=True), first_entry_metrics.reset_index(drop=True)], axis=1)
    else:
        first_entries = first_entries.assign(retain_possession=pd.Series(dtype=bool), enter_box=pd.Series(dtype=bool), create_shot=pd.Series(dtype=bool), move_z14=pd.Series(dtype=bool), move_wide=pd.Series(dtype=bool), lose_possession=pd.Series(dtype=bool), cutback_after=pd.Series(dtype=bool), switch_after=pd.Series(dtype=bool), cross_after=pd.Series(dtype=bool), box_touches_after=pd.Series(dtype=float), xg_after=pd.Series(dtype=float))
    if not entries.empty:
        temp = entries.apply(entry_outcomes, axis=1)
        entries["entry_led_to_shot"] = temp[2]
        entries["entry_led_to_box_touch"] = temp[1]
        entries["entry_xg_after"] = temp[10]
    else:
        entries["entry_led_to_shot"] = pd.Series(dtype=bool)
        entries["entry_led_to_box_touch"] = pd.Series(dtype=bool)
        entries["entry_xg_after"] = pd.Series(dtype=float)

    if not entries.empty:
        entry_players = (
            entries.groupby("player")
            .agg(entry_passes=("entry_type", lambda s: int((s != "Carry").sum())), entry_carries=("entry_type", lambda s: int((s == "Carry").sum())), successful_entries=("id", "count"), entries_leading_to_shot=("entry_led_to_shot", "sum"), entries_leading_to_box_touch=("entry_led_to_box_touch", "sum"), xg_after_entries=("entry_xg_after", "sum"))
            .reset_index()
            .sort_values("successful_entries", ascending=False)
        )
    else:
        entry_players = _empty_df(["player", "entry_passes", "entry_carries", "successful_entries", "entries_leading_to_shot", "entries_leading_to_box_touch", "xg_after_entries"])

    z14_possessions = set(events.loc[events["in_z14_start"] | events["in_z14_end"], ["match_id", "possession"]].itertuples(index=False, name=None)) if not events.empty else set()
    z14_metrics = {
        "z14_entries": int(passes.get("to_z14", pd.Series(dtype=bool)).sum()) + int(carries.get("to_z14", pd.Series(dtype=bool)).sum()),
        "z14_touches": int((events["in_z14_start"] | events["in_z14_end"]).sum()) if not events.empty else 0,
        "z14_receipts": int(receipts.get("in_z14", pd.Series(dtype=bool)).sum()) if not receipts.empty else 0,
        "shots_after_z14": int(sum((m, p) in z14_possessions for m, p in zip(shots.get("match_id", []), shots.get("possession", [])))) if not shots.empty else 0,
    }

    flank_rows = []
    for flank in ["Left", "Right"]:
        flank_entries = entries[entries["entry_lane"].eq(flank)] if not entries.empty else entries
        entry_keys = set(flank_entries[["match_id", "possession"]].itertuples(index=False, name=None)) if not flank_entries.empty else set()
        post_passes = passes[passes.apply(lambda r: (r["match_id"], r["possession"]) in entry_keys, axis=1)] if entry_keys and not passes.empty else passes.iloc[0:0]
        flank_rows.append({
            "flank": flank,
            "attacks": int(len(flank_entries)),
            "cross_frequency": int(post_passes.get("cross", pd.Series(dtype=bool)).sum()) if not post_passes.empty else 0,
            "cutback_frequency": int(post_passes.get("cutback", pd.Series(dtype=bool)).sum()) if not post_passes.empty else 0,
        })
    flank_metrics = pd.DataFrame(flank_rows)

    assist_passes = pd.DataFrame()
    if not shots.empty and "shot_key_pass_id" in shots.columns and not passes.empty and "id" in passes.columns:
        shot_assist_cols = ["id", "shot_key_pass_id", "shot_statsbomb_xg", "player", "location_x", "y_std", "is_goal", "is_sot", "in_box", "shot_distance"]
        shot_side = shots[[c for c in shot_assist_cols if c in shots.columns]].copy()
        assist_passes = passes.merge(shot_side, left_on="id", right_on="shot_key_pass_id", how="inner", suffixes=("_pass", "_shot"))
        xA_map = shot_side.dropna(subset=["shot_key_pass_id"]).groupby("shot_key_pass_id")["shot_statsbomb_xg"].sum()
        passes["key_pass"] = passes["id"].isin(shots["shot_key_pass_id"].dropna())
        passes["shot_assist"] = passes["key_pass"]
        passes["xA"] = passes["id"].map(xA_map).fillna(0)
    else:
        passes["key_pass"] = False
        passes["shot_assist"] = False
        passes["xA"] = 0.0

    group_recov = {k: g for k, g in recoveries.groupby(["match_id", "possession"], dropna=True)} if not recoveries.empty else {}
    group_events = {k: g.sort_values("index") for k, g in events.groupby(["match_id", "possession"], dropna=True)} if not events.empty else {}

    def classify_shot_source(row: pd.Series) -> str:
        if "shot_key_pass_id" in row.index and pd.notna(row.get("shot_key_pass_id")) and not passes.empty:
            prev_pass = passes[passes["id"].eq(row["shot_key_pass_id"])]
            if prev_pass.empty:
                return "Other"
            prev = prev_pass.iloc[0]
            if bool(prev.get("cross", False)):
                return "Cross"
            if bool(prev.get("cutback", False)):
                return "Cutback"
            if bool(prev.get("through_ball", False)):
                return "Through Ball"
            return "Frontal Pass"
        grp = group_events.get((row["match_id"], row["possession"]))
        if grp is None or grp.empty:
            return "Other"
        prev2 = grp[grp["index"] < row["index"]].tail(2)
        if row.get("shot_follows_dribble", False) or (((prev2["player"] == row["player"]) & prev2["type"].isin(["Carry", "Dribble"])).any()):
            return "Carry/Dribble"
        rec = group_recov.get((row["match_id"], row["possession"]))
        if rec is not None and not rec.empty and (rec["location_x"] >= 60).any():
            return "High Regain"
        if (prev2["type"] == "Shot").any():
            return "Second Ball / Rebound"
        shot_type = str(row.get("shot_type", ""))
        if "Free Kick" in shot_type or "Penalty" in shot_type or "Corner" in shot_type:
            return "Set Piece"
        return "Other"

    shots["chance_source"] = shots.apply(classify_shot_source, axis=1) if not shots.empty else pd.Series(dtype=object)

    pos_xg = shots.groupby(["match_id", "possession"])["xg"].sum().rename("possession_xg").reset_index() if not shots.empty else _empty_df(["match_id", "possession", "possession_xg"])
    involved = events.groupby(["match_id", "possession", "player"]).size().rename("events_in_poss").reset_index() if not events.empty else _empty_df(["match_id", "possession", "player", "events_in_poss"])
    if not involved.empty:
        involved = involved.merge(pos_xg, on=["match_id", "possession"], how="left").fillna({"possession_xg": 0})
        player_xg_chain = involved.groupby("player")["possession_xg"].sum()
    else:
        player_xg_chain = pd.Series(dtype=float)

    minute_frames = []
    for df in [passes, receipts, carries, shots, dribbles, recoveries, dispossessed, miscontrols, foul_wons]:
        if not df.empty and {"match_id", "player", "minute"}.issubset(df.columns):
            minute_frames.append(df[["match_id", "player", "minute"]])
    all_player_events = pd.concat(minute_frames, ignore_index=True) if minute_frames else _empty_df(["match_id", "player", "minute"])
    if not all_player_events.empty:
        pm = all_player_events.groupby(["match_id", "player"])["minute"].agg(["min", "max"]).reset_index()
        pm["minutes_est"] = pm["max"] - pm["min"] + 1
        player_minutes = pm.groupby("player")["minutes_est"].sum()
    else:
        player_minutes = pd.Series(dtype=float)

    if any(not df.empty and "player" in df.columns for df in [passes, receipts, carries, shots, dribbles, recoveries, dispossessed, miscontrols, foul_wons]):
        players_list = sorted(set(pd.concat([df["player"] for df in [passes, receipts, carries, shots, dribbles, recoveries, dispossessed, miscontrols, foul_wons] if not df.empty and "player" in df.columns], ignore_index=True).dropna().unique().tolist()))
    else:
        players_list = []

    player_stats = pd.DataFrame({"player": players_list}).set_index("player")
    player_stats["minutes"] = player_minutes
    player_stats["possessions_involved"] = events.groupby("player").apply(lambda g: g[["match_id", "possession"]].drop_duplicates().shape[0]) if not events.empty else 0
    player_stats["receptions"] = receipts.groupby("player").size() if not receipts.empty else 0
    player_stats["progressive_passes"] = passes.groupby("player")["progressive_pass"].sum() if not passes.empty else 0
    player_stats["progressive_carries"] = carries.groupby("player")["progressive_carry"].sum() if not carries.empty else 0
    player_stats["final_third_entries"] = (passes.groupby("player")["entry_to_z3"].sum() if not passes.empty else pd.Series(dtype=float)).add(carries.groupby("player")["entry_to_z3"].sum() if not carries.empty else pd.Series(dtype=float), fill_value=0)
    player_stats["box_entries"] = (passes.groupby("player")["box_entry"].sum() if not passes.empty else pd.Series(dtype=float)).add(carries.groupby("player")["box_entry"].sum() if not carries.empty else pd.Series(dtype=float), fill_value=0)
    player_stats["z14_receptions"] = receipts.groupby("player")["in_z14"].sum() if not receipts.empty else 0
    player_stats["key_passes"] = passes.groupby("player")["key_pass"].sum() if not passes.empty else 0
    player_stats["shot_assists"] = passes.groupby("player")["shot_assist"].sum() if not passes.empty else 0
    player_stats["xA"] = passes.groupby("player")["xA"].sum() if not passes.empty else 0.0
    player_stats["crosses"] = passes.groupby("player")["cross"].sum() if not passes.empty else 0
    player_stats["cutbacks"] = passes.groupby("player")["cutback"].sum() if not passes.empty else 0
    player_stats["through_balls"] = passes.groupby("player")["through_ball"].sum() if not passes.empty else 0
    player_stats["touches_attacking_third"] = events.groupby("player").apply(lambda g: int(((g["x"] >= 80) | (g["end_x"] >= 80)).sum())) if not events.empty else 0
    player_stats["touches_in_box"] = events.groupby("player").apply(lambda g: int((g["in_box_start"] | g["in_box_end"]).sum())) if not events.empty else 0
    player_stats["shots"] = shots.groupby("player").size() if not shots.empty else 0
    player_stats["SoT"] = shots.groupby("player")["is_sot"].sum() if not shots.empty else 0
    player_stats["xG"] = shots.groupby("player")["xg"].sum() if not shots.empty else 0.0
    player_stats["xGOT"] = np.nan
    player_stats["goals"] = shots.groupby("player")["is_goal"].sum() if not shots.empty else 0
    player_stats["avg_shot_distance"] = shots.groupby("player")["shot_distance"].mean() if not shots.empty else np.nan
    player_stats["xG_Chain"] = player_xg_chain
    player_stats["turnovers"] = (dispossessed.groupby("player").size() if not dispossessed.empty else pd.Series(dtype=float)).add(miscontrols.groupby("player").size() if not miscontrols.empty else pd.Series(dtype=float), fill_value=0)
    player_stats["retention_rate_after_reception"] = all_receipts.groupby("player")["retained_after_reception"].mean() if not all_receipts.empty else np.nan
    player_stats = player_stats.fillna(0).reset_index()
    player_stats["per90_factor"] = np.where(player_stats["minutes"] > 0, 90 / player_stats["minutes"], 0.0)
    per90_cols = ["progressive_passes", "progressive_carries", "final_third_entries", "box_entries", "key_passes", "crosses", "cutbacks", "through_balls", "touches_attacking_third", "touches_in_box", "shots", "SoT", "xG", "goals", "xG_Chain"]
    for col in per90_cols:
        player_stats[f"{col}_p90"] = player_stats[col] * player_stats["per90_factor"]
    player_stats = player_stats.sort_values(["minutes", "xG", "key_passes"], ascending=[False, False, False])

    mini_tables = {
        "Build-up players": player_stats.sort_values(["receptions", "progressive_passes"], ascending=False).head(8),
        "Final-third progressors": player_stats.sort_values(["final_third_entries", "box_entries"], ascending=False).head(8),
        "Chance creators": player_stats.sort_values(["key_passes", "xA"], ascending=False).head(8),
        "Finishers": player_stats.sort_values(["xG", "shots"], ascending=False).head(8),
    }

    # ── Build-up type split: compute BEFORE overview so we can embed the values ──
    build_split = pos_df[pos_df["is_build_up"]]["build_up_type"].value_counts(normalize=True).mul(100) if not pos_df.empty else pd.Series(dtype=float)

    overview = {
        "matches": int(len(selected_matches)),
        "possessions": int(len(pos_df)),
        "offensive_possessions": int(pos_df["is_offensive_possession"].sum()) if not pos_df.empty else 0,
        "build_up_possessions": int(pos_df["is_build_up"].sum()) if not pos_df.empty else 0,
        "build_short_pct": float(build_split.get("Short", 0)),
        "build_long_pct": float(build_split.get("Long", 0)),
        "z3_entries": int(len(entries)),
        "shots": int(len(shots)),
        "xg": float(shots["xg"].sum()) if not shots.empty else 0.0,
        "xgot": None,
        "xg_chain_team": None,
        "crosses": int(passes["cross"].sum()) if not passes.empty else 0,
        "cutbacks": int(passes["cutback"].sum()) if not passes.empty else 0,
        "final_third_entry_pct": 100 * safe_div(pos_df["has_final_third"].sum(), len(pos_df)) if not pos_df.empty else 0.0,
        "shot_ending_possession_pct": 100 * safe_div(pos_df["has_shot"].sum(), len(pos_df)) if not pos_df.empty else 0.0,
        "entries_per_match": safe_div(len(entries), len(selected_matches)),
        "xg_per_match": safe_div(shots["xg"].sum(), len(selected_matches)) if not shots.empty else 0.0,
        "xg_per_shot": safe_div(shots["xg"].sum(), len(shots)) if not shots.empty else 0.0,
        "xg_per_possession": safe_div(shots["xg"].sum(), len(pos_df)) if not pos_df.empty else 0.0,
        "shot_conversion_pct": 100 * safe_div(shots["is_goal"].sum(), len(shots)) if not shots.empty else 0.0,
    }
    first_prog_split = first_progression["end_lane"].value_counts(normalize=True).mul(100) if not first_progression.empty else pd.Series(dtype=float)
    recv_profile = build_receipts["role_group"].value_counts(normalize=True).mul(100) if not build_receipts.empty else pd.Series(dtype=float)
    entry_lane_split = entries["entry_lane"].value_counts(normalize=True).mul(100) if not entries.empty else pd.Series(dtype=float)
    shot_source_split = shots["chance_source"].value_counts(normalize=True).mul(100) if not shots.empty else pd.Series(dtype=float)

    def _top_text(series: pd.Series, fallback: str = "unavailable") -> str:
        if series.empty:
            return fallback
        return f"{str(series.idxmax()).lower()} ({float(series.max()):.1f}%)"

    overview_insights = [
        f"England reached the final third in {overview['final_third_entry_pct']:.1f}% of possessions and produced {overview['z3_entries']} total entries ({overview['entries_per_match']:.1f} per match).",
        f"Build-up split: {float(build_split.get('Short', 0)):.1f}% short and {float(build_split.get('Long', 0)):.1f}% long/direct.",
        f"Primary first build-up lane: {_top_text(first_prog_split)}.",
        f"Most common build-up receiver group: {_top_text(recv_profile)}.",
        f"Final-third access leaned toward {_top_text(entry_lane_split)}.",
        f"England generated {overview['xg']:.2f} xG from {overview['shots']} shots ({overview['xg_per_shot']:.3f} xG/shot).",
        f"Main chance source share: {_top_text(shot_source_split)}.",
    ]

    current_code_invalid = [
        "The reported `competition_stage` crash is caused by re-merging a column that already exists in `shots`, creating suffixed columns (`competition_stage_x`, `competition_stage_y`) and then grouping by a non-existent unsuffixed name.",
        "Pages use hardcoded fields without schema validation or placeholders.",
        "Several pages call `.iloc[0]`, `.idxmax()`, or arithmetic on filtered frames that can be empty after sidebar filters.",
        "The current codebase has no final report page or export layer, and no reusable safe merge / safe field utilities.",
    ]
    page_breaks = [
        "Overview can break on the stage xG chart because of duplicate-column merge suffixing.",
        "Build-up can break when filtered data leaves `build_receivers` or `first_progression` empty.",
        "Final Third can break when `entries` is empty and code requests `.iloc[0]` for dominant lane/type.",
        "Chance Creation can break when best-chance chains or assist-pass tables are empty.",
    ]

    audit = {
        "file_names": [name for name in project_data["raw"].keys()],
        "detected_tables": {name: list(df.columns) for name, df in project_data["raw"].items()},
        "coordinate_system": "StatsBomb-style 120 x 80 coordinates detected from location fields.",
        "possessions_exist": True,
        "xg_exists": "shot_statsbomb_xg present in shots",
        "xgot_exists": False,
        "xg_chain_exists": False,
        "missing_columns_used_by_current_code": current_code_invalid,
        "broken_pages_current_code": page_breaks,
    }

    best_chance_chains = []
    if not pos_df.empty:
        top_possessions = pos_df.sort_values(["xg", "has_shot"], ascending=False).head(8)
        match_label_map = selected_matches.set_index("match_id")["match_label"].to_dict() if not selected_matches.empty else {}
        for _, row in top_possessions.iterrows():
            grp = grouped_pos.get((int(row["match_id"]), int(row["possession"])))
            if grp is None or grp.empty:
                continue
            chain = grp[["minute", "type", "player", "subtype", "x", "y", "end_x", "end_y", "xg"]].copy()
            chain["match_id"] = int(row["match_id"])
            chain["match_label"] = match_label_map.get(int(row["match_id"]), f"Match {int(row['match_id'])}")
            chain["possession"] = int(row["possession"])
            chain["possession_xg"] = float(row["xg"])
            best_chance_chains.append(chain)

    availability = {
        "xGOT": False,
        "xG Chain team KPI": False,
        "xT after reception": False,
        "official minutes": False,
        "lineups": False,
    }

    data_caveats = [
        "The uploaded files include match metadata and event-type workbooks, but no lineup / substitution table in the working project extract.",
        "xG is available through `shot_statsbomb_xg`; xGOT is not present.",
        "xG Chain is derived from player involvement in possessions that ended with xG, not loaded directly from source data.",
        "Minutes are estimated from each player's first and last event minute in the filtered sample.",
        "The dashboard uses StatsBomb-style 120 x 80 coordinates and mirrors the y-axis for a conventional attacking-view pitch display.",
    ]

    export_tables = {
        "player_stats": player_stats.copy(),
        "entry_players": entry_players.copy(),
        "build_receivers": build_receivers.copy(),
        "matches": selected_matches.copy(),
        "chance_creators": player_stats.sort_values(["key_passes", "xA"], ascending=False).head(20).copy(),
    }

    return {
        "match": selected_matches,
        "passes": passes,
        "receipts": receipts,
        "carries": carries,
        "shots": shots,
        "dribbles": dribbles,
        "recoveries": recoveries,
        "dispossessed": dispossessed,
        "miscontrols": miscontrols,
        "foul_wons": foul_wons,
        "events": events,
        "pos_df": pos_df,
        "build_passes": build_passes,
        "build_carries": build_carries,
        "build_receipts": build_receipts,
        "build_receivers": build_receivers,
        "build_combo_table": build_combo_table,
        "first_progression": first_progression,
        "entries": entries,
        "first_entries": first_entries,
        "entry_players": entry_players,
        "z14_metrics": z14_metrics,
        "flank_metrics": flank_metrics,
        "assist_passes": assist_passes,
        "player_stats": player_stats,
        "mini_tables": mini_tables,
        "overview": overview,
        "overview_insights": overview_insights,
        "audit": audit,
        "availability": availability,
        "data_caveats": data_caveats,
        "best_chance_chains": best_chance_chains,
        "export_tables": export_tables,
        "meta_map": project_data["meta_map"],
    }


def compute_data_audit(project_data: dict[str, Any]) -> dict[str, Any]:
    raw = project_data["raw"]
    match_meta = project_data["match_meta"]
    file_names = []
    raw_columns = {}
    null_rates = {}
    for name, df in raw.items():
        file_names.append(name)
        raw_columns[name] = list(df.columns)
        null_rates[name] = {} if df.empty else {k: v for k, v in (df.isna().mean() * 100).round(2).to_dict().items()}

    code_issues = [
        "No schema validation layer before page rendering.",
        "Duplicate-column merge pattern causes the `competition_stage` KeyError on the overview page.",
        "Several page narratives assume non-empty filtered datasets and can fail on narrow filters.",
        "No final report or PDF export implementation in the original project.",
    ]

    return {
        "file_names": file_names,
        "detected_tables": raw_columns,
        "null_rates": null_rates,
        "coordinate_system": "StatsBomb 120x80",
        "possessions_exist": all("possession" in df.columns for name, df in raw.items() if name != "matchdetails" and not df.empty),
        "xg_exists": "shot_statsbomb_xg" in raw.get("shots", pd.DataFrame()).columns,
        "xgot_exists": any("xgot" in c.lower() for c in raw.get("shots", pd.DataFrame()).columns),
        "xg_chain_exists": any("chain" in c.lower() for c in raw.get("shots", pd.DataFrame()).columns),
        "lineups_present": False,
        "current_code_files": ["app.py", "preprocess_england_data.py", "requirements.txt"],
        "current_code_issues": code_issues,
        "match_meta_columns": list(match_meta.columns),
    }


def export_tables_excel(tables: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in tables.items():
            df.copy().to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return output.getvalue()
