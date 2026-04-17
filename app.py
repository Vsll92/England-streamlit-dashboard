from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import APP_TITLE, TEAM_NAME, DATA_DIR
from data_loader import load_project_data
from metrics import compute_analysis_bundle
from plots import (
    assist_arrow_map,
    base_pitch,
    binned_zone_map,
    grouped_route_map,
    heatmap_on_pitch,
    pass_network,
    shot_result_map,
    stacked_count_pct,
    summary_bar,
    tactical_summary_pitch,
)
from report_generator import build_final_report_dict, report_to_html, report_to_markdown, report_to_pdf_bytes
from schema_utils import module_available
from wyscout_loader import load_wyscout_team_stats
from dashboard_helpers import (
    build_build_up_outcomes,
    build_pattern_table,
    build_route_families,
    make_export_bytes,
    summarize_best_chances,
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title=APP_TITLE, page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    /* ── Global spacing ── */
    .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px;}
    section[data-testid="stSidebar"] {width: 290px !important;}

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 14px 16px;
        min-height: 100px;
        box-shadow: 0 8px 24px rgba(2,6,23,0.22);
    }
    .metric-label {font-size: 0.82rem; opacity: 0.78; margin-bottom: 0.2rem; letter-spacing: 0.02em;}
    .metric-value {font-size: 1.55rem; font-weight: 700; line-height: 1.15;}
    .metric-sub {font-size: 0.74rem; opacity: 0.62; margin-top: 0.3rem;}

    /* ── Section helpers ── */
    .hero-caption {color:#94a3b8; font-size:0.92rem; margin-bottom:0.6rem;}
    .section-note {color:#94a3b8; font-size:0.88rem; margin-top:-0.1rem; margin-bottom:0.8rem;}
    .info-box {
        background:#0f172a; border:1px solid rgba(255,255,255,0.08); border-radius:12px;
        padding:10px 14px; color:#cbd5e1; font-size:0.88rem; line-height: 1.5;
    }

    /* ── Plotly chart breathing room ── */
    .stPlotlyChart {margin-bottom: 0.4rem;}

    /* ── Dataframe tweaks ── */
    .stDataFrame {margin-top: 0.2rem;}

    /* ── Tab labels ── */
    button[data-baseweb="tab"] {font-size: 0.92rem !important;}

    /* ── Remove Streamlit header padding collision ── */
    h3 {margin-top: 1.2rem !important; margin-bottom: 0.4rem !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fmt_int(v: Any) -> str:
    if pd.isna(v):
        return "—"
    return f"{int(round(float(v))):,}"


def fmt_pct(v: Any) -> str:
    if pd.isna(v):
        return "—"
    return f"{float(v):.1f}%"


def fmt_num(v: Any, d: int = 2) -> str:
    if pd.isna(v):
        return "—"
    return f"{float(v):,.{d}f}"


def metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def get_project_data() -> dict[str, Any]:
    return load_project_data(DATA_DIR)


@st.cache_data(show_spinner=False)
def get_wyscout() -> dict[str, Any]:
    return load_wyscout_team_stats()


@st.cache_data(show_spinner=False)
def get_analysis(match_ids: tuple[int, ...]) -> dict[str, Any]:
    bundle = compute_analysis_bundle(get_project_data(), match_ids)
    ws = get_wyscout()["team_stats"].copy()
    if not ws.empty:
        ws["match_date"] = pd.to_datetime(ws["match_date"], errors="coerce")
        bundle["wyscout_match_ref"] = bundle["match"].merge(ws, on="match_date", how="left")
    else:
        bundle["wyscout_match_ref"] = pd.DataFrame()
    return bundle


# ══════════════════════════════════════════════════════════════════════════════
# SAFE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def filter_match_scope(df: pd.DataFrame, match_id: int | None) -> pd.DataFrame:
    if df is None or df.empty or match_id is None:
        return df
    if "match_id" not in df.columns:
        return df
    return df[df["match_id"].eq(match_id)].copy()


def safe_chain_lookup(chains: list[pd.DataFrame], target_match_id: int, target_possession: int) -> pd.DataFrame:
    """Robustly find the right chain — handles missing match_id gracefully."""
    for c in chains:
        if c.empty:
            continue
        # Primary: use match_id + possession if both present
        if "match_id" in c.columns and "possession" in c.columns:
            try:
                if int(c["match_id"].iloc[0]) == target_match_id and int(c["possession"].iloc[0]) == target_possession:
                    return c
            except (ValueError, TypeError, IndexError):
                continue
        # Fallback: match on possession only (less precise but won't crash)
        elif "possession" in c.columns:
            try:
                if int(c["possession"].iloc[0]) == target_possession:
                    return c
            except (ValueError, TypeError, IndexError):
                continue
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# TACTICAL NOTES
# ══════════════════════════════════════════════════════════════════════════════

def tactical_summary_notes(bundle: dict[str, Any]) -> tuple[str, float, str | None, float | None, list[str]]:
    first_prog = bundle["first_progression"]
    build_receivers = bundle["build_receivers"]
    pos_df = bundle["pos_df"]
    if first_prog.empty:
        return "Center", 0.0, None, None, ["No build-up progression actions in the current filter."]
    lane_share = first_prog["end_lane"].value_counts(normalize=True).mul(100)
    primary_lane = str(lane_share.idxmax())
    primary_pct = float(lane_share.max())
    secondary_lane, secondary_pct = None, None
    if len(lane_share) > 1:
        secondary_lane = str(lane_share.index[1])
        secondary_pct = float(lane_share.iloc[1])
    gk_involved = 0
    if not bundle["build_passes"].empty:
        gk_involved = int((bundle["build_passes"]["role_group"] == "Goalkeeper").sum())
    notes = [
        f"Short build-up: {bundle['overview'].get('build_short_pct', 0):.1f}% · Long/direct: {bundle['overview'].get('build_long_pct', 0):.1f}%",
        f"Top receiver: {build_receivers.iloc[0]['player']} ({int(build_receivers.iloc[0]['receptions'])} receptions)" if not build_receivers.empty else "No stable build-up receiver in the filter.",
        f"Build-up possessions: {int(pos_df['is_build_up'].sum())} of {len(pos_df)} total",
        f"GK involvements: {gk_involved} passing actions" if gk_involved else "Limited goalkeeper involvement",
    ]
    return primary_lane, primary_pct, secondary_lane, secondary_pct, notes


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION TABLE
# ══════════════════════════════════════════════════════════════════════════════

def progression_validation_table(bundle: dict[str, Any], ws_ref: pd.DataFrame) -> pd.DataFrame:
    passes = bundle["passes"]
    carries = bundle["carries"]
    rows = [
        {
            "metric": "Progressive passes",
            "event_total": int(passes["progressive_pass"].sum()) if not passes.empty else 0,
            "matches": int(passes["match_id"].nunique()) if not passes.empty else 0,
            "wyscout_ref": int(ws_ref["ws_progressive_passes_accurate"].sum()) if not ws_ref.empty and "ws_progressive_passes_accurate" in ws_ref.columns else "—",
            "definition": "Forward progress via zone-based x-gain thresholds on 120×80 pitch",
        },
        {
            "metric": "Progressive carries",
            "event_total": int(carries["progressive_carry"].sum()) if not carries.empty else 0,
            "matches": int(carries["match_id"].nunique()) if not carries.empty else 0,
            "wyscout_ref": "—",
            "definition": "Carries meeting the same progressive x-gain thresholds",
        },
        {
            "metric": "Passes into Z3",
            "event_total": int(passes["entry_to_z3"].sum()) if not passes.empty else 0,
            "matches": int(passes["match_id"].nunique()) if not passes.empty else 0,
            "wyscout_ref": int(ws_ref["ws_passes_to_final_third_accurate"].sum()) if not ws_ref.empty and "ws_passes_to_final_third_accurate" in ws_ref.columns else "—",
            "definition": "Completed passes from x < 80 into x ≥ 80",
        },
        {
            "metric": "Carries into Z3",
            "event_total": int(carries["entry_to_z3"].sum()) if not carries.empty else 0,
            "matches": int(carries["match_id"].nunique()) if not carries.empty else 0,
            "wyscout_ref": "—",
            "definition": "Carries from x < 80 into x ≥ 80",
        },
        {
            "metric": "Carries into box",
            "event_total": int(carries["box_entry"].sum()) if not carries.empty else 0,
            "matches": int(carries["match_id"].nunique()) if not carries.empty else 0,
            "wyscout_ref": "—",
            "definition": "Carries ending inside the penalty box",
        },
    ]
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PITCH VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def best_chance_pitch(chain_df: pd.DataFrame) -> go.Figure:
    fig = base_pitch("Selected high-value possession", "Event sequence for the chosen possession", height=500, show_z14_label=True)
    if chain_df.empty:
        return fig
    use = chain_df.copy().sort_values(["minute"]).reset_index(drop=True)
    colors = {"Pass": "#38bdf8", "Carry": "#22c55e", "Shot": "#ef4444"}
    for i, row in use.iterrows():
        if pd.notna(row.get("end_x")) and pd.notna(row.get("end_y")) and pd.notna(row.get("x")) and pd.notna(row.get("y")):
            fig.add_annotation(
                x=float(row["end_x"]), y=float(row["end_y"]),
                ax=float(row["x"]), ay=float(row["y"]),
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1,
                arrowwidth=2.2, arrowcolor=colors.get(row["type"], "#94a3b8"), opacity=0.75,
            )
            xg_val = row.get("xg", np.nan)
            xg_text = f"xG: {xg_val:.3f}" if pd.notna(xg_val) and float(xg_val) > 0 else ""
            fig.add_trace(go.Scatter(
                x=[float(row["x"])], y=[float(row["y"])], mode="markers+text",
                text=[str(i + 1)], textposition="top center",
                marker=dict(size=10, color=colors.get(row["type"], "#94a3b8"), line=dict(color="#0f172a", width=1)),
                hovertemplate=f"{row.get('player', '?')}<br>{row.get('type', '?')} · {row.get('subtype', '')}<br>Min {row.get('minute', '?')}<br>{xg_text}<extra></extra>",
                showlegend=False,
            ))
    return fig


def pattern_pitch(pattern_row: pd.Series) -> go.Figure:
    fig = base_pitch("Common build-up pattern", "Blue = pass, green = carry", height=500)
    steps = pattern_row.get("steps", []) if pattern_row is not None else []
    if not steps:
        return fig
    colors = {"Pass": "#38bdf8", "Carry": "#22c55e"}
    for idx, step in enumerate(steps):
        x0, y0, x1, y1 = step.get("x"), step.get("y"), step.get("end_x"), step.get("end_y")
        if pd.notna(x0) and pd.notna(y0):
            fig.add_trace(go.Scatter(
                x=[x0], y=[y0], mode="markers+text", text=[str(idx + 1)], textposition="top center",
                marker=dict(size=11, color=colors.get(step.get("type"), "#94a3b8")),
                hovertemplate=f"{step.get('player')}<br>{step.get('type')}<extra></extra>",
                showlegend=False,
            ))
        if pd.notna(x0) and pd.notna(y0) and pd.notna(x1) and pd.notna(y1):
            fig.add_annotation(
                x=float(x1), y=float(y1), ax=float(x0), ay=float(y0),
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=2.2,
                arrowcolor=colors.get(step.get("type"), "#94a3b8"), opacity=0.8,
            )
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#38bdf8", width=3), name="Pass"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#22c55e", width=3), name="Carry"))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# BUILD-UP MAP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_map_dataset(bundle: dict[str, Any], map_name: str, match_id: int | None, player_name: str | None) -> tuple[pd.DataFrame, str, str]:
    build_passes = filter_match_scope(bundle["build_passes"], match_id)
    build_carries = filter_match_scope(bundle["build_carries"], match_id)
    build_receipts = filter_match_scope(bundle["build_receipts"], match_id)
    if player_name and player_name != "All players":
        build_passes = build_passes[build_passes["player"].eq(player_name)] if not build_passes.empty else build_passes
        build_carries = build_carries[build_carries["player"].eq(player_name)] if not build_carries.empty else build_carries
        build_receipts = build_receipts[build_receipts["player"].eq(player_name)] if not build_receipts.empty else build_receipts

    if map_name == "All build-up touches":
        df = pd.concat([
            build_passes.assign(plot_x=build_passes["location_x"], plot_y=build_passes["y_std"], plot_kind="Pass start"),
            build_carries.assign(plot_x=build_carries["location_x"], plot_y=build_carries["y_std"], plot_kind="Carry start"),
            build_receipts.assign(plot_x=build_receipts["location_x"], plot_y=build_receipts["y_std"], plot_kind="Receipt"),
        ], ignore_index=True)
        return df, "All build-up touches", "Pass starts, carry starts, and receptions inside build-up possessions."
    if map_name == "Successful build-up actions":
        succ = build_passes[build_passes["successful"]] if not build_passes.empty else build_passes
        df = pd.concat([
            succ.assign(plot_x=succ["location_x"], plot_y=succ["y_std"], plot_kind="Completed pass"),
            build_carries.assign(plot_x=build_carries["location_x"], plot_y=build_carries["y_std"], plot_kind="Carry"),
        ], ignore_index=True)
        return df, "Successful build-up actions", "Completed passes and carries during build-up possessions."
    if map_name == "Unsuccessful build-up actions":
        fail = build_passes[~build_passes["successful"]] if not build_passes.empty else build_passes
        df = fail.assign(plot_x=fail["location_x"], plot_y=fail["y_std"], plot_kind="Incomplete pass")
        return df, "Unsuccessful build-up actions", "Incomplete passes in build-up possessions."
    if map_name == "Build-up receptions":
        df = build_receipts.assign(plot_x=build_receipts["location_x"], plot_y=build_receipts["y_std"], plot_kind="Receipt")
        return df, "Build-up receptions", "Ball receipts inside build-up possessions."
    if map_name == "Build-up pass starts":
        df = build_passes.assign(plot_x=build_passes["location_x"], plot_y=build_passes["y_std"], plot_kind="Pass start")
        return df, "Build-up pass starts", "Start locations of build-up passes."
    if map_name == "Build-up carries":
        df = build_carries.assign(plot_x=build_carries["location_x"], plot_y=build_carries["y_std"], plot_kind="Carry start")
        return df, "Build-up carries", "Carry start locations in build-up possessions."
    if map_name == "Build-up turnovers / losses":
        df = pd.concat([
            filter_match_scope(bundle["dispossessed"], match_id).assign(plot_x=lambda d: d["location_x"], plot_y=lambda d: d["y_std"], plot_kind="Dispossessed"),
            filter_match_scope(bundle["miscontrols"], match_id).assign(plot_x=lambda d: d["location_x"], plot_y=lambda d: d["y_std"], plot_kind="Miscontrol"),
        ], ignore_index=True)
        if player_name and player_name != "All players" and not df.empty:
            df = df[df["player"].eq(player_name)]
        return df, "Build-up turnovers / losses", "Locations where England lost the ball during build-up."
    if map_name == "Build-up outcome zones":
        df = pd.concat([
            build_passes.assign(plot_x=build_passes["pass_end_location_x"], plot_y=build_passes["end_y_std"], plot_kind=np.where(build_passes["successful"], "Completed pass end", "Incomplete pass end")),
            build_carries.assign(plot_x=build_carries["carry_end_location_x"], plot_y=build_carries["end_y_std"], plot_kind="Carry end"),
        ], ignore_index=True)
        return df, "Build-up outcome zones", "Where build-up actions ended on the pitch."
    return pd.DataFrame(), "Build-up map", "No data."


def render_build_map(df: pd.DataFrame, title: str, subtitle: str, display_mode: str) -> go.Figure:
    fig = base_pitch(title, subtitle, height=500)
    if df.empty:
        return fig
    if display_mode == "Binned zones":
        return binned_zone_map(df, "plot_x", "plot_y", title, subtitle, hover_cols=["player", "plot_kind"])
    if display_mode == "Heatmap + points":
        return heatmap_on_pitch(fig, df["plot_x"], df["plot_y"], name="Actions", overlay_points=True)
    if display_mode == "Scatter points":
        fig.add_trace(go.Scatter(
            x=df["plot_x"], y=df["plot_y"], mode="markers",
            marker=dict(size=8, color="#38bdf8", opacity=0.72),
            text=df.get("player"), hovertemplate="%{text}<br>%{customdata}<extra></extra>",
            customdata=df.get("plot_kind"), name="Actions",
        ))
        return fig
    if display_mode == "Action paths" and {"location_x", "y_std"}.issubset(df.columns):
        x1 = "pass_end_location_x" if "pass_end_location_x" in df.columns else ("carry_end_location_x" if "carry_end_location_x" in df.columns else None)
        y1 = "end_y_std" if "end_y_std" in df.columns else None
        if x1 and y1:
            use = df.dropna(subset=["location_x", "y_std", x1, y1]).head(120)
            for _, row in use.iterrows():
                fig.add_annotation(
                    x=float(row[x1]), y=float(row[y1]), ax=float(row["location_x"]), ay=float(row["y_std"]),
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.8,
                    arrowcolor="#38bdf8", opacity=0.35,
                )
        return fig
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOAD & SIDEBAR FILTERS
# ══════════════════════════════════════════════════════════════════════════════

project_data = get_project_data()
meta = project_data["match_meta"].copy()
meta["match_date"] = pd.to_datetime(meta["match_date"], errors="coerce")

st.title(APP_TITLE)
st.markdown(
    '<div class="hero-caption">England-focused dashboard built from StatsBomb-style Euro 2024 event data, '
    'reconciled with Wyscout team-stat Excel tables where available.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Filters")
    st.selectbox("Team", [TEAM_NAME], index=0)
    competitions = sorted(meta["competition"].dropna().unique().tolist())
    stages = sorted(meta["competition_stage"].dropna().unique().tolist())
    opponents = sorted(meta["opponent"].dropna().unique().tolist())
    competition_sel = st.multiselect("Competition", competitions, default=competitions)
    stage_sel = st.multiselect("Stage", stages, default=stages)
    opp_sel = st.multiselect("Opponent", opponents, default=opponents)
    date_min = meta["match_date"].min().date()
    date_max = meta["match_date"].max().date()
    date_sel = st.slider("Date range", min_value=date_min, max_value=date_max, value=(date_min, date_max))
    candidate = meta[
        meta["competition"].isin(competition_sel)
        & meta["competition_stage"].isin(stage_sel)
        & meta["opponent"].isin(opp_sel)
        & meta["match_date"].dt.date.between(date_sel[0], date_sel[1])
    ].copy()
    match_labels = candidate["match_label"].tolist()
    selected_labels = st.multiselect("Match", match_labels, default=match_labels)

selected_match_ids = tuple(candidate[candidate["match_label"].isin(selected_labels)]["match_id"].tolist())
if not selected_match_ids:
    st.warning("No matches match the current filters.")
    st.stop()

bundle = get_analysis(tuple(sorted(selected_match_ids)))
match = bundle["match"]
passes = bundle["passes"]
receipts = bundle["receipts"]
carries = bundle["carries"]
shots = bundle["shots"]
events = bundle["events"]
pos_df = bundle["pos_df"]
ws_ref = bundle.get("wyscout_match_ref", pd.DataFrame())
route_families = build_route_families(bundle)
build_summary, build_by_match = build_build_up_outcomes(bundle)
pattern_table = build_pattern_table(bundle)
best_chances = summarize_best_chances(bundle)
validation_df = progression_validation_table(bundle, ws_ref)

bundle["wyscout_match_ref"] = ws_ref
report = build_final_report_dict(bundle)
report_md = report_to_markdown(report)
report_html = report_to_html(report)
report_pdf = report_to_pdf_bytes(report, export_tables=bundle.get("export_tables", {}))


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

overview_tab, build_tab, z3_tab, chance_tab, players_tab, maps_tab, audit_tab, report_tab = st.tabs(
    ["Overview", "Build-up Phase", "Final Third", "Chance Creation", "Players", "Maps", "Audit", "Report"]
)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  OVERVIEW TAB                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with overview_tab:
    r1 = st.columns(6)
    cards = [
        ("Possessions", fmt_int(bundle["overview"]["possessions"]), "England possessions in the filtered sample"),
        ("Build-up poss.", fmt_int(bundle["overview"]["build_up_possessions"]), f'{bundle["overview"]["build_short_pct"]:.0f}% short · {bundle["overview"]["build_long_pct"]:.0f}% long'),
        ("Z3 entries", fmt_int(bundle["overview"]["z3_entries"]), f"{bundle['overview']['entries_per_match']:.1f} per match"),
        ("Shots", fmt_int(bundle["overview"]["shots"]), f"{bundle['overview']['shot_ending_possession_pct']:.1f}% of possessions → shot"),
        ("Event xG", fmt_num(bundle["overview"]["xg"], 2), f"{bundle['overview']['xg_per_match']:.2f} xG per match"),
        ("WS shots / SoT", f"{fmt_int(ws_ref['ws_shots'].sum()) if not ws_ref.empty and 'ws_shots' in ws_ref.columns else '—'} / {fmt_int(ws_ref['ws_shots_on_target'].sum()) if not ws_ref.empty and 'ws_shots_on_target' in ws_ref.columns else '—'}", "Wyscout team-stat reference"),
    ]
    for col, card in zip(r1, cards):
        with col:
            metric_card(*card)

    st.markdown("")  # spacer
    lane, lane_pct, lane2, lane2_pct, notes = tactical_summary_notes(bundle)
    oc1, oc2 = st.columns([1.15, 0.85])
    with oc1:
        st.plotly_chart(tactical_summary_pitch(lane, lane_pct, notes, "Pitch-based analyst summary", lane2, lane2_pct), use_container_width=True)
    with oc2:
        st.markdown("### Sample context")
        st.markdown(
            '<div class="info-box">This view combines event-derived possessions and pitch actions with uploaded '
            'Wyscout team tables, so headline numbers are supported by match-level reference totals.</div>',
            unsafe_allow_html=True,
        )
        for bullet in bundle["overview_insights"]:
            st.markdown(f"- {bullet}")

    st.markdown("")
    prog_profile = pd.DataFrame([
        {"metric": "Progressive passes", "count": int(passes["progressive_pass"].sum()), "subtitle": "Successful progressive passes"},
        {"metric": "Progressive carries", "count": int(carries["progressive_carry"].sum()), "subtitle": "Progressive carries"},
        {"metric": "Passes into Z3", "count": int(passes["entry_to_z3"].sum()), "subtitle": "Completed passes entering the final third"},
        {"metric": "Carries into Z3", "count": int(carries["entry_to_z3"].sum()), "subtitle": "Carries entering the final third"},
        {"metric": "Carries into box", "count": int(carries["box_entry"].sum()), "subtitle": "Carries ending inside the box"},
    ])
    oc3, oc4 = st.columns([1.0, 1.0])
    with oc3:
        st.plotly_chart(summary_bar(prog_profile, "metric", "count", "Progression profile", "Total actions across all filtered matches", color="#38bdf8", text_fmt="d"), use_container_width=True)
    with oc4:
        chance_source = shots["chance_source"].value_counts().reset_index()
        chance_source.columns = ["source", "count"]
        chance_source["share_pct"] = np.where(chance_source["count"].sum() > 0, chance_source["count"] / chance_source["count"].sum() * 100, 0)
        st.plotly_chart(stacked_count_pct(chance_source, "source", "count", "share_pct", "Chance source mix", "Shot counts plus share of England's total shots"), use_container_width=True)

    st.markdown("### Progression validation")
    st.dataframe(validation_df, use_container_width=True, hide_index=True)

    if not ws_ref.empty:
        ws_overview = ws_ref[[c for c in ["match_date", "opponent", "competition_stage", "ws_progressive_passes_accurate", "ws_passes_to_final_third_accurate", "ws_crosses", "ws_crosses_accurate", "ws_avg_shot_distance", "ws_long_pass_pct"] if c in ws_ref.columns]].copy()
        if not ws_overview.empty:
            st.markdown("### Wyscout match reference")
            st.dataframe(ws_overview.sort_values("match_date"), use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BUILD-UP PHASE TAB                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with build_tab:
    st.markdown("### Build-up phase")
    st.markdown(
        '<div class="section-note"><b>Build-up</b> = possessions starting in the defensive third (x&lt;40) or '
        'initiated by a goalkeeper / centre-back below x=60. <b>Short</b> = first 3 passes are all short (&lt;30 m) '
        'and none reaches x≥70. <b>Long/Direct</b> = at least one of the first 3 passes is ≥30 m or reaches x≥70.</div>',
        unsafe_allow_html=True,
    )

    # ── Controls row (separated from charts) ──
    ctrl1, ctrl2 = st.columns([0.55, 0.45])
    with ctrl1:
        match_scope_options = {"All filtered matches": None}
        match_scope_options.update({row.match_label: int(row.match_id) for row in match.itertuples()})
        match_scope_label = st.selectbox("Match scope", list(match_scope_options.keys()), index=0, key="build_match_scope")
    with ctrl2:
        map_player_options = ["All players"] + sorted(
            pd.concat([bundle["build_passes"]["player"], bundle["build_receipts"]["player"], bundle["build_carries"]["player"]], ignore_index=True)
            .dropna().astype(str).unique().tolist()
        )
        build_player = st.selectbox("Player filter", map_player_options, index=0, key="build_player")
    selected_scope_match = match_scope_options[match_scope_label]

    scope_build_passes = filter_match_scope(bundle["build_passes"], selected_scope_match)
    scope_build_receipts = filter_match_scope(bundle["build_receipts"], selected_scope_match)
    scope_build_carries = filter_match_scope(bundle["build_carries"], selected_scope_match)
    scope_route_families = build_route_families({**bundle, "build_passes": scope_build_passes})
    scope_patterns = build_pattern_table({
        **bundle,
        "events": filter_match_scope(bundle["events"], selected_scope_match),
        "pos_df": bundle["pos_df"][bundle["pos_df"]["match_id"].eq(selected_scope_match)] if selected_scope_match else bundle["pos_df"],
    })

    # ── KPI cards ──
    st.markdown("")
    bkpi = st.columns(5)
    gk_build = int((scope_build_passes["role_group"] == "Goalkeeper").sum()) if not scope_build_passes.empty else 0
    with bkpi[0]:
        metric_card("Build-up poss.", fmt_int(bundle["overview"]["build_up_possessions"]), f"of {bundle['overview']['possessions']} total possessions")
    with bkpi[1]:
        metric_card("Short build-up", fmt_pct(bundle["overview"].get("build_short_pct", 0)), "Short passing sequences")
    with bkpi[2]:
        metric_card("Long / direct", fmt_pct(bundle["overview"].get("build_long_pct", 0)), "Long ball or direct progression")
    with bkpi[3]:
        metric_card("GK actions", fmt_int(gk_build), "Goalkeeper passes in build-up")
    with bkpi[4]:
        retain_pct = build_summary.loc[build_summary["label"] == "Keep possession", "pct"].iloc[0] if not build_summary.empty and "Keep possession" in build_summary["label"].values else 0
        metric_card("Retention", fmt_pct(retain_pct), "Keep ball after opening phase")

    # ── First progression lanes + Wyscout ref ──
    st.markdown("")
    build_lane_share = bundle["first_progression"]["end_lane"].value_counts().reset_index()
    build_lane_share.columns = ["lane", "count"]
    build_lane_share["share_pct"] = np.where(build_lane_share["count"].sum() > 0, build_lane_share["count"] / build_lane_share["count"].sum() * 100, 0)
    br1, br2 = st.columns([1.0, 1.0])
    with br1:
        st.plotly_chart(stacked_count_pct(build_lane_share, "lane", "count", "share_pct", "First progression lanes", "Lane of the first forward build-up action"), use_container_width=True)
    with br2:
        ws_build = ws_ref[[c for c in ["match_date", "opponent", "competition_stage", "ws_long_pass_pct", "ws_avg_passes_per_possession", "ws_avg_pass_length", "ws_goal_kicks", "ws_progressive_passes_accurate"] if c in ws_ref.columns]].copy() if not ws_ref.empty else pd.DataFrame()
        if not ws_build.empty:
            st.markdown("**Wyscout build-up reference**")
            st.dataframe(ws_build.sort_values("match_date"), use_container_width=True, hide_index=True)
        else:
            st.info("Wyscout build-up reference tables are not available for the current filter.")

    # ── Build-up route families (redesigned) ──
    st.markdown("### Build-up route families")
    st.markdown('<div class="section-note">Arrow width = route volume · colour = completion rate (blue ≥70%, green 45–70%, amber &lt;45%). Numbers show attempt counts.</div>', unsafe_allow_html=True)
    lp1, lp2 = st.columns([1.15, 0.85])
    with lp1:
        st.plotly_chart(grouped_route_map(scope_route_families, "Build-up route families", "Top 10 zone-to-zone passing routes by volume"), use_container_width=True)
    with lp2:
        route_table = scope_route_families[["route", "attempts", "completed", "completion_pct", "main_players", "match_count"]].copy() if not scope_route_families.empty else pd.DataFrame()
        if not route_table.empty:
            st.dataframe(route_table.head(12), use_container_width=True, hide_index=True)
        else:
            st.info("No route data available.")

    # ── Build-up map explorer ──
    st.markdown("### Build-up map explorer")
    mc1, mc2 = st.columns(2)
    with mc1:
        build_map_name = st.selectbox("Map type", [
            "All build-up touches", "Successful build-up actions", "Unsuccessful build-up actions",
            "Build-up receptions", "Build-up pass starts", "Build-up carries",
            "Build-up turnovers / losses", "Build-up outcome zones",
        ], key="build_map_type")
    with mc2:
        build_display_mode = st.selectbox("Display", ["Binned zones", "Heatmap + points", "Scatter points", "Action paths"], index=0, key="build_display")
    map_df, map_title, map_sub = build_map_dataset(bundle, build_map_name, selected_scope_match, build_player)
    st.plotly_chart(render_build_map(map_df, map_title, map_sub, build_display_mode), use_container_width=True)

    # ── Expanders ──
    with st.expander("Average build-up positions", expanded=False):
        avg_pos = pd.concat([
            scope_build_passes[["player", "location_x", "y_std"]].rename(columns={"location_x": "x", "y_std": "y"}),
            scope_build_receipts[["player", "location_x", "y_std"]].rename(columns={"location_x": "x", "y_std": "y"}),
            scope_build_carries[["player", "location_x", "y_std"]].rename(columns={"location_x": "x", "y_std": "y"}),
        ], ignore_index=True).dropna()
        fig = base_pitch("Average build-up positions", "Mean on-ball locations inside build-up possessions", height=500)
        if not avg_pos.empty:
            avg_pos = avg_pos.groupby("player").agg(x=("x", "mean"), y=("y", "mean"), actions=("x", "count")).reset_index()
            fig.add_trace(go.Scatter(
                x=avg_pos["x"], y=avg_pos["y"], mode="markers+text",
                text=avg_pos["player"], textposition="top center",
                marker=dict(size=np.clip(avg_pos["actions"] * 0.55, 10, 28), color="#f8fafc", line=dict(color="#0f172a", width=1)),
                hovertemplate="%{text}<br>Actions: %{marker.size:.0f}<extra></extra>", showlegend=False,
            ))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Build-up passing network", expanded=False):
        net_passes = scope_build_passes[scope_build_passes["successful"]] if not scope_build_passes.empty else scope_build_passes
        st.plotly_chart(pass_network(net_passes, "Build-up passing network", "Selected build-up scope only"), use_container_width=True)

    # ── Outcomes ──
    st.markdown("### What happens after build-up?")
    bo1, bo2 = st.columns([1.15, 0.85])
    with bo1:
        st.plotly_chart(stacked_count_pct(build_summary, "label", "count", "pct", "Outcomes after build-up", "Share of build-up possessions in the current filter"), use_container_width=True)
    with bo2:
        if not build_by_match.empty:
            st.dataframe(build_by_match.sort_values("match_date"), use_container_width=True, hide_index=True)

    # ── Patterns ──
    st.markdown("### Common build-up patterns")
    patt_display = scope_patterns[scope_patterns["pattern"].str.contains("→", na=False)].copy() if not scope_patterns.empty else pd.DataFrame()
    if patt_display.empty:
        patt_display = scope_patterns.copy()
    if patt_display.empty:
        st.info("No repeatable build-up pattern in the current sample.")
    else:
        pattern_options = [f"{row.pattern} | freq {int(row.frequency)}" for row in patt_display.itertuples()]
        selected_pattern_label = st.selectbox("Pattern selector", pattern_options, index=0)
        selected_row = patt_display.iloc[pattern_options.index(selected_pattern_label)]
        pp1, pp2 = st.columns([1.05, 0.95])
        with pp1:
            st.plotly_chart(pattern_pitch(selected_row), use_container_width=True)
        with pp2:
            st.dataframe(patt_display[["pattern", "frequency", "match_count", "sample_match_id", "sample_possession"]], use_container_width=True, hide_index=True)

    # ── Analyst summary ──
    top_receiver_text = f"{bundle['build_receivers'].iloc[0]['player']} led build-up receptions." if not bundle['build_receivers'].empty else "No clear build-up reception leader."
    st.markdown("### Analyst summary")
    st.markdown(
        f"- England used a mixed build-up profile: **{bundle['overview'].get('build_short_pct', 0):.1f}%** short, "
        f"**{bundle['overview'].get('build_long_pct', 0):.1f}%** long/direct."
    )
    st.markdown(f"- The first progression most often travelled through the **{lane.lower()}** lane ({lane_pct:.1f}%).")
    st.markdown(f"- {top_receiver_text}")
    st.markdown(f"- Goalkeeper contributed {gk_build} build-up passes in the selected scope.")
    st.markdown(f"- England retained possession after the opening build-up in **{retain_pct:.1f}%** of cases.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FINAL THIRD TAB                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with z3_tab:
    st.markdown("### Final third access")
    z1 = st.columns(5)
    with z1[0]:
        metric_card("Possessions → Z3", fmt_pct(pos_df["has_final_third"].mean() * 100 if not pos_df.empty else 0), "Share reaching x ≥ 80")
    with z1[1]:
        metric_card("Total entries", fmt_int(len(bundle["entries"])), "Pass + carry entries into Z3")
    with z1[2]:
        metric_card("Pass entries", fmt_int(bundle["passes"]["entry_to_z3"].sum() if not passes.empty else 0), "Completed passes into x ≥ 80")
    with z1[3]:
        metric_card("Carry entries", fmt_int(bundle["carries"]["entry_to_z3"].sum() if not carries.empty else 0), "Carries into x ≥ 80")
    with z1[4]:
        ws_z3_val = fmt_int(ws_ref["ws_passes_to_final_third_accurate"].sum()) if not ws_ref.empty and "ws_passes_to_final_third_accurate" in ws_ref.columns else "—"
        metric_card("WS Z3 passes", ws_z3_val, "Wyscout reference")

    st.markdown("")
    entry_lane = bundle["entries"]["entry_lane"].value_counts().reset_index()
    entry_lane.columns = ["lane", "count"]
    entry_lane["share_pct"] = np.where(entry_lane["count"].sum() > 0, entry_lane["count"] / entry_lane["count"].sum() * 100, 0)

    entry_type = bundle["entries"]["entry_type"].value_counts().reset_index()
    entry_type.columns = ["entry_type", "count"]
    entry_type["share_pct"] = np.where(entry_type["count"].sum() > 0, entry_type["count"] / entry_type["count"].sum() * 100, 0)

    post_entry = pd.DataFrame({
        "outcome": ["Retain possession", "Enter box", "Create shot", "Move into Z14", "Move wide", "Lose possession", "Cross after entry", "Cutback after entry"],
        "count": [
            bundle["first_entries"]["retain_possession"].sum() if not bundle["first_entries"].empty else 0,
            bundle["first_entries"]["enter_box"].sum() if not bundle["first_entries"].empty else 0,
            bundle["first_entries"]["create_shot"].sum() if not bundle["first_entries"].empty else 0,
            bundle["first_entries"]["move_z14"].sum() if not bundle["first_entries"].empty else 0,
            bundle["first_entries"]["move_wide"].sum() if not bundle["first_entries"].empty else 0,
            bundle["first_entries"]["lose_possession"].sum() if not bundle["first_entries"].empty else 0,
            bundle["first_entries"]["cross_after"].sum() if not bundle["first_entries"].empty else 0,
            bundle["first_entries"]["cutback_after"].sum() if not bundle["first_entries"].empty else 0,
        ],
    })
    first_entry_count = max(len(bundle["first_entries"]), 1)
    post_entry["share_pct"] = post_entry["count"] / first_entry_count * 100

    z21, z22 = st.columns([1.0, 1.0])
    with z21:
        dom_lane = entry_lane.sort_values("count", ascending=False).iloc[0] if not entry_lane.empty else pd.Series({"lane": "Center", "share_pct": 0})
        st.plotly_chart(tactical_summary_pitch(str(dom_lane["lane"]), float(dom_lane["share_pct"]), [
            f"Total Z3 entries: {len(bundle['entries'])}",
            f"Pass entries: {int(passes['entry_to_z3'].sum() if not passes.empty else 0)} · Carry entries: {int(carries['entry_to_z3'].sum() if not carries.empty else 0)}",
            f"Z14 entries: {int(bundle['z14_metrics']['z14_entries'])}",
            f"Shots after Z14: {int(bundle['z14_metrics']['shots_after_z14'])}",
        ], "Final-third summary pitch"), use_container_width=True)
    with z22:
        st.plotly_chart(stacked_count_pct(entry_lane, "lane", "count", "share_pct", "Where England enter Z3", "Counts and shares of Z3 entries"), use_container_width=True)

    z23, z24 = st.columns([1.0, 1.0])
    with z23:
        st.plotly_chart(stacked_count_pct(entry_type, "entry_type", "count", "share_pct", "How England enter Z3", "Entry action type breakdown"), use_container_width=True)
    with z24:
        st.plotly_chart(stacked_count_pct(post_entry, "outcome", "count", "share_pct", "After first Z3 entry", "What happens next — denominator = first entry events"), use_container_width=True)

    z14_df = pd.DataFrame({
        "metric": ["Entries into Z14", "Touches in Z14", "Passes received in Z14", "Shots after Z14"],
        "count": [bundle["z14_metrics"]["z14_entries"], bundle["z14_metrics"]["z14_touches"], bundle["z14_metrics"]["z14_receipts"], bundle["z14_metrics"]["shots_after_z14"]],
        "per_match": [bundle["z14_metrics"]["z14_entries"] / max(len(match), 1), bundle["z14_metrics"]["z14_touches"] / max(len(match), 1), bundle["z14_metrics"]["z14_receipts"] / max(len(match), 1), bundle["z14_metrics"]["shots_after_z14"] / max(len(match), 1)],
    })
    z25, z26 = st.columns([1.0, 1.0])
    with z25:
        st.plotly_chart(summary_bar(z14_df, "metric", "count", "Z14 involvement", "Totals across filtered matches", color="#f59e0b"), use_container_width=True)
    with z26:
        flank_df = bundle["flank_metrics"].copy()
        if not flank_df.empty:
            flank_df["crosses"] = flank_df["cross_frequency"]
            flank_df["cutbacks"] = flank_df["cutback_frequency"]
            st.markdown("**Flank metrics**")
            st.dataframe(flank_df, use_container_width=True, hide_index=True)
        st.markdown("**Z14 detail**")
        st.dataframe(z14_df, use_container_width=True, hide_index=True)

    if not ws_ref.empty:
        ws_z3 = ws_ref[[c for c in ["match_date", "opponent", "competition_stage", "ws_passes_to_final_third", "ws_passes_to_final_third_accurate", "ws_smart_passes", "ws_smart_passes_accurate", "ws_crosses", "ws_crosses_accurate"] if c in ws_ref.columns]].copy()
        if not ws_z3.empty:
            st.markdown("### Wyscout final-third reference")
            st.dataframe(ws_z3.sort_values("match_date"), use_container_width=True, hide_index=True)

    st.markdown("### Analyst summary")
    if not entry_lane.empty:
        st.markdown(f"- England entered Z3 most often through the **{entry_lane.sort_values('count', ascending=False).iloc[0]['lane'].lower()}** lane.")
    if not entry_type.empty:
        st.markdown(f"- The dominant entry action was **{entry_type.sort_values('count', ascending=False).iloc[0]['entry_type'].lower()}**.")
    st.markdown(f"- Z14: {bundle['z14_metrics']['z14_entries']} entries, {bundle['z14_metrics']['shots_after_z14']} shots after Z14 involvement.")
    if not post_entry.empty:
        ret_pct = post_entry.loc[post_entry["outcome"] == "Retain possession", "share_pct"]
        shot_pct = post_entry.loc[post_entry["outcome"] == "Create shot", "share_pct"]
        st.markdown(f"- After first Z3 entry: retained possession {ret_pct.iloc[0]:.1f}%, created shot {shot_pct.iloc[0]:.1f}%.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CHANCE CREATION TAB                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with chance_tab:
    st.markdown("### Chance creation")
    ck = st.columns(5)
    with ck[0]:
        metric_card("Shot-ending poss.", fmt_pct(bundle["overview"]["shot_ending_possession_pct"]), "Share of possessions → shot")
    with ck[1]:
        metric_card("Event xG", fmt_num(bundle["overview"]["xg"], 2), "Sum of shot_statsbomb_xg")
    with ck[2]:
        metric_card("xG per shot", fmt_num(bundle["overview"]["xg_per_shot"], 3), "Average shot quality")
    with ck[3]:
        metric_card("WS shots / SoT", f"{fmt_int(ws_ref['ws_shots'].sum()) if not ws_ref.empty and 'ws_shots' in ws_ref.columns else '—'} / {fmt_int(ws_ref['ws_shots_on_target'].sum()) if not ws_ref.empty and 'ws_shots_on_target' in ws_ref.columns else '—'}", "Wyscout match-table totals")
    with ck[4]:
        metric_card("Avg shot dist.", fmt_num(ws_ref['ws_avg_shot_distance'].mean(), 1) if not ws_ref.empty and 'ws_avg_shot_distance' in ws_ref.columns else "—", "Wyscout average")

    st.markdown("")
    shot_source = shots["chance_source"].value_counts().reset_index()
    shot_source.columns = ["source", "shots"]
    xg_by_source = shots.groupby("chance_source")["shot_statsbomb_xg"].sum().reset_index().rename(columns={"chance_source": "source", "shot_statsbomb_xg": "xg"})
    shot_source = shot_source.merge(xg_by_source, on="source", how="left")
    shot_source["share_pct"] = np.where(shot_source["shots"].sum() > 0, shot_source["shots"] / shot_source["shots"].sum() * 100, 0)
    cc1, cc2 = st.columns([1.0, 1.0])
    with cc1:
        st.plotly_chart(stacked_count_pct(shot_source, "source", "shots", "share_pct", "Shot source mix", "Counts plus share of England's shots"), use_container_width=True)
    with cc2:
        st.dataframe(shot_source, use_container_width=True, hide_index=True)

    st.markdown("")
    ca1, ca2 = st.columns([1.0, 1.0])
    with ca1:
        st.plotly_chart(shot_result_map(shots, "Shot map by result", "Size = xG, colour = outcome"), use_container_width=True)
    with ca2:
        assist_types = (["All"] + sorted(bundle["assist_passes"]["subtype"].fillna("Other").astype(str).replace({"Pass": "Other", "Long Pass": "Other"}).unique().tolist())) if not bundle["assist_passes"].empty else ["All"]
        selected_assist_type = st.selectbox("Shot-assist filter", assist_types, index=0, key="assist_filter")
        assist_df = bundle["assist_passes"].copy()
        if selected_assist_type != "All" and not assist_df.empty:
            assist_df = assist_df[assist_df["subtype"].fillna("Other").replace({"Pass": "Other", "Long Pass": "Other"}).eq(selected_assist_type)]
        st.plotly_chart(assist_arrow_map(assist_df, "Shot-assist map", "Pass origin → shot location"), use_container_width=True)

    # ── Best chances ──
    st.markdown("### Best chances: highest-xG possessions")
    st.markdown(
        '<div class="info-box"><b>Possession xG</b> = sum of shot xG values inside the same England possession. '
        'The table ranks the most dangerous full attacking chains rather than only the single biggest shot.</div>',
        unsafe_allow_html=True,
    )
    if not best_chances.empty:
        best_filter_opp = st.multiselect(
            "Filter by opponent",
            sorted(best_chances["opponent"].dropna().unique().tolist()),
            default=sorted(best_chances["opponent"].dropna().unique().tolist()),
            key="best_opp_filter",
        )
        show_best = best_chances[best_chances["opponent"].isin(best_filter_opp)] if best_filter_opp else best_chances.copy()
        display_best = show_best.drop(columns=["match_id"], errors="ignore")
        st.dataframe(display_best, use_container_width=True, hide_index=True)

        best_options = [f"{r.match_date} | {r.opponent} | Poss {int(r.possession)} | xG {r.possession_xg:.3f}" for r in show_best.itertuples()]
        if best_options:
            selected_chain_label = st.selectbox("Inspect possession", best_options, index=0, key="chain_select")
            selected_chain_row = show_best.iloc[best_options.index(selected_chain_label)]

            # ── FIXED: safe chain lookup using match_id with fallback ──
            chain_df = safe_chain_lookup(
                bundle["best_chance_chains"],
                int(selected_chain_row["match_id"]),
                int(selected_chain_row["possession"]),
            )

            cb1, cb2 = st.columns([1.0, 1.0])
            with cb1:
                st.plotly_chart(best_chance_pitch(chain_df), use_container_width=True)
            with cb2:
                if not chain_df.empty:
                    display_cols = [c for c in ["minute", "player", "type", "subtype", "x", "y", "end_x", "end_y", "xg"] if c in chain_df.columns]
                    st.dataframe(chain_df[display_cols], use_container_width=True, hide_index=True)
                else:
                    st.info("Chain data not available for this possession.")
    else:
        st.info("No shot-ending possessions in the current filter.")

    if not ws_ref.empty:
        ws_attack = ws_ref[[c for c in ["match_date", "opponent", "competition_stage", "ws_xg", "ws_shots", "ws_shots_on_target", "ws_crosses", "ws_crosses_accurate", "ws_positional_attacks", "ws_positional_attacks_with_shots", "ws_counterattacks", "ws_counterattacks_with_shots"] if c in ws_ref.columns]].copy()
        if not ws_attack.empty:
            st.markdown("### Wyscout attacking reference")
            st.dataframe(ws_attack.sort_values("match_date"), use_container_width=True, hide_index=True)

    top_creator = bundle["player_stats"].sort_values(["key_passes", "xA"], ascending=False).head(1)
    st.markdown("### Analyst summary")
    if not shot_source.empty:
        st.markdown(f"- England's shots came most often from **{shot_source.sort_values('shots', ascending=False).iloc[0]['source'].lower()}** actions.")
    if not top_creator.empty:
        st.markdown(f"- **{top_creator.iloc[0]['player']}** led England on key-pass volume and xA exposure.")
    st.markdown(f"- England produced **{bundle['overview']['xg']:.2f}** event xG from {bundle['overview']['shots']} shots ({bundle['overview']['xg_per_shot']:.3f} xG/shot).")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PLAYERS TAB                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with players_tab:
    st.markdown("### Player tables")
    st.markdown('<div class="section-note">Event-derived player stats. Wyscout workbooks are match-level team tables only.</div>', unsafe_allow_html=True)
    ps = bundle["player_stats"].copy()

    pc1, pc2 = st.columns([0.4, 0.6])
    with pc1:
        player_search = st.text_input("Search player", value="", key="player_search")
    with pc2:
        min_minutes = st.slider("Min. estimated minutes", min_value=0, max_value=int(ps["minutes"].max()) if not ps.empty else 0, value=0, key="min_min")
    if player_search:
        ps = ps[ps["player"].str.contains(player_search, case=False, na=False)]
    ps = ps[ps["minutes"] >= min_minutes]

    total_cols = [c for c in ["player", "minutes", "possessions_involved", "receptions", "progressive_passes", "progressive_carries", "final_third_entries", "box_entries", "z14_receptions", "key_passes", "shot_assists", "crosses", "cutbacks", "through_balls", "touches_attacking_third", "touches_in_box", "shots", "SoT", "xG", "xG_Chain", "turnovers", "retention_rate_after_reception"] if c in ps.columns]
    ptab1, ptab2, ptab3 = st.tabs(["Totals", "Per 90", "Role snapshots"])
    with ptab1:
        st.dataframe(ps[total_cols], use_container_width=True, hide_index=True)
    with ptab2:
        per90_cols = [c for c in ["player", "minutes", "progressive_passes_p90", "progressive_carries_p90", "final_third_entries_p90", "box_entries_p90", "key_passes_p90", "crosses_p90", "cutbacks_p90", "through_balls_p90", "touches_attacking_third_p90", "touches_in_box_p90", "shots_p90", "SoT_p90", "xG_p90", "xG_Chain_p90"] if c in ps.columns]
        st.dataframe(ps[per90_cols], use_container_width=True, hide_index=True)
    with ptab3:
        cols = st.columns(4)
        for col, (name, df) in zip(cols, bundle["mini_tables"].items()):
            with col:
                st.markdown(f"**{name}**")
                mini_cols = [c for c in ["player", "minutes", "receptions", "progressive_passes", "progressive_carries", "final_third_entries", "key_passes", "xG", "xG_Chain"] if c in df.columns]
                st.dataframe(df[mini_cols], use_container_width=True, hide_index=True)

    export_kind, export_bytes, export_msg = make_export_bytes({"player_tables": ps[total_cols]})
    st.info(export_msg)
    st.download_button("Download player table export", data=export_bytes, file_name=f"england_player_tables.{export_kind}", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if export_kind == "xlsx" else "application/zip")

    if not ps.empty:
        st.markdown("### Analyst summary")
        st.markdown(f"- **{ps.sort_values('progressive_passes', ascending=False).iloc[0]['player']}** led progressive-passing volume.")
        st.markdown(f"- **{ps.sort_values('final_third_entries', ascending=False).iloc[0]['player']}** drove the biggest final-third entry load.")
        st.markdown(f"- **{ps.sort_values('xG', ascending=False).iloc[0]['player']}** was the main finisher by event xG.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAPS TAB                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with maps_tab:
    st.markdown("### Heatmaps & maps")
    st.markdown('<div class="section-note">Shot locations are on the Chance Creation page; this page covers touch density, progressive actions, and zone involvement.</div>', unsafe_allow_html=True)

    # Controls in a clean row above the chart
    m1, m2 = st.columns(2)
    with m1:
        map_type = st.selectbox("Map selector", ["Team touch density", "Reception density", "Progressive pass starts", "Carry density", "Cross origins", "Build-up starts", "Z14 involvement", "Box touches"], key="map_sel")
        map_match = st.selectbox("Match filter", ["All filtered matches"] + match["match_label"].tolist(), index=0, key="map_match")
    with m2:
        map_mode = st.selectbox("Display mode", ["Heatmap + points", "Binned zones", "Scatter points"], index=0, key="map_mode")
        all_players_list = sorted(pd.concat([events["player"], passes["player"], carries["player"], receipts["player"]], ignore_index=True).dropna().astype(str).unique().tolist())
        map_player = st.selectbox("Player filter", ["All players"] + all_players_list, index=0, key="map_player")
    map_match_id = None if map_match == "All filtered matches" else int(match.loc[match["match_label"].eq(map_match), "match_id"].iloc[0])

    target_df = pd.DataFrame()
    title, subtitle = map_type, ""
    if map_type == "Team touch density":
        target_df = filter_match_scope(events, map_match_id)
        title, subtitle = "Team touch heatmap", "All England offensive event locations"
        xcol, ycol = "x", "y"
    elif map_type == "Reception density":
        target_df = filter_match_scope(receipts, map_match_id)
        title, subtitle = "Reception heatmap", "Ball receipt locations"
        xcol, ycol = "location_x", "y_std"
    elif map_type == "Progressive pass starts":
        target_df = filter_match_scope(passes[passes["progressive_pass"]] if not passes.empty else passes, map_match_id)
        title, subtitle = "Progressive pass start map", "Start locations of progressive passes"
        xcol, ycol = "location_x", "y_std"
    elif map_type == "Carry density":
        target_df = filter_match_scope(carries[carries["progressive_carry"]] if not carries.empty else carries, map_match_id)
        title, subtitle = "Progressive carry map", "Start locations of progressive carries"
        xcol, ycol = "location_x", "y_std"
    elif map_type == "Cross origins":
        target_df = filter_match_scope(passes[passes["cross"]] if not passes.empty else passes, map_match_id)
        title, subtitle = "Cross origin map", "Start locations of cross attempts"
        xcol, ycol = "location_x", "y_std"
    elif map_type == "Build-up starts":
        target_df = filter_match_scope(bundle["build_passes"], map_match_id)
        title, subtitle = "Build-up start map", "Pass starts in build-up possessions"
        xcol, ycol = "location_x", "y_std"
    elif map_type == "Z14 involvement":
        target_df = filter_match_scope(events[(events["in_z14_start"]) | (events["in_z14_end"])] if not events.empty else events, map_match_id)
        title, subtitle = "Z14 involvement map", "Actions touching Zone 14"
        xcol, ycol = "x", "y"
    else:
        target_df = filter_match_scope(events[(events["in_box_start"]) | (events["in_box_end"])] if not events.empty else events, map_match_id)
        title, subtitle = "Box touch map", "Actions starting or ending inside the penalty box"
        xcol, ycol = "x", "y"

    if map_player != "All players" and not target_df.empty and "player" in target_df.columns:
        target_df = target_df[target_df["player"].eq(map_player)]

    if map_mode == "Binned zones":
        fig = binned_zone_map(target_df, xcol, ycol, title, subtitle)
    else:
        fig = base_pitch(title, subtitle, height=500, show_z14_label=(map_type == "Z14 involvement"))
        if map_mode == "Heatmap + points":
            fig = heatmap_on_pitch(fig, target_df[xcol] if not target_df.empty else [], target_df[ycol] if not target_df.empty else [], name=title, overlay_points=True)
        else:
            if not target_df.empty:
                fig.add_trace(go.Scatter(x=target_df[xcol], y=target_df[ycol], mode="markers", marker=dict(size=8, color="#38bdf8", opacity=0.72), text=target_df.get("player"), hovertemplate="%{text}<extra></extra>", name=title))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Analyst summary")
    st.markdown(f"- Showing **{len(target_df)}** England actions. {subtitle}.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  AUDIT TAB                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with audit_tab:
    st.markdown("### Data audit")
    ad1, ad2 = st.columns(2)
    with ad1:
        st.markdown("**Project files**")
        st.write(sorted([p.name for p in Path(__file__).resolve().parent.iterdir() if p.is_file()]))
        st.markdown("**Event sources**")
        st.write(bundle["audit"]["file_names"])
        st.markdown("**Coordinate & possession audit**")
        st.write({
            "Coordinate system": bundle["audit"]["coordinate_system"],
            "Possession IDs present": bundle["audit"]["possessions_exist"],
            "xG": bundle["audit"]["xg_exists"],
            "xGOT": "Not available",
            "xG Chain": "Derived from player possession involvement",
        })
    with ad2:
        st.markdown("**Wyscout audit**")
        st.write(get_wyscout()["audit"])
        st.markdown("**Dependencies**")
        st.write({
            "openpyxl": module_available("openpyxl"),
            "xlsxwriter": module_available("xlsxwriter"),
            "streamlit": module_available("streamlit"),
        })

    st.markdown("### Build-up metric validation")
    st.markdown(
        '<div class="info-box">'
        f'<b>Build-up possessions:</b> {bundle["overview"]["build_up_possessions"]} of {bundle["overview"]["possessions"]} total<br>'
        f'<b>Short build-up:</b> {bundle["overview"]["build_short_pct"]:.1f}% — first 3 passes are all short (&lt;30 m) and none reaches x≥70<br>'
        f'<b>Long/direct:</b> {bundle["overview"]["build_long_pct"]:.1f}% — at least one of the first 3 passes is ≥30 m or reaches x≥70<br>'
        f'<b>Classification rule:</b> Possession starts in own third (x&lt;40) or is initiated by GK/CB below x=60'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Wyscout integration notes")
    st.markdown("- **Build-up:** long-pass share, avg passes per possession, avg pass length, goal kicks, progressive passes.")
    st.markdown("- **Final Third:** passes to final third, smart passes, crosses — accurate totals.")
    st.markdown("- **Chance Creation:** team xG, shots, SoT, positional attacks, counterattacks, cross accuracy.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  REPORT TAB                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with report_tab:
    st.markdown("### Final report")
    st.markdown(report_md)
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.download_button("Download PDF", data=report_pdf, file_name="england_final_report.pdf", mime="application/pdf")
    with rc2:
        st.download_button("Download HTML", data=report_html, file_name="england_final_report.html", mime="text/html")
    with rc3:
        st.download_button("Download Markdown", data=report_md, file_name="england_final_report.md", mime="text/markdown")
