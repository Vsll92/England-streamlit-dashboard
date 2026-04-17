from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import PITCH_LENGTH, PITCH_WIDTH


ZONE_CENTERS = {
    "Defensive Third | Left": (20, 66),
    "Defensive Third | Center": (20, 40),
    "Defensive Third | Right": (20, 14),
    "Middle Third | Left": (60, 66),
    "Middle Third | Center": (60, 40),
    "Middle Third | Right": (60, 14),
    "Final Third | Left": (100, 66),
    "Final Third | Center": (100, 40),
    "Final Third | Right": (100, 14),
}

# ── Consistent dark theme tokens ──
_BG = "#0b1220"
_FONT = "#e5e7eb"
_TITLE_FONT = "#f8fafc"
_LINE = "#dbe4f0"
_GUIDE = "rgba(148,163,184,0.28)"
_ACCENT_BLUE = "#38bdf8"
_ACCENT_GREEN = "#22c55e"
_ACCENT_AMBER = "#f59e0b"
_ACCENT_RED = "#ef4444"


def base_pitch(
    title: str = "",
    subtitle: str = "",
    height: int = 460,
    show_zone_labels: bool = True,
    show_z14_label: bool = False,
) -> go.Figure:
    fig = go.Figure()
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=PITCH_LENGTH, y1=PITCH_WIDTH, line=dict(color=_LINE, width=1.5)),
        dict(type="line", x0=PITCH_LENGTH / 2, y0=0, x1=PITCH_LENGTH / 2, y1=PITCH_WIDTH, line=dict(color=_LINE, width=1)),
        dict(type="circle", x0=50, y0=30, x1=70, y1=50, line=dict(color=_LINE, width=1)),
        dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color=_LINE, width=1)),
        dict(type="rect", x0=102, y0=18, x1=PITCH_LENGTH, y1=62, line=dict(color=_LINE, width=1)),
        dict(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color=_LINE, width=1)),
        dict(type="rect", x0=114, y0=30, x1=PITCH_LENGTH, y1=50, line=dict(color=_LINE, width=1)),
        dict(type="line", x0=40, y0=0, x1=40, y1=PITCH_WIDTH, line=dict(color=_GUIDE, width=0.8, dash="dot")),
        dict(type="line", x0=80, y0=0, x1=80, y1=PITCH_WIDTH, line=dict(color=_GUIDE, width=0.8, dash="dot")),
        dict(type="line", x0=0, y0=PITCH_WIDTH / 3, x1=PITCH_LENGTH, y1=PITCH_WIDTH / 3, line=dict(color=_GUIDE, width=0.8, dash="dot")),
        dict(type="line", x0=0, y0=2 * PITCH_WIDTH / 3, x1=PITCH_LENGTH, y1=2 * PITCH_WIDTH / 3, line=dict(color=_GUIDE, width=0.8, dash="dot")),
        dict(type="rect", x0=80, y0=30, x1=102, y1=50, line=dict(color="rgba(245,158,11,0.7)", width=1, dash="dot"), fillcolor="rgba(245,158,11,0.05)"),
    ]
    annotations = []
    if show_zone_labels:
        annotations.extend([
            dict(x=20, y=78, text="Defensive third", showarrow=False, font=dict(color="rgba(226,232,240,0.6)", size=10)),
            dict(x=60, y=78, text="Middle third", showarrow=False, font=dict(color="rgba(226,232,240,0.6)", size=10)),
            dict(x=100, y=78, text="Final third", showarrow=False, font=dict(color="rgba(226,232,240,0.6)", size=10)),
        ])
    if show_z14_label:
        annotations.append(dict(x=91, y=40, text="Z14", showarrow=False, font=dict(color="rgba(251,191,36,0.88)", size=11)))

    title_text = f"<b>{title}</b><br><span style='font-size:12px;color:#94a3b8'>{subtitle}</span>" if subtitle else f"<b>{title}</b>"

    fig.update_layout(
        title={"text": title_text, "x": 0.02, "font": {"color": _TITLE_FONT, "size": 17}},
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        xaxis=dict(range=[-2, PITCH_LENGTH + 2], visible=False),
        yaxis=dict(range=[-4, PITCH_WIDTH + 4], visible=False, scaleanchor="x", scaleratio=1),
        shapes=shapes,
        annotations=annotations,
        margin=dict(l=10, r=10, t=68, b=10),
        height=height,
        font=dict(color=_FONT),
        hoverlabel=dict(bgcolor="#111827", font=dict(color="#f8fafc")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
        modebar=dict(orientation="v"),
    )
    return fig


def heatmap_on_pitch(
    fig: go.Figure, x: Iterable[float], y: Iterable[float],
    bins: tuple[int, int] = (20, 14), name: str = "Density", overlay_points: bool = False,
) -> go.Figure:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return fig
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, PITCH_LENGTH], [0, PITCH_WIDTH]])
    fig.add_trace(go.Heatmap(
        x=(xedges[:-1] + xedges[1:]) / 2,
        y=(yedges[:-1] + yedges[1:]) / 2,
        z=H.T,
        colorscale=[
            [0.0, "rgba(30,41,59,0.05)"],
            [0.2, "rgba(59,130,246,0.25)"],
            [0.45, "rgba(16,185,129,0.45)"],
            [0.7, "rgba(245,158,11,0.68)"],
            [1.0, "rgba(239,68,68,0.9)"],
        ],
        showscale=False, opacity=0.9,
        hovertemplate=f"{name}: %{{z}} actions<extra></extra>", name=name,
    ))
    if overlay_points:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=4, color="rgba(241,245,249,0.45)"),
            name="Actions", hoverinfo="skip", showlegend=False,
        ))
    return fig


def binned_zone_map(
    df: pd.DataFrame, x: str, y: str, title: str, subtitle: str,
    hover_cols: list[str] | None = None,
) -> go.Figure:
    fig = base_pitch(title, subtitle)
    if df.empty:
        return fig
    temp = df[[x, y] + ([c for c in hover_cols if c in df.columns] if hover_cols else [])].copy().dropna(subset=[x, y])
    if temp.empty:
        return fig
    temp["bin_x"] = pd.cut(temp[x], bins=[0, 40, 80, 120], labels=[20, 60, 100], include_lowest=True).astype(float)
    temp["bin_y"] = pd.cut(temp[y], bins=[0, PITCH_WIDTH / 3, 2 * PITCH_WIDTH / 3, PITCH_WIDTH], labels=[14, 40, 66], include_lowest=True).astype(float)
    grp = temp.groupby(["bin_x", "bin_y"]).size().reset_index(name="count")
    fig.add_trace(go.Scatter(
        x=grp["bin_x"], y=grp["bin_y"], mode="markers+text",
        text=grp["count"], textposition="middle center",
        marker=dict(
            size=np.clip(grp["count"] * 1.4, 24, 70),
            color=grp["count"], colorscale="YlOrRd", opacity=0.88,
            line=dict(color="#0f172a", width=1),
        ),
        hovertemplate="Count: %{text}<extra></extra>", name="Zone count",
    ))
    return fig


def grouped_route_map(
    routes: pd.DataFrame, title: str, subtitle: str, color_by: str = "completion_pct",
) -> go.Figure:
    """Redesigned build-up route family map.

    Uses graduated arrow width for volume and a blue→green→amber colour ramp for
    completion rate, making it easy to spot high-traffic and high-accuracy routes.
    Only shows the top-10 routes and eliminates the mid-point dot clutter.
    """
    fig = base_pitch(title, subtitle, height=500)
    if routes.empty:
        return fig

    use = routes.head(10).copy()
    max_attempts = use["attempts"].max() if not use.empty else 1

    for _, row in use.iterrows():
        # Width proportional to volume (1.5 → 5.5)
        vol_frac = float(row.get("attempts", 1)) / max(max_attempts, 1)
        width = 1.5 + 4.0 * vol_frac

        # Colour ramp: low completion → amber, high → bright blue
        comp = float(row.get(color_by, 0))
        if comp >= 70:
            color = f"rgba(56,189,248,{0.45 + 0.45 * min(comp, 100) / 100:.2f})"
        elif comp >= 45:
            color = f"rgba(34,197,94,{0.40 + 0.40 * min(comp, 100) / 100:.2f})"
        else:
            color = f"rgba(245,158,11,{0.35 + 0.40 * min(comp, 100) / 100:.2f})"

        sx, sy = float(row["start_x"]), float(row["start_y"])
        ex, ey = float(row["end_x"]), float(row["end_y"])

        fig.add_annotation(
            x=ex, y=ey, ax=sx, ay=sy,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2,
            arrowwidth=width, arrowcolor=color, opacity=0.85,
        )

        # Volume label at midpoint — small and subtle
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        fig.add_trace(go.Scatter(
            x=[mx], y=[my], mode="text",
            text=[str(int(row["attempts"]))],
            textfont=dict(size=10, color="rgba(248,250,252,0.8)"),
            hovertemplate=(
                f"<b>{row['route']}</b><br>"
                f"Attempts: {int(row['attempts'])}<br>"
                f"Completed: {int(row['completed'])}<br>"
                f"Completion: {row['completion_pct']:.1f}%<br>"
                f"Players: {row.get('main_players', '')}<extra></extra>"
            ),
            showlegend=False,
        ))

    # Legend entries for colour ramp
    for label, c in [("≥70% completion", _ACCENT_BLUE), ("45–70%", _ACCENT_GREEN), ("<45%", _ACCENT_AMBER)]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=c, width=3), name=label,
        ))

    return fig


def shot_result_map(shots: pd.DataFrame, title: str, subtitle: str) -> go.Figure:
    fig = base_pitch(title, subtitle, height=480)
    if shots.empty:
        return fig
    temp = shots.copy()
    result_map = {
        "Goal": "Goal", "Saved": "Saved / on target", "Saved to Post": "Saved / on target",
        "Off T": "Off target", "Wayward": "Off target", "Blocked": "Blocked", "Post": "Woodwork",
    }
    temp["shot_result_group"] = temp["shot_outcome"].map(result_map).fillna(temp["shot_outcome"].fillna("Other"))
    colors = {
        "Goal": _ACCENT_GREEN, "Saved / on target": _ACCENT_BLUE,
        "Blocked": _ACCENT_AMBER, "Off target": _ACCENT_RED,
        "Woodwork": "#a855f7", "Other": "#94a3b8",
    }
    for label, df in temp.groupby("shot_result_group"):
        fig.add_trace(go.Scatter(
            x=df["location_x"], y=df["y_std"], mode="markers", name=label,
            marker=dict(
                size=np.clip(df["shot_statsbomb_xg"].fillna(0) * 120 + 10, 10, 38),
                color=colors.get(label, "#94a3b8"), opacity=0.82,
                line=dict(color="#0f172a", width=1),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>vs %{customdata[1]}<br>"
                "Minute: %{customdata[2]}<br>xG: %{customdata[3]:.3f}<br>"
                "Result: %{customdata[4]}<br>Body: %{customdata[5]}<br>"
                "Source: %{customdata[6]}<extra></extra>"
            ),
            customdata=df[["player", "opponent", "minute", "shot_statsbomb_xg", "shot_outcome", "shot_body_part", "chance_source"]].fillna("Unknown"),
        ))
    return fig


def assist_arrow_map(df: pd.DataFrame, title: str, subtitle: str) -> go.Figure:
    fig = base_pitch(title, subtitle, height=500)
    if df.empty:
        return fig
    type_colors = {
        "Cross": "#f97316", "Cutback": "#eab308", "Through Ball": "#10b981",
        "Other": _ACCENT_BLUE, "Long Pass": "#8b5cf6", "Pass": _ACCENT_BLUE,
    }
    temp = df.copy()
    temp["assist_type_group"] = temp["subtype"].replace({"Pass": "Other", "Long Pass": "Other"}).fillna("Other")
    for label, grp in temp.groupby("assist_type_group"):
        color = type_colors.get(label, _ACCENT_BLUE)
        for _, row in grp.iterrows():
            fig.add_annotation(
                x=float(row["location_x_shot"]), y=float(row["y_std_shot"]),
                ax=float(row["location_x_pass"]), ay=float(row["y_std_pass"]),
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.8,
                arrowcolor=color, opacity=0.55,
            )
        fig.add_trace(go.Scatter(
            x=grp["location_x_pass"], y=grp["y_std_pass"], mode="markers", name=f"{label} origin",
            marker=dict(size=8, color=color, symbol="circle", line=dict(color="#0f172a", width=0.8)),
            customdata=grp[["player_pass", "player_shot", "assist_type_group", "shot_statsbomb_xg", "opponent", "competition_stage"]].fillna("Unknown"),
            hovertemplate="<b>%{customdata[0]}</b> → %{customdata[1]}<br>Type: %{customdata[2]}<br>xG: %{customdata[3]:.3f}<br>vs %{customdata[4]}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=temp["location_x_shot"], y=temp["y_std_shot"], mode="markers", name="Shot end",
        marker=dict(
            size=np.clip(temp["shot_statsbomb_xg"].fillna(0) * 120 + 9, 10, 32),
            color="#f8fafc", symbol="x", line=dict(color="#0f172a", width=1),
        ),
        hovertemplate="Shot location<br>xG: %{customdata[0]:.3f}<extra></extra>",
        customdata=np.c_[temp["shot_statsbomb_xg"].fillna(0)],
    ))
    return fig


def summary_bar(
    df: pd.DataFrame, x: str, y: str, title: str, subtitle: str = "",
    color: str | None = None, text_fmt: str = ".1f",
) -> go.Figure:
    title_text = f"<b>{title}</b><br><span style='font-size:12px;color:#94a3b8'>{subtitle}</span>" if subtitle else f"<b>{title}</b>"
    fig = px.bar(df, x=x, y=y, title=title_text, text_auto=text_fmt)
    fig.update_traces(marker_color=color or _ACCENT_BLUE)
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG, font=dict(color=_FONT),
        margin=dict(l=10, r=10, t=72, b=10), height=370,
        xaxis_title=None, yaxis_title=None,
    )
    return fig


def stacked_count_pct(
    df: pd.DataFrame, x: str, y_count: str, y_pct: str, title: str, subtitle: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[x], y=df[y_count], name="Count",
        marker_color=_ACCENT_BLUE, hovertemplate="Count: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y_pct], name="Share %",
        mode="lines+markers+text",
        text=[f"{v:.1f}%" for v in df[y_pct]], textposition="top center",
        yaxis="y2", line=dict(color=_ACCENT_AMBER, width=2.5),
        hovertemplate="Share: %{y:.1f}%<extra></extra>",
    ))
    title_text = f"<b>{title}</b><br><span style='font-size:12px;color:#94a3b8'>{subtitle}</span>"
    fig.update_layout(
        title=title_text,
        paper_bgcolor=_BG, plot_bgcolor=_BG, font=dict(color=_FONT),
        height=400, margin=dict(l=10, r=10, t=72, b=10),
        xaxis_title=None,
        yaxis=dict(title="Count"),
        yaxis2=dict(title="Share %", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def tactical_summary_pitch(
    lane: str, pct: float, notes: list[str], title: str,
    secondary_lane: str | None = None, secondary_pct: float | None = None,
) -> go.Figure:
    fig = base_pitch(title, "Key tactical findings shown on the pitch", height=480)
    lane_band = {
        "Right": (0, PITCH_WIDTH / 3),
        "Center": (PITCH_WIDTH / 3, 2 * PITCH_WIDTH / 3),
        "Left": (2 * PITCH_WIDTH / 3, PITCH_WIDTH),
    }
    y0, y1 = lane_band.get(lane, (0, PITCH_WIDTH))
    fig.add_shape(type="rect", x0=0, x1=PITCH_LENGTH, y0=y0, y1=y1, fillcolor="rgba(59,130,246,0.16)", line=dict(color="rgba(59,130,246,0.0)"))
    fig.add_annotation(x=96, y=(y0 + y1) / 2, ax=28, ay=(y0 + y1) / 2, showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=6, arrowcolor="rgba(59,130,246,0.85)")
    fig.add_annotation(x=95, y=(y0 + y1) / 2 + 6, text=f"Primary lane: {lane} ({pct:.1f}%)", showarrow=False, font=dict(color="#f8fafc", size=13))
    if secondary_lane and secondary_pct is not None and secondary_lane in lane_band:
        sy0, sy1 = lane_band[secondary_lane]
        fig.add_shape(type="rect", x0=55, x1=PITCH_LENGTH, y0=sy0, y1=sy1, fillcolor="rgba(245,158,11,0.10)", line=dict(color="rgba(245,158,11,0.0)"))
        fig.add_annotation(x=94, y=(sy0 + sy1) / 2 - 6, text=f"Secondary: {secondary_lane} ({secondary_pct:.1f}%)", showarrow=False, font=dict(color="#fbbf24", size=11))
    # Notes positioned at bottom-left of pitch to avoid collisions
    y_note = -2
    for note in notes[:4]:
        fig.add_annotation(x=1, y=y_note, text=note, showarrow=False, xanchor="left", font=dict(color="#94a3b8", size=10))
        y_note -= 5
    # Expand bottom margin so notes are visible
    fig.update_layout(yaxis=dict(range=[-22, PITCH_WIDTH + 4]))
    return fig


def pass_network(
    df: pd.DataFrame, title: str,
    subtitle: str = "Average touch positions and strongest pass links", height: int = 480,
) -> go.Figure:
    fig = base_pitch(title, subtitle, height=height)
    if df.empty or "pass_recipient" not in df.columns:
        return fig
    touches = pd.concat([
        df[["player", "location_x", "y_std"]].rename(columns={"location_x": "x", "y_std": "y"}),
        df[["pass_recipient", "pass_end_location_x", "end_y_std"]].dropna(subset=["pass_recipient"]).rename(columns={"pass_recipient": "player", "pass_end_location_x": "x", "end_y_std": "y"}),
    ], ignore_index=True).dropna()
    if touches.empty:
        return fig
    nodes = touches.groupby("player").agg(x=("x", "mean"), y=("y", "mean"), touches=("x", "count")).reset_index()
    links = (
        df.dropna(subset=["pass_recipient"]).groupby(["player", "pass_recipient"]).size().reset_index(name="count")
        .sort_values("count", ascending=False).head(18)
    )
    node_map = nodes.set_index("player")[["x", "y"]].to_dict("index")
    for _, row in links.iterrows():
        if row["player"] not in node_map or row["pass_recipient"] not in node_map:
            continue
        p1, p2 = node_map[row["player"]], node_map[row["pass_recipient"]]
        fig.add_trace(go.Scatter(
            x=[p1["x"], p2["x"]], y=[p1["y"], p2["y"]], mode="lines",
            line=dict(color="rgba(96,165,250,0.35)", width=max(1.0, float(row["count"]) / 4.0)),
            hovertemplate=f"{row['player']} → {row['pass_recipient']}<br>Passes: {int(row['count'])}<extra></extra>",
            showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=nodes["x"], y=nodes["y"], mode="markers+text",
        text=nodes["player"], textposition="top center",
        marker=dict(size=np.clip(nodes["touches"] * 0.55, 10, 28), color="#f8fafc", opacity=0.92, line=dict(color="#0f172a", width=1)),
        hovertemplate="%{text}<br>Touches: %{marker.size:.0f}<extra></extra>",
        showlegend=False,
    ))
    return fig
