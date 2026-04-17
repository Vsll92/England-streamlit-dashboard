from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config import TEAM_NAME, APP_TITLE


def _safe_top(df: pd.DataFrame, col: str, sort_cols: list[str]) -> str:
    if df.empty or col not in df.columns:
        return "Unavailable"
    row = df.sort_values(sort_cols, ascending=False).head(1)
    if row.empty:
        return "Unavailable"
    return str(row.iloc[0][col])


def build_final_report_dict(bundle: dict[str, Any]) -> dict[str, Any]:
    overview = bundle["overview"]
    player_stats = bundle["player_stats"]
    build_receivers = bundle["build_receivers"]
    entries = bundle["entries"]
    shots = bundle["shots"]
    first_prog = bundle["first_progression"]
    first_entries = bundle["first_entries"]
    caveats = bundle["data_caveats"]
    match = bundle["match"]

    build_split = bundle["pos_df"][bundle["pos_df"]["is_build_up"]]["build_up_type"].value_counts(normalize=True).mul(100) if not bundle["pos_df"].empty else pd.Series(dtype=float)
    first_lane = first_prog["end_lane"].value_counts(normalize=True).mul(100) if not first_prog.empty else pd.Series(dtype=float)
    entry_lane = entries["entry_lane"].value_counts(normalize=True).mul(100) if not entries.empty else pd.Series(dtype=float)
    entry_type = entries["entry_type"].value_counts(normalize=True).mul(100) if not entries.empty else pd.Series(dtype=float)
    shot_source = shots["chance_source"].value_counts(normalize=True).mul(100) if not shots.empty else pd.Series(dtype=float)

    def top_pct(series: pd.Series) -> str:
        if series.empty:
            return "Unavailable"
        return f"{series.idxmax()} ({series.max():.1f}%)"

    wyscout = bundle.get("wyscout_match_ref", pd.DataFrame())
    wyscout_note = []
    if wyscout is not None and not wyscout.empty:
        valid_cross = wyscout.get("ws_cross_accuracy_pct")
        valid_prog = wyscout.get("ws_progressive_passes_accurate")
        valid_ft = wyscout.get("ws_passes_to_final_third_accurate")
        if valid_cross is not None and pd.Series(valid_cross).notna().any():
            wyscout_note.append(f"Wyscout team tables show {pd.Series(valid_cross).mean():.1f}% cross accuracy across the filtered sample.")
        if valid_prog is not None and pd.Series(valid_prog).notna().any():
            wyscout_note.append(f"Wyscout records {int(pd.Series(valid_prog).sum())} accurate progressive passes across the filtered matches.")
        if valid_ft is not None and pd.Series(valid_ft).notna().any():
            wyscout_note.append(f"Wyscout records {int(pd.Series(valid_ft).sum())} accurate passes into the final third across the filtered matches.")

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "team": TEAM_NAME,
        "match_count": len(match),
        "executive_summary": [
            f"England generated {overview['xg']:.2f} xG from {overview['shots']} shots across {len(match)} filtered matches.",
            f"They reached the final third in {overview['final_third_entry_pct']:.1f}% of possessions and ended {overview['shot_ending_possession_pct']:.1f}% of possessions with a shot.",
            f"Their build-up split was {build_split.get('Short', 0):.1f}% short and {build_split.get('Long', 0):.1f}% long/direct.",
            f"Final-third access leaned toward {top_pct(entry_lane)} and their main entry type was {top_pct(entry_type)}.",
        ],
        "strengths": [
            f"Strong final-third access volume: {bundle['overview']['z3_entries']} entries, {bundle['overview']['entries_per_match']:.1f} per match.",
            f"High box conversion from entry: {first_entries['enter_box'].mean() * 100:.1f}% of first Z3 entries led to box access." if not first_entries.empty else "Box-conversion after entry unavailable in the current filter.",
            f"Zone 14 remained important with {bundle['z14_metrics']['z14_entries']} entry actions into Z14.",
        ],
        "risks": [
            "xGOT is unavailable in the uploaded dataset, so finishing quality relies on xG, shot conversion, and SoT splits.",
            "Minutes are estimated because no lineup/substitution extract is present in the working project data.",
            "Several role judgments are based on event positions and event involvement rather than tactical freeze-frame structure.",
        ],
        "build_up": [
            f"England build-up was split {build_split.get('Short', 0):.1f}% short vs {build_split.get('Long', 0):.1f}% long/direct.",
            f"Preferred first progression lane: {top_pct(first_lane)}.",
            f"Main build-up receiver: {_safe_top(build_receivers, 'player', ['receptions', 'progressive_receptions'])}.",
            f"Most common early build-up combination: {bundle['build_combo_table'].iloc[0]['pattern']}." if not bundle["build_combo_table"].empty else "Recurring build-up combinations were not stable enough to report in this filter.",
        ],
        "final_third": [
            f"England reached Z3 in {overview['final_third_entry_pct']:.1f}% of possessions.",
            f"Main entry route: {top_pct(entry_lane)}.",
            f"Main entry action type: {top_pct(entry_type)}.",
            f"After first entry, they retained possession {first_entries['retain_possession'].mean() * 100:.1f}% of the time and created a shot {first_entries['create_shot'].mean() * 100:.1f}% of the time." if not first_entries.empty else "Post-entry outcome rates unavailable in the current filter.",
        ],
        "chance_creation": [
            f"Main source of shots: {top_pct(shot_source)}.",
            f"xG per shot: {overview['xg_per_shot']:.3f}.",
            f"Shot conversion: {overview['shot_conversion_pct']:.1f}%.",
            f"Average shot distance: {shots['shot_distance'].mean():.1f} in StatsBomb coordinates." if not shots.empty else "Average shot distance unavailable.",
        ],
        "key_players": {
            "build_up": _safe_top(bundle["build_receivers"], "player", ["receptions", "progressive_receptions"]),
            "progressor": _safe_top(player_stats, "player", ["final_third_entries", "box_entries"]),
            "creator": _safe_top(player_stats, "player", ["key_passes", "xA"]),
            "finisher": _safe_top(player_stats, "player", ["xG", "shots"]),
        },
        "tactical_conclusions": [
            f"England’s early possession is balanced rather than extreme, with short and long build-up both used in meaningful volume ({build_split.get('Short', 0):.1f}% vs {build_split.get('Long', 0):.1f}%).",
            f"The first progression phase trends away from the centre and toward {top_pct(first_lane)}.",
            f"Final-third access is structurally asymmetric, with {top_pct(entry_lane)} leading the route profile.",
            f"Short passing remains the dominant entry mechanism whenever England cross into Z3.",
            f"Zone 14 remains a meaningful connector zone before shots and box entries.",
            f"England’s shot output is volume-driven by possession access and repeated entries rather than a single isolated source.",
            f"The strongest creators are best identified through key passes, xA, and xG Chain rather than xGOT, which is not available.",
        ],
        "data_caveats": caveats + wyscout_note,
    }


def report_to_markdown(report: dict[str, Any]) -> str:
    lines = [f"# {APP_TITLE} — Final report", "", f"Generated: {report['generated_at']}", "", "## Executive Summary"]
    lines.extend([f"- {item}" for item in report["executive_summary"]])
    lines.extend(["", "## Key Strengths"])
    lines.extend([f"- {item}" for item in report["strengths"]])
    lines.extend(["", "## Main Risks / Limitations"])
    lines.extend([f"- {item}" for item in report["risks"]])
    lines.extend(["", "## Build-up Analysis"])
    lines.extend([f"- {item}" for item in report["build_up"]])
    lines.extend(["", "## Final Third Access"])
    lines.extend([f"- {item}" for item in report["final_third"]])
    lines.extend(["", "## Chance Creation"])
    lines.extend([f"- {item}" for item in report["chance_creation"]])
    lines.extend(["", "## Key Player Profiles"])
    for role, player in report["key_players"].items():
        lines.append(f"- {role.title()}: {player}")
    lines.extend(["", "## Tactical Conclusions"])
    lines.extend([f"- {item}" for item in report["tactical_conclusions"]])
    lines.extend(["", "## Data Caveats"])
    lines.extend([f"- {item}" for item in report["data_caveats"]])
    return "\\n".join(lines)


def report_to_html(report: dict[str, Any]) -> str:
    def bullets(items: list[str]) -> str:
        return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"
    kp = "".join(f"<li><b>{role.title()}</b>: {player}</li>" for role, player in report["key_players"].items())
    return f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <style>
          body {{ font-family: Arial, sans-serif; padding: 28px; color: #111827; }}
          h1, h2 {{ color: #0f172a; }}
          .meta {{ color: #475569; margin-bottom: 16px; }}
          .box {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px; margin-bottom: 18px; }}
        </style>
      </head>
      <body>
        <h1>{APP_TITLE} — Final report</h1>
        <div class="meta">Generated: {report['generated_at']}</div>
        <div class="box"><h2>Executive Summary</h2>{bullets(report['executive_summary'])}</div>
        <div class="box"><h2>Key Strengths</h2>{bullets(report['strengths'])}</div>
        <div class="box"><h2>Main Risks / Limitations</h2>{bullets(report['risks'])}</div>
        <div class="box"><h2>Build-up Analysis</h2>{bullets(report['build_up'])}</div>
        <div class="box"><h2>Final Third Access</h2>{bullets(report['final_third'])}</div>
        <div class="box"><h2>Chance Creation</h2>{bullets(report['chance_creation'])}</div>
        <div class="box"><h2>Key Player Profiles</h2><ul>{kp}</ul></div>
        <div class="box"><h2>Tactical Conclusions</h2>{bullets(report['tactical_conclusions'])}</div>
        <div class="box"><h2>Data Caveats</h2>{bullets(report['data_caveats'])}</div>
      </body>
    </html>
    """


def report_to_pdf_bytes(report: dict[str, Any], export_tables: dict[str, pd.DataFrame] | None = None) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=16*mm, leftMargin=16*mm, topMargin=16*mm, bottomMargin=14*mm)
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h_style = styles["Heading2"]
    body = ParagraphStyle("Body", parent=styles["BodyText"], leading=15, fontSize=9.5, spaceAfter=5)
    small = ParagraphStyle("Small", parent=styles["BodyText"], leading=12, fontSize=8.5, textColor=colors.HexColor("#475569"))
    story = [Paragraph(f"{APP_TITLE} — Final report", title_style), Paragraph(f"Generated: {report['generated_at']}", small), Spacer(1, 6)]

    sections = [("Executive Summary", report["executive_summary"]), ("Key Strengths", report["strengths"]), ("Main Risks / Limitations", report["risks"]), ("Build-up Analysis", report["build_up"]), ("Final Third Access", report["final_third"]), ("Chance Creation", report["chance_creation"]), ("Tactical Conclusions", report["tactical_conclusions"]), ("Data Caveats", report["data_caveats"])]
    for title, items in sections:
        story.append(Paragraph(title, h_style))
        for item in items:
            story.append(Paragraph(f"• {item}", body))
        story.append(Spacer(1, 4))

    story.append(Paragraph("Key Player Profiles", h_style))
    kp_data = [["Role", "Player"]] + [[role.title(), player] for role, player in report["key_players"].items()]
    kp_table = Table(kp_data, colWidths=[45*mm, 110*mm])
    kp_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]), ("FONTSIZE", (0, 0), (-1, -1), 9), ("LEADING", (0, 0), (-1, -1), 11)]))
    story.extend([kp_table, Spacer(1, 8)])

    if export_tables:
        for name in ["player_stats", "entry_players", "build_receivers"]:
            df = export_tables.get(name)
            if df is None or df.empty:
                continue
            sample = df.head(8).copy().astype(str)
            story.append(Paragraph(name.replace("_", " ").title(), h_style))
            table_data = [list(sample.columns)] + sample.values.tolist()
            table = Table(table_data, repeatRows=1)
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]), ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("LEADING", (0, 0), (-1, -1), 9)]))
            story.extend([table, Spacer(1, 8)])

    doc.build(story)
    return buf.getvalue()
