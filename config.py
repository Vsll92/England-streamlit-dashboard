from pathlib import Path

APP_TITLE = "England Euro 2024 — Build-up & Offensive Phases"
TEAM_NAME = "England"

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

DATA_DIR = Path(__file__).resolve().parent / "data"

STAGE_CANDIDATES = [
    "competition_stage",
    "stage",
    "match_stage",
    "competition_round",
    "round",
    "match_round",
]

DATE_CANDIDATES = [
    "match_date",
    "date",
    "game_date",
    "kickoff_date",
]

COMPETITION_CANDIDATES = [
    "competition",
    "tournament",
    "league",
    "competition_name",
]

HOME_TEAM_CANDIDATES = ["home_team", "home_team_name"]
AWAY_TEAM_CANDIDATES = ["away_team", "away_team_name"]

HOME_SCORE_CANDIDATES = ["home_score", "home_goals", "score_home"]
AWAY_SCORE_CANDIDATES = ["away_score", "away_goals", "score_away"]
