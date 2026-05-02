"""
NBA Data Collection Script — Comprehensive Version
===================================================
Pulls game-level (per-game-log) data to enable proper rolling-window feature engineering

Data collected
--------------
1. player_gamelogs.csv      — every player's stats for every game they played
2. player_advanced.csv      — season-level advanced stats (USG%, PIE, ratings)
3. lineup_synergy.csv       — 5-man lineup advanced stats (net rating, synergy)
4. game_results.csv         — team-level game results (home vs. away merged)
5. player_rolling.csv       — rolling-window features derived from gamelogs
                              (shift-1 so no same-game leakage)

Requirements
------------
    pip install nba_api pandas tqdm
"""

import argparse
import random
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from nba_api.stats.endpoints import (
    leaguedashlineups,
    leaguedashplayerstats,
    leaguegamelog,
    playergamelogs,
)

# ── Default configuration ──────────────────────────────────────────────────────
DEFAULT_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
SEASON_TYPE     = "Regular Season"
BASE_DELAY      = 1.5          # seconds between API calls
JITTER          = 1.0          # additional random jitter (0 ~ JITTER seconds)
MIN_PLAYER_MIN  = 10           # filter: minimum minutes per game for player logs
MIN_LINEUP_GP   = 5            # filter: minimum games played for lineup entries
ROLLING_WINDOW  = 10           # rolling mean window size (games)
ROLLING_MIN_OBS = 3            # minimum observations for rolling mean to be valid

# ── Stat columns collected at the game-log level ───────────────────────────────
GAMELOG_STAT_COLS = [
    "MIN",
    # Scoring
    "PTS", "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    # Boards
    "OREB", "DREB", "REB",
    # Playmaking / defense
    "AST", "TOV", "STL", "BLK", "BLKA",
    # Fouls
    "PF", "PFD",
    # Overall
    "PLUS_MINUS",
    "NBA_FANTASY_PTS",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _sleep():
    """Polite delay with jitter to avoid NBA API rate-limiting."""
    time.sleep(BASE_DELAY + random.uniform(0, JITTER))


def _safe_fetch(fn, label: str, retries: int = 3):
    """Call fn() with retries; returns None on persistent failure."""
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            print(f"    ⚠  {label} — attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(BASE_DELAY * attempt * 2)
    print(f"    ✗  {label} — all retries exhausted, skipping.")
    return None


# ── Fetchers ───────────────────────────────────────────────────────────────────

def fetch_player_gamelogs(season: str) -> pd.DataFrame:
    """
    Pull every player's individual game log for the season.
    This is the raw material for rolling-window feature engineering.
    """
    print(f"  [Player GameLogs] {season} ...")
    _sleep()

    def _call():
        ep = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable=SEASON_TYPE,
            per_mode_simple_nullable="PerGame",
        )
        return ep.get_data_frames()[0]

    df = _safe_fetch(_call, f"PlayerGameLogs {season}")
    if df is None or df.empty:
        return pd.DataFrame()

    # Standardise column names
    rename = {
        "PLAYER_ID":         "player_id",
        "PLAYER_NAME":       "player_name",
        "TEAM_ID":           "team_id",
        "TEAM_ABBREVIATION": "team",
        "GAME_ID":           "game_id",
        "GAME_DATE":         "game_date",
        "MATCHUP":           "matchup",
        "WL":                "win_loss",
        "MIN":               "min",
        "PTS":               "pts",
        "FGM":               "fgm",       "FGA":    "fga",    "FG_PCT":  "fg_pct",
        "FG3M":              "fg3m",      "FG3A":   "fg3a",   "FG3_PCT": "fg3_pct",
        "FTM":               "ftm",       "FTA":    "fta",    "FT_PCT":  "ft_pct",
        "OREB":              "oreb",      "DREB":   "dreb",   "REB":     "reb",
        "AST":               "ast",       "TOV":    "tov",
        "STL":               "stl",       "BLK":    "blk",    "BLKA":    "blk_against",
        "PF":                "fouls",     "PFD":    "fouls_drawn",
        "PLUS_MINUS":        "plus_minus",
        "NBA_FANTASY_PTS":   "fantasy_pts",
    }
    available = {k: v for k, v in rename.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    # Type cleanup
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["win"]       = (df["win_loss"] == "W").astype(int)
    df["is_home"]   = df["matchup"].str.contains("vs.", na=False).astype(int)
    df["season"]    = season

    # Filter: meaningful minutes only
    df = df[df["min"] >= MIN_PLAYER_MIN].reset_index(drop=True)

    # Derived per-game features
    df["ast_to_tov"]    = df["ast"]  / (df["tov"]  + 0.001)
    df["true_shooting"] = df["pts"]  / (2 * (df["fga"] + 0.44 * df["fta"]) + 0.001)
    df["usage_proxy"]   = (df["fga"] + 0.44 * df["fta"] + df["tov"]) / (df["min"] + 0.001)
    df["def_impact"]    = df["stl"]  + df["blk"] + df["dreb"]

    return df.sort_values(["player_id", "game_date"]).reset_index(drop=True)


def fetch_player_advanced(season: str) -> pd.DataFrame:
    """
    Season-level advanced stats: USG%, PIE, ON/OFF ratings, etc.
    Used as supplementary features (not for rolling window).
    """
    print(f"  [Advanced Stats] {season} ...")
    _sleep()

    def _call():
        ep = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star=SEASON_TYPE,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
        )
        return ep.get_data_frames()[0]

    df = _safe_fetch(_call, f"AdvancedStats {season}")
    if df is None or df.empty:
        return pd.DataFrame()

    rename = {
        "PLAYER_ID":  "player_id",
        "PLAYER_NAME":"player_name",
        "TEAM_ABBREVIATION": "team",
        "OFF_RATING": "off_rating",
        "DEF_RATING": "def_rating",
        "NET_RATING": "net_rating",
        "AST_PCT":    "ast_pct",
        "AST_TO":     "ast_to",
        "AST_RATIO":  "ast_ratio",
        "OREB_PCT":   "oreb_pct",
        "DREB_PCT":   "dreb_pct",
        "REB_PCT":    "reb_pct",
        "TM_TOV_PCT": "tov_pct",
        "EFG_PCT":    "efg_pct",
        "TS_PCT":     "ts_pct",
        "USG_PCT":    "usg_pct",
        "PACE":       "pace",
        "PIE":        "pie",
    }
    available = {k: v for k, v in rename.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)
    df["season"] = season
    return df


def fetch_lineup_stats(season: str) -> pd.DataFrame:
    """
    5-man lineup advanced stats: net rating, pace, synergy metrics.
    Used to capture cross-player complementarity (Output B features).
    """
    print(f"  [Lineup Stats] {season} ...")
    _sleep()

    def _call():
        ep = leaguedashlineups.LeagueDashLineups(
            season=season,
            season_type_all_star=SEASON_TYPE,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            group_quantity=5,
        )
        return ep.get_data_frames()[0]

    df = _safe_fetch(_call, f"LineupStats {season}")
    if df is None or df.empty:
        return pd.DataFrame()

    rename = {
        "GROUP_ID":          "lineup_id",
        "GROUP_NAME":        "lineup_names",
        "TEAM_ABBREVIATION": "team",
        "GP":                "games_played",
        "MIN":               "min",
        "OFF_RATING":        "off_rating",
        "DEF_RATING":        "def_rating",
        "NET_RATING":        "net_rating",
        "AST_PCT":           "ast_pct",
        "OREB_PCT":          "oreb_pct",
        "DREB_PCT":          "dreb_pct",
        "REB_PCT":           "reb_pct",
        "TM_TOV_PCT":        "tov_pct",
        "EFG_PCT":           "efg_pct",
        "TS_PCT":            "ts_pct",
        "PACE":              "pace",
        "PIE":               "pie",
        "POSS":              "possessions",
    }
    available = {k: v for k, v in rename.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)
    df["season"] = season
    df = df[df["games_played"] >= MIN_LINEUP_GP].reset_index(drop=True)
    return df


def fetch_game_results(season: str) -> pd.DataFrame:
    """
    Team-level game log (one row per team per game).
    Merged into home-vs-away pairs for training labels.
    """
    print(f"  [Game Results] {season} ...")
    _sleep()

    def _call():
        ep = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=SEASON_TYPE,
            player_or_team_abbreviation="T",
        )
        return ep.get_data_frames()[0]

    df = _safe_fetch(_call, f"GameResults {season}")
    if df is None or df.empty:
        return pd.DataFrame()

    rename = {
        "GAME_ID":           "game_id",
        "GAME_DATE":         "game_date",
        "TEAM_ABBREVIATION": "team",
        "MATCHUP":           "matchup",
        "WL":                "win_loss",
        "MIN":               "min",
        "PTS":               "pts",
        "FGM": "fgm", "FGA": "fga", "FG_PCT": "fg_pct",
        "FG3M": "fg3m", "FG3A": "fg3a", "FG3_PCT": "fg3_pct",
        "FTM": "ftm", "FTA": "fta", "FT_PCT": "ft_pct",
        "OREB": "oreb", "DREB": "dreb", "REB": "reb",
        "AST": "ast", "STL": "stl", "BLK": "blk",
        "TOV": "tov", "PF": "pf",
        "PLUS_MINUS": "point_diff",
    }
    available = {k: v for k, v in rename.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["is_home"]   = df["matchup"].str.contains("vs.", na=False).astype(int)
    df["win"]       = (df["win_loss"] == "W").astype(int)
    df["season"]    = season
    return df


# ── Post-processing ────────────────────────────────────────────────────────────

def build_game_pairs(game_results: pd.DataFrame) -> pd.DataFrame:
    """
    Merge each game from two rows (home team + away team) into a single row.
    Output has one row per game with home_* and away_* prefixed columns.
    """
    print("\n[Processing] Building home vs. away matchup records ...")

    stat_cols = [
        "pts", "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
        "ftm", "fta", "ft_pct", "oreb", "dreb", "reb",
        "ast", "stl", "blk", "tov", "pf", "point_diff",
    ]

    home = (
        game_results[game_results["is_home"] == 1]
        .drop_duplicates(subset=["game_id"])
        .copy()
    )
    away = (
        game_results[game_results["is_home"] == 0]
        .drop_duplicates(subset=["game_id"])
        .copy()
    )

    home = home.rename(columns={**{c: f"home_{c}" for c in stat_cols},
                                 "team": "home_team", "win": "home_win"})
    away = away.rename(columns={**{c: f"away_{c}" for c in stat_cols},
                                 "team": "away_team"})

    home_keep = (["game_id", "game_date", "season", "home_team", "home_win"]
                 + [f"home_{c}" for c in stat_cols])
    away_keep = ["game_id", "away_team"] + [f"away_{c}" for c in stat_cols]

    merged = pd.merge(
        home[[c for c in home_keep if c in home.columns]],
        away[[c for c in away_keep if c in away.columns]],
        on="game_id",
        how="inner",
    )
    return merged.sort_values("game_date").reset_index(drop=True)


def build_rolling_features(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling-window (last N games) averages per player.

    Key design decisions
    --------------------
    * shift(1)  — the current game is excluded from its own rolling mean,
                  preventing same-game data leakage.
    * min_periods=ROLLING_MIN_OBS — rows with too few prior games return NaN
                  rather than unreliable estimates.
    * Sorted by (player_id, game_date) before groupby to ensure correct order.
    """
    print("\n[Processing] Computing rolling-window player features ...")

    roll_cols = [
        "pts", "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
        "ftm", "fta", "ft_pct",
        "oreb", "dreb", "reb",
        "ast", "tov", "stl", "blk",
        "fouls", "fouls_drawn",
        "plus_minus", "min",
        # Derived
        "ast_to_tov", "true_shooting", "usage_proxy", "def_impact",
    ]

    # Keep only columns that actually exist
    roll_cols = [c for c in roll_cols if c in gamelogs.columns]

    df = gamelogs.sort_values(["player_id", "game_date"]).copy()

    def rolling_mean(series: pd.Series) -> pd.Series:
        return (
            series
            .shift(1)                                        # exclude current game
            .rolling(window=ROLLING_WINDOW,
                     min_periods=ROLLING_MIN_OBS)
            .mean()
        )

    rolled = (
        df.groupby("player_id")[roll_cols]
        .transform(rolling_mean)
        .add_suffix(f"_roll{ROLLING_WINDOW}")
    )

    result = pd.concat(
        [df[["player_id", "player_name", "team", "game_id", "game_date",
             "season", "win", "is_home"]],
         rolled],
        axis=1,
    )

    # Drop rows where all rolling features are NaN (first few games of career)
    roll_feature_cols = rolled.columns.tolist()
    result = result.dropna(subset=roll_feature_cols, how="all").reset_index(drop=True)
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main(seasons: list[str], output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_gamelogs, all_advanced, all_lineups, all_games = [], [], [], []

    print("=" * 60)
    print("NBA Comprehensive Data Collection")
    print(f"Seasons : {seasons}")
    print(f"Output  : {out.resolve()}")
    print("=" * 60)

    for season in tqdm(seasons, desc="Season Progress"):
        gl = fetch_player_gamelogs(season)
        if not gl.empty:
            all_gamelogs.append(gl)

        adv = fetch_player_advanced(season)
        if not adv.empty:
            all_advanced.append(adv)

        lu = fetch_lineup_stats(season)
        if not lu.empty:
            all_lineups.append(lu)

        gr = fetch_game_results(season)
        if not gr.empty:
            all_games.append(gr)

    if not all_gamelogs:
        print("\n✗  No data collected — check network / nba_api version.")
        return

    # ── Concatenate ───────────────────────────────────────────────────────────
    df_gamelogs  = pd.concat(all_gamelogs,  ignore_index=True)
    df_advanced  = pd.concat(all_advanced,  ignore_index=True) if all_advanced  else pd.DataFrame()
    df_lineups   = pd.concat(all_lineups,   ignore_index=True) if all_lineups   else pd.DataFrame()
    df_games_raw = pd.concat(all_games,     ignore_index=True) if all_games     else pd.DataFrame()

    df_game_pairs = build_game_pairs(df_games_raw) if not df_games_raw.empty else pd.DataFrame()
    df_rolling    = build_rolling_features(df_gamelogs)

    # ── Export ────────────────────────────────────────────────────────────────
    files = {
        "player_gamelogs.csv":  df_gamelogs,
        "player_advanced.csv":  df_advanced,
        "lineup_synergy.csv":   df_lineups,
        "game_results.csv":     df_game_pairs,
        "player_rolling.csv":   df_rolling,
    }

    print("\n[Export] Writing CSV files ...")
    for fname, df in files.items():
        if df.empty:
            print(f"{fname} — empty, skipped")
            continue
        path = out / fname
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{fname:30s}  {len(df):>8,} rows × {df.shape[1]:>3} cols")

    print("\n Done.")



# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA comprehensive data collection")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=DEFAULT_SEASONS,
        help='Seasons to fetch, e.g. --seasons 2022-23 2023-24',
    )
    parser.add_argument(
        "--output",
        default="./",
        help="Output directory (created if it does not exist)",
    )
    args = parser.parse_args()
    main(seasons=args.seasons, output_dir=args.output)
