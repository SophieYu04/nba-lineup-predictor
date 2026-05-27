"""
predict_lineup.py — Inference for arbitrary lineup matchups.

Usage
-----
    from predict_lineup import predict_matchup, find_player

    home_ids = [find_player("LeBron James"), find_player("Anthony Davis"), ...]
    away_ids = [find_player("Jayson Tatum"), find_player("Jaylen Brown"), ...]

    result = predict_matchup(home_ids, away_ids, snapshot_date="2025-04-01")
    print(result["P_home_win"])
    print(result["per_player_shap_home"])
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import shap


def _normalize(s: str) -> str:
    """Strip diacritics for accent-insensitive name matching."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    ).lower()

DATA = Path("nba_data")
MODELS = Path("outputs/models")


# ──────────────────────────────────────────────────────────────────────
# Load artifacts (lazy singletons)
# ──────────────────────────────────────────────────────────────────────
class _Artifacts:
    _loaded = False

    @classmethod
    def load(cls):
        if cls._loaded:
            return
        cls.xgb_clf   = joblib.load(MODELS / "xgb_output_A.joblib")
        cls.logit_B   = joblib.load(MODELS / "logit_output_B.joblib")
        cls.final_clf = joblib.load(MODELS / "final_logit.joblib")
        cls.scaler_A  = joblib.load(MODELS / "scaler_A.joblib")
        cls.scaler_B  = joblib.load(MODELS / "scaler_B.joblib")
        cls.scaler_pm = joblib.load(MODELS / "scaler_pm.joblib")
        with open(MODELS / "feature_meta.json") as f:
            cls.meta = json.load(f)
        cls.p80 = pd.read_csv(MODELS / "p80_thresholds.csv")
        cls.rolling = pd.read_csv(DATA / "player_rolling.csv")
        cls.rolling["game_date"] = pd.to_datetime(cls.rolling["game_date"])
        # name lookup (last seen team-season — enough for resolution)
        cls.name_map = (
            cls.rolling[["player_id", "player_name"]]
            .drop_duplicates("player_id")
            .set_index("player_id")["player_name"]
            .to_dict()
        )
        cls.explainer = shap.TreeExplainer(cls.xgb_clf)
        cls._loaded = True


def find_player(name_substring: str) -> int:
    """Resolve a name substring to a single player_id (case- and accent-insensitive)."""
    _Artifacts.load()
    hits = (
        _Artifacts.rolling[["player_id", "player_name"]]
        .drop_duplicates("player_id")
        .copy()
    )
    needle = _normalize(name_substring)
    hits["normalized"] = hits["player_name"].map(_normalize)
    matches = hits[hits["normalized"].str.contains(needle, regex=False)]
    if len(matches) == 0:
        raise ValueError(f"No player matches {name_substring!r}")
    if len(matches) > 1:
        opts = ", ".join(f"{r.player_name}({r.player_id})" for r in matches.itertuples())
        raise ValueError(f"Ambiguous: {name_substring!r} → {opts}")
    return int(matches.iloc[0]["player_id"])


def player_name(pid: int) -> str:
    _Artifacts.load()
    return _Artifacts.name_map.get(int(pid), f"#{pid}")


# ──────────────────────────────────────────────────────────────────────
# Snapshot — get each player's most recent rolling row before snapshot_date
# ──────────────────────────────────────────────────────────────────────
def _snapshot(player_ids: Iterable[int], snapshot_date: pd.Timestamp | None) -> pd.DataFrame:
    """Return one row per player_id: the latest rolling features strictly before snapshot_date."""
    _Artifacts.load()
    df = _Artifacts.rolling[_Artifacts.rolling["player_id"].isin(list(player_ids))].copy()
    if snapshot_date is not None:
        df = df[df["game_date"] < pd.to_datetime(snapshot_date)]
    if df.empty:
        raise ValueError(f"No rolling data found for {list(player_ids)} before {snapshot_date}")
    # take latest per player
    df = df.sort_values("game_date").groupby("player_id", as_index=False).tail(1)
    # validate every requested id found
    missing = set(player_ids) - set(df["player_id"])
    if missing:
        missing_names = [player_name(m) for m in missing]
        raise ValueError(f"No rolling data before {snapshot_date} for: {missing_names}")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Per-team scoring
# ──────────────────────────────────────────────────────────────────────
def _score_team(snapshot: pd.DataFrame, season_for_p80: str) -> dict:
    """Compute Score_A (sum SHAP), Score_B (logit), team_pm, and per-player SHAP for one team."""
    A = _Artifacts
    feats_A = A.meta["output_A_features"]
    dummy_stats = A.meta["output_B_dummy_stats"]

    # ─ Output A: per-player XGB + SHAP, sum across team ───────────────
    X_A = snapshot[feats_A].fillna(snapshot[feats_A].median())
    shap_vals = A.explainer.shap_values(X_A)  # (n_players, n_feats)
    player_shap = shap_vals.sum(axis=1)
    score_A = float(player_shap.sum())

    # ─ Output B: binarize → max/sum aggregate → 36 interactions → logit
    p80_row = A.p80[A.p80["season"] == season_for_p80]
    if p80_row.empty:
        # fall back to latest available season's thresholds
        p80_row = A.p80.sort_values("season").tail(1)
    p80_vec = p80_row.iloc[0][dummy_stats]

    dummies = (snapshot[dummy_stats] > p80_vec).astype(int)  # (n_players, 9)
    agg_max = dummies.max(axis=0).add_prefix("d_").add_suffix("_max")
    agg_sum = dummies.sum(axis=0).add_prefix("d_").add_suffix("_sum")
    team_row = pd.concat([agg_max, agg_sum])

    # 36 interactions: sum × sum
    dummy_cols = [f"d_{c}" for c in dummy_stats]
    for i, a in enumerate(dummy_cols):
        for b in dummy_cols[i + 1:]:
            team_row[f"x_{a}_{b}"] = team_row[f"{a}_sum"] * team_row[f"{b}_sum"]

    # align to training feature order
    X_B = team_row.reindex(A.meta["output_B_features"]).fillna(0).to_frame().T
    score_B = float(A.logit_B.decision_function(X_B)[0])

    # ─ team_pm = mean of plus_minus_roll10
    team_pm = float(snapshot["plus_minus_roll10"].mean())

    return {
        "score_A": score_A,
        "score_B": score_B,
        "team_pm": team_pm,
        "per_player_shap": pd.DataFrame({
            "player_id": snapshot["player_id"].astype(int).values,
            "player_name": snapshot["player_name"].values,
            "min_roll10": snapshot["min_roll10"].values,
            "shap_sum": player_shap,
            "snapshot_date": snapshot["game_date"].dt.date.values,
        }).sort_values("shap_sum", ascending=False).reset_index(drop=True),
    }


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────
def predict_matchup(
    home_player_ids: Iterable[int],
    away_player_ids: Iterable[int],
    snapshot_date: str | pd.Timestamp | None = None,
    is_home: int = 1,
) -> dict:
    """Predict P(home wins) for an arbitrary lineup matchup.

    Parameters
    ----------
    home_player_ids, away_player_ids : lists of player_id
        Use find_player("name") to resolve names.
    snapshot_date : str | Timestamp, optional
        Each player's stats come from their latest game strictly before this date.
        If None, uses each player's most recent available game.
    is_home : int, default 1
        Set to 0 for "neutral floor" hypothetical (drops home-court factor).

    Returns
    -------
    dict with P_home_win, score components, per-player SHAP for both teams.
    """
    _Artifacts.load()

    if snapshot_date is None:
        # use each player's most recent
        snap_h = _snapshot(home_player_ids, snapshot_date=None)
        snap_a = _snapshot(away_player_ids, snapshot_date=None)
        season_for_p80 = snap_h["season"].iloc[0]  # use any snapshot's season
    else:
        snapshot_date = pd.to_datetime(snapshot_date)
        snap_h = _snapshot(home_player_ids, snapshot_date)
        snap_a = _snapshot(away_player_ids, snapshot_date)
        # determine season from snapshot_date
        m = snapshot_date.month
        y = snapshot_date.year
        if m >= 10:
            season_for_p80 = f"{y}-{str(y + 1)[2:]}"
        else:
            season_for_p80 = f"{y - 1}-{str(y)[2:]}"

    home = _score_team(snap_h, season_for_p80)
    away = _score_team(snap_a, season_for_p80)

    delta_A_raw = home["score_A"] - away["score_A"]
    delta_B_raw = home["score_B"] - away["score_B"]
    delta_pm_raw = home["team_pm"] - away["team_pm"]

    A = _Artifacts
    delta_A  = float(A.scaler_A.transform(pd.DataFrame({"delta_A_raw":  [delta_A_raw]}))[0, 0])
    delta_B  = float(A.scaler_B.transform(pd.DataFrame({"delta_B_raw":  [delta_B_raw]}))[0, 0])
    delta_pm = float(A.scaler_pm.transform(pd.DataFrame({"delta_pm_raw":[delta_pm_raw]}))[0, 0])

    final_x = pd.DataFrame(
        [[delta_A, delta_B, delta_pm, is_home]],
        columns=A.meta["final_feature_order"],
    )
    P_home = float(A.final_clf.predict_proba(final_x)[0, 1])

    # decompose final logit into per-component contributions for explainability
    α, β, δ, γ = A.final_clf.coef_[0]
    b = A.final_clf.intercept_[0]
    contributions = {
        "from_score_A":  α * delta_A,
        "from_score_B":  β * delta_B,
        "from_delta_pm": δ * delta_pm,
        "from_is_home":  γ * is_home,
        "from_intercept": b,
    }

    return {
        "P_home_win": P_home,
        "P_away_win": 1 - P_home,
        "snapshot_date": str(snapshot_date.date()) if snapshot_date is not None else "latest",
        "season_for_p80": season_for_p80,
        "is_home": is_home,
        # raw scores
        "score_A_home": home["score_A"],
        "score_A_away": away["score_A"],
        "score_B_home": home["score_B"],
        "score_B_away": away["score_B"],
        "pm_home": home["team_pm"],
        "pm_away": away["team_pm"],
        # standardized deltas
        "delta_A":  delta_A,
        "delta_B":  delta_B,
        "delta_pm": delta_pm,
        # additive contributions (sum + sigmoid → P_home)
        "logit_contributions": contributions,
        # per-player breakdown
        "per_player_shap_home": home["per_player_shap"],
        "per_player_shap_away": away["per_player_shap"],
    }


def summary_table(result: dict) -> pd.DataFrame:
    """Compact one-row summary of a matchup result."""
    return pd.DataFrame([{
        "P_home_win": result["P_home_win"],
        "Score_A_home": result["score_A_home"],
        "Score_A_away": result["score_A_away"],
        "ΔA_z": result["delta_A"],
        "Score_B_home": result["score_B_home"],
        "Score_B_away": result["score_B_away"],
        "ΔB_z": result["delta_B"],
        "pm_home": result["pm_home"],
        "pm_away": result["pm_away"],
        "Δpm_z": result["delta_pm"],
    }])
