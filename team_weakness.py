"""
Team weakness analysis — verify that trained Score A / Score B artifacts
can surface which skill dimensions each team is weak in.

Lens 1: per-feature SHAP from xgb_output_A.joblib
  → "where do this team's players lose win-prob the most?"
Lens 2: role coverage from P80 dummies + agg
  → "which elite-player roles is this team missing most often?"
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

DATA   = Path("nba_data")
OUT    = Path("outputs")
MODELS = OUT / "models"

SEASON = "2024-25"
TEAMS  = ["BOS", "LAL", "GSW", "WAS"]

print("Loading artifacts ...")
# joblib loads are safe here: outputs/models/ is produced by score_ab.py
# in this same repo, never from external sources.
xgb_clf = joblib.load(MODELS / "xgb_output_A.joblib")
with open(MODELS / "feature_meta.json") as f:
    meta = json.load(f)
PLAYER_FEATS = meta["output_A_features"]
DUMMY_STATS  = meta["output_B_dummy_stats"]

p80 = pd.read_csv(MODELS / "p80_thresholds.csv")
p80_row = p80[p80["season"] == SEASON].iloc[0]

rolling = pd.read_csv(DATA / "player_rolling.csv")
rolling = rolling.dropna(subset=PLAYER_FEATS).copy()

explainer = shap.TreeExplainer(xgb_clf)


def analyze(team: str) -> None:
    rows = rolling[(rolling["team"] == team) & (rolling["season"] == SEASON)].copy()
    if rows.empty:
        print(f"\n[{team}]  no rows for {SEASON}")
        return

    n_games = rows["game_id"].nunique()
    n_pgs   = len(rows)

    print(f"\n{'='*72}")
    print(f"  {team} — {SEASON}   ({n_games} games, {n_pgs:,} player-game rows)")
    print(f"{'='*72}")

    # Lens 1: per-feature SHAP, weighted by min_roll10 (same as training).
    X = rows[PLAYER_FEATS]
    sv = explainer.shap_values(X)                       # (n_rows, n_feats)
    w  = rows["min_roll10"].fillna(rows["min_roll10"].median()).clip(lower=1.0).values
    w  = w / w.sum()
    avg_shap = (sv * w[:, None]).sum(axis=0)            # 24 numbers

    fs = (pd.DataFrame({"feature": PLAYER_FEATS, "avg_shap": avg_shap})
          .sort_values("avg_shap"))

    print(f"\n  [Lens 1] Per-feature SHAP (min_roll10-weighted)")
    print(f"  ── 5 weakest (negative = drags win-prob down):")
    for _, r in fs.head(5).iterrows():
        print(f"      {r['feature']:28s} {r['avg_shap']:+.4f}")
    print(f"  ── 5 strongest:")
    for _, r in fs.tail(5).iloc[::-1].iterrows():
        print(f"      {r['feature']:28s} {r['avg_shap']:+.4f}")

    # Lens 2: role coverage. For each game, "did the team have an elite X?"
    for c in DUMMY_STATS:
        rows[f"d_{c}"] = (rows[c] > p80_row[c]).astype(int)

    game_max = rows.groupby("game_id")[[f"d_{c}" for c in DUMMY_STATS]].max()
    cov = (pd.DataFrame({"skill": DUMMY_STATS, "coverage": game_max.mean().values})
           .sort_values("coverage"))

    print(f"\n  [Lens 2] Elite-role coverage (% games with ≥1 P80 player)")
    print(f"  ── 3 most-missing roles:")
    for _, r in cov.head(3).iterrows():
        print(f"      {r['skill']:28s} {r['coverage']*100:5.1f}%  "
              f"(gap = {(1-r['coverage'])*100:.0f}% of games)")
    print(f"  ── 3 best-covered roles:")
    for _, r in cov.tail(3).iloc[::-1].iterrows():
        print(f"      {r['skill']:28s} {r['coverage']*100:5.1f}%")


for t in TEAMS:
    analyze(t)

print("\nDone.")
