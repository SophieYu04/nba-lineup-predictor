"""
Design B: Score A (XGB+SHAP) + Score B (logit, team-internal) + Score C
(L1 logit, cross-team) → final.

Pipeline
--------
1. Time-based split: train = 2020-21 ~ 2023-24, test = 2024-25
2. Sample alignment: intersect game_ids that both models can handle
3. Output A: per-player XGBoost → SHAP → Σ per team-game → Score_A
4. Output B: lineup dummy max/sum + 36 within-team interactions → logit → Score_B
5. Output C: 9×9 cross-team sum×sum interactions → L1 logit → Score_C (per matchup)
6. Final calibration: P = σ(α·ΔA + β·ΔB + ζ·C + δ·Δpm + γ·is_home + b)

Outputs
-------
- predictions_test.csv : per-matchup predictions incl. Score_A/B/C and P_home
- player_shap.csv      : per-(game, team, player) SHAP contributions (Score_A 解釋性)
- final_weights.json   : α, β, ζ, δ, γ, intercept
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

OOF_FOLDS = 5

DATA = Path("nba_data")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]
TEST_SEASON   = "2024-25"

# Per-player rolling features for Output A (plus_minus 排除 — 它是隊伍強度代理，不是個人技能)
PLAYER_FEATS = [
    "pts_roll10", "fgm_roll10", "fga_roll10", "fg_pct_roll10",
    "fg3m_roll10", "fg3a_roll10", "fg3_pct_roll10",
    "ftm_roll10", "fta_roll10", "ft_pct_roll10",
    "oreb_roll10", "dreb_roll10", "reb_roll10",
    "ast_roll10", "tov_roll10", "stl_roll10", "blk_roll10",
    "fouls_roll10", "fouls_drawn_roll10",
    "ast_to_tov_roll10", "true_shooting_roll10",
    "usage_proxy_roll10", "def_impact_roll10",
    "min_roll10",
]

# Stats for Output B dummies (用賽季級 P80 二值化)
DUMMY_STATS = [
    "pts_roll10", "ast_roll10", "reb_roll10",
    "stl_roll10", "blk_roll10", "fg3m_roll10",
    "true_shooting_roll10", "usage_proxy_roll10", "def_impact_roll10",
]


# ──────────────────────────────────────────────────────────────────────
# 1. Load & split by season
# ──────────────────────────────────────────────────────────────────────
print("[1] Loading data ...")
games = pd.read_csv(DATA / "game_results.csv")
rolling = pd.read_csv(DATA / "player_rolling.csv")
gamelogs = pd.read_csv(DATA / "player_gamelogs.csv",
                       usecols=["game_id", "team", "player_id", "win", "min"])

games["game_date"] = pd.to_datetime(games["game_date"])
rolling["game_date"] = pd.to_datetime(rolling["game_date"])

train_mask_g = games["season"].isin(TRAIN_SEASONS)
test_mask_g = games["season"].eq(TEST_SEASON)
print(f"  games: train={train_mask_g.sum()}  test={test_mask_g.sum()}")


# ──────────────────────────────────────────────────────────────────────
# 2. Build per-player training rows for Output A
# ──────────────────────────────────────────────────────────────────────
print("[2] Building per-player rows for Output A ...")

# keep only rolling rows whose features are all present
feats_present = [c for c in PLAYER_FEATS if c in rolling.columns]
print(f"  using {len(feats_present)} per-player features (plus_minus excluded)")

player_rows = rolling.dropna(subset=feats_present).copy()

# attach team_win label (broadcast from gamelogs)
labels = (gamelogs
          .groupby(["game_id", "team"])["win"]
          .first()
          .reset_index()
          .rename(columns={"win": "team_win"}))
player_rows = player_rows.merge(labels, on=["game_id", "team"], how="inner")

# sample weight: min_roll10 (replace NaN with median to avoid losing rows)
player_rows["min_roll10"] = player_rows["min_roll10"].fillna(player_rows["min_roll10"].median())
player_rows["sample_weight"] = player_rows["min_roll10"].clip(lower=1.0)

A_train = player_rows[player_rows["season"].isin(TRAIN_SEASONS)]
A_test  = player_rows[player_rows["season"].eq(TEST_SEASON)]
print(f"  per-player rows: train={len(A_train):,}  test={len(A_test):,}")


# ──────────────────────────────────────────────────────────────────────
# 3. Train XGBoost (Output A), compute SHAP, aggregate to team-game
# ──────────────────────────────────────────────────────────────────────
print("[3] Training Output A XGBoost (OOF for train + single fit for test) ...")

def build_xgb():
    return xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )

# OOF SHAP on training rows (so train-side Score_A reflects out-of-sample quality)
train_game_ids = A_train["game_id"].unique()
kf = KFold(n_splits=OOF_FOLDS, shuffle=True, random_state=42)
A_train = A_train.reset_index(drop=True)
A_train["shap_sum"] = np.nan

for fold, (tr_g, va_g) in enumerate(kf.split(train_game_ids), 1):
    tr_games = set(train_game_ids[tr_g])
    va_games = set(train_game_ids[va_g])
    tr_idx = A_train.index[A_train["game_id"].isin(tr_games)]
    va_idx = A_train.index[A_train["game_id"].isin(va_games)]

    clf = build_xgb()
    clf.fit(
        A_train.loc[tr_idx, feats_present], A_train.loc[tr_idx, "team_win"],
        sample_weight=A_train.loc[tr_idx, "sample_weight"],
    )
    expl = shap.TreeExplainer(clf)
    shap_va = expl.shap_values(A_train.loc[va_idx, feats_present])
    A_train.loc[va_idx, "shap_sum"] = shap_va.sum(axis=1)
    print(f"  fold {fold}/{OOF_FOLDS}: train {len(tr_idx):,} → OOF {len(va_idx):,}")

# Refit on full train for test prediction
xgb_clf = build_xgb()
xgb_clf.fit(
    A_train[feats_present], A_train["team_win"],
    sample_weight=A_train["sample_weight"],
)
explainer = shap.TreeExplainer(xgb_clf)
shap_test = explainer.shap_values(A_test[feats_present])
A_test["shap_sum"] = shap_test.sum(axis=1)

# Aggregate Σ SHAP per team-game = Score A
score_A = (pd.concat([A_train, A_test])
           .groupby(["game_id", "team"])["shap_sum"]
           .sum()
           .reset_index()
           .rename(columns={"shap_sum": "score_A"}))
print(f"  Score_A team-games: {len(score_A):,}")

# Also save per-player SHAP for explainability later
per_player_shap = pd.concat([A_train, A_test])[
    ["game_id", "game_date", "team", "player_id", "player_name",
     "season", "min_roll10", "shap_sum"]
].rename(columns={"shap_sum": "player_shap"})
per_player_shap.to_csv(OUT / "player_shap.csv", index=False)


# ──────────────────────────────────────────────────────────────────────
# 4. Build Output B features (lineup dummies)
# ──────────────────────────────────────────────────────────────────────
print("[4] Building Output B lineup dummy features ...")

dummy_src = rolling.dropna(subset=DUMMY_STATS).copy()
# Season-wise P80 dummies
for c in DUMMY_STATS:
    dummy_src[f"d_{c}"] = (
        dummy_src.groupby("season")[c]
        .transform(lambda s: (s > s.quantile(0.80)).astype(int))
    )

dummy_cols = [f"d_{c}" for c in DUMMY_STATS]

# Aggregate to team-game: max (有無) and sum (厚度)
agg = (dummy_src
       .groupby(["game_id", "team"])[dummy_cols]
       .agg(["max", "sum"]))
agg.columns = [f"{a}_{b}" for a, b in agg.columns]
agg = agg.reset_index()

# Pairwise interaction terms — sum × sum for ALL 9 skill pairs (C(9,2) = 36)
# L2 regularization (smaller C) handles overfitting from the larger feature space.
ALL_DUMMIES = [f"d_{c}" for c in DUMMY_STATS]
for i, a in enumerate(ALL_DUMMIES):
    for b in ALL_DUMMIES[i + 1:]:
        agg[f"x_{a}_{b}"] = agg[f"{a}_sum"] * agg[f"{b}_sum"]

team_feat_cols_B = [c for c in agg.columns if c not in ("game_id", "team")]
print(f"  Output B team-level features: {len(team_feat_cols_B)}")


# ──────────────────────────────────────────────────────────────────────
# 5. Sample alignment: intersect Score_A coverage and Output B coverage
# ──────────────────────────────────────────────────────────────────────
games_A = set(zip(score_A["game_id"], score_A["team"]))
games_B = set(zip(agg["game_id"], agg["team"]))
common = games_A & games_B
print(f"[5] Aligned team-games: {len(common):,}  (A={len(games_A):,}, B={len(games_B):,})")


# ──────────────────────────────────────────────────────────────────────
# 6. Train Output B logistic regression
# ──────────────────────────────────────────────────────────────────────
print("[6] Training Output B logit (OOF for train + single fit for test) ...")
agg = agg.merge(labels, on=["game_id", "team"], how="inner")
agg = agg.merge(games[["game_id", "season"]].drop_duplicates(),
                on="game_id", how="inner")

B_train = agg[agg["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
B_test  = agg[agg["season"].eq(TEST_SEASON)].reset_index(drop=True)

B_train["score_B"] = np.nan
train_game_ids_B = B_train["game_id"].unique()
for fold, (tr_g, va_g) in enumerate(kf.split(train_game_ids_B), 1):
    tr_games = set(train_game_ids_B[tr_g])
    va_games = set(train_game_ids_B[va_g])
    tr_idx = B_train.index[B_train["game_id"].isin(tr_games)]
    va_idx = B_train.index[B_train["game_id"].isin(va_games)]
    lg = LogisticRegression(C=1.0, max_iter=2000)
    lg.fit(B_train.loc[tr_idx, team_feat_cols_B], B_train.loc[tr_idx, "team_win"])
    B_train.loc[va_idx, "score_B"] = lg.decision_function(B_train.loc[va_idx, team_feat_cols_B])

# Refit on full train for test prediction
logit_B = LogisticRegression(C=1.0, max_iter=2000)
logit_B.fit(B_train[team_feat_cols_B], B_train["team_win"])
B_test["score_B"] = logit_B.decision_function(B_test[team_feat_cols_B])

score_B = pd.concat([
    B_train[["game_id", "team", "score_B"]],
    B_test[["game_id", "team", "score_B"]],
], ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# 6b. Team-mean plus_minus_roll10 (隊伍歷史強度代理) — 不放進 per-player XGB
#     避免 leak，僅在 final layer 作為獨立特徵使用
# ──────────────────────────────────────────────────────────────────────
print("[6b] Aggregating team-mean plus_minus_roll10 ...")
pm_src = rolling.dropna(subset=["plus_minus_roll10"]).copy()
team_pm = (pm_src
           .groupby(["game_id", "team"])["plus_minus_roll10"]
           .mean()
           .reset_index()
           .rename(columns={"plus_minus_roll10": "team_pm"}))
print(f"  team-games with team_pm: {len(team_pm):,}")


# ──────────────────────────────────────────────────────────────────────
# 6c. Output C: per-matchup cross-team skill interactions (L1 logit)
#     9 home skills × 9 away skills = 81 interactions; L1 picks survivors.
#     score_C is matchup-level (already directional), no diff needed.
# ──────────────────────────────────────────────────────────────────────
print("[6c] Building Output C cross-team interaction features ...")

sum_cols = [f"d_{c}_sum" for c in DUMMY_STATS]
team_sums = agg[["game_id", "team"] + sum_cols].copy()

home_sums = team_sums.rename(
    columns={"team": "home_team", **{c: f"h_{c}" for c in sum_cols}}
)
away_sums = team_sums.rename(
    columns={"team": "away_team", **{c: f"a_{c}" for c in sum_cols}}
)

cross_frame = games[["game_id", "season", "home_team", "away_team", "home_win"]].copy()
cross_frame = cross_frame.merge(home_sums, on=["game_id", "home_team"], how="inner")
cross_frame = cross_frame.merge(away_sums, on=["game_id", "away_team"], how="inner")

c_feat_cols = []
for h_skill in DUMMY_STATS:
    for a_skill in DUMMY_STATS:
        col = f"c_h_{h_skill}_a_{a_skill}"
        cross_frame[col] = (
            cross_frame[f"h_d_{h_skill}_sum"] * cross_frame[f"a_d_{a_skill}_sum"]
        )
        c_feat_cols.append(col)

print(f"  cross-team features: {len(c_feat_cols)} (9x9)")

C_train = cross_frame[cross_frame["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
C_test  = cross_frame[cross_frame["season"].eq(TEST_SEASON)].reset_index(drop=True)
print(f"  matchups: train={len(C_train):,}  test={len(C_test):,}")


def build_logit_C():
    # L1 needs scaled inputs; pipeline keeps train/inference consistent.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(
            Cs=np.logspace(-3, 1, 20),
            penalty="l1",
            solver="saga",
            max_iter=5000,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42,
        )),
    ])


C_train["score_C_raw"] = np.nan
train_game_ids_C = C_train["game_id"].unique()
for fold, (tr_g, va_g) in enumerate(kf.split(train_game_ids_C), 1):
    tr_games = set(train_game_ids_C[tr_g])
    va_games = set(train_game_ids_C[va_g])
    tr_idx = C_train.index[C_train["game_id"].isin(tr_games)]
    va_idx = C_train.index[C_train["game_id"].isin(va_games)]
    pipe = build_logit_C()
    pipe.fit(C_train.loc[tr_idx, c_feat_cols], C_train.loc[tr_idx, "home_win"])
    C_train.loc[va_idx, "score_C_raw"] = pipe.decision_function(
        C_train.loc[va_idx, c_feat_cols]
    )
    print(f"  fold {fold}/{OOF_FOLDS}: train {len(tr_idx):,} → OOF {len(va_idx):,}")

logit_C = build_logit_C()
logit_C.fit(C_train[c_feat_cols], C_train["home_win"])
C_test["score_C_raw"] = logit_C.decision_function(C_test[c_feat_cols])

clf_C = logit_C.named_steps["clf"]
nonzero = int((clf_C.coef_[0] != 0).sum())
print(f"  L1 kept {nonzero}/{len(c_feat_cols)} features  (C={clf_C.C_[0]:.4f})")

score_C_all = pd.concat([
    C_train[["game_id", "home_team", "away_team", "score_C_raw"]],
    C_test[["game_id", "home_team", "away_team", "score_C_raw"]],
], ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# 7. Build matchup-level features (ΔA, ΔB, Δpm, is_home)
# ──────────────────────────────────────────────────────────────────────
print("[7] Building matchup-level features ...")
matchup = games[["game_id", "game_date", "season", "home_team", "away_team", "home_win"]].copy()

# home side scores
matchup = matchup.merge(
    score_A.rename(columns={"team": "home_team", "score_A": "score_A_home"}),
    on=["game_id", "home_team"], how="inner",
)
matchup = matchup.merge(
    score_B.rename(columns={"team": "home_team", "score_B": "score_B_home"}),
    on=["game_id", "home_team"], how="inner",
)
# away side scores
matchup = matchup.merge(
    score_A.rename(columns={"team": "away_team", "score_A": "score_A_away"}),
    on=["game_id", "away_team"], how="inner",
)
matchup = matchup.merge(
    score_B.rename(columns={"team": "away_team", "score_B": "score_B_away"}),
    on=["game_id", "away_team"], how="inner",
)
# team_pm merges
matchup = matchup.merge(
    team_pm.rename(columns={"team": "home_team", "team_pm": "pm_home"}),
    on=["game_id", "home_team"], how="inner",
)
matchup = matchup.merge(
    team_pm.rename(columns={"team": "away_team", "team_pm": "pm_away"}),
    on=["game_id", "away_team"], how="inner",
)

# score_C is per-matchup (directional), merge directly — no diff
matchup = matchup.merge(
    score_C_all, on=["game_id", "home_team", "away_team"], how="inner",
)

matchup["delta_A_raw"]  = matchup["score_A_home"] - matchup["score_A_away"]
matchup["delta_B_raw"]  = matchup["score_B_home"] - matchup["score_B_away"]
matchup["delta_pm_raw"] = matchup["pm_home"]      - matchup["pm_away"]
matchup["is_home"] = 1  # training frame is fixed; final layer learns its weight

M_train = matchup[matchup["season"].isin(TRAIN_SEASONS)].copy()
M_test  = matchup[matchup["season"].eq(TEST_SEASON)].copy()
print(f"  matchups: train={len(M_train):,}  test={len(M_test):,}")

# Standardize all Δ / Score features using train-set statistics only
scaler_A  = StandardScaler().fit(M_train[["delta_A_raw"]])
scaler_B  = StandardScaler().fit(M_train[["delta_B_raw"]])
scaler_pm = StandardScaler().fit(M_train[["delta_pm_raw"]])
scaler_C  = StandardScaler().fit(M_train[["score_C_raw"]])
for df in (M_train, M_test):
    df["delta_A"]  = scaler_A.transform(df[["delta_A_raw"]]).ravel()
    df["delta_B"]  = scaler_B.transform(df[["delta_B_raw"]]).ravel()
    df["delta_pm"] = scaler_pm.transform(df[["delta_pm_raw"]]).ravel()
    df["score_C"]  = scaler_C.transform(df[["score_C_raw"]]).ravel()


# ──────────────────────────────────────────────────────────────────────
# 8. Final calibration: P = σ(α·ΔA + β·ΔB + ζ·C + δ·Δpm + γ·is_home + b)
# ──────────────────────────────────────────────────────────────────────
print("[8] Final calibration logit ...")
FINAL_FEATS = ["delta_A", "delta_B", "score_C", "delta_pm", "is_home"]
final_clf = LogisticRegression(C=1.0, max_iter=1000)
final_clf.fit(M_train[FINAL_FEATS], M_train["home_win"])

alpha, beta, zeta, delta_pm_coef, gamma = final_clf.coef_[0]
intercept = final_clf.intercept_[0]

print(f"  α (Score_A weight)        = {alpha:+.4f}")
print(f"  β (Score_B weight)        = {beta:+.4f}")
print(f"  ζ (Score_C cross-team)    = {zeta:+.4f}")
print(f"  δ (plus_minus_diff weight)= {delta_pm_coef:+.4f}")
print(f"  γ (home advantage)        = {gamma:+.4f}")
print(f"  intercept                 = {intercept:+.4f}")


# ──────────────────────────────────────────────────────────────────────
# 9. Evaluate on hold-out 2024-25
# ──────────────────────────────────────────────────────────────────────
print("\n[9] Evaluation on hold-out 2024-25:")
P_test = final_clf.predict_proba(M_test[FINAL_FEATS])[:, 1]
y_test = M_test["home_win"].values
y_pred = (P_test >= 0.5).astype(int)

print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  AUC       : {roc_auc_score(y_test, P_test):.4f}")
print(f"  Baseline  : {max(y_test.mean(), 1 - y_test.mean()):.4f}")
print(classification_report(y_test, y_pred, digits=3))

# Component ablations
def auc_for_cols(cols):
    p = LogisticRegression(C=1.0, max_iter=1000).fit(
        M_train[cols], M_train["home_win"]
    ).predict_proba(M_test[cols])[:, 1]
    return roc_auc_score(y_test, p)

print(f"\n  Ablation AUC:")
print(f"    Score_A only  + is_home     : {auc_for_cols(['delta_A','is_home']):.4f}")
print(f"    Score_B only  + is_home     : {auc_for_cols(['delta_B','is_home']):.4f}")
print(f"    Score_C only  + is_home     : {auc_for_cols(['score_C','is_home']):.4f}")
print(f"    Δpm only      + is_home     : {auc_for_cols(['delta_pm','is_home']):.4f}")
print(f"    A + B         + is_home     : {auc_for_cols(['delta_A','delta_B','is_home']):.4f}")
print(f"    A + Δpm       + is_home     : {auc_for_cols(['delta_A','delta_pm','is_home']):.4f}")
print(f"    B + Δpm       + is_home     : {auc_for_cols(['delta_B','delta_pm','is_home']):.4f}")
print(f"    A + B + Δpm   + is_home     : {auc_for_cols(['delta_A','delta_B','delta_pm','is_home']):.4f}")
print(f"    A + C + Δpm   + is_home     : {auc_for_cols(['delta_A','score_C','delta_pm','is_home']):.4f}")
print(f"    B + C + Δpm   + is_home     : {auc_for_cols(['delta_B','score_C','delta_pm','is_home']):.4f}")
print(f"    A + B + C     + is_home     : {auc_for_cols(['delta_A','delta_B','score_C','is_home']):.4f}")
print(f"    A + B + C + Δpm + is_home   : {roc_auc_score(y_test, P_test):.4f}")


# ──────────────────────────────────────────────────────────────────────
# 10. Export
# ──────────────────────────────────────────────────────────────────────
out_test = M_test[["game_id", "game_date", "home_team", "away_team",
                   "score_A_home", "score_A_away", "delta_A_raw", "delta_A",
                   "score_B_home", "score_B_away", "delta_B_raw", "delta_B",
                   "score_C_raw", "score_C",
                   "pm_home", "pm_away", "delta_pm_raw", "delta_pm",
                   "is_home", "home_win"]].copy()
out_test["P_home_win"] = P_test
out_test["pred_home_win"] = y_pred
out_test.to_csv(OUT / "predictions_test.csv", index=False)

with open(OUT / "final_weights.json", "w") as f:
    json.dump({
        "alpha_score_A": float(alpha),
        "beta_score_B": float(beta),
        "zeta_score_C": float(zeta),
        "delta_plus_minus": float(delta_pm_coef),
        "gamma_is_home": float(gamma),
        "intercept": float(intercept),
        "train_seasons": TRAIN_SEASONS,
        "test_season": TEST_SEASON,
        "n_train_matchups": int(len(M_train)),
        "n_test_matchups": int(len(M_test)),
        "output_B_total_features": int(len(team_feat_cols_B)),
        "output_B_interaction_pairs": int(len(ALL_DUMMIES) * (len(ALL_DUMMIES) - 1) // 2),
        "output_C_total_features": int(len(c_feat_cols)),
        "output_C_l1_kept_features": int(nonzero),
    }, f, indent=2)

print(f"\n  Saved: {OUT / 'predictions_test.csv'}")
print(f"  Saved: {OUT / 'player_shap.csv'}")
print(f"  Saved: {OUT / 'final_weights.json'}")


# ──────────────────────────────────────────────────────────────────────
# 11. Persist trained models & metadata for inference
# ──────────────────────────────────────────────────────────────────────
print("\n[11] Saving models for inference ...")
MODELS = OUT / "models"
MODELS.mkdir(exist_ok=True)

# Per-season P80 thresholds used to binarize Output B dummies at inference
p80_thresholds = (
    dummy_src.groupby("season")[DUMMY_STATS]
    .quantile(0.80)
    .reset_index()
)
p80_thresholds.to_csv(MODELS / "p80_thresholds.csv", index=False)

# Save fitted artifacts via joblib
joblib.dump(xgb_clf,    MODELS / "xgb_output_A.joblib")
joblib.dump(logit_B,    MODELS / "logit_output_B.joblib")
joblib.dump(logit_C,    MODELS / "logit_output_C.joblib")
joblib.dump(final_clf,  MODELS / "final_logit.joblib")
joblib.dump(scaler_A,   MODELS / "scaler_A.joblib")
joblib.dump(scaler_B,   MODELS / "scaler_B.joblib")
joblib.dump(scaler_C,   MODELS / "scaler_C.joblib")
joblib.dump(scaler_pm,  MODELS / "scaler_pm.joblib")

# Column-order metadata for all three feature sets
with open(MODELS / "feature_meta.json", "w") as f:
    json.dump({
        "output_A_features": feats_present,
        "output_B_features": team_feat_cols_B,
        "output_B_dummy_stats": DUMMY_STATS,
        "output_B_interaction_pairs": [
            f"{a}|{b}"
            for i, a in enumerate(ALL_DUMMIES) for b in ALL_DUMMIES[i + 1:]
        ],
        "output_C_features": c_feat_cols,
        "output_C_pairs": [
            f"home:{h}|away:{a}" for h in DUMMY_STATS for a in DUMMY_STATS
        ],
        "final_feature_order": FINAL_FEATS,
    }, f, indent=2)

# Interpretability: export Score C survivors with their L1 coefficients.
# Rows sorted by |coef| so reviewers can scan the strongest cross-team
# pairs first; zero-weight features dropped (they're not survivors).
cross_team_pairs = pd.DataFrame({
    "feature":   c_feat_cols,
    "home_skill": [f"d_{h}" for h in DUMMY_STATS for _ in DUMMY_STATS],
    "away_skill": [f"d_{a}" for _ in DUMMY_STATS for a in DUMMY_STATS],
    "coef":      clf_C.coef_[0],
})
cross_team_pairs = cross_team_pairs[cross_team_pairs["coef"] != 0].copy()
cross_team_pairs["abs_coef"] = cross_team_pairs["coef"].abs()
cross_team_pairs = cross_team_pairs.sort_values("abs_coef", ascending=False)
cross_team_pairs.to_csv(OUT / "cross_team_pairs.csv", index=False)

print(f"  Saved: {MODELS}/ (8 models + 2 metadata files)")
print(f"  Saved: {OUT / 'cross_team_pairs.csv'}")
print("\nDone.")
