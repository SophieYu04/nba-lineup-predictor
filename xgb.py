import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt


# =========================
# 1. Read CSV files
# =========================
games = pd.read_csv("nba_data/game_results.csv")
rolling = pd.read_csv("nba_data/player_rolling.csv")

games["game_date"] = pd.to_datetime(games["game_date"])
rolling["game_date"] = pd.to_datetime(rolling["game_date"])


# =========================
# 2. Choose rolling features
# =========================
rolling_features = [
    "pts_roll10",
    "fg3m_roll10",
    "fg3a_roll10",
    "fg3_pct_roll10",
    "fg_pct_roll10",
    "ft_pct_roll10",
    "reb_roll10",
    "ast_roll10",
    "tov_roll10",
    "stl_roll10",
    "blk_roll10",
    "plus_minus_roll10",
    "true_shooting_roll10",
    "usage_proxy_roll10",
    "def_impact_roll10",
    "min_roll10"
]

# only keep columns that actually exist
rolling_features = [c for c in rolling_features if c in rolling.columns]

print("Using rolling features:")
print(rolling_features)


# =========================
# 3. Aggregate player rolling stats into team-game stats
# =========================
team_game = (
    rolling
    .groupby(["game_id", "team"])[rolling_features]
    .mean()
    .reset_index()
)


# =========================
# 4. Merge home team features
# =========================
home_features = team_game.copy()

home_features = home_features.rename(
    columns={
        "team": "home_team",
        **{c: "home_" + c for c in rolling_features}
    }
)

df = games.merge(
    home_features,
    on=["game_id", "home_team"],
    how="inner"
)


# =========================
# 5. Merge away team features
# =========================
away_features = team_game.copy()

away_features = away_features.rename(
    columns={
        "team": "away_team",
        **{c: "away_" + c for c in rolling_features}
    }
)

df = df.merge(
    away_features,
    on=["game_id", "away_team"],
    how="inner"
)

print("Merged dataframe columns:")
print(df.columns.tolist())


# =========================
# 6. Build home-away difference features
# =========================
X = pd.DataFrame()

for c in rolling_features:
    home_col = "home_" + c
    away_col = "away_" + c

    if home_col in df.columns and away_col in df.columns:
        diff_name = c.replace("_roll10", "") + "_diff"
        X[diff_name] = df[home_col] - df[away_col]
    else:
        print("Missing:", home_col, away_col)

y = df["home_win"]

data = pd.concat([df[["game_id", "game_date", "home_team", "away_team"]], X, y], axis=1).dropna()

meta = data[["game_id", "game_date", "home_team", "away_team"]]
X = data.drop(columns=["game_id", "game_date", "home_team", "away_team", "home_win"])
y = data["home_win"]

print("Final X columns:")
print(X.columns.tolist())
print("X shape:", X.shape)
print("y shape:", y.shape)



X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X,
    y,
    meta,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 8. Train XGBoost
# =========================
model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# 9. Evaluate
# =========================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
result = meta_test.copy()
result["actual_home_win"] = y_test.values
result["pred_home_win"] = y_pred
result["pred_home_win_prob"] = y_prob

result["actual_result"] = result["actual_home_win"].map({
    1: "Home win",
    0: "Away win"
})

result["predicted_result"] = result["pred_home_win"].map({
    1: "Home win",
    0: "Away win"
})

print(result.head(20))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))


# =========================
# 10. Feature importance
# =========================
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values("importance", ascending=False)

print("\nTop 15 important features:")
print(importance.head(15))


# =========================
# 11. Plot feature importance
# =========================
top_n = 15
top_features = importance.head(top_n)

plt.figure(figsize=(10, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()