# NBA Lineup Predictor

Predict NBA matchup win probability from player-level rolling stats, lineup skill coverage, cross-team interactions, and team-strength priors.

## What this project does

- Builds per-player XGBoost scores with SHAP explanations.
- Builds lineup-level logistic scores from elite skill dummies and interaction terms.
- Adds cross-team skill interaction features and team plus-minus priors.
- Calibrates everything into `P(home wins)` on a 2024-25 hold-out season.

## Current result

| Metric | Value |
|---|---:|
| Test AUC | 0.7280 |
| Test Accuracy | 0.6759 |
| Train seasons | 2020-21 to 2023-24 |
| Test season | 2024-25 |

## Architecture

```text
player_rolling.csv
  -> Output A: per-player XGBoost + SHAP -> Score_A
  -> Output B: P80 skill dummies + lineup interactions -> Score_B
  -> Output C: cross-team skill interactions -> Score_C
  -> team mean plus_minus_roll10 -> delta_pm
  -> final logistic calibration -> P(home wins)
```

## Reports

- [Project summary](../PROJECT_SUMMARY.md)
- [Design B/C results](../design_b_results.md)
- [Data validation report](../data_validation_report.md)

## Important files

- `score_ab.py`: training pipeline and artifact export.
- `predict_lineup.py`: arbitrary matchup inference API.
- `outputs/predictions_test.csv`: hold-out predictions.
- `outputs/player_shap.csv`: player-level SHAP contributions.
- `outputs/cross_team_pairs.csv`: non-zero cross-team interaction coefficients.
- `outputs/models/`: persisted model artifacts.

## GitHub Pages structure

- `assets/css/`: page styling.
- `assets/js/`: optional interactivity.
- `assets/img/`: charts, screenshots, and diagrams.
- `data/`: small public-facing data extracts.
- `reports/`: rendered report pages or copied markdown summaries.
