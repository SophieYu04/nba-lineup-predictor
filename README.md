# NBA Lineup Predictor

Predict NBA matchup win probability from arbitrary player groups, then explain the result by player contribution and lineup-level factors.

## Current Demo-Ready Version

Latest run after the demo fixes:

| Metric | Value |
|---|---:|
| Test AUC | 0.7280 |
| Test Accuracy | 0.6759 |
| Train seasons | 2020-21 to 2023-24 |
| Test season | 2024-25 |
| Output B L1 survivors | 13 / 54 |
| Output C L1 survivors | 31 / 81 |

The main demo is [demo.ipynb](demo.ipynb). It has been executed end to end with the latest artifacts.

## What The Model Does

Given two groups of NBA players, the project predicts `P(home wins)` and explains the prediction through:

- `Score_A`: per-player XGBoost + SHAP contribution.
- `Score_B`: lineup skill coverage and within-team skill interactions.
- `Score_C`: cross-team skill interaction terms.
- `delta_pm`: team-level recent plus-minus strength prior.
- `is_home`: home-court advantage.

The inference API lives in [predict_lineup.py](predict_lineup.py):

```python
from predict_lineup import find_player, predict_matchup

home_ids = [find_player("LeBron James"), find_player("Anthony Davis")]
away_ids = [find_player("Jayson Tatum"), find_player("Jaylen Brown")]

result = predict_matchup(home_ids, away_ids, snapshot_date="2025-03-01")
print(result["P_home_win"])
print(result["per_player_shap_home"])
```

For neutral team-strength comparisons, use `symmetric=True`:

```python
result = predict_matchup(home_ids, away_ids, snapshot_date="2025-03-01", symmetric=True)
```

## Training Pipeline

The full training/evaluation pipeline is [score_ab.py](score_ab.py):

```text
player_rolling.csv
  -> Output A: per-player XGBoost + SHAP -> Score_A
  -> Output B: P80 skill dummies + L1 lineup logit -> Score_B
  -> Output C: cross-team skill interactions -> Score_C
  -> team mean plus_minus_roll10 -> delta_pm
  -> final logistic calibration -> P(home wins)
```

Recent fixes:

- Output B now uses L1 logistic regression instead of keeping all interaction terms with L2.
- OOF folds are time-aware rather than random.
- P80 thresholds are recomputed inside each OOF fold to avoid feature leakage.
- Inference now includes Score C and matches the final model feature order.
- Demo includes `symmetric=True` for neutral strength comparisons.
- Calibration diagnostics are exported for the 0.45-0.50 probability bucket.

## Important Files

| File | Purpose |
|---|---|
| [demo.ipynb](demo.ipynb) | Presentation/demo notebook |
| [score_ab.py](score_ab.py) | Main training pipeline |
| [predict_lineup.py](predict_lineup.py) | Arbitrary matchup inference |
| [outputs/predictions_test.csv](outputs/predictions_test.csv) | Hold-out predictions |
| [outputs/player_shap.csv](outputs/player_shap.csv) | Player-level SHAP outputs |
| [outputs/calibration_buckets.csv](outputs/calibration_buckets.csv) | Calibration by probability bucket |
| [outputs/calibration_bucket_045_050.csv](outputs/calibration_bucket_045_050.csv) | Diagnostic slice for the weak 0.45-0.50 bucket |
| [outputs/models](outputs/models) | Persisted model artifacts |

## Report Files

These are intentionally separate:

- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md): full project summary / final report style writeup.
- [design_b_results.md](design_b_results.md): technical modeling details and ablations.
- [data_validation_report.md](data_validation_report.md): data feasibility and join validation.
- [docs/index.md](docs/index.md): GitHub Pages landing page.

## Setup

```bash
python -m pip install -r requirements.txt
```

Then run:

```bash
python score_ab.py
```

