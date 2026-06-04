# NBA 球員組合勝負預測 — 資料可行性驗證報告

> 對應提案：`Final Report Proposal.pdf` / `Final Report Proposal.md`
> 驗證範圍：`nba_data/` 內 5 份 CSV 是否足以支撐 **Output B（陣容技能互補 Logistic Regression）** 的完整訓練流程
> 結論：**✅ 資料完全足以執行 Logit / Output B**，並已用 toy pipeline 端到端跑通驗證。

---

## 一、資料盤點

### 1.1 檔案與規模

| 檔案 | rows | cols | 用途（對應提案章節） |
|---|---:|---:|---|
| [`nba_data/player_gamelogs.csv`](nba_data/player_gamelogs.csv) | 108,101 | 37 | 重建每場 (game, team) 的上場球員名單；Output A 的原始訓練樣本 |
| [`nba_data/game_results.csv`](nba_data/game_results.csv) | 5,995 | 44 | 勝負標籤（home_win）、主客場 box-score |
| [`nba_data/player_advanced.csv`](nba_data/player_advanced.csv) | 2,825 | 18 | 賽季級 advanced 指標 → Output B 的 dummy 分位數閾值來源 |
| [`nba_data/player_rolling.csv`](nba_data/player_rolling.csv) | 105,488 | 33 | shift-1 滾動平均特徵，已防同場洩漏（提案 5.3 對應） |
| [`nba_data/lineup_synergy.csv`](nba_data/lineup_synergy.csv) | 6,259 | 18 | 5-man lineup 賽季表現，未來可作 Output B 結果驗證 |

### 1.2 賽季覆蓋

所有 5 份 CSV **均覆蓋一致的 5 個賽季**：

```
2020-21  2021-22  2022-23  2023-24  2024-25
```

每季 player_advanced 球員樣本數：

| 賽季 | player-seasons |
|---|---:|
| 2020-21 | 540 |
| 2021-22 | 605 |
| 2022-23 | 539 |
| 2023-24 | 572 |
| 2024-25 | 569 |

---

## 二、Output B（Logit）關鍵需求逐項驗證

提案 §3 Output B 步驟為：
1. 每位球員依分位數產生技能 dummy `D_{k,i}`
2. 隊伍層級 max / sum 聚合
3. 跨球員技能交乘項
4. Logistic Regression + L2 正則

執行任一步驟前，資料須滿足以下條件 —— **全部通過**：

### 2.1 名單可重建性（最關鍵）

> 對每一場 (game, team) 必須能列出哪些球員上場，否則無法聚合到隊伍層級。

| 指標 | 結果 |
|---|---|
| team-games 數 | **12,000** |
| 平均每隊每場上場人數（已套用 MIN ≥ 10 過濾） | **9.01** |
| 最小 / 最大 | 6 / 15 |
| 上場人數 < 5 的場次 | **0** |

→ 每個 team-game 都有足夠球員可組成 5-man+ 陣容。

### 2.2 名單 ↔ 比賽結果可 Join

> 沒對上 game_results 的話，等於缺勝負標籤。

| 指標 | 結果 |
|---|---|
| roster 中 (game_id, team) 鍵 | 12,000 |
| 與 game_results 比對成功 | **11,990 / 12,000（99.92%）** |
| 缺漏 | 10 場（可忽略，或全棒球記錄末尾） |

### 2.3 勝負標籤一致性（Sanity Check）

> 同一 team-game 內每位球員的 `win` 欄位應該完全相同。

| 指標 | 結果 |
|---|---|
| 勝負標籤不一致的 team-game | **0**（預期 0）|

### 2.4 賽季級分位數閾值可用性

> Output B 的 dummy 來自 `D_{k,i} = 1[player i 的指標 k > P80(k)]`，需要賽季級 advanced 指標。

`player_advanced.csv` 已包含完整指標（2023-24 樣本 P80）：

| 指標 | P80 (2023-24) |
|---|---:|
| `usg_pct` | 0.224 |
| `ast_pct` | 0.215 |
| `reb_pct` | 0.121 |
| `ts_pct` | 0.612 |
| `def_rating` | 116.20（低分為佳，正式建模時要倒轉） |
| `pie` | 0.116 |

### 2.5 主客場側別解析

> 聚合到 team-game 後要分配給 home / away 才能做差值特徵。

| 指標 | 結果 |
|---|---|
| player_rolling × game_results join rows | 105,398 |
| 解析為 home | 52,665 |
| 解析為 away | 52,733 |
| 解析失敗（team 對不上） | **0** |

### 2.6 防洩漏設計

> 提案 5.3 強調 Rolling Window 滑動平均，每場僅用該日期之前的歷史。

[`nba_data_collection.py:362-370`](nba_data_collection.py#L362-L370) 已用 `shift(1)` 配合 `min_periods=3` 確保排除當場、且早期樣本不會給出不穩估計：

```python
def rolling_mean(series: pd.Series) -> pd.Series:
    return (
        series
        .shift(1)                                        # exclude current game
        .rolling(window=ROLLING_WINDOW, min_periods=ROLLING_MIN_OBS)
        .mean()
    )
```

`player_rolling.csv` 已是這個流程的產物，所有 `*_roll10` 欄位可直接拿來建模。

---

## 三、端到端 Toy Logit 驗證

為了證明資料真的能跑通 logit（不只是「欄位看起來夠」），我建了一個最小版本的 Output B pipeline。

### 3.1 流程

1. 對 9 個球員指標（`pts_roll10`, `ast_roll10`, `reb_roll10`, `stl_roll10`, `blk_roll10`, `fg3m_roll10`, `true_shooting_roll10`, `usage_proxy_roll10`, `def_impact_roll10`），按 **賽季分組**取 P80 → dummy `d_*`
2. 對每個 (game_id, team) 做 `max` 與 `sum` 兩種聚合 → 隊伍層級特徵
3. 與 game_results join，分 home / away
4. 計算 18 維差值特徵：`diff_d_{stat}_{max|sum} = home − away`
5. 5-fold cross-validation Logistic Regression（L2 正則）

### 3.2 結果

| 指標 | 數值 |
|---|---:|
| 訓練樣本數 | **5,947** |
| 特徵維度 | 18 |
| Home 勝率（baseline） | 0.552 |
| 5-fold CV **AUC** | **0.626** |
| 5-fold CV Accuracy | **0.594** |

> 注意：這個 toy 版本**還沒包含提案的核心「技能交乘項」**，也沒用個人強度（Output A 的 Score A）。光是 dummy 的 max/sum 差值就已穩定贏 home-court baseline 4 個百分點，代表：
> 1. Pipeline 流程完全跑得通
> 2. 資料中確實有 lineup-level 訊號
> 3. 加入交乘項與 Score A 後，效能還有上升空間

---

## 四、需要留意的缺口

雖然 Logit / Output B 已可直接開做，但仍有三點要先處理：

### 4.1 xgboost 無法在本機載入 ⚠️

```
xgboost.core.XGBoostError: Library not loaded: @rpath/libomp.dylib
```

→ 解法：

```bash
brew install libomp
```

僅影響 Output A（XGBoost + SHAP）。Output B 不需要 xgboost。

### 4.2 部分指標只有賽季級，沒有 game-level ⚠️

下列欄位**只存在 `player_advanced.csv`，不在 `player_gamelogs.csv`**，因此 `player_rolling.csv` 也無法包含：

- `pace`
- `pie`
- `off_rating`
- `def_rating`
- `net_rating`

**對 Output B 的影響**：無。dummy 是賽季級閾值，照常可用。
**對 Output A 的影響**：game-level XGBoost 拿不到這幾欄當每場特徵。

→ 若 Output A 需要這些指標，須在 [`nba_data_collection.py`](nba_data_collection.py) 增加 box-score advanced endpoint（如 `boxscoreadvancedv2`）。

### 4.3 負向特徵方向尚未統一 ⚠️

提案 §4.7 規定：

> 失誤、犯規等負向特徵差值方向相反，建議在特徵工程時乘以 -1 統一方向。

目前 `player_rolling.csv` 的 `tov_roll10`, `fouls_roll10`, `def_rating`（若加入）等仍是原始值。特徵工程階段需要：

```python
NEGATIVE_FEATS = ["tov", "fouls", "def_rating"]
for c in NEGATIVE_FEATS:
    df[f"{c}_signed"] = -df[c]
```

或直接讓 XGBoost 自學（提案有提此選項，但 Logit 一定要先處理，否則係數方向會反）。

---

## 五、可立即開工的下一步

按優先順序：

1. **特徵工程模組**：將上述 toy pipeline 包成穩定函式（dummy 生成 → 隊伍聚合 → 主客差值）
2. **加入技能交乘項**：依提案 §3 Step 3，產生跨球員 dummy 兩兩交乘特徵，套 L2 正則對抗維度爆炸
3. **校正欄位語意**：負向特徵乘 -1、`def_rating` 反轉、`is_home` 顯式輸入
4. **建立分賽季 train/test 切分**：避免時序洩漏（提案 §5.3），建議用 2020-21 ~ 2023-24 訓練、2024-25 做 hold-out
5. **`brew install libomp` 解決 xgboost**，為 Output A 鋪路

---

## 附錄 A：驗證使用的環境

```
Python 3.14（.venv）
pandas 3.0.3
numpy  2.4.6
scikit-learn 1.8.0
xgboost 已安裝但 libomp 缺失（待 brew install libomp）
```

## 附錄 B：驗證腳本

驗證邏輯保留在本次 session 的 bash 內聯腳本中（未落地為檔案）。如需重新執行，可重建為 `scripts/validate_data.py`，核心步驟：

1. 載入 5 份 CSV
2. 對 `(game_id, team)` groupby 檢查 roster 規模、勝負一致性
3. 與 `game_results` 做 inner join 檢查覆蓋率
4. 對 `player_advanced` 算分位數確認 dummy 閾值合理
5. 跑 toy logistic regression 取 5-fold AUC / accuracy
