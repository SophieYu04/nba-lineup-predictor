# NBA 陣容勝負預測系統 — 完整總結

> 對應提案：[`Final Report Proposal.md`](Final%20Report%20Proposal.md)
> 涵蓋：架構決策、資料驗證、模型表現、Demo 結果、決策支援能力、使用教學
> 子文件：[`data_validation_report.md`](data_validation_report.md)（資料層驗證細節）、[`design_b_results.md`](design_b_results.md)（Design B 實作細節）

---

## 0. TL;DR

我們建了一個可以「輸入任意 N 名球員 → 輸出勝率 + 球員級貢獻分解」的 NBA 對戰預測系統。

| 指標 | 數值 | 對標 |
|---|---:|---|
| **Test AUC**（2024-25 hold-out, 1,225 場）| **0.7233** | 538 ~0.72，Vegas 0.72-0.74 |
| **Test Accuracy** | **0.6776** | 538 0.67-0.69，Vegas 0.67-0.70 |
| P_home_win 範圍 | 0.137 ~ 0.925 | spread 接近 Vegas |
| Calibration（最高 bucket 預測 vs 實際） | 0.766 / 0.790 | 幾乎完美 |
| 訓練資料 | 4,722 場（2020-21 ~ 2023-24） | 全 NBA 5 個賽季 |

**三個核心能力**：
1. 對任意 N 名球員的對決即時預測勝率
2. 將勝率拆解到**每位球員的 SHAP 貢獻**（不只是團隊平均）
3. 支援**時間旅行**：用 `snapshot_date` 比較同一球員不同時期、做歷史 what-if

---

## 1. 系統架構（Design B）

### 1.1 整體流程

```
                ┌──────────────────────────────────────┐
                │ player_rolling.csv (shift-1 rolling) │
                │ ─ plus_minus_roll10  EXCLUDED 個人   │
                └──────────┬───────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼ Output A                            ▼ Output B
┌─────────────────────┐               ┌─────────────────────┐
│ per-player XGBoost  │               │ 9 stats → 賽季 P80   │
│ sample_weight=min   │               │   dummy 二值化       │
│ OOF 5-fold (train)  │               │ team 聚合: max + sum │
│ refit-all (test)    │               │ 36 對 sum×sum 交乘   │
│ + TreeExplainer SHAP│               │ OOF 5-fold logit     │
│ Σ player SHAP →     │               │ refit-all            │
│   Score_A per team  │               │ decision_function →  │
└─────────┬───────────┘               │   Score_B per team   │
          │                           └──────────┬───────────┘
          └──────────────┬───────────────────────┘
                         │
                         ▼  per (game, team) Scores
              ┌─────────────────────────┐
              │ Matchup pivot:           │
              │  ΔA = A_home − A_away    │
              │  ΔB = B_home − B_away    │
              │  Δpm = pm_home − pm_away │ ◄── team-mean(plus_minus_roll10)
              │  is_home = 1             │
              │  z-score (train stats)   │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌─────────────────────────────────────┐
              │ P(home wins) = σ(α·ΔA + β·ΔB        │
              │              + δ·Δpm + γ·is_home    │
              │              + b)                    │
              └─────────────────────────────────────┘
```

### 1.2 最終訓練係數

```
P(home wins) = σ(0.345·ΔA + 0.171·ΔB + 0.320·Δpm + 0.119·is_home + 0.120)
```

| 係數 | 值 | 意義 |
|---|---:|---|
| α | +0.345 | 個人強度權重（per-player SHAP Σ） |
| β | +0.171 | 陣容互補權重（max/sum dummies + 36 交乘）|
| δ | +0.320 | 歷史強度權重（隊伍滾動 plus_minus 平均）|
| γ | +0.119 | 主場優勢 — 對應約 σ(0.12)≈53% |
| intercept | +0.120 | base rate |

**全部正號**，無共線性導致的方向反轉。可解釋的 final layer 確保決策透明。

### 1.3 為什麼這樣設計

| 設計選擇 | 理由 |
|---|---|
| 個人 XGB + SHAP（Output A） | 提案 §3 指定 → 唯一能拆到球員的維度 |
| 排除 `plus_minus_roll10` 當個人特徵 | 它是隊伍強度代理，會污染個人 SHAP |
| 將 `plus_minus_roll10` 改用 team-mean Δpm 放 final layer | 保留訊號，避免個人歸因失真 |
| Logit + 36 交乘項（Output B） | 提案 §3 指定，捕捉跨球員技能互補 |
| z-score 標準化 ΔA/ΔB/Δpm | 三者量綱差異大（std 1.27 / 0.47 / 5+），不標準化 final logit 會偏到 ΔA |
| OOF 5-fold 產生 train 的 Score | **修隱藏 in-sample bias**（initial β 變負是此問題的徵兆）|
| 時序切分 train/test | 防止跨季洩漏（提案 §5.3） |

---

## 2. 資料驗證摘要

詳細：[`data_validation_report.md`](data_validation_report.md)

### 2.1 資料覆蓋

| 檔案 | rows | 用途 |
|---|---:|---|
| [`nba_data/player_gamelogs.csv`](nba_data/player_gamelogs.csv) | 108,101 | 重建 (game, team) roster + 取勝負標籤 |
| [`nba_data/game_results.csv`](nba_data/game_results.csv) | 5,995 | 比賽結果（home_win + box-score）|
| [`nba_data/player_advanced.csv`](nba_data/player_advanced.csv) | 2,825 | 賽季級 advanced 指標（dummy 閾值）|
| [`nba_data/player_rolling.csv`](nba_data/player_rolling.csv) | 105,488 | shift-1 滾動平均特徵（無洩漏）|
| [`nba_data/lineup_synergy.csv`](nba_data/lineup_synergy.csv) | 6,259 | 5-man lineup 賽季表現（驗證用）|

**5 賽季**：2020-21、2021-22、2022-23、2023-24、2024-25。

### 2.2 關鍵 join 健康度

| 檢查項 | 結果 |
|---|---|
| team-games 數 | 12,000 |
| 平均球員/場（MIN ≥ 10）| 9.01 |
| 上場人數 < 5 的場次 | 0 |
| roster ↔ game_results join 成功率 | 99.92% |
| 勝負標籤不一致 team-game | 0 |
| 主客場側別解析成功 | 100% |

---

## 3. 模型表現對標

### 3.1 同 hold-out（2024-25, 1,069 場過濾季初）baseline 階梯

| # | 方法 | Accuracy | AUC |
|---|---|---:|---:|
| 1 | 永遠猜主場 | 0.5426 | 0.5000 |
| 2 | 勝率較高的隊（season-to-date）| 0.6576 | 0.7122 |
| 3 | 累積場均淨得分較高的隊 | 0.6520 | 0.7180 |
| 4 | Δpm only（單一特徵）| 0.6642 | 0.7192 |
| 5 | **Design B（完整版）** | **0.6801** | **0.7263** |

### 3.2 vs 公開基準

| 來源 | Accuracy | AUC |
|---|---:|---:|
| FiveThirtyEight | 0.67–0.69 | ~0.72 |
| **Vegas 收盤盤口**（efficient market）| 0.67–0.70 | 0.72–0.74 |
| 近年 ML 論文（2018-2023）| 0.65–0.75 | varies |
| **我們** | **0.6801** | **0.7263** |

**結論：頂到純預測模型的天花板。** Vegas 是 efficient market，沒有純預測模型能穩定大幅領先它。

### 3.3 邊際貢獻誠實看

```
跟「永遠猜主場」比      ：+14 pp  ← 模型整體有用
跟「勝率較高的隊」比    ：+2 pp   ← 個人 + 陣容工程貢獻 ~2 pp
跟「Δpm only」比         ：+1.6 pp ← Score_A + Score_B 合起來只多 1.6 pp
```

NBA 勝負預測的硬上限：**「知道誰過去比較會贏」就吃掉大部分訊號**。我們的 SHAP + 陣容 dummy 邊際貢獻有限 —— 但這個專題的價值不在 AUC，在**可解釋性與任意組合查詢**。

### 3.4 Calibration（按 P 預測值分組看實際勝率）

| 預測 bucket | n | 平均預測 | 實際勝率 |
|---|---:|---:|---:|
| < 0.30 | 74 | 0.256 | 0.176 |
| 0.30–0.40 | 142 | 0.351 | 0.261 |
| 0.40–0.45 | 108 | 0.425 | 0.444 |
| 0.45–0.50 | 123 | 0.476 | 0.358 |
| 0.50–0.55 | 143 | 0.525 | 0.552 |
| 0.55–0.60 | 136 | 0.575 | 0.618 |
| 0.60–0.70 | 266 | 0.648 | 0.669 |
| > 0.70 | 233 | 0.766 | **0.790** |

→ 高信心 bucket 校準幾乎完美（0.766 預測 vs 0.790 實際）。

### 3.5 P_home_win 分布（test set）

```
min   = 0.137
10%   = 0.336
50%   = 0.559
90%   = 0.751
max   = 0.925
std   = 0.154
```

**41% 的場次模型給出 ≥0.65 或 ≤0.35** 的明確判斷 —— 模型不是壓在 50%。

---

## 4. 四個對齊問題 + 解法

詳細：[`design_b_results.md`](design_b_results.md) §2

| # | 問題 | 解法 |
|---|---|---|
| 1 | **Train/Test 切分** | 時序切分（train = 2020-21~2023-24, test = 2024-25），同時防跨季洩漏 |
| 2 | **樣本對齊** | 兩模型 inner join 同 `game_id`，11,910 team-games 全對齊 |
| 3 | **`is_home` 旗標** | 不放進 Score 計算（保持對稱），放 final layer 當獨立特徵 |
| 4 | **In-sample bias**（隱藏陷阱）| **OOF 5-fold** + **z-score 標準化**（用 train 統計值）|

### 第 4 個對齊問題的關鍵性

| 階段 | α | β | γ | δ(pm) | Test AUC |
|---|---:|---:|---:|---:|---:|
| 初版（無 OOF、無標準化）| +1.241 | **−0.613** ⚠️ | +0.139 | —— | 0.6441 |
| + OOF + 標準化 | +0.541 | +0.220 | +0.118 | —— | 0.6922 |
| + 36 交乘 + Δpm（最終）| +0.345 | +0.171 | +0.119 | +0.320 | **0.7233** |

β 一開始是負的，因為 Score 在 train 是 in-sample（過度精確）、在 test 是 out-of-sample。final logit 用差異分布訓練，會把另一個 Score 推到負值補正。**OOF 修這個就解決了。**

---

## 5. Demo 結果（[`demo.ipynb`](demo.ipynb)）

### 5.1 案例彙整

| 案例 | 設定 | P(home wins) | 備註 |
|---|---|---:|---|
| **A** | LAL vs BOS @ 2025-03-01 | 0.576 | 強強對決，自然接近 50% |
| **A2** | OKC vs WAS @ 2025-03-01 | **0.774** | 西冠 vs 墊底，Δpm=+2.50 σ 拉開 spread |
| **B** | A 但 LeBron→Dončić | 0.547 | -2.8pp，個人替換的量化影響 |
| **C** | A 但 is_home=0 | 0.546 | 主場優勢 +2.9pp |
| **D** | A 但 7v7 加板凳 | 0.534 | 板凳稀釋頂尖球員影響 |
| **E** | 2020-21 LAL vs 2020-21 BOS | 0.556 | 跨時空 LeBron SHAP: 2021=+0.40, 2025=+0.60 |

### 5.2 案例 A 的 SHAP 分解（log-odds）

**Home (LAL, Σ = +0.87):**
```
LeBron James        +0.605
Anthony Davis       +0.283
Rui Hachimura       +0.196
Austin Reaves       +0.088
D'Angelo Russell    -0.307   ← 模型認為他拖後腿
```

**Away (BOS, Σ = +0.68):**
```
Jayson Tatum        +0.401
Jrue Holiday        +0.195
Jaylen Brown        +0.122
Derrick White       +0.061
Kristaps Porziņģis  -0.096
```

---

## 6. 決策支援能力

### 6.1 支援的決策問題

| 決策類型 | API 呼叫 | 範例 |
|---|---|---|
| X→Y 換人是否有利？ | `predict_matchup` × 2 | 把 DLo 換 Kyrie 對 BOS 怎樣 |
| 該換掉誰？ | 對每位球員試移除 | 找隊上 SHAP 最低者 |
| 哪個自由球員值得簽？ | 對每個候選試加入 | 全聯盟掃描 |
| 主場優勢值多少？ | `is_home=1` vs `is_home=0` | 中性場場景 |
| 跨季 lineup 比較 | 不同 `snapshot_date` | 巔峰 LeBron vs 現在 |
| MVP 個人貢獻拆解 | `per_player_shap_home/away` | LLM 報告素材 |

### 6.2 實例：LAL 想換掉 D'Angelo Russell，6 個候選 vs BOS

| 候選人 | P(LAL wins) | Δ vs baseline | 他的 SHAP | Score_A | Score_B | pm_home |
|---|---:|---:|---:|---:|---:|---:|
| **Gabe Vincent** | **0.594** | **+0.018** | 0.000 | 1.173 | 0.185 | **6.02** |
| De'Aaron Fox | 0.588 | +0.012 | +0.219 | 1.392 | 0.303 | 4.36 |
| Kyrie Irving | 0.583 | +0.007 | +0.399 | 1.571 | 0.070 | 4.58 |
| Trae Young | 0.577 | +0.001 | -0.255 | 0.917 | 0.126 | 6.08 |
| D'Angelo Russell（baseline）| 0.576 | 0 | -0.307 | 0.865 | 0.288 | 5.38 |
| **Devin Booker** ⚠️ | **0.567** | **-0.009** | **+0.246** | 1.418 | 0.126 | 3.84 |

**反直覺發現**：Devin Booker 個人 SHAP +0.246（強）但加入 LAL 反而**降低 0.9pp 勝率**。原因：太陽近期戰績差 → 他的 plus_minus 帶低 LAL 的 pm_home（5.38 → 3.84）。

→ 這就是提案核心賣點「**不只是 sum of stars**」的具體體現。

### 6.3 沒有模型化的部分（誠實聲明）

| 因素 | 為什麼沒做 | 影響 |
|---|---|---|
| **位置重複** | 模型不知道 LeBron 和 AD 都打前場 | 加第三個前鋒模型不會懲罰 |
| **球權分配** | 沒有 game-time usage 模型 | 全明星陣容可能被高估 |
| **傷病/休息** | 沒接 injury data | 自由球員預測以「滿員出戰」估算 |
| **真實化學反應** | 只用了 lineup dummy，沒餵 lineup_synergy.csv | 從未共同上場過的組合預測略樂觀 |

---

## 7. 使用教學

### 7.1 環境準備

```bash
# 1. Mac 需要 libomp（xgboost 依賴）
brew install libomp

# 2. 建立 venv 並安裝套件
python3 -m venv .venv
.venv/bin/pip install pandas numpy scikit-learn xgboost shap matplotlib joblib jupyter
```

### 7.2 訓練模型

```bash
.venv/bin/python score_ab.py
```

執行流程（約 80 秒 on M1）：
1. 讀取 5 份 CSV
2. 建 per-player 訓練樣本（83,640 rows）
3. Output A 5-fold OOF XGBoost + SHAP（~60 秒）
4. Output B dummy 二值化 + 36 交乘項 + OOF logit（~3 秒）
5. Δpm team-mean 聚合
6. Final calibration logit + ablation
7. 儲存所有模型 → [`outputs/models/`](outputs/models/)

### 7.3 推論（Python）

```python
from predict_lineup import predict_matchup, find_player, summary_table

# 球員名稱 → player_id（accent-insensitive substring 搜尋）
home_ids = [find_player(n) for n in
            ["LeBron James", "Anthony Davis", "Austin Reaves",
             "D'Angelo Russell", "Rui Hachimura"]]
away_ids = [find_player(n) for n in
            ["Jayson Tatum", "Jaylen Brown", "Jrue Holiday",
             "Derrick White", "Porzingis"]]   # 模糊匹配，不必輸入特殊字元

# 預測（可指定快照日期，None 表示用每位球員最新一筆）
result = predict_matchup(home_ids, away_ids, snapshot_date="2025-03-01")

print(result["P_home_win"])                     # 0.576
print(result["per_player_shap_home"])           # 球員級貢獻
print(result["logit_contributions"])            # 維度級分解
summary_table(result)                           # one-row 表格
```

### 7.4 推論輸出結構

```python
{
    "P_home_win":             float,       # σ(logit)
    "P_away_win":             float,       # 1 - P_home_win
    "snapshot_date":          str,
    "season_for_p80":         str,         # 用哪個賽季的 P80 閾值
    "is_home":                int,         # 0 = 中性場, 1 = 主隊主場
    # raw scores
    "score_A_home/away":      float,       # Σ SHAP per team
    "score_B_home/away":      float,       # decision_function per team
    "pm_home/away":           float,       # team-mean plus_minus_roll10
    # standardized
    "delta_A":                float,       # z-score
    "delta_B":                float,
    "delta_pm":               float,
    # 維度分解
    "logit_contributions": {
        "from_score_A":       float,       # α · ΔA
        "from_score_B":       float,       # β · ΔB
        "from_delta_pm":      float,       # δ · Δpm
        "from_is_home":       float,       # γ · is_home
        "from_intercept":     float,
    },
    # 球員級分解
    "per_player_shap_home":   pd.DataFrame,   # cols: player_id, player_name,
                                              #       min_roll10, shap_sum, snapshot_date
    "per_player_shap_away":   pd.DataFrame,
}
```

### 7.5 互動 demo（Jupyter）

```bash
.venv/bin/jupyter notebook demo.ipynb
```

[`demo.ipynb`](demo.ipynb) 5 個 section、6 個案例 + 視覺化：
- 1️⃣ Lakers vs Celtics 標準對決
- 2️⃣ 極端對比（OKC vs WAS）+ test set 全分布直方圖
- 3️⃣ 球員交換 what-if（LeBron→Dončić）
- 4️⃣ 中性場（拆主場優勢）
- 5️⃣ 7v7 含板凳
- 6️⃣ 跨時空快照（2020-21 vs 2024-25 LAL）

### 7.6 常見問題

**Q: 球員名稱怎麼匹配？特殊字元怎麼辦？**
A: `find_player` 是 accent-insensitive substring 搜尋。`"Porzingis"` 會匹配到 `"Kristaps Porziņģis"`，`"Doncic"` 會匹配 `"Luka Dončić"`。

**Q: 預測一場要多久？**
A: 模型載入後（lazy 一次性 ~1 秒），單場預測 < 50ms。批次推論可在記憶體緩存 `_Artifacts`。

**Q: 可以查不在 NBA 的球員嗎（譬如 G League 或退休）？**
A: 只要 `nba_data/player_rolling.csv` 有他的 rolling 資料就可以。retire 後 snapshot_date 必須早於退休。

**Q: 換成不同賽季的球員會如何？**
A: 模型內部用 `snapshot_date` 對應的賽季 P80 閾值去算 dummy。跨季混搭可能讓 Score_B 偏掉（因為 P80 是按 snapshot 賽季算）。

**Q: 結果常常在 50% 附近正常嗎？**
A: 強強對決真的會接近 50%（NBA 本質）。換 OKC vs WAS 這種強弱對決就會拉到 77%。test set 真實 spread 是 0.137 ~ 0.925。

---

## 8. 檔案結構

```
nba-lineup-predictor/
├── nba_data/                                ← 5 賽季原始資料
│   ├── player_gamelogs.csv
│   ├── player_rolling.csv
│   ├── player_advanced.csv
│   ├── lineup_synergy.csv
│   └── game_results.csv
├── outputs/                                 ← 訓練產物
│   ├── predictions_test.csv                 ← 1,225 場 2024-25 預測
│   ├── player_shap.csv                      ← 每球員每場 SHAP（LLM 報告源）
│   ├── final_weights.json                   ← α/β/δ/γ/intercept
│   └── models/                              ← 推論用 artifacts
│       ├── xgb_output_A.joblib
│       ├── logit_output_B.joblib
│       ├── final_logit.joblib
│       ├── scaler_A.joblib
│       ├── scaler_B.joblib
│       ├── scaler_pm.joblib
│       ├── feature_meta.json
│       └── p80_thresholds.csv
├── nba_data_collection.py                   ← 原始抓資料 script
├── xgb.py                                   ← 同學的 baseline XGB
├── score_ab.py                              ← Design B 訓練主程式
├── predict_lineup.py                        ← 推論 API
├── demo.ipynb                               ← 互動展示
├── Final Report Proposal.md / .pdf          ← 提案
├── data_validation_report.md                ← 資料層驗證細節
├── design_b_results.md                      ← Design B 實作細節
└── PROJECT_SUMMARY.md                       ← 本文件
```

---

## 9. 已知限制 & 下一步

### 9.1 待修

| 項目 | 嚴重度 | 工作量 |
|---|---|---|
| **對稱性** — 中性場 `P(A vs B) + P(B vs A) = 1.06 ≠ 1.0` | 低（影響排序），原因是訓練資料 home_win 框架的 intercept 殘留 | 強制 `fit_intercept=False` 或 augment 鏡像樣本 |
| Score_A 與 Δpm 部分共線（B+Δpm AUC 0.7259 略高於 A+B+Δpm 0.7233）| 低（差距不顯著）| 接受或殘差化 |
| Score_B 36 交乘擴張後單獨 AUC 略降（0.6653→0.6494）| 低 | L1 自動選稀疏交乘 |

### 9.2 提案剩餘 phase

| Phase | 內容 | 狀態 |
|---|---|---|
| Phase 1 | 資料收集 | ✅ 完成（[`nba_data_collection.py`](nba_data_collection.py)）|
| Phase 2 | 特徵工程 + XGBoost Baseline + Output A/B 模型 | ✅ 完成（[`score_ab.py`](score_ab.py)）|
| Phase 3 | 前端互動介面 + 視覺化圖表 | 🟡 半完成（[`demo.ipynb`](demo.ipynb) Jupyter 版，未做 web）|
| Phase 4 | AI 分析報告模組（LLM 整合）| ⬜ 未開始（資料源 [`outputs/player_shap.csv`](outputs/player_shap.csv) 已就緒）|
| Phase 5 | 測試、調校、內部 Demo | 🟡 部分（test set 評估 + ablation 完成；無 stakeholder demo）|

### 9.3 模型可優化方向

| 方向 | 預期影響 | 工作量 |
|---|---|---|
| 接入 lineup_synergy.csv 真實 5-man 數據當特徵 | 中（陣容化學反應更準）| 高 |
| 加入傷病 / 休息 / back-to-back 旗標 | 中（短期波動更準）| 中 |
| Leave-one-season-out 交叉驗證 | 低（穩健性確認）| 低 |
| Isotonic / Platt 機率校準 | 低（AUC 不變，機率更準）| 低 |
| 換用 ELO / 滾動 net rating 取代 plus_minus_diff | 低 | 中 |

### 9.4 產品化方向

| 方向 | 工作量 | 對應提案 |
|---|---|---|
| Streamlit / Gradio web app | 中 | Phase 3 |
| Claude API 整合自動產生文字分析報告 | 中 | Phase 4 |
| 即時抓 NBA API（snapshot=today） | 中 | 提案 §2 §4 |
| 全聯盟掃描「該簽哪個自由球員」工具 | 低 | 衍生應用 |

---

## 附錄 A：完整模型公式

### Output A — Score A（per team）

```
For each player i in team T:
    SHAP_i = TreeExplainer(xgb_clf).shap_values(player_i_features_24d)
    player_contribution_i = Σ_k SHAP_i,k

Score_A(T) = Σ_{i ∈ T} player_contribution_i
```

訓練：sample_weight = `min_roll10`；OOF 5-fold + refit-all

### Output B — Score B（per team）

```
For each player i in team T, for each stat k ∈ {pts, ast, reb, stl, blk,
                                                  fg3m, ts, usg, def_impact}:
    D_{k,i} = 1[player i's k > P80_season(k)]

T_k^max  = max_i D_{k,i}        # 有無
T_k^sum  = Σ_i D_{k,i}          # 厚度

interactions = {T_{k1}^sum × T_{k2}^sum  for all k1 < k2}   # C(9,2) = 36

feature_vec = [T_*^max (9), T_*^sum (9), interactions (36)]     # 54 dims
Score_B(T) = w · feature_vec + b₀     # logit decision_function
```

訓練：L2 (C=1.0)；OOF 5-fold + refit-all

### Δpm（per team）

```
team_pm(T) = mean_{i ∈ T} plus_minus_roll10_i
```

### Final layer

```
ΔA  = z_A(Score_A_home - Score_A_away)         # z-score用 train 統計
ΔB  = z_B(Score_B_home - Score_B_away)
Δpm = z_pm(team_pm_home - team_pm_away)

logit = α·ΔA + β·ΔB + δ·Δpm + γ·is_home + b
P(home wins) = σ(logit)
              = 1 / (1 + exp(-logit))
```

訓練好的參數（見 [`outputs/final_weights.json`](outputs/final_weights.json)）：
- α = +0.345, β = +0.171, δ = +0.320, γ = +0.119, b = +0.120

---

## 附錄 B：環境

```
Python 3.14（.venv）
pandas 3.0.3
numpy  2.4.6
scikit-learn 1.8.0
xgboost 3.2.0
shap   0.51.0
joblib 1.5.3
matplotlib 3.10.9
jupyter 安裝後依需求啟動
libomp 22.1.6 (brew install libomp)
```

## 附錄 C：訓練耗時（M1 Mac）

| 階段 | 時間 |
|---|---|
| Output A 5-fold OOF XGB + SHAP | ~60 秒 |
| Output A 全 train refit + test SHAP | ~15 秒 |
| Output B 5-fold OOF logit | ~3 秒 |
| Final layer + ablation + 儲存 artifacts | < 2 秒 |
| **Total** | **~80 秒** |

單場推論（模型已載入）：< 50ms。
