# 設計 B 實作結果 — Score A (XGBoost+SHAP) + Score B (Logit) + Δpm

> 對應提案：`Final Report Proposal.pdf` §3「核心模型：XGBoost + Regression」
> 實作檔：[`score_ab.py`](score_ab.py)
> 評估方式：時序切分 **train = 2020-21 ~ 2023-24（4,722 場）/ test = 2024-25（1,225 場）**

---

## 一、最終結果

```
P(home wins) = σ(0.345·ΔA + 0.171·ΔB + 0.320·Δpm + 0.119·is_home + 0.120)
```

| 指標 | 數值 |
|---|---:|
| **Test AUC** | **0.7233** |
| **Test Accuracy** | **0.6776** |
| Baseline（永遠猜 home win） | 0.5445 |
| Train 樣本數 | 4,722 matchups |
| Test 樣本數 | 1,225 matchups |
| Output A 訓練樣本（per-player） | 83,640 rows |
| Output B 特徵維度 | 54（18 max+sum × 9 stats + 36 交乘對） |

---

## 二、四個對齊問題的解法

提案合併兩個模型時的隱藏陷阱，逐項解：

| # | 對齊問題 | 處理方法 | 程式位置 |
|---|---|---|---|
| 1 | **Train/Test 切分** | 時序切分：train = 4 個賽季、test = 2024-25。同時解決提案 §5.3 時序洩漏。 | [score_ab.py:39-40](score_ab.py#L39-L40) |
| 2 | **樣本對齊** | 兩模型 inner join 同一批 `game_id`。11,910 team-games 全對齊。 | [score_ab.py:178-181](score_ab.py#L178-L181) |
| 3 | **`is_home` 旗標** | 不放進 Score 計算（保持對稱），改放到 final layer 當獨立特徵。`γ` 就是主場優勢。 | [score_ab.py:262-268, 284-285](score_ab.py#L284) |
| 4 | **In-sample bias**（新發現 ⚠️） | **OOF 5-fold** 產生 train 的 Score → final layer 看到的訓練 Score 跟 test set 同等級。同時用 train statistics 做 **z-score 標準化**。 | [score_ab.py:117-145](score_ab.py#L117-L145), [score_ab.py:222-241](score_ab.py#L222-L241), [score_ab.py:280-289](score_ab.py#L280-L289) |

### 為什麼問題 #4 是隱藏陷阱

第一版實作 train AUC 看起來合理，但 test AUC 比單獨模型還差，且 β 係數為 **負**：

| 階段 | α | β | γ | δ(pm) | Test AUC |
|---|---:|---:|---:|---:|---:|
| 初版（無 OOF、無標準化） | +1.241 | **−0.613** ⚠️ | +0.139 | —— | 0.6441 |
| + OOF + 標準化 | +0.541 | +0.220 | +0.118 | —— | 0.6922 |
| + 36 交乘 + Δpm（最終） | +0.345 | +0.171 | +0.119 | +0.320 | **0.7233** |

原因：Score 在 train 是 in-sample fit（精度過高），在 test 是 out-of-sample（有泛化誤差）。final logit 對著兩個分布完全不同的訊號學係數，會把第二個 Score 推到負值來「修正」第一個 Score 在 train 上的 overshoot。OOF 預測讓兩邊分布一致就解了。

---

## 三、Pipeline 結構

```
                ┌──────────────────────────────────────┐
                │ player_rolling.csv (24 features)     │
                │ ─ plus_minus_roll10  EXCLUDED        │
                └──────────┬───────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼ Output A                            ▼ Output B
┌─────────────────────┐               ┌─────────────────────┐
│ per-player XGBoost  │               │ 9 stats → 賽季 P80   │
│ sample_weight =     │               │   dummy 二值化       │
│   min_roll10        │               │ team 聚合: max + sum │
│ OOF 5-fold for train│               │ 36 交乘對 (sum×sum)  │
│ refit-all for test  │               │ OOF 5-fold logit     │
│ + SHAP (margin)     │               │ refit-all for test   │
│ Σ player SHAP →     │               │ decision_function →  │
│   Score_A team-game │               │   Score_B team-game  │
└─────────┬───────────┘               └──────────┬───────────┘
          │                                      │
          └──────────────┬───────────────────────┘
                         │
                         ▼  per (game, team) Scores
              ┌─────────────────────────┐
              │ Matchup pivot:           │
              │  ΔA = A_home − A_away    │
              │  ΔB = B_home − B_away    │
              │  Δpm = pm_home − pm_away │ ◄── team-mean(plus_minus_roll10)
              │  is_home = 1             │
              │  z-score with train stats │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌─────────────────────────────────────┐
              │ Final calibration logit              │
              │ P = σ(α·ΔA + β·ΔB + δ·Δpm           │
              │       + γ·is_home + b)               │
              └─────────────────────────────────────┘
```

---

## 四、Output A：per-player XGBoost → SHAP

### 4.1 訓練設定

| 項目 | 設定 |
|---|---|
| 樣本單位 | (game_id, team, player_id) — 每筆是「某球員某場某隊」 |
| 標籤 | team_win（從 gamelogs broadcast，同隊球員共享） |
| 特徵 | 24 個 rolling stats（**排除 plus_minus**） |
| sample_weight | `min_roll10`（替補貢獻自動縮小） |
| 模型 | XGBClassifier(n=400, depth=4, lr=0.05) |

### 4.2 為什麼排除 `plus_minus_roll10`

你同學原版 xgb.py 的 importance:
```
plus_minus_diff       0.207  ← 第一名，比第二名高 3 倍
def_impact_diff       0.066
pts_diff              0.064
true_shooting_diff    0.064
ast_diff              0.063
```

`plus_minus`（球員在場時隊伍淨得分差）本質是**隊伍強度的代理**而非個人技能。在 team-mean 版本還算合理（當作 team-strength prior），但在**per-player SHAP 分解**情境下會產生循環：贏球隊的所有球員 plus_minus 都正 → SHAP 都正 → Σ 大 → 預測贏。這只是學到「贏的隊會贏」，沒有個人歸因的語意。

→ **解法**：從 per-player feature 排除（避免污染 SHAP），改在 final layer 當 team-mean Δpm 用（保留歷史強度先驗）。

### 4.3 Score_A 計算

對每位球員的 SHAP value 是 margin-space（log-odds 加性）：

```
Score_A_{team, game} = Σ_{player ∈ team} SHAP(player_features)
```

team-game 總和直接加進 final layer，**保留 σ 的對稱性**。

---

## 五、Output B：陣容技能互補 Logit

### 5.1 Dummy 生成

對 9 個 rolling 指標，按賽季取 P80 閾值：

```python
DUMMY_STATS = ["pts_roll10", "ast_roll10", "reb_roll10",
               "stl_roll10", "blk_roll10", "fg3m_roll10",
               "true_shooting_roll10", "usage_proxy_roll10", "def_impact_roll10"]

D_{k,i} = 1[球員 i 的 k > P80_season(k)]
```

### 5.2 隊伍層級聚合（提案 §3 Step 2）

每位球員的 dummy 聚合到 (game_id, team)：

- **max**：「隊伍是否有至少一名頂級 k 技能球員」（有無）
- **sum**：「隊伍擁有頂級 k 技能的人數」（厚度）

→ 9 × 2 = **18 個基礎特徵**

### 5.3 跨球員技能交乘（提案 §3 Step 3，核心設計）

C(9, 2) = **36 對 sum × sum 交乘項**，刻畫「兩種技能跨球員的協同效果」：

```python
x_{d_pts_d_ast}   = T_pts^depth × T_ast^depth    # 得分手 × 組織者
x_{d_fg3m_d_blk}  = T_3pt^depth × T_blk^depth    # 射手 × 護框
... (36 pairs)
```

→ Output B 總特徵維度 = 18 + 36 = **54**

### 5.4 訓練

- LogisticRegression(C=1.0, max_iter=2000) + L2
- 5-fold OOF 產生 train 的 Score_B
- 用全部 train 重 fit 後對 test 預測
- 輸出 `decision_function`（logit value，與 Score_A margin-space 同尺度）

---

## 六、Δpm：團隊歷史強度（後加）

| 項目 | 設定 |
|---|---|
| 計算 | team_pm = mean(plus_minus_roll10) per (game_id, team) |
| 對齊到 matchup | Δpm = pm_home − pm_away |
| 標準化 | z-score with train statistics |
| 用途 | final layer 第三個 Score-level 特徵 |

**為什麼放在最後而不是中間**：
- 不能放進 per-player XGBoost（會污染個人 SHAP，見 §4.2）
- 不能放進 Output B 的 dummy 集（它是連續值不是技能 dummy）
- → 唯一合理位置：final calibration layer

---

## 七、Ablation 分析

| 組合 | Test AUC | Δ vs baseline | 解讀 |
|---|---:|---:|---|
| Baseline（永遠 home win） | 0.5445 | — | 無模型 |
| Score_A only + is_home | 0.6661 | +0.122 | 個人強度 |
| Score_B only + is_home | 0.6494 | +0.105 | 陣容互補 |
| **Δpm only + is_home** | **0.7164** | **+0.172** | 最強單一訊號 |
| A + B + is_home | 0.6864 | +0.142 | 提案原版（無 Δpm） |
| A + Δpm + is_home | 0.7175 | +0.173 | |
| B + Δpm + is_home | 0.7259 | +0.181 | |
| **A + B + Δpm + is_home（完整）** | **0.7233** | **+0.179** | **採用** |

### 重要發現

**Δpm 部分吸收了 Score_A 的預測力**：
- B + Δpm（無 A）= 0.7259
- A + B + Δpm（有 A）= 0.7233
- 差距 0.003 在 1,225 樣本下不顯著

→ Score_A 與 Δpm 高度共線（兩者都在量「球員整體強度」）

### 為什麼仍保留 Score_A

雖然 AUC-wise 邊際貢獻不顯著，**Score_A 是唯一可拆解到個別球員的維度**：

| 維度 | 可解釋到球員？ | 用途 |
|---|---|---|
| Score_A (SHAP) | **✅ 可**（[outputs/player_shap.csv](outputs/player_shap.csv)） | 提案 UI 的「MVP / 哪位球員貢獻多少勝率」、雷達圖 |
| Score_B | ❌ 隊伍級交乘項 | 陣容化學效應解釋 |
| Δpm | ❌ 隊伍級平均 | 歷史強度先驗 |

提案的賣點不只是預測準度，還包括 **AI 文字報告**和**球員貢獻視覺化**。Score_A 是這部分的資料源，移除會讓 LLM 報告失去「個人歸因」的能力。

---

## 八、最終係數解讀

```
P(home wins) = σ(0.345·ΔA + 0.171·ΔB + 0.320·Δpm + 0.119·is_home + 0.120)
```

| 係數 | 值 | 意義 |
|---|---:|---|
| **α** | +0.345 | 個人強度（Score_A）權重 |
| **β** | +0.171 | 陣容互補（Score_B）權重 |
| **δ** | +0.320 | 歷史強度（Δpm）權重 |
| **γ** | +0.119 | 主場優勢 — 對應約 σ(0.12)≈53% > 50%，與 NBA 經驗一致 |
| intercept | +0.120 | base rate |

所有係數**為正且符合預期方向**，無共線性導致的反向。

### 給 UI 用的概念權重歸一化

如果要顯示「三個維度各佔多少」（不含主場），按係數 × ΔX 的 std 比例：

```
α · std(ΔA)   = 0.345 × 1.000 = 0.345   (z-score 後 std=1)
β · std(ΔB)   = 0.171 × 1.000 = 0.171
δ · std(Δpm)  = 0.320 × 1.000 = 0.320
```

→ 個人強度 41% / 陣容互補 21% / 歷史強度 38%

---

## 九、產出檔案（[outputs/](outputs/)）

| 檔案 | 內容 | 用途 |
|---|---|---|
| [`predictions_test.csv`](outputs/predictions_test.csv) | 1,225 場 2024-25 預測，含 Score_A/B/pm（raw + 標準化）、ΔA/B/pm、P_home_win、pred、actual | demo、誤差分析 |
| [`player_shap.csv`](outputs/player_shap.csv) | 每 (game, team, player) 的 SHAP 貢獻 + min_roll10 | AI 報告的「個人貢獻」資料源 |
| [`final_weights.json`](outputs/final_weights.json) | α/β/δ/γ/intercept、訓練/測試樣本數、特徵維度 | API 載入時參數 |

---

## 十、剩下可優化的方向

### 1. plus_minus 換成更乾淨的隊伍強度代理（小工）

team-mean(plus_minus_roll10) 還是有「隊友噪音」(球員 A 的 plus_minus 受隊友 B 表現影響)。可以改用：

- 隊伍 Elo rating（賽季滾動更新）
- 隊伍滾動 net rating（從 game_results 直接算）

兩者預期略增穩定性，AUC 變化 ±0.01。

### 2. Score_B 交乘特徵選擇（中工）

目前 36 對全進。可改用 L1 Lasso 或 Group Lasso 自動選稀疏交乘 → 提升可解釋性（哪些技能組合真的有 synergy），論文寫作友善。

### 3. 校準（calibration）（小工）

Logit 輸出機率可能 over/under-confident。用 isotonic 或 Platt scaling 在 hold-out 上再校準 → 預期 Brier score 改善，AUC 不變但機率顯示更合理。

### 4. 跨賽季 OOF（中工）

目前 OOF 是 random K-fold。可改成 **leave-one-season-out** 更嚴格驗證跨季穩健性。

---

## 附錄 A：環境

```
Python 3.14（.venv）
pandas 3.0.3
numpy  2.4.6
scikit-learn 1.8.0
xgboost 3.2.0
shap   0.51.0
libomp 22.1.6 (brew)
```

## 附錄 B：訓練耗時

| 階段 | 時間（粗估）|
|---|---|
| Output A — 5 折 OOF XGB + SHAP | ~60 秒 |
| Output A — 全 train refit | ~15 秒 |
| Output B — 5 折 OOF logit | ~3 秒 |
| Final layer + ablation | <1 秒 |
| **Total（M1 Mac）** | **~80 秒** |
