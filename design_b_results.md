# 設計 B 實作結果 — Score A (XGBoost+SHAP) + Score B (Logit) + Score C (L1 cross-team) + Δpm

> 對應提案：`Final Report Proposal.pdf` §3「核心模型：XGBoost + Regression」
> 實作檔：[`score_ab.py`](score_ab.py)
> 評估方式：時序切分 **train = 2020-21 ~ 2023-24 / test = 2024-25（1,225 場）**；final layer 使用 time-aware OOF 有分數的 3,935 場訓練 matchup

---

## 2026-06-04 demo-ready revision

這份文件原本記錄 Score C 加入後的第一版 Design B/C 結果。demo 前已完成新版修正：

- Score B 從 L2 logit 改成 `StandardScaler + LogisticRegressionCV(penalty="l1")`，54 個 features 留下 13 個。
- Output A/B/C 的 OOF 都改成按 `game_date` 排序的 `TimeSeriesSplit`。
- Score B 的 P80 threshold 改成每個 fold 的 train split 內部計算，測試季使用訓練期最新 threshold fallback。
- `predict_lineup.py` 已補上 Score C inference，並新增 `symmetric=True` 做中性場強弱比較。
- 新增 calibration bucket 診斷輸出，特別追蹤 0.45-0.50 bucket。

新版 hold-out 結果：

| 指標 | 數值 |
|---|---:|
| **Test AUC** | **0.7280** |
| **Test Accuracy** | **0.6759** |
| Baseline（永遠猜 home win） | 0.5445 |
| Final train matchups | 3,935 |
| Test matchups | 1,225 |
| Output B 特徵維度 | 54，L1 保留 13 |
| Output C 特徵維度 | 81，L1 保留 31 |

新版 final layer：

```
P(home wins) = σ(0.1885·ΔA + 0.1810·ΔB − 0.0179·C
                 + 0.4818·Δpm + 0.1280·is_home + 0.1290)
```

新版 ablation：

| 組合 | Test AUC |
|---|---:|
| Score_A only + is_home | 0.6661 |
| Score_B only + is_home | 0.6624 |
| Score_C only + is_home | 0.5832 |
| Δpm only + is_home | 0.7164 |
| A + B + is_home | 0.6941 |
| B + Δpm + is_home | 0.7264 |
| A + B + Δpm + is_home | 0.7280 |
| A + B + C + Δpm + is_home | 0.7280 |

解讀：AUC 提升到 0.7280，但 Accuracy 在 0.5 threshold 下略降。Score C 仍然不是主要預測訊號；它的價值比較偏解釋與報告敘事。

---

## 一、最終結果

```
P(home wins) = σ(0.1885·ΔA + 0.1810·ΔB − 0.0179·C + 0.4818·Δpm + 0.1280·is_home + 0.1290)
```

| 指標 | 數值 |
|---|---:|
| **Test AUC** | **0.7280** |
| **Test Accuracy** | **0.6759** |
| Baseline（永遠猜 home win） | 0.5445 |
| Train 樣本數 | 3,935 matchups |
| Test 樣本數 | 1,225 matchups |
| Output A 訓練樣本（per-player） | 83,640 rows |
| Output B 特徵維度 | 54（18 max+sum × 9 stats + 36 內部交乘對），L1 保留 13 |
| Output C 特徵維度 | 81（9 主隊技能 × 9 客隊技能），L1 保留 31 |

> **Score C 加入後仍幾乎不提升 AUC**。新版 A+B+Δpm 與 A+B+C+Δpm 都是 0.7280；Score C 保留為解釋性產出（`cross_team_pairs.csv` 列出 31 個非零 L1 係數），不是主要預測來源。

---

## 二、四個對齊問題的解法

提案合併兩個模型時的隱藏陷阱，逐項解：

| # | 對齊問題 | 處理方法 | 程式位置 |
|---|---|---|---|
| 1 | **Train/Test 切分** | 時序切分：train = 4 個賽季、test = 2024-25。同時解決提案 §5.3 時序洩漏。 | [score_ab.py:39-40](score_ab.py#L39-L40) |
| 2 | **樣本對齊** | 兩模型 inner join 同一批 `game_id`。11,910 team-games 全對齊。 | [score_ab.py:178-181](score_ab.py#L178-L181) |
| 3 | **`is_home` 旗標** | 不放進 Score 計算（保持對稱），改放到 final layer 當獨立特徵。`γ` 就是主場優勢。 | [score_ab.py:262-268, 284-285](score_ab.py#L284) |
| 4 | **In-sample bias**（新發現 ⚠️） | **Time-aware OOF 5-fold** 產生 train 的 Score → final layer 看到的訓練 Score 跟 test set 同等級。同時用 train statistics 做 **z-score 標準化**。 | [score_ab.py](score_ab.py) |
| 5 | **P80 threshold leakage** | Score B 的 P80 dummy 在每個 OOF fold 內只用 train split 重算；test season 使用訓練期最新 threshold fallback。 | [score_ab.py](score_ab.py) |

### 為什麼問題 #4 是隱藏陷阱

以下表格保留第一版 debugging 歷程；最新版結果請以上方 demo-ready revision 為準。

| 階段 | α | β | γ | δ(pm) | ζ(C) | Test AUC |
|---|---:|---:|---:|---:|---:|---:|
| 初版（無 OOF、無標準化） | +1.241 | **−0.613** ⚠️ | +0.139 | —— | —— | 0.6441 |
| + OOF + 標準化 | +0.541 | +0.220 | +0.118 | —— | —— | 0.6922 |
| + 36 內部交乘 + Δpm | +0.345 | +0.171 | +0.119 | +0.320 | —— | 0.7233 |
| + Score C (81 跨隊 + L1)（前版） | +0.348 | +0.176 | +0.119 | +0.319 | **−0.023** | **0.7236** |
| **+ L1 Score B + time-aware OOF + fold 內 P80（新版）** | +0.1885 | +0.1810 | +0.1280 | +0.4818 | **−0.0179** | **0.7280** |

原因：Score 在 train 是 in-sample fit（精度過高），在 test 是 out-of-sample（有泛化誤差）。final logit 對著兩個分布完全不同的訊號學係數，會把第二個 Score 推到負值來「修正」第一個 Score 在 train 上的 overshoot。OOF 預測讓兩邊分布一致就解了。

---

## 三、Pipeline 結構

```
                ┌──────────────────────────────────────┐
                │ player_rolling.csv (24 features)     │
                │ ─ plus_minus_roll10  EXCLUDED        │
                └──────────┬───────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼ Output A         ▼ Output B         ▼ Output C
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ per-player XGB   │  │ 9 stats → 賽季    │  │ B 的 sum 特徵     │
│ sample_w=min     │  │   P80 dummy      │  │ → 9×9 跨隊        │
│ time-aware OOF   │  │ team max + sum   │  │   sum×sum 交乘    │
│ refit-all test   │  │ 36 內部交乘       │  │ Pipeline:        │
│ + SHAP (margin)  │  │ time-aware L1    │  │   Std → L1 logit │
│ Σ player SHAP →  │  │ refit-all test   │  │   (CV 選 C)       │
│   Score_A        │  │ decision_func →  │  │ refit-all test   │
│   team-game      │  │   Score_B        │  │ decision_func →  │
│                  │  │   team-game      │  │   Score_C        │
│                  │  │                  │  │   matchup-level  │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                               ▼  per-team / per-matchup Scores
                  ┌─────────────────────────┐
                  │ Matchup frame:           │
                  │  ΔA = A_home − A_away    │
                  │  ΔB = B_home − B_away    │
                  │  C  = score_C (directional)
                  │  Δpm = pm_home − pm_away │ ◄ team-mean(plus_minus_roll10)
                  │  is_home = 1             │
                  │  z-score with train stats│
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌─────────────────────────────────────┐
                  │ Final calibration logit              │
                  │ P = σ(α·ΔA + β·ΔB + ζ·C + δ·Δpm     │
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

對 9 個 rolling 指標，按賽季取 P80 閾值。新版 OOF 會在每個 fold 的 train split 內重算 threshold，避免 validation fold 資訊洩漏：

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

- `StandardScaler + LogisticRegressionCV(penalty="l1")`
- Time-aware 5-fold OOF 產生 train 的 Score_B
- Full train refit 後，54 個 Output B features 保留 13 個非零係數
- 用全部 train 重 fit 後對 test 預測
- 輸出 `decision_function`（logit value，與 Score_A margin-space 同尺度）

---

## 五-B、Score C：跨隊技能交互 Logit（新增）

### 5-B.1 動機

§五 的 36 對交乘是**同一隊內部**的技能搭配（home 射手 × home 護框），抓不到「主隊射手 vs 客隊外圍防守弱」這類**跨隊克制**。Score C 補上這個缺口。

### 5-B.2 特徵設計

對 9 個 Score B 用的 dummy stats，做 **9 × 9 = 81 個跨隊 sum × sum 交乘**：

```python
for h_skill in DUMMY_STATS:
    for a_skill in DUMMY_STATS:
        m[f"c_h_{h_skill}_a_{a_skill}"] = (
            home_features[f"d_{h_skill}_sum"]
            * away_features[f"d_{a_skill}_sum"]
        )
```

→ 包含對稱項（`home_pts × away_def` 跟 `home_def × away_pts` 分別建項，方向相反由資料決定係數）。

### 5-B.3 訓練

| 項目 | 設定 |
|---|---|
| 樣本單位 | per-matchup（每場 1 row，不是 per team-game） |
| 標籤 | home_win |
| 模型 | `Pipeline(StandardScaler → LogisticRegressionCV)`，penalty=L1, solver=saga |
| 正則化 | LogisticRegressionCV 從 `np.logspace(-3, 1, 20)` 自動掃 C（內 5-fold） |
| OOF | Time-aware 5-fold（同 A/B），train-side score_C 為 out-of-sample |
| 結果 | L1 保留 **31/81** 特徵，最佳 C ≈ 0.0785 |

Score C **不做差** — 81 個跨隊交乘本身已是 matchup-level，有方向性，直接進 final layer。

### 5-B.4 結果（負面 — 但有意義）

新版 `ζ = −0.0179`（標準化後），A+B+Δpm 和 A+B+C+Δpm 的 hold-out AUC 都是 0.7280。

**含意**：

| 觀察 | 解讀 |
|---|---|
| Score C only AUC = 0.5832（baseline 0.5445） | 跨隊交乘**單獨**只有有限訊號 |
| 加入 Score C 後 ζ 微負 | 不是 leak（OOF 已防），是 final layer 對 C 殘餘噪音做小幅校正 |
| B + Δpm = 0.7264，B + C + Δpm = 0.7262（近乎持平） | C 在 B+Δpm 之上**幾乎沒有獨立貢獻** |
| L1 保留 31 個特徵 | 抓到一些有方向的對子，但訊號太弱無法影響大局 |

**為什麼會這樣**：
1. **隊伍歷史強度（Δpm）已經吸收了大部分訊號**。NBA 強隊普遍各維度都強，弱隊普遍各維度都弱 — 「主隊各項技能厚度 × 客隊各項技能厚度」的乘積，主要還是在量「強隊 × 弱隊」這個粗粒度訊號，沒有真正的克制細節。
2. **P80 dummy 是 0/1，乘積空間粗糙**。要抓細微克制可能需要連續值或多層交互。
3. **NBA 風格極端互克的情境本來就少**，C(9,2) × 樣本數可能不足以估計細粒度交互。

### 5-B.5 為什麼仍保留

雖然 AUC 不漲，**Score C 的 L1 survivors 提供論文需要的「跨隊克制」敘事素材**：

[outputs/cross_team_pairs.csv](outputs/cross_team_pairs.csv) 列出 31 對 |coef|>0 的跨隊配對 ranked by |coef|。前幾名範例：

| home_skill | away_skill | coef | 籃球直覺 |
|---|---|---:|---|
| `d_def_impact` | `d_reb` | **+0.152** | 主隊防守好 × 客隊籃板強 — 主隊優勢 |
| `d_pts` | `d_def_impact` | −0.076 | 主隊得分手 × 客隊頂防 — 客隊優勢（符合直覺） |
| `d_blk` | `d_fg3m` | −0.063 | 主隊護框 × 客隊射手 — 客隊優勢（射手 bypass 護框） |
| `d_stl` | `d_fg3m` | +0.063 | 主隊抄截 × 客隊射手 — 主隊優勢（抄斷三分傳球） |
| `d_true_shooting` | `d_blk` | +0.058 | 主隊高效率 × 客隊護框 — 主隊優勢 |

→ 這些對子可作為 AI 報告中「為什麼預測主隊贏」的解釋來源，**即使預測力本身沒提升**。

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
| Score_B only + is_home | 0.6624 | +0.118 | 陣容互補（新版 L1） |
| Score_C only + is_home | 0.5832 | +0.039 | 跨隊交互（單獨訊號有限） |
| **Δpm only + is_home** | **0.7164** | **+0.172** | 最強單一訊號 |
| A + B + is_home | 0.6941 | +0.150 | 提案原版（無 Δpm） |
| A + Δpm + is_home | 0.7211 | +0.177 | |
| B + Δpm + is_home | 0.7264 | +0.182 | |
| A + B + Δpm + is_home | 0.7280 | +0.184 | C 以外的主力組合 |
| A + C + Δpm + is_home | 0.7212 | +0.177 | C 不替代 B |
| B + C + Δpm + is_home | 0.7262 | +0.182 | C 不疊加於 B（近乎持平） |
| **A + B + C + Δpm + is_home（最終）** | **0.7280** | **+0.184** | **採用** |

### 重要發現

**1. Δpm 部分吸收了 Score_A 的預測力**：
- B + Δpm（無 A）= 0.7264
- A + B + Δpm（有 A）= 0.7280
- 新版加入 Score_A 後小幅提升 0.0016，表示 Score_A 仍有邊際訊號，但主要 power 仍在 Δpm

→ Score_A 與 Δpm 高度共線（兩者都在量「球員整體強度」）

**2. Score C 與 B + Δpm 完全共線（新發現）**：
- B + Δpm（無 C）= 0.7264
- B + C + Δpm（有 C）= 0.7262
- A + B + Δpm = 0.7280；A + B + C + Δpm = 0.7280
- C 提供的「跨隊技能交互」訊號大多已被 B（同隊內部組合）+ Δpm（隊伍歷史強度）吸收

→ 跨隊 sum × sum 在 NBA 賽季粒度下**沒有獨立預測力**。詳見 §五-B。

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
P(home wins) = σ(0.348·ΔA + 0.176·ΔB − 0.023·C + 0.319·Δpm + 0.119·is_home + 0.120)
```

| 係數 | 值 | 意義 |
|---|---:|---|
| **α** | +0.348 | 個人強度（Score_A）權重 |
| **β** | +0.176 | 陣容互補（Score_B）權重 |
| **ζ** | −0.023 | 跨隊交互（Score_C）權重 — **接近 0，無顯著貢獻** |
| **δ** | +0.319 | 歷史強度（Δpm）權重 |
| **γ** | +0.119 | 主場優勢 — 對應約 σ(0.12)≈53% > 50%，與 NBA 經驗一致 |
| intercept | +0.120 | base rate |

α / β / δ / γ **為正且符合預期方向**；ζ 的微負是 final layer 對 C 殘餘噪音的小幅校正，不是 leak（OOF 已防）。加入 C 後 α / β / δ 跟前一版幾乎不變，表示 C 沒搶走 A / B / Δpm 的訊號。

### 給 UI 用的概念權重歸一化

如果要顯示「四個維度各佔多少」（不含主場），按係數 × ΔX 的 std 比例（z-score 後 std=1）：

```
|α · std(ΔA)|  = 0.348 × 1.000 = 0.348
|β · std(ΔB)|  = 0.176 × 1.000 = 0.176
|ζ · std(C)|   = 0.023 × 1.000 = 0.023   ← 可忽略
|δ · std(Δpm)| = 0.319 × 1.000 = 0.319
```

→ 個人強度 41% / 陣容互補 21% / 跨隊交互 3% / 歷史強度 38%

→ Demo / UI 可以選擇只呈現 3 個維度（A、B、Δpm），把 C 留給「解釋性」用途而非「貢獻佔比」。

---

## 九、產出檔案（[outputs/](outputs/)）

| 檔案 | 內容 | 用途 |
|---|---|---|
| [`predictions_test.csv`](outputs/predictions_test.csv) | 1,225 場 2024-25 預測，含 Score_A/B/C/pm（raw + 標準化）、ΔA/B/Δpm、P_home_win、pred、actual | demo、誤差分析 |
| [`player_shap.csv`](outputs/player_shap.csv) | 每 (game, team, player) 的 SHAP 貢獻 + min_roll10 | AI 報告的「個人貢獻」資料源 |
| [`cross_team_pairs.csv`](outputs/cross_team_pairs.csv) | Score C 的 31 個非零 L1 係數，含 home_skill / away_skill / coef，按 \|coef\| 排序 | 「跨隊技能克制」敘事素材 |
| [`final_weights.json`](outputs/final_weights.json) | α/β/ζ/δ/γ/intercept、訓練/測試樣本數、特徵維度 | API 載入時參數 |
| `outputs/models/logit_output_C.joblib` | Score C 的 L1 logit pipeline（含內部 StandardScaler） | inference 時 reconstruct score_C |
| `outputs/models/scaler_C.joblib` | matchup 層 score_C → z-score | final layer 標準化 |

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

### 5. Score C 的後續變體（中工，預期仍負面）

§五-B 的負面結果是 sum × sum + L1 這個特定設定下的結論。可以試的變體：

- **max × max（81 個）**：抓「有 vs 沒有」而非「幾個 vs 幾個」。預期更接近 0/1，正則化後可能保留更精簡的對子，但訊號量級更低。
- **連續值 × 連續值**：放棄 P80 二值化，直接用 home 隊均 pts × away 隊均 def_impact 等乘積。代價：失去 lineup composition 的語意，比較像通用 power rating。
- **Elastic Net 替代 L1**：當前 L1 在高度相關特徵中只挑一個，可能丟掉同等資訊。Elastic Net 會保留組，便於敘事但可能更稀疏不出來。
- **Group Lasso 按技能分組**：把「home_d_pts_a_*」9 個當一組，學「主隊得分手對哪些客隊技能敏感」。

**建議優先順序**：論文寫作上 §五-B 的負面結果已經夠用（「我們試了 81 對交乘 + L1，發現與 Δpm 共線」），上述變體值不值得跑要看時間。

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
| Output C — 5 折 OOF L1 LogitCV（內含 5-fold C search） | ~30 秒 |
| Final layer + ablation | <1 秒 |
| **Total（M1 Mac）** | **~110 秒** |
