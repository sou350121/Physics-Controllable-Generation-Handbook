# Robot Data Benchmarks

> 量化「生成 / 合成機器人資料」的價值＝用它訓練出的 policy 在 sim 與真機上的下游成功率（不是影片好不好看）。

## 這類在測什麼

生成式機器人資料（video world model 生出的軌跡、sim 程序化生成的 demo、neural trajectory）的 ground truth **不是視覺保真度**，而是 **downstream policy success**：拿生成資料去訓練 / co-train 一個 policy，這個 policy 在標準操作 benchmark（sim）或真機上的任務成功率有沒有提升。

換句話說，「這批生成資料好不好」只有一個誠實答案：**用它訓出來的 policy 會不會成功**。影片再逼真、物理再守恆，若 policy 學完成功率不動，這批資料對機器人學習就是零增益。本頁的 benchmark 全部圍繞這條因果鏈：生成資料 → 訓練 → 成功率。

## Benchmark 全景

| Benchmark | 測什麼 | 主要 metric | 已知缺陷 / 可被 game | 出處 |
|---|---|---|---|---|
| **LIBERO** | sim 操作 + lifelong learning（130 任務 / 4 task suite，程序化生成 + 人類遙操 demo） | per-task / 平均 success rate；forward transfer | 純 sim，無真機；任務集固定，過擬合 suite 可刷高分；分數飽和（SOTA 已 ~98%）後鑑別力下降 | [2306.03310](https://arxiv.org/abs/2306.03310) |
| **RoboCasa** | 大規模 kitchen sim（150+ 物件類、generative AI 資產）+ 合成資料 scaling + sim→real co-train | sim 任務 success rate；真機 co-train 成功率提升 | sim 真實度有限；co-train 結果混入真機 demo（增益難歸因；見下節） | [2406.02523](https://arxiv.org/abs/2406.02523) |
| **CALVIN** | 長程、語言條件操作（連續指令鏈、未見環境 zero-shot） | 連續指令鏈成功率 / 平均完成序列長度 | 純 sim；長鏈成功率對單步失敗極敏感；語言 grounding 可被模板套路刷 | [2112.03227](https://arxiv.org/abs/2112.03227) |
| **SIMPLER** | real-to-sim 評測環境：用 sim 預測真機 policy 表現 | sim success 與 real success 的**相關性** | 自身是「評測器的評測」；對齊有限的 task 集；非 full digital twin，視覺 / control 差異仍殘留 | [2405.05941](https://arxiv.org/abs/2405.05941) |
| **DreamGen Bench** | video world model 生成資料的下游價值（生成 → 回收動作 → 訓 policy） | video 生成分數 與 downstream policy success 的相關性 | benchmark 與真機任務分布綁定；generation 分數對 policy 增益只是相關非保證 | [2505.12705](https://arxiv.org/abs/2505.12705) |
| **RoboArena** | 分散式真機評測：眾包、雙盲、成對 policy 比較 | 成對勝率（pairwise win rate，類 Elo） | 任務 / 環境由評測者自選 → 高 diversity 但難精確復現單一數字；相對排名而非絕對成功率 | [2506.18123](https://arxiv.org/abs/2506.18123) |

關鍵已核對數字（source-grounded）：

- **RoboCasa co-train**：counter↔sink 的 pick-and-place 任務，真機 demo only 成功率 **13.6%**，加入 sim 合成資料 co-train 後升到 **24.4%**（[2406.02523](https://arxiv.org/abs/2406.02523)）。這是「生成 / sim 資料 → 真機成功率提升」最常被引用的單點證據。
- **DreamGen**：humanoid 僅靠**單一 pick-and-place 任務**的遙操資料，經 video world model 生成 + 動作回收後，可在 seen / unseen 環境執行 **22 種新行為**（[2505.12705](https://arxiv.org/abs/2505.12705)）。
- **Cosmos-Policy**（post-train Cosmos Predict-2 world model）：LIBERO 4 suite 平均 **98.5%**；RoboCasa 24 kitchen 任務平均 **67.1%**，且僅用 **50 demo**（對比舊 SOTA 300）（NVlabs/cosmos-policy，README / ROBOCASA.md；論文 arXiv ID `UNVERIFIED`）。
- **π0**（VLA flow model）：報告 laundry folding / table cleaning / box assembly 等真機任務能力，但摘要層未給可逐項核對的成功率數字 → 具體分數標 `UNVERIFIED`（[2410.24164](https://arxiv.org/abs/2410.24164)）。

## 怎麼誠實讀分數

- **sim success ≠ real success**：sim 上的高分不自動轉成真機。SIMPLER 的整個意義就是先量「sim 與 real 的相關性」——只有 correlation 站得住，sim 數字才值得信。沒做 real-to-sim 對齊就拿 LIBERO / RoboCasa sim 分當「真機能力」是過度外推。
- **data-quality confound（更多 vs 更好）**：成功率上升可能只是**資料量變多**，不是**生成資料品質高**。誠實協議要固定總資料量、或設「等量真機 demo」對照組，才能把增益歸給「生成」而非「數量」。
- **co-train 混淆歸因**：RoboCasa 13.6%→24.4% 是 **sim + real 混訓**的結果，真機 demo 仍在配方裡。這證明「生成資料有增量幫助」，但不能讀成「生成資料單獨可訓出可用 policy」。
- **real-robot 數字難復現**：真機評測貴、慢、對 hardware / 場景 / 操作者敏感，單一實驗室的絕對成功率幾乎無法跨團隊復現——這正是 RoboArena 改走**眾包雙盲成對比較 + 相對勝率**的動機。讀真機絕對成功率時，預設它帶大 error bar。

## 現況與缺口

- **最大缺口：沒有標準化的「生成資料增益」協議。** 各家用不同 base policy、不同資料量、不同 co-train 配方報自家提升，彼此**不可比**。缺一個固定「base policy + 固定真機 demo 預算 + 加入 N 條生成資料」的對照標準，無法回答「哪種生成方法產的資料更值錢」。
- **real eval 又貴又難復現。** sim benchmark（LIBERO / RoboCasa / CALVIN）便宜可復現但會飽和、會 game；真機評測有效度高卻昂貴且復現性差。SIMPLER（real-to-sim 相關性）與 RoboArena（分散式真機成對比較）是兩條互補的緩解路線，但都尚未成為社群預設。
- **generation 分數 ≠ policy 增益的因果保證。** DreamGen Bench 顯示 video 生成分數與下游成功率**相關**，但相關非因果；仍需逐案用 downstream success 復驗，不能拿生成品質分當代理指標收工。

## 連回

- [benchmarks/ overview](../overview.md)
- [foundations/evaluation-physics](../../foundations/evaluation-physics/overview.md)
- [foundations/data-engine/robocasa](../../foundations/data-engine/robocasa.md)
- [crossing/sim-vs-gen-data](../../crossing/sim-vs-gen-data/overview.md)
- [use-cases/robotics-data-gen](../../use-cases/robotics-data-gen/overview.md)
- [bridge-to-vla/generative-data-for-vla](../../bridge-to-vla/generative-data-for-vla.md)

## 參考

- LIBERO — Benchmarking Knowledge Transfer for Lifelong Robot Learning（2023）— [arXiv:2306.03310](https://arxiv.org/abs/2306.03310)
- RoboCasa — Large-Scale Simulation of Everyday Tasks for Generalist Robots（2024）— [arXiv:2406.02523](https://arxiv.org/abs/2406.02523)
- SIMPLER — Evaluating Real-World Robot Manipulation Policies in Simulation（2024）— [arXiv:2405.05941](https://arxiv.org/abs/2405.05941)
- CALVIN — A Benchmark for Language-Conditioned Long-Horizon Robot Manipulation（2021）— [arXiv:2112.03227](https://arxiv.org/abs/2112.03227)
- DreamGen — Unlocking Generalization in Robot Learning through Video World Models（2025）— [arXiv:2505.12705](https://arxiv.org/abs/2505.12705)
- π0 — A Vision-Language-Action Flow Model for General Robot Control（2024, RSS 2025）— [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
- RoboArena — Distributed Real-World Evaluation of Generalist Robot Policies（2025）— [arXiv:2506.18123](https://arxiv.org/abs/2506.18123)
- Cosmos-Policy — Fine-Tuning Video Models for Visuomotor Control（NVlabs/cosmos-policy；LIBERO 98.5% / RoboCasa 67.1%；論文 arXiv ID `UNVERIFIED`）
