<!-- ontology-5axis output=action-seq injection=sim-in-loop-train control=action|image-init temporal=latent-rollout domain=robotics -->

# 世界模型即策略 —— Dreamer 在腦內做夢學會行動 解構

> Hafner, _et al._ _Mastering Diverse Domains through World Models_ (DreamerV3). arXiv [2301.04104](https://arxiv.org/abs/2301.04104) · Wu, Escontrela, Hafner, _et al._ _DayDreamer: World Models for Physical Robot Learning._ arXiv [2206.14176](https://arxiv.org/abs/2206.14176)。
>
> **為什麼進 embodied-policy-rollout anchor 名單**：本 handbook 四條 generation 路線裡，latent-WM 這條最反直覺的主張是 ——「策略可以完全不碰真實互動、只在一個學來的世界模型的 latent 夢境裡學會行動」。Dreamer 線是這個主張的 canonical 代表，而且 DayDreamer 把它從模擬器搬到**四台真機**上跑通了。它在 ontology 上乾淨佔住 `temporal=latent-rollout` × `output=action-seq` × `injection=sim-in-loop-train` 這個座標，是 [overview.md](./overview.md) 「Actor-critic on WM」派的 anchor。但「想像即決策」這句話藏著一份**承重契約**（§5）—— 那是本篇的智力核心：想像之所以能換掉真實互動，是因為世界模型的 reward / continue head 與短程動力學「必須對」。看懂這份契約，才看得懂 Dreamer 的貢獻邊界與崩潰邊界。
>
> **與 sibling 的分工**：[../aerial-sim/dream-to-fly.md](../aerial-sim/dream-to-fly.md) 已把 DreamerV3 飛無人機（aerial、HIL、raw-pixel）那一案講透 —— **本篇不重複那個無人機案例**，本篇講 general embodied（DayDreamer 的四足 / 雙臂 / 輪式真機）與「WM-as-policy 的承重契約」這個抽象問題。

## 1. TL;DR

**Dreamer 的核心 thesis：把一個共用的 latent world model（RSSM）學出來，然後讓 actor 與 critic 純粹在這個世界模型「想像」出來的 latent 軌跡上學行為 —— agent 碰真實環境，只為了把經驗填回 replay buffer 去把世界模型訓得更準。** 行為學習期間（imagination horizon **T=16**）梯度從不流經真實 reward、也不流經真實像素，整段 actor-critic 跑在模型的夢裡。DreamerV3 用**一套固定超參**跨 **150+ 任務**全部勝出，並在 Minecraft **從零、無任何人類資料**自己採到鑽石。

**DayDreamer 是這條線最硬的存在證明：把 Dreamer 直接線上接到真實世界的四台機器人上。四足機器人從零、無重置、約 1 小時學會走路，再約 10 分鐘適應被推；雙臂從相機 + 稀疏 reward 學 pick-place；輪式做視覺導航 —— 全程沒有模擬器。** 它之所以能用「極少真實互動」做到，正是因為 actor-critic 是在**想像** rollout 上訓的，真機只負責餵資料。

**一句話的承重契約**：WM-as-policy 的本質是「**把昂貴的真實互動，換成廉價的想像 rollout**」。這筆交換不是免費的 —— 它的代價就是 §5 那三件**必須對**：**reward head、continue（episode-end）head、以及資料分布附近的短程動力學**。錯了其中任何一件，actor 就會去最大化一個「夢裡很高、真實不認」的回報（model exploitation）。DayDreamer 維持誠實的辦法很樸素：**持續把真實資料餵回去**，把 model-exploitation 迴圈閉掉。

## 2. 核心機制（RSSM + imagination horizon T=16 + 三個 head）

Dreamer 由**三個網路**構成：(1) **world model**（RSSM：encoder 把觀察壓成 latent；recurrent 部分維持 deterministic 狀態）、(2) **actor**、(3) **critic**。世界模型的 RSSM 把高維觀察壓成 latent state `s_t = (h_t, z_t)`：`h_t` 是 deterministic recurrent 狀態（攜帶 history），`z_t` 是 discrete latent（捕捉當下不確定性，posterior 由 encoder 給、prior 由 dynamics 給）。世界模型掛**三個 head**：**reward predictor + continuation(episode-end) predictor + decoder（重建觀察）**。

**承重點先講白**：★**actor-critic 的行為學習期間，從不看真實 reward —— 它最大化的是 reward head 想像出來的回報。因此 reward head 與 continue head 是承重件**，它們錯了，想像回報就錯，策略就學歪。decoder 只是世界模型訓練的輔助 target，policy 不依賴它生成的像素。

```
  ┌──────────── WORLD MODEL (RSSM)  ── 只在真實 env steps 上學 ───────────┐
  │                                                                       │
  │   o_t ──Encoder──► z_t   (discrete latent；posterior q(z|h,o))        │
  │                     │                                                 │
  │   h_t = recurrent(h_{t-1}, z_{t-1}, a_{t-1})   ← deterministic state  │
  │   ẑ_t ~ prior p(ẑ|h_t)   ← dynamics 自己預測下一 latent（無 o）       │
  │                     │                                                 │
  │     s_t=(h_t,z_t) ──┼──► Decoder       ──► ô_t   (重建；輔助 target)  │
  │                     ├──► Reward head   ──► r̂_t   ★承重               │
  │                     └──► Continue head ──► ĉ_t   ★承重 (episode-end) │
  └───────────────────────────────────────────────────────────────────┘
                        │
                        ▼  此線以下不碰真實 reward、不碰像素 —— 純 latent
  ┌──────── IMAGINATION（actor-critic 在這裡訓，horizon T=16）──────────┐
  │                                                                     │
  │   s_t ─► actor π ─► a_t ─► RSSM 用 prior 想像 ─► s_{t+1}             │
  │        ─► r̂, v̂  ...（×16 latent steps，全程不回真環境）             │
  │   actor: 最大化想像回報 (λ-return of r̂)   critic: 回歸 v̂            │
  └─────────────────────────────────────────────────────────────────┘
        ▲ 只在這裡碰真環境：採 (o,a,r) 填 replay buffer 訓 world model
        └──── 閉迴路：資料↑ ⇒ world model↑ ⇒ 夢更接近真實 ⇒ 策略更可信 ───
```

迴路四步：**encode**（壓 latent）→ **learn WM**（在真實 env steps 上學 dynamics + reward + continue）→ **imagine**（用 prior 在 latent 裡 rollout 16 步）→ **actor-critic**（只在夢上 backprop）。**真實 env step 的唯一用途是訓世界模型與蒐集 latent 起點 —— 這是 Dreamer 樣本效率結構性勝過 model-free 的根本原因：它把昂貴的真實 rollout 攤提成一次性的世界模型訓練，之後想像要多少 rollout 就有多少。**

DreamerV3 讓「一套固定超參跨域」成立的穩健機制（四件套）：**symlog**（對 reward / value / decoder target 做對稱對數壓縮，免去逐域調 reward scale）、**two-hot** 分布式 reward / value head（把回歸改成分類，吃得下極端尺度）、**free bits**（KL clip 在 **1 nat**，防 posterior collapse）、**percentile return normalization**（用回報的百分位數做正規化，穩定 actor 梯度）。沒有這四件，「固定超參跨 150 任務」這個賣點不成立。

**規模即性能**：DreamerV3 提供 **6 種 size（12M → 400M params，各自單張 A100 即可訓）**，而且**越大越好、且越大需要的資料越少** —— 這跟 model-free 常常 saturate 形成乾淨對照。最戲劇性的存在證明：Minecraft **從零、無任何人類資料**採到鑽石（**100M steps，約 9 天單 GPU**）。

## 3. ⚡ 強 / ❌ 崩

⚡ **強（樣本效率、通用性、真機 DayDreamer）**

- **樣本效率是 enabler，不是 nice-to-have**：把真實互動攤提成世界模型訓練後，想像 rollout 近乎免費。DayDreamer 的真機數字就靠這個 —— 四足**約 1 小時**從零學會走，是 model-based imagination 直接讓「真機上學連續控制」變得可負擔。per-task 對 model-free 的具體加速倍數在原文（標 [UNVERIFIED]，本篇不替論文背書未經查證的數字）。
- **通用性是被證過的**：DreamerV3 **一套固定超參跨 150+ 任務**全勝 —— 把「model-based RL 不通用、得逐域調參」這條 prior 直接打破。
- **真機、無模擬器（DayDreamer 的硬核賣點）**：Dreamer 線上**直接在真實世界的四台機器人**上跑 —— 四足（從零、**無重置**、約 1 小時走、約 10 分鐘適應抗推）、雙臂（相機 + 稀疏 reward 的 pick-place，「approaching human」）、輪式（視覺導航）。**無重置**這點尤其重要：真機沒有「reset 大法」，Dreamer 的閉迴路在這種非理想條件下仍站得住。
- **規模友善**：12M→400M 六擋，**單張 A100** 即可訓任一擋；越大越好且越省資料。學界可負擔，不需 GPU farm。
- **無人類先驗的長程探索**：Minecraft 從零採鑽石（100M steps / ~9 天單 GPU），證明 latent imagination 撐得起 sparse-reward + long-horizon。

❌ **崩（想像 ≠ 真實的代價）**

- **Model exploitation / reward hacking 是結構性風險**：actor 最大化的是 reward head **想像**的回報。只要 reward head 或 continue head 在 actor 探到的 latent 區域估錯，actor 就會直奔「夢裡高分、真實不認」的軌跡。**這不是 bug，是 WM-as-policy 範式的固有張力** —— 整個 Dreamer / MuZero 家族都被指出過。
- **Imagination horizon 短是有原因的**：T=16 不是隨便設的。latent prior 的 rollout 誤差會逐步累積，超過某個步數後想像就偏離真實太多、不能用來學行為。**「把 horizon 拉長」幾乎總是傷性能**（見 §8）—— 這直接界定了 Dreamer 能處理的時間尺度：短程可信、長程不可信。
- **想像只在資料分布附近可信**：世界模型只在 replay buffer 見過的 (o, a) 分布附近準。actor 一旦把策略推到分布外（OOD action / 沒見過的狀態），prior 動力學與 reward head 同時失準 —— 這正是 DayDreamer 必須**持續餵真資料**的原因：把策略探到的新區域不斷拉回「世界模型見過」的範圍。
- **承重件錯則全錯（§5 的反面）**：因為行為學習全靠想像回報，reward head / continue head 一旦在關鍵狀態估歪，沒有任何真實 reward 在訓練迴路裡把它救回來 —— 錯誤會被 actor 放大，不會被自動修正。
- **decoder artifact 不等於 policy 壞**：RSSM decoder 重建常常糊（尤其 UI / 細節），但那是輔助 target；policy 品質看的是 reward curve，不是重建清晰度。新手常把重建糊讀成「沒學好」，是誤判。

## 4. 五軸定位 + 同軸對手

| Axis | **Dreamer (V3 / DayDreamer)** | TD-MPC2 | [V-JEPA-2](../../foundations/latent-world-models/v-jepa-2.md) | MuZero / EfficientZero | [Genie-2](../../foundations/latent-world-models/genie-2.md) |
|---|---|---|---|---|---|
| Output | `action-seq`（actor 直出 action） | `action-seq`（MPC 規劃出 action） | `latent-tokens`（後接 action head） | `action-seq`（tree search 出 action） | `action-seq` / playable WM |
| Injection | **`sim-in-loop-train`**（世界模型即「學來的可微 sim」，想像 rollout 訓 actor） | `sim-in-loop-train`（latent dynamics 供 planning） | `data-only`（self-supervised latent 預測） | `sim-in-loop-train`（latent model 供 MCTS） | `data-only`（VPT/video distribution） |
| Control | `action`（+ `image-init` 起 rollout 開頭） | `action` | `action` | `action` | `action` + `image-init` |
| Temporal | **`latent-rollout`（imagine T=16）** | `latent-rollout`（每步 re-plan） | `latent-rollout` | `latent-rollout`（tree） | `latent-rollout`（autoregress token） |
| Domain | `robotics`（DayDreamer 真機四足/雙臂/輪式；V3 跨 150 任務） | `robotics` | `robotics` | `generalist`（Atari/board） | `generalist`（playable world） |

**同軸對手的關鍵差異**：

- **vs TD-MPC2** —— 兩者共用 latent dynamics，但 **Dreamer 把 planning 攤提（amortize）進 actor，TD-MPC 每步重新 plan**。trade-off：Dreamer **部署便宜**（一次前向出 action）、但對 distribution shift 較脆；TD-MPC **re-plan 對 shift 更 robust**、但 long-horizon planning cost 線性放大。Dreamer 勝在「inference 便宜、policy 多次重用」，MPC 勝在「task 多變、policy 不好預訓」（這條 trade-off 與 [overview.md](./overview.md) 兩派分界一致）。
- **vs V-JEPA-2** —— V-JEPA-2 押注 **representation transfer**（看大量 unlabeled video 學表示，再 attach action head），latent 是「為表示而生」；Dreamer 的 latent 是「**為 imagination rollout 與 control 而生**」、從 day-1 就 close loop。兩者在「offline video → controllable agent」正面對撞，**diff 在 latent 是否一開始就為 dynamics 設計**。
- **vs MuZero / EfficientZero** —— 同屬 latent-WM，但用 **MCTS tree search** 做決策（discrete planning）；Dreamer 用 **stochastic latent + gradient-based policy improvement**。Dreamer 在連續控制（DayDreamer 真機）更自然，tree search 在離散動作 / board game 更強。
- **vs Genie-2** —— Genie 是 interactive **playable** WM，主打人玩；**不在 WM 內做 RL、無 actor-critic**。Dreamer 是 WM 內訓 policy。

> **Cross-axis note**：`temporal=latent-rollout` × `injection=sim-in-loop-train` × `output=action-seq` 這個座標的精髓在 **Injection 軸**（本 handbook 的 USP 軸）—— Dreamer 的世界模型扮演的就是一個「**從資料學出來的、可微的 in-loop simulator**」，actor 的梯度穿過它。這跟 Genesis/MJX 那種「人寫物理方程的可微 sim」是同一個 ontology 格子、不同的物理來源（學來的 vs 寫死的）。

## 5. 「想像要可信」必須對的是什麼

這是本篇的智力核心，也是 WM-as-policy 範式的承重契約。因為 **actor-critic 優化的是想像回報、訓練迴路裡沒有真實 reward 把關**，所以世界模型「必須對」的不是全部 —— 而是精確的三件，且策略必須待在資料分布內：

| 必須對的東西 | 為什麼承重 | 錯了會怎樣 | Dreamer 的保險 |
|---|---|---|---|
| **Reward head**（`r̂`） | actor 整段在最大化它想像的回報，真實 reward 不進訓練迴路 | actor 直奔「夢裡高分、真實不認」的軌跡（reward hacking） | symlog + two-hot 穩住跨域 reward scale；**持續餵真資料**校準（DayDreamer） |
| **Continue / episode-end head**（`ĉ`） | 它決定想像 rollout 何時該終止 / bootstrap value；估錯等於在錯的 horizon 上算回報 | 把「該結束」的狀態當成能繼續累積 reward → 高估回報、學出危險策略 | continue head 與 reward 同步在真實 episode 邊界上訓 |
| **短程動力學（資料分布附近）** | T=16 的想像 rollout 全靠 prior 預測下一 latent；只要本地、短程準就夠 | 分布外 / 長 horizon 下 prior 漂移 → 整段想像失真 → actor 學廢 | **horizon 故意設短（T=16）** + **持續餵真資料**把策略探區拉回分布內 |

**關鍵洞察**：世界模型**不需要全局正確**（它的 decoder 可以一直糊、它對遠處 / 罕見狀態可以一無所知）—— 它只需要在**「策略當前會去的地方、未來 16 步之內」**這個小鄰域裡，**reward / termination / 動力學**三者同時誠實。把「全局保真」誤當成 WM-as-policy 的要求，是常見的概念錯置；真正的要求是**局部、短程、在分布內的誠實**。

**DayDreamer 怎麼守這份契約**：靠**持續把真實互動資料餵回 replay buffer**。每當 actor 把策略推到世界模型沒見過的區域，新真資料就把那塊補進世界模型 —— 這就**把 model-exploitation 迴圈閉掉了**：actor 想鑽 reward head 的漏洞，下一輪真資料就把漏洞補上。換句話說，**「持續餵真資料」不是工程細節，它就是這份承重契約在真機上的執行機制**。

## 6. 跨路線綜合

本 handbook 四條 generation 路線：**pixel-WM / latent-WM / diff-sim / neural-surrogate**。**Dreamer 是 latent-WM 路線在 embodied control 的 anchor** —— 它不出 pixel（RSSM decoder 會重建影像，但那是世界模型訓練的輔助 target，policy 不依賴生成像素），而是在 latent 空間想像出可直接執行的 action。

| Line | Dreamer 的對應位置 |
|---|---|
| pixel-WM（Cosmos / Sora / [Genie-2](../../foundations/latent-world-models/genie-2.md)） | 不走 pixel rollout。pixel-WM 強在人類可視化驗證、弱在 RL 內訓 compute efficiency；Dreamer 證明「latent rollout 就夠 agent control 用」。pixel-WM 可當 Dreamer 的「更真實 renderer」候選，但 streaming latency 是進 control loop 的瓶頸。 |
| **latent-WM（[DreamerV4](../../foundations/latent-world-models/dreamer-v4.md) / [V-JEPA-2](../../foundations/latent-world-models/v-jepa-2.md)）** | **本篇就是這條路在 general embodied 的代表。** RSSM imagination 出 action；**DreamerV4 是更強的世界模型 backbone**（block-causal transformer + offline）—— 但本篇**不重拆 V4 內部**，內部演化見 foundation 頁。 |
| diff-sim（Genesis / MJX / Aerial Gym） | 同屬 `sim-in-loop-train` 格子，但**物理來源不同**：diff-sim 是人寫的可微物理方程，Dreamer 的「sim」是**從資料學來的世界模型**。互補關係：可微 sim 可當 Dreamer 的高保真資料源 / 動力學先驗。 |
| neural-surrogate（FNO / GraphCast） | surrogate 解「給定 PDE 預測下一狀態」、無 action/reward/policy 概念；Dreamer 是 data-only 的 agentic WM。可組合：surrogate 當 simulator → Dreamer 在其上訓 policy。 |

**最關鍵的 cross-line 命題**：**Dreamer 把「latent-WM + 樣本效率」做成了 WM-as-policy 的範本 —— 它證明你可以用廉價的想像 rollout 換掉昂貴的真實互動，但這筆交換的價格寫在 §5 的承重契約裡（reward / continue / 短程動力學必須對 + 策略待在分布內）。**

- 與 [planning-and-trust-contract.md](./planning-and-trust-contract.md)：本篇的「承重契約」就是那篇「信任契約」在 **actor-critic-on-WM** 派的具體實例 —— 「想像 rollout 在什麼條件下值得信任」正是兩篇共享的問題。MPC-on-WM 派（TD-MPC / PETS）面對同一份契約，只是用 re-plan 而非 amortized actor 去守它。
- 與 [../../bridge-to-vla/world-model-as-policy.md](../../bridge-to-vla/world-model-as-policy.md)：Dreamer 是「WM rollout **即決策**」（而非「生成資料訓 policy」）這條 bridge 線的 anchor 之一。
- **與 aerial sibling [../aerial-sim/dream-to-fly.md](../aerial-sim/dream-to-fly.md)**：那篇是同一 DreamerV3 引擎在**無人機**上的案例（HIL / raw-pixel / 視覺 gap 被旁路的星號）；本篇是同引擎在**general embodied 真機**上的案例（DayDreamer / 無模擬器 / 承重契約）。**兩篇 sibling、不重複**：要看「latent-WM 飛無人機」去那篇，要看「latent-WM 的 WM-as-policy 抽象與真機落地」看本篇。
- ontology 5 軸定義：[../../cheat-sheet/ontology.md](../../cheat-sheet/ontology.md)。

## 7. 參考

**Primary**
- Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T. _Mastering Diverse Domains through World Models_ (DreamerV3). arXiv [2301.04104](https://arxiv.org/abs/2301.04104) —— RSSM + imagination + 三 head + 固定超參跨 150 任務 + Minecraft 鑽石的母本。
- Wu, P., Escontrela, A., Hafner, D., _et al._ _DayDreamer: World Models for Physical Robot Learning._ arXiv [2206.14176](https://arxiv.org/abs/2206.14176) —— [VALIDATED 真機、無模擬器]：四足 / 雙臂 / 輪式四台真機線上學。

**Same-repo dissections / anchors**
- WM-as-policy 兩派總圖：[overview.md](./overview.md)。
- 信任契約（同派抽象問題）：[planning-and-trust-contract.md](./planning-and-trust-contract.md)。
- 更強的 latent-WM backbone（V4 內部演化，本篇不重拆）：[../../foundations/latent-world-models/dreamer-v4.md](../../foundations/latent-world-models/dreamer-v4.md)。
- 同軸對手（self-supervised latent + action head）：[../../foundations/latent-world-models/v-jepa-2.md](../../foundations/latent-world-models/v-jepa-2.md)。
- WM-as-policy bridge：[../../bridge-to-vla/world-model-as-policy.md](../../bridge-to-vla/world-model-as-policy.md)。

**Sibling（aerial，DreamerV3 飛無人機 —— 不重複該案例）**
- [../aerial-sim/dream-to-fly.md](../aerial-sim/dream-to-fly.md)。

**Ontology**
- 5 軸定義：[../../cheat-sheet/ontology.md](../../cheat-sheet/ontology.md)。

## §8 踩坑日誌

| # | Severity | Issue | Source | Workaround |
|---|---|---|---|---|
| §8.1 | 🔴 | **Reward head / continue head 是承重件** —— 行為學習全靠想像回報、真實 reward 不進訓練迴路；兩 head 在 actor 探區估錯 → reward hacking（夢裡高分、真實不認） | DreamerV3 arXiv [2301.04104](https://arxiv.org/abs/2301.04104)（reward/continue head 機制） | **持續餵真資料**校準 head（DayDreamer 做法）；symlog + two-hot 穩 reward scale；勿假設世界模型全局正確，只需局部誠實（§5） |
| §8.2 | 🔴 | **Model exploitation 是範式固有張力**，非 bug —— actor 會去鑽世界模型的漏洞；Dreamer/MuZero 家族通病 | DreamerV3 / DayDreamer 範式分析；arXiv [2206.14176](https://arxiv.org/abs/2206.14176)（靠持續真資料閉迴路） | DayDreamer 的解：把真實互動持續餵回 replay buffer，把 model-exploitation 迴圈閉掉；策略探到新區就用新真資料補世界模型 |
| §8.3 | 🟠 | **Imagination horizon 拉長幾乎總是傷性能** —— T=16 是刻意設計；prior rollout 誤差累積，長 horizon 想像失真 | DreamerV3 arXiv [2301.04104](https://arxiv.org/abs/2301.04104)（horizon T=16）；長 horizon 退化的精確步數界 [UNVERIFIED] | 守住短 horizon（T=16 量級）；需更長時間尺度請評估 DreamerV4（context 更長）而非盲目拉 V3 horizon |
| §8.4 | 🟠 | **世界模型只在資料分布附近可信** —— actor 推到 OOD action / 沒見過狀態時 prior 與 reward head 同時失準 | DayDreamer arXiv [2206.14176](https://arxiv.org/abs/2206.14176)（持續餵真資料維持誠實） | 持續真資料注入把策略探區拉回分布內；真機尤其依賴此（無「reset 大法」可重採） |
| §8.5 | 🟡 | **Decoder 重建糊 ≠ policy 壞** —— RSSM decoder 是輔助 target，policy 品質看 reward curve 不看重建清晰度 | DreamerV3 decoder 為 recon 輔助 target（arXiv [2301.04104](https://arxiv.org/abs/2301.04104)） | 訓練中前段 decoder 常糊但 reward 已上升屬正常；別把重建模糊讀成沒學好 |
| §8.6 | 🟡 | **固定超參跨域靠四件套**（symlog / two-hot / free bits=1 nat / percentile return norm）—— 拆掉任一件，「免調參跨 150 任務」賣點就不成立 | DreamerV3 arXiv [2301.04104](https://arxiv.org/abs/2301.04104) | 復現時勿擅自移除穩健機制；reward scale 極端時雖號稱免調仍偶須看 KL / free-bits 曲線 |
| §8.7 | 🟡 | **DayDreamer per-task 對 model-free 的加速倍數未在本篇查證** —— 數字在原文但本篇不替未驗證數字背書 | DayDreamer arXiv [2206.14176](https://arxiv.org/abs/2206.14176)（正文有數字，本篇標 [UNVERIFIED]） | 引用具體加速倍數前回原文 §experiments 核對；勿從二手摘要轉述 |

**[UNVERIFIED] 標記彙總**：(1) §3 / §8.7 DayDreamer per-task 對 model-free 的具體加速倍數（數字在原文，本篇未逐一查證、不背書）；(2) §8.3 imagination horizon 超過多少步開始顯著退化的精確步數界（推論 T=16 是上界附近，論文未在本篇引用範圍內給出精確界）。所有其他數值（imagination horizon **T=16**、三 head = reward/continue/decoder、固定超參 **>150 任務**、6 size **12M→400M params 各單 A100**、Minecraft 從零無人類資料 **100M steps ~9 天單 GPU**、free bits **KL clip 1 nat**、DayDreamer 四足 **~1 小時學會走 / ~10 分鐘適應抗推 / 四台真機 / 無重置**）均出自 primary sources。
