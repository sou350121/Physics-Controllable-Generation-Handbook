# Bridge: Generative Data for VLA — World-Model Rollouts as Action Training Data

> **本倉 (Physics-Gen) = generation 端** · **VLA-Handbook = action 端**
> **交界問題**：生成的 video / latent / sim rollout，何時能**替代真實 demo** 訓出可用的 policy？

**Status:** v1 — opinionated draft. Grounded in [`foundations/foundation-physics-models/cosmos-wfm.md`](../foundations/foundation-physics-models/cosmos-wfm.md)、[`foundations/latent-world-models/v-jepa-2.md`](../foundations/latent-world-models/v-jepa-2.md)、[`use-cases/robotics-data-gen/overview.md`](../use-cases/robotics-data-gen/overview.md). 跨倉 benchmark 數字未直接 grounding 者標 `UNVERIFIED`.

**TL;DR:** VLA 訓練最大瓶頸是**真實 demo 太貴**（每條軌跡要真機 teleop）。Generation 端想用 world-model rollout 當合成資料補。但「video 看起來合理」≠「action 可信」。核心 contract：generation 端必須交付**帶 grounded action label、座標系對齊、且物理一致（尤其 contact）** 的軌跡；VLA 端必須能容忍 generation 端的 distribution shift。最關鍵的單一證據是 [Cosmos-Policy](../foundations/foundation-physics-models/cosmos-wfm.md)（video FM 直接當 policy backbone，LIBERO/RoboCasa 超越 from-scratch baseline，**無架構修改**）與 [V-JEPA-2-AC](../foundations/latent-world-models/v-jepa-2.md)（1M h internet video pretrain + 62 h Droid → zero-shot 真機 Franka）——兩者證明 generated/pretrain signal **能**幫 action，但都撞上同一堵牆：**contact-rich physics 與 closed action vocabulary**。

---

## 1 · 三條合成路線（generation 端能交付什麼）

| 路線 | 代表 | action label 怎麼來 | 物理保證 | 適合 |
|---|---|---|---|---|
| **純 video gen** | [Cosmos-Predict](../foundations/foundation-physics-models/cosmos-wfm.md) / Sora robot variant | **沒有原生 action** → 要 inverse-dynamics 反推 | implicit（data-only，contact 會 silent fail） | 視覺多樣性、photoreal 增廣 |
| **action-conditioned WM** | [V-JEPA-2-AC](../foundations/latent-world-models/v-jepa-2.md) / [Genie-2](../foundations/latent-world-models/genie-2.md) / [DreamerV4](../foundations/latent-world-models/dreamer-v4.md) | **原生 action token**（grounded） | latent rollout，物理隱含在 representation | imitation / RL，action 直接可用 |
| **sim-augmented gen** | Genesis / Isaac + domain randomization | sim 給 **ground-truth** action + force | 顯式（physics engine，含 contact） | sim2real，force-rich task |

關鍵差異在 **action label 的 grounded 程度**：純 video gen 的 action 是**事後反推**（inverse dynamics 的誤差直接污染 label）；action-conditioned WM 的 action 是**生成時的條件**（grounded，但限於訓練過的 action vocabulary）；sim 的 action 是 engine **算出來的真值**（最可信，但 sim2real visual gap 要靠 randomization 補）。

---

## 2 · 兩端契約（interface table）

| 契約欄位 | Physics-Gen 端（generation）保證 | VLA 端（action）需要 | seam |
|---|---|---|---|
| **Action label** | 純 video：無，需 inverse-dynamics 反推；WM：原生 token；sim：ground-truth | 必須有與 policy action space 對齊的 label | 🔴 純 video 路線的 label 是反推來的，誤差不可控 |
| **Coordinate frame** | rollout 的 camera / world frame 約定 | policy 的 obs frame 與 action frame | ⚠ camera-frame vs world-frame mismatch 是 #1 整合 bug |
| **Action vocabulary** | WM 限於訓練過的 gripper / object / camera 分布 | 部署環境的 action 可能 OOD | 🔴 V-JEPA-2-AC 的「Droid-shaped WM」天花板：62h Droid 分布外無保證 |
| **Contact / force** | pixel-video & latent-WM 都**不可信**（contact silent fail）；只有 sim 有 force | 精細接觸任務（grasp / 插拔 / 布料）需 force fidelity | 🔴 結構性 break：generated data 對 contact task 不可單獨用 |
| **Distribution match** | rollout 分布 = 生成模型分布，非真實機器人分布 | policy 部署在真實分布 | ⚠ sim2real / gen2real gap，需真實 demo 混合校準 |
| **Inference cost (若 in-loop)** | Cosmos ~4 min/clip；V-JEPA-2-AC ~16 s/action | closed-loop 需 ≤10–100 ms | 🔴 generation 太慢，只能 **out-of-loop data engine**，不能 in-loop |

---

## 3 · 什麼讓 generated data 對 action 「可信」

「video 合理」是**視覺**判準；「data 可信」是**因果 + 物理**判準。三道閘門：

1. **Action 是 grounded 還是反推的？** 反推（inverse dynamics）會把 video 的視覺誤差翻譯成錯誤 action label，policy 學到的是「錯誤但自洽」的對應。Action-conditioned WM / sim 的 grounded label 沒這個問題——這是為什麼 [Cosmos-Policy](../foundations/foundation-physics-models/cosmos-wfm.md) 走「token 預測頭直接做 visuomotor」而**不額外加 action head**（paper 明列加了反而退化）。

2. **Contact phase 對不對？** Pixel-video / latent-WM 的物理是 implicit——抓取 / 接觸**視覺上合理但 force phase 崩**（Cosmos anchor §9.2、多筆社群 reproduction 報告）。對 grasp / 插拔 / 變形物這類 contact-rich task，generated data **不能單獨用當訓練資料**；要嘛混真實 demo，要嘛與 diff-sim 的 contact label 對齊。

3. **分布有沒有覆蓋部署場景？** V-JEPA-2-AC 證明少量真機 video（62 h）就能把 internet-video pretrain 釘到 zero-shot Franka（grasp 65–75% vs Octo full-Droid behavior-clone 15% `UNVERIFIED 跨倉數字`）——但這是**在 Droid 分布內**。一旦 gripper / object / camera 出分布，整個 stack 無保證。**Generated data 擴的是 pretrain 廣度與 long-tail，不是替你保證部署分布**。

**判準（honest version）**：generated data 在 **long-horizon、多 embodiment 泛化、long-tail 場景、視覺增廣** 上贏；真實 demo 在 **精細接觸、sim2real gap 大、安全攸關** 的場景贏。最佳實務是 **Pareto 混合**（純合成 → 合成+少量真實 → 純真實），不是二選一。具體混合比的 ablation 仍是 open（見 [`use-cases/robotics-data-gen/overview.md`](../use-cases/robotics-data-gen/overview.md) dissection wishlist）。

---

## 4 · 兩條已驗證的 anchor pattern

- **Cosmos-Policy（video FM → policy backbone）**：Predict2-2B 單階段 SFT，8×A100/H100 一晚，LIBERO / RoboCasa 超越 from-scratch diffusion policy / VLA baseline，**不加 action head**。意義：generated-data 不只是「多餵幾條軌跡」，而是 **video FM weights 本身**可當 VLA 的 pretrain 起點。但 grasp / 接觸 task 不要單獨用 Cosmos rollout 當 train data。具體 success-rate 數字 `UNVERIFIED`（Cosmos anchor §5 待升 v1 補）。
- **V-JEPA-2-AC（latent WM → zero-shot 真機）**：latent-prediction（非 pixel reconstruction）路線**唯一有真機 demo 撐場**的證據。inference ~16 s/action（vs Cosmos ~4 min/action，~15× 快）。但 closed action vocabulary、無 language goal、camera 敏感——「Droid-shaped WM」不是 generalist。

兩條都指向同一結論：**generated/pretrain signal 確實能幫 action，但天花板是 contact fidelity 與 action vocabulary，不是資料量**。

---

## 5 · 開放 seam（未解）

- **🔴 Contact label co-training 還沒落地**：把 diff-sim（Genesis / MJX）的 force 信號當 auxiliary loss 餵 video-FM post-train，是補 contact silent-failure 最有希望的路（Cosmos anchor §7.1 predict #1），但截至錨點寫作時尚無公開 end-to-end 實作。
- **🔴 合成/真實混合比沒有定論**：RoboCasa-style 合成 vs 真實 demo 的 Pareto 曲線、PI data engine 的 sim/gen/real 比例，都還是經驗值而非可複現 ablation（[`use-cases/robotics-data-gen/overview.md`](../use-cases/robotics-data-gen/overview.md) wishlist）。
- **⚠ Action vocabulary 開放化**：所有 action-conditioned WM 都困在 closed vocabulary。open-vocab action 的 world model（language-conditioned AC、multi-cam）是下一代必須，VLA-Handbook 與本倉都需對齊這條。
- **⚠ Inference 太慢無法 in-loop**：generation 目前只能當 **out-of-loop data engine**。要當 in-loop simulator（policy 在 WM 裡 rollout 決策），latency 還差 2–4 個數量級——這條歸 [`bridge-to-vla/world-model-as-policy.md`](./world-model-as-policy.md)，與本篇「WM 當資料源」是不同範式。

---

## Boundary

- 「WM rollout 即決策」（policy 在 WM 裡 plan，非當資料源）→ [`bridge-to-vla/world-model-as-policy.md`](./world-model-as-policy.md)
- 「video pretrain → latent embedding → action head」的 backbone 取捨 → [`bridge-to-vla/video-pretraining-for-action.md`](./video-pretraining-for-action.md)
- Generation 端單篇 anchor → [`foundations/foundation-physics-models/cosmos-wfm.md`](../foundations/foundation-physics-models/cosmos-wfm.md) · [`foundations/latent-world-models/v-jepa-2.md`](../foundations/latent-world-models/v-jepa-2.md)
- Robotics data-gen use case → [`use-cases/robotics-data-gen/overview.md`](../use-cases/robotics-data-gen/overview.md)
- Action-head / policy-training / 真機 success-rate 側 → VLA-Handbook [`theory/03-engineering/`](https://github.com/sou350121/VLA-Handbook/tree/main/theory/03-engineering) · [`embodiments/`](https://github.com/sou350121/VLA-Handbook/tree/main/embodiments)

## References

- Cosmos Policy — *Fine-Tuning Video Models for Visuomotor Control and Planning*. 2026-01 · [arXiv:2601.16163](https://arxiv.org/abs/2601.16163)
- Cosmos World Foundation Model Platform — 2025-01 · [arXiv:2501.03575](https://arxiv.org/abs/2501.03575)
- V-JEPA 2 / 2-AC — Assran, Bardes, Ballas, LeCun et al. (Meta FAIR). 2025-06 · [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
- Octo (full-Droid behavior-clone baseline，跨倉對照數字 `UNVERIFIED`) — 詳見 V-JEPA-2 anchor §X-Ray.
