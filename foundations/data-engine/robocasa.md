<!-- ontology-5axis output=N/A injection=sim-in-loop-train control=action|trajectory|image-init temporal=streaming domain=robotics -->

# RoboCasa — Anchor Dissection

> Large-Scale Simulation of Everyday Tasks for Generalist Robots
> Nasiriany, Maddukuri, Zhang, Parikh, Lo, Joshi, Mandlekar, Zhu — **RSS 2024** — arXiv 2406.02523
> 與 Open X-Embodiment（Padalkar et al., ICRA 2024, arXiv 2310.08864）、DROID（Khazatsky et al., RSS 2024, arXiv 2403.12945）並列為 2024 機器人 data-engine 三條主線。

---

## 1. TL;DR

RoboCasa 不是一個生成模型，而是一條 **「sim → demo → policy」的 data engine pipeline**——把廚房這個 long-tail domain 做透：**2,500 個 AI 生成廚房場景、3,200+ 物件、150+ 類別、365 個任務、600+ 小時人類遙操 + 1,600+ 小時 synthetic demo**。它示範了單任務 sim（PickCube, BlockStack 那一票）做不到的事——**domain depth × asset diversity × task taxonomy** 同時放大，把 imitation learning 的 scaling law 從 "more demos of same task" 推到 "more tasks × more scenes × same skill family"。

相對於 DROID（real-data，350h，76k traj，564 場景）走「真實資料堆量」、Open-X（22 embodiments，527 skills，160k tasks，21 機構聚合）走「跨 embodiment 對齊」，RoboCasa 走的是 **「sim 端把 long-tail 廚房窮舉」**——這是 Cosmos / GR00T 之類 foundation 端最缺的 "controllable curriculum" 燃料。

**為什麼進 anchor 名單**：它是 2024 後最常被 cite 為「robot data engine」canonical example 的工作，連 NVIDIA Cosmos / Isaac GR00T 的 data pipeline blog 都把 RoboCasa-style synthetic demo 當成 video pre-train 的 long-tail fodder。

---

## 2. Core mechanism

RoboCasa 建在 **RoboSuite / MuJoCo** 之上（parent ecosystem，Mandlekar / Zhu 同一個 lab line），但把場景生成從手刻 USD/URDF 換成 **三條 generative 來源**：

- **3D assets**：Objaverse 1.0 + LightWheel AI + **Luma AI text-to-3D** → 3,200+ 物件
- **Textures**：**MidJourney** 生成 400 張（100 wall / 100 floor / 100 counter / 100 cabinet panel）
- **Tasks**：**GPT-4** guided task taxonomy → 365 tasks，其中 65 個 atomic skill tasks（pick-place / door / drawer / knob / lever / button / insert / nav / rack / lid）

```
       ┌─────────────────────────────────────────────────────────┐
       │  ASSET LAYER (generative)                                │
       │   text→3D (Luma)    text→tex (MidJourney)   Objaverse   │
       │       │                  │                      │       │
       │       └──── 3,200 obj × 150 cat ─── 400 tex ────┘       │
       └───────────────────────────────┬─────────────────────────┘
                                       ▼
       ┌─────────────────────────────────────────────────────────┐
       │  SCENE COMPOSER                                          │
       │   procedural kitchen layout × 2,500 scenes               │
       │   furniture / appliances / clutter randomization         │
       └───────────────────────────────┬─────────────────────────┘
                                       ▼
       ┌─────────────────────────────────────────────────────────┐
       │  TASK LAYER (GPT-4 curated)                              │
       │   365 tasks = 65 atomic × composition                    │
       └───────────────────────────────┬─────────────────────────┘
                                       ▼
       ┌─────────────────────────────────────────────────────────┐
       │  DEMO LAYER                                              │
       │   600h human teleop ─►  policy seed                      │
       │                          │                               │
       │                          ▼                               │
       │   1,600h synthetic ──── MimicGen-style replay/aug        │
       └───────────────────────────────┬─────────────────────────┘
                                       ▼
                            VLA / BC-Transformer training
```

關鍵是最後一段——**human teleop 是 seed，synthetic demo 是放大**。RoboCasa 引用 MimicGen 思路：少量人類 demo + 大量場景隨機化 → replay 出 trajectory 級的合成資料。這條路是 πSlot、Octo、OpenVLA 後續工作吃 RoboCasa 的主因。

---

## 3. 五軸定位 + 同軸對手

| 軸 | RoboCasa |
|---|---|
| output | **N/A**（不產 pixel/video，產 trajectory + state） |
| injection | **sim-in-loop-train**（policy 訓練端 inject，非 inference 端） |
| control | **action / trajectory / image-init**（廚房場景初始化 + scripted/teleop action） |
| temporal | **streaming**（episodic rollout） |
| domain | **robotics**（廚房 long-tail） |

**同軸對手**：

- vs [Genesis](../differentiable-simulators/genesis.md)：Genesis 是 general-purpose differentiable sim（流體 / 軟體 / rigid 通吃），RoboCasa 是 **domain-specific kitchen content pack** + RoboSuite/MuJoCo backend。互補：Genesis 提供 physics 廣度，RoboCasa 提供 task taxonomy 深度。
- vs [MuJoCo MJX](../differentiable-simulators/mujoco-mjx.md)：MJX 是 batched GPU physics engine（底層），RoboCasa 是上層 content + task DSL。Issue [#174 "Mujoco mjx"](https://github.com/robocasa/robocasa/issues/174) 就是社群在問什麼時候 RoboCasa 會切到 MJX 後端拿 100× throughput。
- vs **DROID**（arXiv 2403.12945）：DROID 走真實資料路線——350h × 76k traj × 564 scene × 50 collectors × 12 months，硬剛 distribution shift；RoboCasa 走 sim 路線，**scale 比 DROID 大一個量級但有 sim2real gap**。實務上兩者是混用：DROID 當 distribution anchor，RoboCasa 補 long-tail。
- vs **Open X-Embodiment**（arXiv 2310.08864，ICRA 2024）：Open-X 是 22 embodiment × 21 機構 × 527 skill × 160k task 的 **跨機構真實 demo 聚合**；RoboCasa 是單機構單 embodiment（Franka 為主）的 **sim 端窮舉**。Open-X 解 "transfer across robots"，RoboCasa 解 "transfer across kitchens"。

---

## 4. ⚡ shines / ❌ breaks

**⚡ shines**

- **Kitchen domain depth**：別人做 BlockStack，RoboCasa 做 lid / knob / drawer / lever 一整套 kitchen-affordance taxonomy。VLA 訓練的時候這套 affordance 比 PickCube 100k 條更值錢。
- **AI asset diversity**：Luma + MidJourney + Objaverse 三來源混合，texture/geometry diversity 比手刻 USD 高一個量級——對 sim2real domain randomization 是直接的彈藥。
- **RoboSuite parent ecosystem**：繼承 robomimic 評估、MimicGen 資料增強、teleop infra；新人不用重造輪子。
- **Scaling trend 證據**：paper 直接 plot synthetic demo 數量 vs policy success rate 曲線，這條曲線是後續所有 robot-data-engine 工作的對標基準。

**❌ breaks**

- **只有廚房**——不是 general-purpose data engine，要做 warehouse / outdoor / surgical 場景請另尋。
- **Sim2real gap 仍在**：paper 自己也說 sim2real 需要 domain randomization；texture diversity 有幫助但不消除 dynamics gap（手指摩擦、軟體變形、流體）。
- **MuJoCo 後端不是 GPU-batched**（未切 MJX，見 issue #174）：throughput 受限，1,600h synthetic 是長時間累積出來的，不是一夜跑完。
- **Pip 安裝痛**（[issue #177](https://github.com/robocasa/robocasa/issues/177) "Robocasa is not pip installable except if cloned locally"）：對 CI/CD pipeline 不友善。
- **Robomimic eval 偶爾 fail**（[issue #166](https://github.com/robocasa/robocasa/issues/166) "Robomimic eval rollouts fail on RoboCasa dataset"，env class detection 報 "code should never reach this point"）。

---

## 5. Reproduction

- **Paper**：arXiv 2406.02523（RSS 2024）
- **Project page**：https://robocasa.ai/
- **Code**：https://github.com/robocasa/robocasa
- **Dataset (RoboCasa365)**：2,500 scenes / 365 tasks / 600h teleop + 1,600h synthetic，從 project page 下載
- **Parent**：RoboSuite（github.com/ARISE-Initiative/robosuite） + MimicGen（github.com/NVlabs/mimicgen）
- **典型訓練流程**：clone repo → install MuJoCo 2.3+ → 下載 dataset → robomimic BC-Transformer baseline → 評估在 65 atomic task 的 success rate

---

## 6. Cross-line synthesis

**RoboCasa × Cosmos**：NVIDIA Cosmos data pipeline（[Cosmos-WFM](../foundation-physics-models/cosmos-wfm.md)）的 robotics video pre-train 燃料有相當比例是 RoboCasa-style 合成 demo——廚房 long-tail 的視覺多樣性正是 web video 缺的。Cosmos 把 RoboCasa 輸出當 "controllable video curriculum"，再透過 [Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md) 蒸餾出可控 world model。這就是 **Sim → Gen → Real** 閉環的具體實例（見 [data-engine overview](./overview.md) 的三層管道）。

**RoboCasa × Open-X**：實務上 VLA / OpenVLA / π0 訓練配方是 **Open-X 當 base（跨 embodiment 廣度）+ DROID 當 in-the-wild anchor + RoboCasa 當 domain depth booster**——三條線各補一塊，沒有單一資料集能取代另外兩條。

**RoboCasa × GR00T**：Isaac GR00T blueprint 把 RoboCasa-style synthetic demo 作為 humanoid manipulation 的早期 curriculum，再切到 Isaac Lab 的全身運動學場景。

---

## 7. References

- **RoboCasa**：Nasiriany et al., "RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots", RSS 2024, arXiv 2406.02523. https://arxiv.org/abs/2406.02523
- **RoboCasa project**：https://robocasa.ai/
- **GitHub**：https://github.com/robocasa/robocasa
- **DROID**：Khazatsky et al., "DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset", RSS 2024, arXiv 2403.12945. https://arxiv.org/abs/2403.12945
- **Open X-Embodiment**：Padalkar et al. (Open X-Embodiment Collaboration), "Open X-Embodiment: Robotic Learning Datasets and RT-X Models", ICRA 2024, arXiv 2310.08864. https://arxiv.org/abs/2310.08864
- **Parent ecosystem**：RoboSuite（Zhu et al., CoRL 2020）、MimicGen（Mandlekar et al., CoRL 2023）

---

## 8. §8 Pitfall log

來自 GitHub issues 與社群實測：

- **[#174 "Mujoco mjx"](https://github.com/robocasa/robocasa/issues/174)** — 社群希望切 MJX 後端拿 GPU batched throughput；目前還是 CPU MuJoCo，1,600h synthetic 累積成本高。
- **[#177 "Robocasa is not pip installable except if cloned locally"](https://github.com/robocasa/robocasa/issues/177)** — 安裝必須本地 clone，CI/CD 不友善。
- **[#186 "Solution for Parallel environment and VLA Integration"](https://github.com/robocasa/robocasa/issues/186)** — 平行 env + VLA inference loop 還在拼接，沒有 first-party 解。
- **[#173 "About VLA integrations for RoboCasa"](https://github.com/robocasa/robocasa/issues/173)** — OpenVLA / Octo / π0 接入需要外部 wrapper。
- **[#169 "env._check_success() return False when regenerate dataset"](https://github.com/robocasa/robocasa/issues/169)** — dataset regenerate 後 success 條件判錯，引起 silent label noise（**對 imitation learning 殺傷力大**）。
- **[#166 "Robomimic eval rollouts fail on RoboCasa dataset"](https://github.com/robocasa/robocasa/issues/166)** — env class detection 報 "code should never reach this point"——robomimic 與 RoboCasa 的 env registry 對齊問題。
- **Sim2real failure mode（社群報告）**：domain randomization 不夠時，AI 生成 texture 反而引入 "too perfect" artifact，policy 在真實 cluttered scene 反而退化——這是 sim asset 過度規整的反向 overfit。
- **Long-horizon task drift**：365 tasks 中組合型任務（multi-step recipe execution）的 success rate 顯著低於 atomic skill；composition gap 仍未解。

---

**TBD**

- [ ] RoboCasa 與 [Cosmos-WFM](../foundation-physics-models/cosmos-wfm.md) 實際資料管道對接細節（NVIDIA 內部 blog 確認版）
- [ ] MJX 後端切換的 throughput 對比數字（待 issue #174 落地）
- [ ] RoboCasa365 vs 原版 RoboCasa 的 policy scaling law 對比曲線
- [ ] 與 [robotics-data-gen overview](../../use-cases/robotics-data-gen/overview.md) sister anchor 的 cross-reference 補完
