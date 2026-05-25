<!-- ontology-5axis output=latent|pixel-video injection=implicit-from-data control=action|image-prompt temporal=temporal-transformer-rolling domain=generalist -->

# Genie 2 — A Large-Scale Foundation World Model (DeepMind, Dec 2024)

## 1. One-paragraph TL;DR

Genie 2 是 DeepMind 在 2024-12-04 公佈的「foundation world model」：給一張 image prompt + 連續鍵盤滑鼠 action，模型 autoregressively 生成最長約 1 分鐘的可互動 3D 視訊環境。它把 Genie 1 (arxiv 2402.15391, Feb 2024) 「從 unsupervised internet video 學出 latent action」的 LAM 範式從 2D 平台跳到 generalist 3D world，主打 **action conditioning native + 短記憶物件持續性**（離開視野再回頭、東西大致還在）。存在的價值是把「video generation」與「agent-control playground」這兩條原本各做各的線焊在一起 — Sora/Veo 不收 action，DreamerV3/V4 不收像素 prompt，Genie 2 兩邊都收。Prior gap = 「scalable, generalist, action-controllable interactive video」。

## 2. Core mechanism

公開資料只給到 block-diagram 等級（無 paper、無 weights）：

```
 image prompt ──┐
                ▼
       ┌──────────────┐    latent frames    ┌────────────────────┐
       │  autoencoder │ ───────────────────▶│ transformer        │── next latent
       └──────────────┘                     │ dynamics (causal   │   frame
                                            │ mask, KV cache)    │
 action (kb/mouse) ─────────────────────────▶                    │
                                            └────────────────────┘
                                                       │
                                            classifier-free guidance
                                                       │
                                                       ▼
                                                  AE decoder ──▶ pixels
```

要點：

- **Autoregressive latent diffusion**：在 latent space 一幀一幀往前 sample；不是一次 joint-rollout 整段 clip。
- **Causal-mask transformer dynamics**：跟 LLM 同 family；KV cache 讓滾動 inference 可行（這是把它放在 `temporal=temporal-transformer-rolling` 而不是 `latent-rollout` 的關鍵）。
- **Latent action 沿用 Genie 1 LAM**：Genie 1 的 LAM 用 ST-transformer encoder 從 (frame_t, frame_{t+1}) 推 latent action token（codebook ≤ 8 actions, fully unsupervised）。Genie 2 沒明說 codebook 大小，但保留「latent action interface」這層抽象 — 訓練時不需要真實 action label，inference 時把鍵盤滑鼠映到同一 latent action 空間。
- **Classifier-free guidance**：為了讓 action 真的「踩得動」物件，inference 時用 CFG 拉開 conditional/unconditional 預測。
- **Distilled real-time variant**：blog 提到有 distilled 版本可即時跑，代價是品質下降；undistilled 才是 demo 中那條「漂亮但離線」的版本。

## 3. 五軸定位 + 同軸對手

| Axis | Value | 說明 |
|---|---|---|
| Output | `latent\|pixel-video` | latent space rollout，decode 出 pixel 才能 demo |
| Injection | `implicit-from-data` | 純靠大型 video dataset 隱式學物理；無 PDE/contact loss |
| Control | `action\|image-prompt` | 鍵鼠 action token + 起始幀 |
| Temporal | `temporal-transformer-rolling` | causal-mask transformer + KV cache 滾動推 |
| Domain | `generalist` | 不鎖 driving/robotics/Minecraft；demo 涵蓋第三人稱、第一人稱、水/植被/角色 |

同軸對手（皆 latent/rolling-window WM）：

| 競爭方法 | 五軸差異 | 對撞點 |
|---|---|---|
| **Decart Oasis** (Oct 2024) | domain=`generalist`(實際偏 Minecraft) · 即時 30fps · Etched 自研硬體 | Genie 2 generality 高，Oasis 速度高 — 工程取捨對撞 |
| **[V-JEPA-2](./v-jepa-2.md)-action** (2025) | output=`latent` (不 decode) · control=`action` · domain=`robotics` | V-JEPA-2 拒絕 decode 像素 (representation-only)，Genie 2 必須 decode 才能 demo |
| **[DreamerV4](./dreamer-v4.md)** (2025) | output=`latent` · temporal=`latent-rollout` · 配 actor-critic | Dreamer 為了 RL agent；Genie 2 為了人玩 + agent 訓練料 |
| **[Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md)-action** (NVIDIA, 2025) | output=`pixel-video` · injection=`implicit-from-data` · temporal=`hierarchical` | Cosmos hierarchical 換 long-horizon，Genie 2 用 KV cache 換 interactivity |

## 4. Where it shines / where it breaks

### ⚡ Shines

- **Action conditioning native**：跟 Sora/Veo 在「拿來訓 agent」這場景上有質差 — 後者不收 action，只能事後加 hindsight labeler。
- **物件短期記憶**：blog 強調可重建「短暫離開視野」的物件，這是 transformer-rolling + 足夠 context window 的好處。
- **Counterfactual rollout**：同 initial image，不同 action 序列 → 不同結果，這是純 video generator 給不了的。
- **Generalist visual diversity**：demo 含水反射、植被擺動、第三/第一人稱切換，超過任何 Minecraft-only WM 的視覺多樣性。
- **生 agent 訓練料**：DeepMind 自家定位是「endless variety of action-controllable environments for training and evaluating embodied agents」— 這是把 WM 當 data engine 而非終端產品的姿勢。

### ❌ Breaks

- **長時 drift**：~10–20s 大部分樣本品質可看；越過 1 分鐘後物理鬆散、物件 morph、空間不一致。autoregressive 機制 + implicit physics 雙重放大誤差。
- **物件 identity 飄移**：因為「不知道牆是牆，只是 predict 一面牆的像素」，鏡頭轉回來時顏色、位置可能換。
- **物理一致性脆**：重力、剛體碰撞、流體只在訓練分布內勉強撐；OOD prompt（如「打開冰箱」）失敗率高 [TBD: verify systematic eval]。
- **未即時 / 即時版品質掉**：undistilled 不能 real-time；distilled real-time 版 blog 自承「a reduction in quality」。
- **完全閉源 + 無 paper**：沒 weights、沒 arxiv、沒 codebook size、沒 dataset 揭露 — community 無法 ablate。
- **被自家迭代壓過**：Genie 3 (2025-08-05) 直接給 24fps real-time, 720p, 數分鐘長度，並把「persistent memory」當成 emergent 賣點 — Genie 2 的角色基本變成「過渡里程碑」。

## 5. Reproduction notes

**官方狀態**：閉源。Limited Research Preview，未公開 weights / code / dataset / 模型大小 / 訓練 compute。

**Public 可用替代**：

- **Genie 1 (arxiv 2402.15391)**：paper 公開，無官方 weights，但社群有 reference 實作（如 1x-technologies/genie，多為玩具 size 200M–1B）。可以重建 LAM + ST-transformer dynamics + MaskGIT 三件套，跑 2D Platformer 等級 demo。
- **Oasis (Decart/Etched)**：對應 Genie 2 級別，weights 部分釋出；硬體強相關（Etched Sohu / 大 H100 cluster）。
- **MineWorld** (arxiv 2504.08388, 2025-04)：open-source real-time interactive Minecraft WM，可當作 Genie 2 在受限 domain 的 ablate 平台。

**最小可跑 setup（Genie 1-style reproduction）**：8× A100/H100；datasets = Platformer / Atari / OpenX (250–500h)；訓 LAM (~50M) + dynamics (~200M–1B)；典型踩坑：
1. LAM codebook collapse（latent action 退化到全用同一 code）— 用 commitment loss + 小 codebook (≤ 8) 抑。
2. Dynamics teacher-forcing → autoregressive inference 的 exposure bias。
3. KV cache + classifier-free guidance 同用時，cond/uncond 兩條 stream 的 cache 要分開維護。

## 6. Cross-line synthesis

| 路線 | 怎麼接 |
|---|---|
| **Pixel-WM** ([Sora](../video-world-models/sora.md)/[Veo](../video-world-models/veo.md)/[Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md)) | Pixel-WM 不收 action，但 Genie 2 的 LAM 可當「action labeler」反向給 pixel-WM 補 action label，再 distill 出 action-conditioned pixel-WM |
| **Latent-WM** ([Dreamer](./dreamer-v4.md) / [V-JEPA-2](./v-jepa-2.md)) | Dreamer 走 latent rollout + actor-critic，Genie 2 走 latent rolling-window + 人玩；可用 Genie 2 當「環境」、Dreamer 當「agent」配對訓練 |
| **Diff-sim** ([Genesis](../differentiable-simulators/genesis.md) / Brax / [MuJoCo MJX](../differentiable-simulators/mujoco-mjx.md)) | Diff-sim 有 ground-truth physics 但視覺差；可把 Genie 2 當「visual renderer」、diff-sim 當「dynamics engine」做混合 — 但目前無公開實作 |
| **Neural surrogate** ([FNO](../neural-surrogates/fno.md) / [GraphCast](../neural-surrogates/graphcast.md)) | 不直接接；surrogate 走場域物理，Genie 2 走 generalist 視覺，跨域只在「都是 implicit physics learner」這層哲學對話 |

Composition pattern：**Genie 2-as-data-engine** — 把它當作生 (image, action, next_image) triplet 的 infinite source，下游接 VLA / IL / RL，這是 DeepMind blog 本身主打的 usage，比 Sora 直接拿來播給人看更接近「生產用途」。

## 7. References

**Canonical**：

1. [Genie 2: A large-scale foundation world model — Google DeepMind blog (2024-12-04)](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/)
2. [Bruce et al., "Genie: Generative Interactive Environments" (arxiv 2402.15391, Feb 2024)](https://arxiv.org/abs/2402.15391) — Genie 1 paper，唯一公開的架構細節來源
3. [Genie 3: A new frontier for world models — DeepMind blog (2025-08-05)](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) — 後繼者，反推 Genie 2 限制

**Secondary**：

4. [MarkTechPost — "An Autoregressive Latent Diffusion Model" (2024-12-04)](https://www.marktechpost.com/2024/12/04/google-deepmind-introduces-genie-2-an-autoregressive-latent-diffusion-model-for-virtual-world-and-game-creation-with-minimal-input/) — autoregressive + CFG 細節
5. [Simon Willison — Genie 2 notes (2024-12-04)](https://simonwillison.net/2024/Dec/4/genie-2/) — 第三方第一時間 takeaway
6. [Cosmo-Edge — Genie 60-second limit technical analysis](https://cosmo-edge.com/project-genie-60-second-limit-technical-analysis/) — State drift 機制分析
7. [Ben Dickson — A critical look at Genie 3 (2025-08)](https://bdtechtalks.substack.com/p/a-critical-look-at-deepminds-genie) — Genie 3 critique 反推 Genie 2 gap

## 8. §8 Pitfall log

> 因 Genie 2 閉源，本節 pitfalls 以 DeepMind blog 自承 + 二手實測為主；無 GitHub issue 可引。

### §8.1 State Drift — physics dissolution beyond 1 min

- **Source**：DeepMind blog 原文："can generate consistent worlds for up to a minute, with the majority of examples shown lasting 10-20s"；Cosmo-Edge 二手分析「micro-errors compound into State Drift」
- **Severity**：HIGH — 是 1-min hard cap 的直接成因
- **Mechanism**：autoregressive loop + implicit-from-data → 每幀微小機率偏差 = 下一幀的訓練分布 OOD
- **Workaround**：縮短 episode 長度、reset frequently；長 horizon 改走 hierarchical (Cosmos-Predict) 或 latent rollout (DreamerV4)

### §8.2 Object identity flip on re-entry

- **Source**：DeepMind blog 強調「retains information for extended periods」是賣點 → 反證默認失效；二手「doesn't 'know' a wall is there」
- **Severity**：MEDIUM — demo 挑選的 case 通過，但邊界外不保證
- **Workaround**：用 3D-scene WM (World Labs / GaussianAnything) 顯式存幾何；或加 retrieval-augmented memory token

### §8.3 Distilled real-time = quality drop

- **Source**：DeepMind blog 原文「a reduction in quality of the outputs」
- **Severity**：MEDIUM — 即時版不是 demo 看到那條
- **Workaround**：兩條 pipeline（offline 高品質生 data、real-time 低品質給 agent inference）

### §8.4 OOD prompt → physical inconsistency

- **Source**：[TBD: verify with DeepMind paper if released] — blog 未列具體 failure modes；二手分析推測 OOD 物件交互（容器開合、流體溢出）失敗率高
- **Severity**：HIGH for downstream agent training（training distribution leakage）
- **Workaround**：對下游 agent 加 dynamics-consistency filter，丟掉違反守恆的 rollout

### §8.5 完全閉源 → 無法 ablate

- **Source**：無 paper、無 weights、無 dataset card
- **Severity**：HIGH for academic reproduction；MEDIUM for industry（可用 Oasis / MineWorld 替代）
- **Workaround**：Genie 1 paper + MineWorld (arxiv 2504.08388) 重建工程細節

### §8.6 被 Genie 3 取代

- **Source**：DeepMind 2025-08-05 Genie 3 blog
- **Severity**：LOW（學術價值仍在），HIGH（生產選型應直接看 Genie 3）
- **Note**：Genie 3 號稱 720p / 24fps / 數分鐘 / emergent persistent memory；但同樣閉源 + Research Preview，trade-off 未變
