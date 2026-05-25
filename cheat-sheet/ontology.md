# Ontology v2.0 — 5-axis taxonomy of Physics-Controllable Generation

**版本**：v2.0 · 2026-05-26 · 採納 [v1.1 review](./ontology-v1.1-review.md) 全部改動，並補一條 cross-axis 強制檢查。
**前版**：v1.0 (2026-05-25) — 見 git history。
**核心動機**：v1 把 `mesh`/`3d-scene` 重疊、`latent` 一詞兩用、`score-conditioned` 與 `constraint-loss` 概念糾纏 — v2 拆乾淨並對齊 2025 前沿（V-JEPA 2 · GAIA-2 · NewtonGen · Force Prompting · PhysDiff · Generative Gaussian Splatting）。

每篇 dissection 頂部必附：

```html
<!-- ontology-5axis
     output=<value>[|<value>...] injection=<value>[|<value>...]
     control=<value>[|<value>...] temporal=<value>[|<value>...]
     domain=<value> -->
```

`scripts/handbook_audit.py` Check 9 強制 enforce 列舉值；Check 9b 強制 cross-axis 相容矩陣；Check 9c 強制 `domain=generalist` 白名單。

> **`N/A` 是合法值**：對於非生成模型的條目（sim like Genesis / MJX / Aerial Gym → output=N/A）以及純評測 benchmark（VBench-Physics → output=N/A injection=N/A）允許 N/A。N/A 表示「此軸不適用於該條目」，不代表「未標」。
>
> **多值用 `|` 分隔**：例如 `output=pixel-video|action-seq`。v2 移除 `multi` 字面 keyword（v1 留下的 hack）。

---

## Axis 1 — Output space

模型直接交付給下游使用的「東西」存在哪個空間。**重點：不看內部 latent，看使用者拿到的最終輸出**（latent diffusion 但 decode pixel video 的應標 `pixel-video`，不是 `latent-tokens`）。

| Value | 描述 | Canonical anchor (per-axis) | 本倉 dissection |
|---|---|---|---|
| `pixel-video` | RGB 像素影片，直接 decode 出 frame | Cosmos 2501.03575 (NVIDIA, Jan 2025) | [Sora](../foundations/video-world-models/sora.md) · [Veo](../foundations/video-world-models/veo.md) · [Cosmos-Predict](../foundations/foundation-physics-models/cosmos-wfm.md) · [GAIA-2](../foundations/video-world-models/gaia-2.md) |
| `latent-tokens` | 非 decode 的 planning latent / world-model token，供下游 policy / planner 消費（不是 latent-diffusion 的中間狀態） | V-JEPA 2 2506.09985 (Meta, Jun 2025); DreamerV3 (Nature 2025) | [V-JEPA-2](../foundations/latent-world-models/v-jepa-2.md) · [DreamerV4](../foundations/latent-world-models/dreamer-v4.md) |
| `3d-explicit` | 顯式 3D 表徵：3DGS / mesh / point cloud / occupancy（v1 `3d-scene` + v1 `mesh` 合併） | Generative Gaussian Splatting 2503.13272 (Mar 2025) | World Labs gen-3D · GaussianAnything |
| `3d-implicit` | 隱式場 3D：SDF / NeRF / occupancy field（v2 新增；從 v1 `3d-scene` 拆出） | DeepSDF 1901.05103 (Park 2019) | (待 anchor) |
| `particle` | 粒子集 / MPM / SPH / 原子座標 | Neural-MPM 2408.15753 (Aug 2024) | NeuralMPM · ContactGen-particle · AlphaFold-3 (atoms) |
| `field` | 連續場（velocity / stress / temperature / pressure） | GraphCast (Science 2023); FNO 2010.08895 | [GraphCast](../foundations/neural-surrogates/graphcast.md) · [FNO](../foundations/neural-surrogates/fno.md) |
| `action-seq` | 動作序列（state-action token） | Genie 2402.15391 (Feb 2024) | [Genie-2](../foundations/latent-world-models/genie-2.md) · Decart latent-act |
| `motion` | 骨架 / 關節角 / SMPL pose 序列（v2 新增；PhysDiff 等不再硬塞 `action-seq` 或 `pixel-video`） | PhysDiff 2212.02500 | [PhysDiff](../foundations/physics-conditioning/physdiff.md) |

> **v1→v2 遷移**：v1 用 `output=mesh` 的條目改 `output=3d-explicit`；v1 用 `output=3d-scene` 同樣改 `output=3d-explicit`（內部用 SDF/NeRF 的改 `output=3d-implicit`）；v1 用 `output=latent` 的條目逐一審視 — 若實際 decode pixel（Cosmos / SVD），改 `output=pixel-video`；若純供 policy 消費（V-JEPA / Dreamer），改 `output=latent-tokens`。

---

## Axis 2 — Physics injection

物理規律怎麼進入模型。**本 handbook USP 軸**，2025 最熱戰場（NewtonGen · Force Prompting · PhysDiff · V-JEPA-2-reward · Cosmos-Reason）。

**v2 核心調整**：v1 `score-conditioned` 與 `constraint-loss` 概念糾纏（兩者都是「加物理梯度」，只差訓練 vs 推理）→ 合併為 `guidance-gradient`，另外把 `sim-in-loop` 拆 train / infer。並新增 `architecture-bias-soft` 對應 NewtonGen / MeshGraphNet 這類「有 physics flavor 但不保證守恆」的網絡。

| Value | 描述 | 訊號強度 | Canonical anchor | 本倉 dissection |
|---|---|---|---|---|
| `data-only` | 物理規律靠大量真實/模擬影片隱式學會。無 loss / 無架構偏置 / 無 sim 介入 | 最弱 | Cosmos 2501.03575; Sora tech report (OpenAI 2024) | [Sora](../foundations/video-world-models/sora.md) · [Veo](../foundations/video-world-models/veo.md) · [Cosmos-Predict](../foundations/foundation-physics-models/cosmos-wfm.md) |
| `aux-loss` | 訓練時加 PDE 殘差 / 守恆 / contact-penalty 等輔助 loss（v1 `constraint-loss` 改名） | 中 | PINN 1711.10561 (Raissi Nov 2017) | **[PINN](../foundations/physics-conditioning/pinn.md)** ★ · [PhysGen aux](../foundations/physics-conditioning/physgen.md) |
| `sim-in-loop-train` | 訓練時可微 sim 提供 gradient 或 GT trajectory | 強 | Genesis 2412.18608 (Dec 2024); MuJoCo MJX | [Genesis](../foundations/differentiable-simulators/genesis.md) · [MJX](../foundations/differentiable-simulators/mujoco-mjx.md) |
| `sim-in-loop-infer` | 推理時 sim 介入 denoising loop 或 rollout 校正（PhysDiff 把 sim 塞進每步 diffusion denoising） | 強 | PhysDiff 2212.02500 (Yuan ICCV 2023) | [PhysDiff](../foundations/physics-conditioning/physdiff.md) · Cosmos-Reason eval rollout |
| `guidance-gradient` | 物理梯度作為 score function：訓練（PINN 式 aux gradient）或推理（classifier-free / classifier-guided diffusion）。v1 `score-conditioned` 改名並擴大涵義 | 中強 | Classifier-Free Guidance 2207.12598 | [PhysDiff](../foundations/physics-conditioning/physdiff.md) (同時 sim-in-loop-infer) · [Force Prompting](../foundations/physics-conditioning/force-prompting.md) |
| `architecture-bias-soft` | 網絡架構帶 physics-flavor inductive bias 但**不保證**守恆 / 等價 / 對稱（neural dynamics head, message-passing GNN）。v2 新增 | 中 | NewtonGen 2509.21309 (Sep 2025); MeshGraphNet 2010.03409 | [Force Prompting + NewtonGen](../foundations/physics-conditioning/force-prompting.md) |
| `hard-constraint` | 架構天然滿足某類 PDE / 守恆律 / 等價性（symplectic / equivariant network）。v1 `hard-PDE` 改名擴大 | 最強 | E(3)-equivariant DeepH 2210.13955; Hamiltonian NN 1906.01563 | **[Hamiltonian / Lagrangian NN](../foundations/physics-conditioning/hamiltonian-lagrangian-nn.md)** ★ |

> **`energy-based` 已 demote 為 footnote**：v1 把 EBM 當獨立 injection 機制，但實務上 EBM 已被 score-based model 吸收（LeCun JEPA position paper 雖把 JEPA 框成 latent-EBM，V-JEPA 2 的物理仍是 `data-only`，EBM 是 loss family 而非 physics mechanism）。**Tag 規則**：按物理機制標，不按 loss family；JEPA-style EBM 不自動推導為 `hard-constraint`。
>
> **`v1→v2 遷移`**：
> - `implicit-from-data` → `data-only`
> - `constraint-loss` → `aux-loss`
> - `sim-in-loop` → `sim-in-loop-train` 或 `sim-in-loop-infer`（PhysDiff/Cosmos-Reason 屬後者）
> - `score-conditioned` → `guidance-gradient`
> - `hard-PDE` → `hard-constraint`
> - `energy-based` → 重新評估 → 通常改 `data-only` 或 `hard-constraint`

設計權衡：訊號強度 ↑ → fidelity ↑ → generalization ↓。詳見 [`crossing/controllability-vs-fidelity/`](../crossing/controllability-vs-fidelity/)。

---

## Axis 3 — Controllability input

模型接受的 conditioning 輸入。**多選**（多模態 controllability 是 2026 主戰場：Cosmos / GAIA-2 / Generative-GS 都同時收 text + camera + 3d-init + layout）。

| Value | 描述 | Canonical anchor | 本倉/cross-倉 |
|---|---|---|---|
| `text` | 自由文字 prompt | Cosmos 2501.03575 | Sora / Veo / Cosmos / GAIA-2 |
| `action` | 離散 / 連續動作 token（agent 視角） | Genie 2402.15391; DreamerV3 (Nature 2025) | Genie-2 · DreamerV4 · V-JEPA-2-action |
| `trajectory` | 路徑（2D/3D points over time） | GAIA-2 2503.20523 | GAIA-2 trajectory cond · Cosmos-Drive |
| `force` | 力 / 力矩 / 接觸力分布（agent-physics ladder 中段） | Force Prompting 2505.19386 (May 2025) | [Force Prompting](../foundations/physics-conditioning/force-prompting.md) · ContactGen-force |
| `contact` | 接觸圖 / 接觸時序 | PhysDiff 2212.02500 (motion-contact); ContactNets 2009.11193 | [PhysDiff](../foundations/physics-conditioning/physdiff.md) · CC-Diff |
| `image-init` | 起始幀 / reference image（initial condition，非真控制；v1 `image-prompt` 改名） | Stable Video Diffusion 2311.15127 | SVD · Cosmos-img2vid |
| `3d-init` | 起始 3D scene / mesh / 3DGS（initial condition；v1 `3d-prompt` 改名） | Generative Gaussian Splatting 2503.13272 | World Labs gen-3D conditioning |
| `camera` | 相機 pose / trajectory（v2 新增；2025 video gen 主流 — Cosmos / GAIA-2 / Generative-GS 都暴露） | Cosmos 2501.03575 (camera-conditioned post-training) | Cosmos · GAIA-2 |
| `layout` | 結構化 scene-graph / BEV / road-graph（v2 新增；不同於 `text`，GAIA-2 / Cosmos-Drive 主用） | GAIA-2 2503.20523 | GAIA-2 · Cosmos-Drive |
| `param` | 顯式物理參數（mass / stiffness / viscosity / friction；v1 `physical-param` 改名簡化） | NewtonGen 2509.21309 | NeuralPhysics · ParamControlNet |

> **v2 移除 `multi`** — v1 留下的 hack；直接用 `|` 分隔多值，例如 `control=text|action|trajectory|camera`。
>
> **`v1→v2 遷移`**：`image-prompt` → `image-init` · `3d-prompt` → `3d-init` · `physical-param` → `param`；`multi` 刪除（每個值列出來）。

---

## Axis 4 — Temporal paradigm

時間維度怎麼處理 — 直接決定 long-horizon 能不能撐住。

| Value | 描述 | 典型問題 | Canonical anchor |
|---|---|---|---|
| `single-frame` | 一張圖（無時間） | 不適用 video physics；audit 應排除 temporal 評估 | 任一 T2I |
| `streaming` | Sim / 連續時間 ODE 解算，無 frame 概念 | Sim2real gap、contact discontinuity | Genesis · MJX |
| `autoregressive` | 一幀一幀往前生成，下一幀 condition 上一幀 | Drift 累積、exposure bias | VideoGPT 2104.10157; GraphCast (6h autoregressive) |
| `clip-parallel` | 一次生成整段 clip（如 16/24/48 幀，固定窗口）。v1 `joint-rollout` 改名（更清楚說「整 clip 平行」） | 長度受限，跨 clip 銜接難 | Stable Video Diffusion 2311.15127; Cosmos-Predict |
| `latent-rollout` | latent space 中 rollout，最後再 decode 像素（或不 decode 直接給 policy） | 省 compute，drift 仍存在 | DreamerV3 (Nature 2025); V-JEPA 2 |
| `streaming-cache` | sliding window transformer + KV cache，串流式滾動（v1 `temporal-transformer-rolling` 改名） | KV cache 過期 / 記憶遺忘 | Genie / Genie-2 (DeepMind Dec 2024); Decart Oasis |
| `hierarchical` | 兩層（或更多）：高層慢時間規劃、低層快時間細節。**v2 文件化為 compositional**：允許與其他 paradigm `|` 共標，例 `temporal=latent-rollout|hierarchical` | 兩層介面難對齊；planner-renderer drift | TECO 2210.02396; Cosmos-Reason+Predict |

> **`v1→v2 遷移`**：`joint-rollout` → `clip-parallel` · `temporal-transformer-rolling` → `streaming-cache`；`hierarchical` 可與其他值共標。

---

## Axis 5 — Domain coupling

物理「在哪個世界」 — 影響 inductive bias 與 evaluation。

| Value | 描述 | Canonical anchor |
|---|---|---|
| `generalist` | 通用世界（沒明顯場域限定）。**v2 audit 規則**：Check 9c 只允許 Cosmos / Sora / Veo / Cosmos-Predict 等 foundation video 模型使用；其餘必須宣告具體 domain | Sora (OpenAI 2024); Cosmos 2501.03575 |
| `robotics` | 機器人操作 / 雙臂 / 移動 / VLA | V-JEPA 2 2506.09985 (Franka deploy); RT-2 2307.15818 |
| `driving` | 自駕 / 道路場景 | GAIA-2 2503.20523 |
| `fluid` | 液體 / 氣體 CFD（與 `weather` 分流是社區習慣，不是技術差異 — 同公式不同 benchmark） | FNO 2010.08895; Neural-MPM-fluid 2505.18926 |
| `rigid` | 剛體（contact-rich, mechanism） | PhysGen 2409.18964 |
| `soft` | 軟體 / 布料 / 變形 | PhysWorld 2510.21447 (deformable, Oct 2025) |
| `granular` | 顆粒 / 粉體 / 沙 | Inverse-granular-GNS 2401.13695 |
| `bio` | 蛋白質結構 / 分子動力學 / 細胞 / 神經組織（過載；待 v2.x 拆 sub-tag） | AlphaFold 3 (Nature 2024); ESMFold 2206.13517 |
| `weather` | 全球大氣 / 海洋 forecasting | GraphCast (Science 2023); Pangu-Weather 2211.02556 |
| `astro` | 天文 / N-body / 宇宙學 surrogate（v2 新增，小規模文獻但真實 — CAMELS / CMD） | CAMELS suite 2010.00619 |

> **`fluid` ⊃ liquid/gas CFD；`weather` = global atmospheric/oceanic forecasting**。重疊故意，按社區/benchmark 分；不要強行合併。

---

## Cross-axis 強制檢查（v2 新增 Check 9b/9c）

v1 review 列出 5 條 cross-axis 議題，v2 把其中 3 條變成 audit 強制檢查；2 條留 descriptive note。

### 9b — Output × Injection 相容矩陣（強制）

| Output ↓ × Injection → | data-only | aux-loss | sim-in-loop-train | sim-in-loop-infer | guidance-gradient | arch-bias-soft | hard-constraint |
|---|---|---|---|---|---|---|---|
| `pixel-video` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ (too high-dim) |
| `latent-tokens` | ✓ | ✓ | ✗ (rare) | ✗ | ✓ | ✓ | ✓ |
| `3d-explicit` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `3d-implicit` | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| `particle` | ✗ (rare) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `field` | ✗ (rare) | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| `action-seq` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `motion` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

> **Check 9b 規則**：任何 `output=pixel-video + injection=hard-constraint` 的 dissection 必須在 §8 pitfall 解釋如何在像素空間做 exact constraint（否則 audit fail）。

### 9c — `domain=generalist` 白名單（強制）

僅允許以下 dissection 標 `domain=generalist`：Sora · Veo · Cosmos-Predict · Cosmos-WFM。其餘必須宣告具體 domain（否則 audit fail）。

### Descriptive notes（不強制，但寫進每篇 §8）

- **Injection × Temporal**：`sim-in-loop-infer` 只對 iterative paradigm 有意義（`autoregressive` / `latent-rollout` / `streaming-cache`）；PhysDiff 是特例（每步 denoising 當 iteration）。
- **Control × Domain**：`force`/`contact` 通常 `robotics`/`rigid`/`soft`；`layout`/`trajectory` 通常 `driving`。Audit 看到反例應在 §8 解釋。
- **Score / EBM 概念漏洞**：tag 看物理 mechanism，不看 loss family；JEPA 的 EBM structure ≠ `hard-constraint`。已寫進 Axis 2 docstring。

---

## 為什麼是 5 軸不是 4 / 6 / 7

- **4 軸**（砍 domain）→ GraphCast 跟 Cosmos 同類，失去誤差/驗證視角
- **6 軸**（加 training-paradigm: self-sup / RLHF / curated）→ 跟 Axis 2 重疊（self-sup 通常 `data-only`）
- **6 軸**（加 refinement-loop: 區分 PhysDiff 的 per-step denoising 是不是時間維度）→ v1.1 review 投票 do not；用 §8 註記特殊性即可
- **7 軸**（同時拆 injection 為 mechanism × time）→ v1.1 review 提出但 v2 暫不採（dissection cost 太高，等 30+ dissection 後再評估）
- 5 軸是 sister Spatial-Handbook 同套方法論延伸；三冊一致便於 cross-ref

---

## 與 Spatial Ontology v3.2 的對應

| Physics v2 軸 | Spatial 對應 | 對應度 |
|---|---|---|
| Axis 1 Output | Spatial Axis 2 Representation（生成 vs 感知視角；3d-explicit / 3d-implicit 對齊 Spatial 的 3dgs / nerf / sdf） | 部分對齊 |
| Axis 2 Injection | **新增** — Spatial 沒有；本倉獨有 USP | 無對應 |
| Axis 3 Control | Spatial Axis 3 Sensor + Axis 4 Paradigm 子集（force/contact/camera 對齊；text/layout/action 是生成端新加） | 部分對齊 |
| Axis 4 Temporal | Spatial Axis 5 Time | 對齊 |
| Axis 5 Domain | Spatial Axis 1 Problem 但範圍不同（physics-gen 多 fluid/weather/astro，少 detection/segmentation） | 部分對齊 |

---

## v1 → v2 完整 rename 表（dissection header 重簽必讀）

| Axis | v1 value | v2 value | 影響本倉 dissection 數 (估) |
|---|---|---|---|
| 1 | `mesh` | `3d-explicit` | 1-2 |
| 1 | `3d-scene` | `3d-explicit` 或 `3d-implicit` | 1 |
| 1 | `latent` | `pixel-video` 或 `latent-tokens` | 3 (V-JEPA / Dreamer / Cosmos-Predict) |
| 1 | — | + `motion` 新增 | 1 (PhysDiff) |
| 1 | — | + `3d-implicit` 新增 | 0 |
| 2 | `implicit-from-data` | `data-only` | 3 (Sora / Veo / Cosmos-Predict) |
| 2 | `constraint-loss` | `aux-loss` | 2 (PINN / PhysGen) |
| 2 | `sim-in-loop` | `sim-in-loop-train` 或 `sim-in-loop-infer` | 3 (Genesis-train / PhysDiff-infer / Cosmos-Reason-infer) |
| 2 | `score-conditioned` | `guidance-gradient` | 2 (PhysDiff / Force Prompting) |
| 2 | `hard-PDE` | `hard-constraint` | 1 (HNN/LNN) |
| 2 | `energy-based` | demote → 重新評估 | 0 |
| 2 | — | + `architecture-bias-soft` 新增 | 1 (NewtonGen) |
| 3 | `image-prompt` | `image-init` | 4 |
| 3 | `3d-prompt` | `3d-init` | 2 |
| 3 | `physical-param` | `param` | 2 |
| 3 | `multi` | 刪除 → 改 `|` 分隔 | 全部 multi-value 條目 |
| 3 | — | + `camera` 新增 | 3-5 (Cosmos / GAIA-2 / Gen-GS) |
| 3 | — | + `layout` 新增 | 2 (GAIA-2 / Cosmos-Drive) |
| 4 | `joint-rollout` | `clip-parallel` | 3 (SVD / Cosmos-Predict) |
| 4 | `temporal-transformer-rolling` | `streaming-cache` | 2 (Genie / Decart) |
| 5 | — | + `astro` 新增 | 0 (待 anchor) |

**遷移工作量估**：14 篇 dissection × 平均 3 軸有改 ≈ 42 處 header re-sign；用 `scripts/handbook_audit.py --migrate-v1-to-v2` 半自動（待實作）。

---

## v2 後續 TODO（不阻塞當前採納）

- [ ] 寫 `scripts/handbook_audit.py --migrate-v1-to-v2` 半自動重簽 14 篇 dissection header
- [ ] Check 9b/9c 加進 audit script 並 backtest 現有 14 篇
- [ ] Phase 3 抵 30+ dissection 後評估是否需要 v3：拆 Axis 2 為 mechanism × time 兩個子軸
- [ ] `bio` 拆 sub-tag（protein / molecular-dynamics / cell / tissue）
- [ ] `astro` 配 anchor dissection（CAMELS 或更新的 cosmology surrogate）
- [ ] `3d-implicit` 配本倉 dissection（DeepSDF 或 NeRF-physics）

---

## Changelog

- **v2.0 (2026-05-26)** — 採納 [v1.1 review](./ontology-v1.1-review.md) 全部 diff。Axis 1：合 mesh / 拆 3d-scene / 限 latent / 加 motion + 3d-implicit。Axis 2：rename 5 values + split sim-in-loop + 加 architecture-bias-soft + demote energy-based。Axis 3：rename 3 values + 加 camera/layout + 刪 multi。Axis 4：rename 2 values + 文件化 hierarchical 可 compositional。Axis 5：加 astro + 文件化 fluid/weather overlap + generalist 白名單。Cross-axis：新增 Check 9b (Output×Injection 相容矩陣) + 9c (generalist 白名單)。
- **v1.0 (2026-05-25)** — 初版（5 軸 / 6 injection values）。
