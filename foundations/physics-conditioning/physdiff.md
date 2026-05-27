<!-- ontology-5axis
output: motion | 3d-explicit
injection: guidance-gradient + sim-in-loop-infer
control: text + contact
temporal: clip-parallel (denoise) + autoregressive (sim rollout)
domain: robotics | rigid (humanoid + ground plane)
ref: ../../cheat-sheet/ontology.md §5
-->

# PhysDiff 解構（Physics-Guided Human Motion Diffusion）

> **發布時間**：2022-12 · arXiv [2212.02500](https://arxiv.org/abs/2212.02500) · **ICCV 2023 (Oral)**
> **論文**：*PhysDiff: Physics-Guided Human Motion Diffusion Model*
> **作者**：Ye Yuan, Jiaming Song, Umar Iqbal, Arash Vahdat, Jan Kautz（NVIDIA Research）
> **核心定位**：v2 ontology 上把 `guidance-gradient` 與 `sim-in-loop-infer` **同時**標記的鼻祖 paper —— 用 non-differentiable humanoid simulator 在 reverse diffusion 末段做投影，把純資料 score model 生成的 floating/penetration/sliding artifact 一次性壓下一個量級。

**Status:** v0.5 — 解構基於 paper 全文 + project page + 二手分析；UHC/PHC 互換性的長期穩定性、video-domain 移植可行性仍是 open。完整 GPU latency / Isaac Gym contact solver 細節由維護者升 v1。
**TL;DR:** PhysDiff 不重訓 score network，只在 reverse diffusion 末段塞一個 UHC（Universal Humanoid Controller）simulator 做 **physics projection**，把 candidate motion rollout 成「真能站著走」的軌跡再回灌 denoising chain；HumanML3D + MDM baseline 上 **ground penetration 11.29mm → 0.998mm**（× 11 改進）、floating 18.88 → 2.60mm、sliding 1.41 → 0.51mm，整體 Phys-Err -86%；schedule "End 4, Space 1" 是 fragile tuning 但 reproducible —— 對 v2 ontology 而言這是 score-guidance + iterative sim-projection 同時用的 **canonical reference**。

**X-Ray.** Motion diffusion（MDM 一脈）證明了「text → SMPL clip」可以純資料學，但留下三個工程坑全部源自一件事 —— **denoising 過程裡完全沒有任何接觸、重力、慣性訊號**：(a) 腳穿過地板 (ground penetration)、(b) 整人懸空 (floating)、(c) 站定時還在滑 (foot sliding)。MDM 自己塞了一個 foot-contact soft loss，但它是 training-time penalty，OOD prompt 一來就崩。PhysDiff 的洞察是 **物理規律不該進 architecture（HNN 那條）也不該進 training loss（PINN 那條），它應該進 inference 迴圈** —— 因為對 contact-discontinuity 來說，只有 simulator rollout 是 ground truth，hand-crafted PDE loss 永遠 approximate。把這條 line 放回 v2 ontology：PhysDiff 是 **第一個把 `guidance-gradient` (score step 拉樣本) 和 `sim-in-loop-infer` (sim 投影回 manifold) 串成單一 reverse pass** 的方法；下游 PhysHOI / PhysGen / NewtonGen / Force Prompting 全部繞著「sim 到底插進 train 還是 infer、插一次還是 N 次」這條軸做變奏。對 Sora-級 video generation 那個臭名昭著的 foot-physics 問題，PhysDiff 是目前範式上最直接可借鑒的解 —— 但 UHC 是 humanoid sim、video pixel 沒對應 simulator，搬過去要先解 video↔mesh 閉環，**這就是論文打不開的 envelope**。

## 📍 研究全景時間線

```ascii
   2022-05         2022-12                    2023-12               2024-?            2025-?
   MDM ──────────► PhysDiff ───────────────► PhysHOI ────────────► PhysGen ────────► Force Prompting / NewtonGen
   ICLR 2023       YOU ARE HERE              arXiv 2312.04393       CVPR 2024         2025-? (score-conditioned)
   pure data       diffusion + sim-in-loop   + object interaction   pipeline sim      force as condition
   foot loss       (humanoid only)           (pure RL imitation)    (static→video)    (no sim)
   = soft penalty  ★ guidance-gradient
   ground penetr.    + sim-in-loop-infer
   11.29mm           = 0.998mm
                     foot slide 0.51mm
   └─ score-only ─────► score + projector ─────► projector + HOI ─────► sim-pipeline ─────► force-condition ───►
                        (this paper)              (no diffusion)         (no inference loop) (no simulator)
```

★ = 主要新點：**sim-in-loop 在 inference 迭代投影，不重訓 score**。**仍未解：video-level UHC（video pixel ↔ humanoid sim 閉環）、object interaction、real-time latency** —— 全部留給下一代。

---

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs MDM (baseline)

| 維度 | MDM (Tevet 2022) | PhysDiff |
|---|---|---|
| **物理訊號注入點** | Training-time foot-contact loss (soft penalty) | **Inference-time UHC rollout projection** (hard via sim) |
| **Score network** | 1000-step DDPM, sample-prediction $\hat x_0$ | **完全沿用 MDM**，不改 weight |
| **每步輸出** | $\hat x_{t-1} = \text{denoise}(x_t)$ | $x_{t-1} = \text{PhysProj}(\hat x_{t-1})$ on selected timesteps |
| **Projection schedule** | — | **"End 4, Space 1"** — 末段 4 個 timestep、間隔 1（晚做、少做、做精準）|
| **Simulator backend** | — | Isaac Gym + UHC (PPO-trained imitator, PD ctrl + residual force) |
| **Differentiable?** | yes (loss backprop) | **No** — sim 是黑箱，所以**不能用來訓 score** |
| **Ground penetration (HumanML3D)** | 11.29 mm | **0.998 mm** |
| **Floating** | 18.88 mm | **2.60 mm** |
| **Foot sliding** | 1.41 mm | **0.51 mm** |

### 1.2 ⚡ Eureka Moment

> **每一步 denoise 出來的 $\hat x_0$（MDM 是 sample-prediction，正好是一段「可被 imitate 的動作」）餵給 UHC 跑一段物理 rollout，rollout 出來的軌跡就是 physically valid manifold 上最接近的點 —— 拿這個乾淨版本回灌下一步 denoising 的 $x_{t-1}$，等於把 score chain 強制鎖在物理流形上。** 不改 score、不加 loss、不要求 differentiable sim —— 純後處理投影。

關鍵直覺：**晚做、少做、做精準**。早期高 noise 階段做 projection 反而傷品質 —— simulator 無法 imitate 一團 noise（PD controller 會直接失敗或飛出 box），結果 score chain 被一個失敗 rollout 拉出高密度區，後續 denoise 救不回來。這跟 PINN 路線「每步 timestep 都施加物理 loss」的本質區別 —— **PhysDiff 把物理懲罰當成 endgame 校正，不是 throughout regularizer**。

### 1.3 信息流（架構圖）

```ascii
              MDM (baseline)                              PhysDiff
   ──────────────────────────────              ───────────────────────────────────
                                                                                  
   text c                                       text c
     │                                            │
     ▼                                            ▼
   x_T (noise)                                  x_T (noise)
     │                                            │
     ▼                                            ▼
   ┌──────────────┐                             ┌──────────────┐    ┌──────────────────┐
   │ denoise(t)   │  ──► x_{t-1}                │ denoise(t)   │──► x̂  │ Isaac Gym + UHC  │
   │ MDM score    │     (with foot-contact     │ MDM score    │       │ PD ctrl + res-F  │
   │ + soft loss  │      loss baked in         │ (unchanged)  │       │ PPO imitator     │
   └──────────────┘      at TRAIN time)        └──────────────┘       └──────┬───────────┘
        ▲                                            ▲                       │
        │                                            │      if t ∈ {T-3..T}  ▼
        │  loop t = T → 0                            │      ◄── PhysProj ──  x′ (physical)
        │                                            │      else: x̂ passthrough
        x_{t-1}                                      x_{t-1}
                                                                                  
   Phys-Err: 11.29 / 18.88 / 1.41 mm           Phys-Err: 0.998 / 2.60 / 0.51 mm
   (penetration / floating / sliding)          ★ -86% overall
```

注意：PhysProj **非可微**（Isaac Gym contact solver 是黑盒），所以這條 line **不能反傳 gradient 訓 score**；想反傳就得換 Genesis / MuJoCo MJX 那條 differentiable sim 路線（§3 / §7）。

---

## §2 · 數學層

### 📌 Napkin Formula

```
   Reverse diffusion with sim-in-loop projection:

      x_t   ──denoise──►   x̂_{t-1}  =  D_θ(x_t, t)        ← score step (MDM)
                                                              guidance-gradient
                                                              
      x_{t-1}  =  PhysProj(x̂_{t-1})    if t ∈ EndWindow      ← sim-in-loop-infer
                  x̂_{t-1}              otherwise              (only "End 4, Space 1")
      
      PhysProj(m) := UHC_rollout(m)  via Isaac Gym
                   = PPO_π(s_t | m_ref)  → PD-control loop → SMPL_t+1
   
   Cost:  N_sim_steps × ContactSolve   ≈  10-20× score step FLOPs
          per projection step
   
   Total inference: ~30s / clip   (vs MDM ~2s / clip)
```

**直覺**：`guidance-gradient` 那一行是 score 對 $\log p_{\text{data}}$ 的拉力；`PhysProj` 那一行是 sim 對 $p_{\text{physical}}$ manifold 的投影。兩者**乘**起來等於 $p_{\text{data}} \cap p_{\text{physical}}$ —— 不是 product of experts 那種 soft mixing，是 hard intersection（每次都先 score 後投影）。**這就是 v2 ontology 把這對 axis 同時標起來的原因 —— PhysDiff 是同時用兩種 injection 的最早 paper**。

### 2.1 為什麼 MDM 的 sample-prediction 是天然搭檔

DDPM 原版 noise-prediction 輸出 $\epsilon_\theta(x_t)$，要 project 得先反算 $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta) / \sqrt{\bar\alpha_t}$ 才有 motion 可 imitate；MDM 直接輸出 $\hat x_0$（一段完整 motion clip），UHC 拿到就能 rollout —— 省一層 cost，更穩。**這就是 PhysDiff 為什麼接 MDM 而不接 DDPM**。

### 2.2 Projection schedule 的 hyperparameter sensitivity

`End k, Space s` ablation（paper §5.3）顯示：
- `End 4, Space 1`：FID 持平、Phys-Err -86%（最佳）
- `End 8, Space 1`：FID 上升（manifold drift）、Phys-Err 略好
- `End 50, Space 1`（整段都投）：FID 大幅惡化 —— simulator 拉樣本拉到 score 接不回來
- `End 1`：Phys-Err 救不回（單次投影不夠）

**結論**：projection 是 fragile tuning，換 denoiser（DDIM step 數 / classifier-free guidance scale）必須重調 —— ratio 不是絕對步數。

---

## §3 · 數據層 / 訓練 scale

| 資料 | 用法 | 規模 |
|---|---|---|
| **HumanML3D** | text → SMPL motion (主) | 14,616 sequences |
| **HumanAct12** | action label → motion | 1,191 sequences |
| **UESTC** | action → motion | 25,600 sequences |
| **AMASS (UHC 用)** | imitator pre-training reference | ~9000 sequences |

**關鍵事實**：PhysDiff 自己**不做 score training**，所有 motion data 都是 MDM 預訓的；UHC 是另一階段在 AMASS 上做 PPO imitation pre-training。**整體 paper 的 contribution 在 inference 階段** —— 這也是它為什麼能套到任何 pretrained motion diffusion model 上的根本原因。

**對讀者意義**：想複現 PhysDiff 不必有 GPU cluster，**只要能跑 MDM inference + Isaac Gym headless 就行**（單張 A6000 / 4090 足夠）。這跟訓一條 Force Prompting / NewtonGen 條件 video diffusion 完全不是同一個成本級。

---

## §4 · 代碼層

| 項 | 狀態 |
|---|---|
| **官方 repo** | ❌ 截至 2026-05 NVlabs 未開源 PhysDiff `[TBD: verify]` |
| **MDM denoiser** | ✅ [GuyTevet/motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model)（HumanML3D / KIT / HumanAct12 全 ckpt）|
| **UHC simulator** | ✅ [ZhengyiLuo/UHC](https://github.com/ZhengyiLuo/UHC)（MuJoCo 原版；論文後續升 Isaac Gym）|
| **更穩 imitator** | ✅ [PHC (ICCV 2023)](https://arxiv.org/abs/2305.06456) Perpetual Humanoid Control —— 推薦現在重現用 PHC 替代 UHC |
| **License** | MDM: MIT; UHC: research-only |
| **Inference GPU** | 1× A6000 / 4090, ~16GB sweet spot |
| **Inference latency** | ~30s / clip（vs MDM ~2s / clip, 約 15× slowdown）|
| **Streaming** | ❌ batch-only（projection schedule 依賴整段 clip 已被部分 denoise）|
| **Differentiable sim?** | ❌ Isaac Gym contact solver 是黑盒 |

**社群重現 glue code**（PhysDiff 沒開源，自寫約 200-400 LOC）：
```
denoise_step(MDM) → SMPL pose-axis-angle → 
   convert to UHC reference (25-30 fps resample) → 
      UHC rollout N sim-steps → 
         SMPL back → next denoise_step
```

踩坑點：
1. UHC 25/30 fps vs MDM 20 fps —— 時間重採樣，線性插值 IK 抖動
2. SMPL pose ↔ rotation-matrix 軸定義差異，root 容易翻 180°
3. Isaac Gym headless docker 需 X11 / `libGL`，CI 不友好
4. "End 4, Space 1" 是 1000-step DDPM ratio，搬 50-step DDIM 要 rescale

---

## §5 · 評測 / Benchmark

### 5.1 PhysDiff 報的核心數字

| Benchmark | Metric | MDM baseline | PhysDiff | Δ |
|---|---|---|---|---|
| **HumanML3D** | Ground penetration (mm) | **11.29** | **0.998** | **× 11.3 改進** |
| HumanML3D | Floating (mm) | 18.88 | 2.60 | × 7.3 改進 |
| HumanML3D | Foot sliding (mm) | 1.41 | 0.51 | × 2.8 改進 |
| HumanML3D | Overall Phys-Err | 100% | 14% | **-86%** |
| HumanML3D | FID (motion quality) | baseline | ≈ baseline | persistent |
| HumanAct12 | Action recognition Acc | baseline | ≈ baseline | persistent |
| UESTC | Action recognition Acc | baseline | ≈ baseline | persistent |

### 5.2 解讀 —— 哪部分是真 capability、哪部分是 sim floor？

`Ground penetration 0.998mm` 看起來是「幾乎不穿地」，但 **Isaac Gym 自身 rigid-rigid contact solver 有 ~5mm 等級的數值穿插容忍** —— 真實「物理零」根本不在 0.998mm 量級。所以 PhysDiff 報的下限**有相當部分是 sim floor，不是物理真值**。要更精得換 MuJoCo MPR / Bullet hard-contact，但這兩個 GPU acceleration 都差。

⚠️ **不要外推**：HumanML3D 的 motion distribution 是「日常動作 + 簡單武打」—— PhysDiff 在這個分布內降 86% Phys-Err 是真的；但在 **dance / acrobatics / cooking（手部）/ 雜技** 上，UHC imitator 直接失敗，projection 退化為 raw MDM 輸出（甚至更糟 —— projection 中途打斷 score chain，結果比沒做還壞）。**這是 §6 hidden assumption 列表的根**。

---

## §6 · Issues & Limitations

### 6.1 論文自述 limitations

1. **Compute overhead ~15×** —— 從 MDM 的 ~2s/clip 拉到 ~30s/clip，real-time 應用基本不可行
2. **Projection schedule fragile** —— "End 4, Space 1" 是 DDPM 1000-step 調出來的，換 DDIM / CFG scale 全部重來
3. **Humanoid + ground plane only** —— object interaction、不平地形、手部精細 (MANO) 完全沒解
4. **Simulator manifold ≠ data manifold** —— projection 過密反傷 FID

### 6.2 Hidden Assumptions（隱含假設）

- **UHC 能 imitate 任何 OOD motion** —— 實際上 dance / contact juggling 直接失敗
- **Sample-prediction MDM 是必要前提** —— noise-prediction DDPM 用 PhysDiff 要先反算 $\hat x_0$
- **Isaac Gym contact 數值穩** —— 但 rigid-rigid contact 5mm 穿插是 sim floor
- **沒 imitator confidence gate** —— UHC 失敗的處理是黑盒，paper 沒詳述失敗率
- **20 fps motion ↔ 25/30 fps sim 重採樣不損失動態細節** —— 高速動作（揮拳、彈跳）插值會抖
- **Ground plane 是水平且 infinite** —— 上樓梯 / 跨水溝 / 抓欄杆全部超出 UHC 訓練分布

### 6.3 失敗模式（atlas 聯動 conservation-violation）

| 失敗模式 | 觸發條件 | 嚴重度 |
|---|---|---|
| **OOD motion → projection 拒收** | text prompt 描述 UHC 沒見過的動作（moonwalk / jugglіng）| 🔴 退化為 raw MDM (更糟) |
| **Manifold drift** | projection schedule 過密（End >8 / Space 0）| 🟠 FID 上升 sample 拉離 score 高密度 |
| **手部 / 物件互動完全缺席** | 拿杯 / 開門 / 踢球 | 🔴 UHC 不支援 MANO + object |
| **Sim accuracy floor** | 報的 0.998mm 是 Isaac floor 不是物理零 | 🟡 多數應用夠用，精細科研不夠 |
| **Real-time inference** | ~30s/clip | 🔴 production 不可行 |
| **Code 未開源** | NVlabs 沒 release | 🟠 重現難度被低估（自寫 glue 200-400 LOC）|

**Maintainer 響應度**：官方 repo 不存在；MDM repo 由 [GuyTevet](https://github.com/GuyTevet/motion-diffusion-model) 維護中（active）；UHC repo 由 ZhengyiLuo 維護中但已被 PHC 取代。

---

## §7 · 比較 & 面試 Tip

| 同軸對手 | Injection (Axis 2) | Sim differentiable? | Streaming? | Train-time or Infer-time? | Status |
|---|---|---|---|---|---|
| **PhysDiff** | `guidance-gradient` + `sim-in-loop-infer` | ❌ black-box Isaac Gym | ❌ batch | **Infer-time projection** | ICCV 2023 |
| PINN ([./pinn.md](./pinn.md)) | `pde-loss` | n/a | ✅ (no sim) | Train-time loss | shipped |
| Hamiltonian/Lagrangian NN ([./hamiltonian-lagrangian-nn.md](./hamiltonian-lagrangian-nn.md)) | `architecture-conserve` | n/a | ✅ | **Architecture-level** | shipped |
| Genesis-train ([../differentiable-simulators/genesis.md](../differentiable-simulators/genesis.md)) | `sim-in-loop-train` | ✅ differentiable | n/a | **Train-time backprop** | shipped |
| MuJoCo MJX ([../differentiable-simulators/mujoco-mjx.md](../differentiable-simulators/mujoco-mjx.md)) | `sim-in-loop-train` | ✅ differentiable | n/a | Train-time backprop | shipped |
| PhysGen ([./physgen.md](./physgen.md)) | `sim-pipeline` (img→sim→video) | partial | ❌ | Pipeline (one-shot sim) | CVPR 2024 |
| Force Prompting ([./force-prompting.md](./force-prompting.md)) | `data-only` + force condition | ❌ no sim | ✅ | None (cond input only) | 2025 |
| MDM ([baseline 2209.14916](https://arxiv.org/abs/2209.14916)) | `data-only` + foot-contact soft loss | n/a | ❌ batch | Train-time soft penalty | ICLR 2023 |
| PhysHOI (2312.04393) | `sim-in-loop` (pure RL, no diffusion) | ❌ | ✅ | RL imitation | arXiv 2023 |

> **🎤 Interview Tip.** 「PhysDiff 適合 Sora 級 video gen 嗎？」**正確答**：「**結構上不適合 —— UHC 是 humanoid mesh-level sim，video pixel 沒有對應 simulator，要先解 video↔mesh↔sim 的閉環，每一段都是當前 SOTA 邊界。**短期可行的是 partial projection（只投影 character body region 像 inpainting），但這已經不是 PhysDiff 原版方法。長期看 Force Prompting / NewtonGen 把力當條件輸入是更可行的工程路徑，雖然犧牲了 hard physical guarantee。」**錯答**：「直接套 PhysDiff 思路在 video diffusion 上 projection 一下就好。」—— video 沒有 SMPL 的 well-defined imitator 對象，這個答案暴露對 sim-in-loop 抽象層次的不理解。

### 7.1 Falsifiable predictions

1. **2027-06 前**：第一篇 video-level PhysDiff 變種會出現，但不是直接套 UHC —— 它會走 「video → SMPL 提取 → UHC project → SMPL → video 重繪」 partial pipeline，並且只覆蓋 character region。
2. **2027-12 前**：差異化 simulator 路線（Genesis / MJX-train）在 contact-rich humanoid 場景上仍**不會**取代 PhysDiff 風格的 sim-in-loop-infer —— differentiable contact 的數值穩定性在 hard contact 上還沒解。
3. **2028-12 前不會發生**：PhysDiff 風格 inference-projection 進入 production real-time motion gen pipeline（30s/clip 量級就是上限，distillation 路線雖然可行但會丟掉 hard physical guarantee，跟 raw MDM 就沒有本質區別）。

---

## §8 · For the Reader（按 persona 分流）

- **影片生成工程師 (Sora-line)** —— ⭐ **本篇對你最重要的單一啟示**：foot-physics / object permanence 不是 model size 問題，是 **inference-time 缺少投影到物理流形的環節**。短期不要奢望直接套 PhysDiff（沒對應 video-domain UHC），先研究 partial projection / character-region inpainting；長期看 [`crossing/conservation-violation-atlas/`](../../crossing/conservation-violation-atlas/) 列出的 foot-contact / ground-penetration sub-violation 表 —— 那是這條 line 最值得照搬的具體 KPI 清單。
- **VLA / robot policy 工程師** —— PhysDiff 的 UHC 跟你的 sim2real pipeline 是兄弟工具：都依賴 imitator + PD ctrl。如果你已有 Isaac Gym / MuJoCo MJX setup，**加 PhysDiff 風格 sample-time projection 在 motion planner 輸出上**是低成本實驗（不重訓 policy）。
- **影像生成 / 物理 conditioning 研究者** —— 把 PhysDiff 當成 v2 ontology `guidance-gradient + sim-in-loop-infer` 雙軸**最 cleanest 的對照組**。你的 paper 想 claim「我在 inference 用 simulator」就要先把跟 PhysDiff 的差異講清楚 —— 是 differentiable 嗎？是 train-time 還是 infer-time？是 humanoid 還是 fluid / soft body？這四維界定了你的 contribution slot。
- **Animation / motion-cap 工程師** —— production 場景 PhysDiff 直接用，但要換 **PHC 替代 UHC**（更穩 imitator）+ **schedule 重調**（你的 DDIM step 數通常 < 50）。預期 inference 仍是分鐘級，dance / acro 動作要 imitator-confidence gate。
- **神經 PDE / surrogate 研究者** —— PhysDiff 跟 PINN / HNN 是**三條互補路線**而非競爭：architecture-level（HNN）→ train-time loss（PINN）→ infer-time projection（PhysDiff）。對 contact-discontinuity 場景，infer-time 是目前唯一可行的；對 smooth conservation（fluid / pendulum），architecture-level 更乾淨。**選哪條看你的物理是 closed system 還是 contact-rich**。
- **Research 學生** —— 注意 §7.1 三條預測 + §6.3 失敗 atlas。PhysDiff 是 anchor paper，所有後續 sim-in-loop 變種（PhysHOI / PhysGen / NewtonGen）都繞著它的「sim 插在 train 還是 infer / 插一次還是 N 次 / 可微還是黑盒」三個軸做變奏 —— 把這三軸畫成 2×2×2 立方體就是 2024-2027 整片 design space。

---

## References

- **PhysDiff (canonical)** — Yuan, Song, Iqbal, Vahdat, Kautz. *PhysDiff: Physics-Guided Human Motion Diffusion Model*. **ICCV 2023 Oral** · arXiv [2212.02500](https://arxiv.org/abs/2212.02500) · [project page](https://nvlabs.github.io/PhysDiff/)
- **MDM baseline** — Tevet, Raab, Gordon, Shafir, Cohen-Or, Bermano. *Human Motion Diffusion Model*. **ICLR 2023** · arXiv [2209.14916](https://arxiv.org/abs/2209.14916) · [code: GuyTevet/motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model)
- **UHC simulator** — Luo et al. *Universal Humanoid Control* (Kinpoly NeurIPS 2021 / EmbodiedPose NeurIPS 2022) · [code: ZhengyiLuo/UHC](https://github.com/ZhengyiLuo/UHC)
- **PHC (recommended imitator replacement)** — Luo et al. *Perpetual Humanoid Control for Real-time Simulated Avatars*. **ICCV 2023** · arXiv [2305.06456](https://arxiv.org/abs/2305.06456)
- **PhysHOI (object interaction extension)** — Wang et al. *PhysHOI: Physics-Based Imitation of Dynamic Human-Object Interaction*. arXiv [2312.04393](https://arxiv.org/abs/2312.04393)
- **UMR (universal motion repr.)** — Luo et al. *Universal Humanoid Motion Representations for Physics-Based Control*. arXiv [2310.04582](https://arxiv.org/abs/2310.04582)
- **Datasets** — HumanML3D, HumanAct12, UESTC（標準 motion-text benchmarks）
- **Third-party reproduction notes** — 社群實作（無單一 canonical reference）

---

## Boundary

- 完整 PINN 解構（train-time PDE loss）→ [`./pinn.md`](./pinn.md)
- 完整 Hamiltonian / Lagrangian NN 解構（architecture-level conservation）→ [`./hamiltonian-lagrangian-nn.md`](./hamiltonian-lagrangian-nn.md)
- PhysGen（rigid-body image → physics video, pipeline 而非 inference loop）→ [`./physgen.md`](./physgen.md)
- Force Prompting（force as conditioning input, no sim）→ [`./force-prompting.md`](./force-prompting.md)
- 可微 sim 對照組（train-time backprop 路線）→ [`../differentiable-simulators/genesis.md`](../differentiable-simulators/genesis.md) / [`../differentiable-simulators/mujoco-mjx.md`](../differentiable-simulators/mujoco-mjx.md)
- Conservation violation 具體目標清單（foot-contact / ground-penetration sub-violation specs）→ [`crossing/conservation-violation-atlas/`](../../crossing/conservation-violation-atlas/)
- 與 5 axis 全景 → [`cheat-sheet/ontology.md`](../../cheat-sheet/ontology.md)

---

## ✍️ 維護者註（v0.5 → v1 升級清單）

本 v0.5 基於 paper 全文 + project page + 社群重現經驗。下次升 v1 時補：

1. ⏳ 驗證 NVlabs 是否曾 release official PhysDiff repo（截至 2026-05 未見）
2. ⏳ Isaac Gym contact solver 的實際 mm-level 穿插 floor 量化
3. ⏳ "End 4, Space 1" schedule 在 50-step DDIM / classifier-free guidance 下重調的 sweep result
4. ⏳ PHC 替代 UHC 的 reproduction Phys-Err 對照（社群是否已有？）
5. ⏳ Video-domain partial projection 第一篇 follow-up paper 列出
6. ⏳ UHC imitator confidence gate 的開源實作（OOD motion graceful fallback）
7. ⏳ Status v0.5 → v1，刪本節

---

[← Back to Physics Conditioning](./overview.md)

Sources:
- [PhysDiff arXiv 2212.02500](https://arxiv.org/abs/2212.02500)
- [PhysDiff project page](https://nvlabs.github.io/PhysDiff/)
- [MDM arXiv 2209.14916](https://arxiv.org/abs/2209.14916)
- [GuyTevet/motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model)
- [ZhengyiLuo/UHC](https://github.com/ZhengyiLuo/UHC)
- [PHC arXiv 2305.06456](https://arxiv.org/abs/2305.06456)
- [PhysHOI arXiv 2312.04393](https://arxiv.org/abs/2312.04393)
