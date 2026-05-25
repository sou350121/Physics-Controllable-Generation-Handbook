<!-- ontology-5axis output=pixel-video injection=score-conditioned|constraint-loss control=force|contact|physical-param|text temporal=joint-rollout domain=rigid|generalist -->

# Force Prompting & NewtonGen — 2025 顯式力/牛頓條件視頻生成雙錨

> **雙錨 dissection**：本篇同時拆解 Force Prompting (arxiv 2505.19386, NeurIPS 2025) 與 NewtonGen (arxiv 2509.21309, ICLR 2026)。兩者都把「力 / 牛頓狀態」當成顯式 conditioning token 注入 pretrained video diffusion，是 2025 年「explicit force as control」線上的兩篇 canonical paper，必須一起看才能拼出 design space。
>
> 注：本倉 dissection wishlist 中的 `physgen.md`、`physdiff.md`、`pinn.md` 暫未落地，下文凡指向這三檔者為 forward ref；交叉路線觀點仍可在 [`overview.md`](./overview.md) 與 `crossing/` 找到 v1 摘要。

---

## 1. TL;DR — 2025 force-conditioning wave 為什麼必要

2024 年以前的 video gen 兩個陣營都解不了「可控物理」這個老問題：

- **Text-only Sora-class**（→ [`../video-world-models/sora.md`](../video-world-models/sora.md)）：fidelity 漂亮、但 prompt "蘋果掉到桌上" 出來的可能是漂浮、可能反向加速；用戶沒有 dial 可以調「掉多快」或「往哪個方向施一個 5N 的力」。
- **Hard-sim pipeline (PhysGen 線, → `./physgen.md`)**：把 image → 3D 剛體 → PyBullet → render，物理對但脆，碰到非剛體 / 非典型場景就崩；artist workflow 永遠繞不開 asset prep。

**Force Prompting (Brown × Google DeepMind, May 2025, NeurIPS 2025)** 把「力向量 (location, angle, magnitude)」直接做為 conditioning channel 加進 pretrained video diffusion；不掛 simulator，不做 3D 重建，只用 ~15k–23k Blender 合成 force-paired videos 微調，結果**跨材質、物體、affordance 都 generalize**，包含一些對 mass 的隱式理解。

**NewtonGen (Purdue, Sep 2025, ICLR 2026)** 再往上一格：與其只給「當下這一刻的力」，乾脆訓一個 **Neural Newtonian Dynamics (NND)** 頭——一個 physics-informed neural ODE——把完整 9-dim 牛頓狀態 `[x, y, vx, vy, θ, ω, s, l, a]` 解算出未來軌跡，再把它轉成 optical flow，餵進 CogVideoX-5B + Go-with-the-Flow 的 noise-warping 管線。換言之 NewtonGen 把 dynamics solver 內嵌成可訓練模組，而不是當前 frame 的 conditioning。

兩篇放在一起，定義了 **explicit-force conditioning 的兩種架構姿勢**：

| 維度 | Force Prompting | NewtonGen |
|---|---|---|
| 條件粒度 | 單一時刻力向量 (location/angle/magnitude) | 完整 9-dim 牛頓狀態 + 軌跡 rollout |
| 物理機制 | data-only + conditioning channel | architecture-bias-soft（neural ODE 解算物理） |
| 注入點 | diffusion conditioning（類似 ControlNet branch） | optical-flow guided noise warping（pre-diffusion） |
| 基模 | 未明說（Brown-palm 實作走 CogVideoX 系列） | CogVideoX-5B + Go-with-the-Flow LoRA |
| 控制方式 | 用戶在 image 上指力 | 用戶設初始狀態（位置/速度/角速度/質量近似） |
| 物理一致性 | 隱式（靠 pretrained visual prior） | 顯式（ODE solver 保證 trajectory consistency） |

**為什麼這對 controllability-vs-fidelity 是架構級回答**：以前可控性的代價是 fidelity（hard-sim 看起來假），fidelity 的代價是可控性（Sora 不接受力 token）。Force Prompting 證明 conditioning channel 加 force 不必崩 fidelity；NewtonGen 證明把 dynamics 拆成獨立可學模組能進一步壓住「對 mass / 加速度的隱式建模誤差」。這條線是 [`crossing/controllability-vs-fidelity/`](../../crossing/controllability-vs-fidelity/) 的活樣本。

---

## 2. Core mechanism — 兩種 force/state token 注入

### 2.1 Force Prompting：force vector as conditioning channel

```
user input                conditioning encoder            video diffusion
──────────                ────────────────────             ────────────────
image  ────────────────┐
                       ├──▶ force map (spatial)  ──▶ cross-attn + ControlNet-like branch ──▶ pixel video
force (loc/angle/mag) ─┘                                   (pretrained backbone frozen / LoRA-tuned)
```

- **Local force**：在 image 上指一個點 + 角度 + 量值（poking a plant），編成 sparse 2D map。
- **Global force**：均勻 wind field，編成 dense vector field（wind blowing on fabric）。
- 訓練資料：Blender 合成，local force ~11k 植物 + ~12k 球；global wind ~15k。**全部走合成 → real generalization 靠 pretrained visual prior 撐**。
- 訓練成本：四張 A100，一天。

關鍵在於沒有顯式物理 loss、沒有 simulator —— Axis 2 嚴格意義上是 `data-only` + 一個 force 條件通道。能 generalize 到不同材質/物體，是因為 pretrained backbone 已經吃過足夠多 plausible motion，只是需要被「指定一個力」這件事重塑 conditioning。

### 2.2 NewtonGen：Neural Newtonian Dynamics 內嵌 ODE

```
initial state Z₀ = [x, y, vx, vy, θ, ω, s, l, a]
                   │
                   ▼
        Physics-informed Neural ODE
        az z̈ + bz ż + cz z + dz + MLP(Z) = 0
        ──────────────────────────         ───
            linear (learnable)           nonlinear residual
                   │
                   ▼  odeint
        Z(t) trajectory  ─────▶  rasterize to per-frame optical flow
                                         │
                                         ▼
                                Go-with-the-Flow noise warping
                                         │
                                         ▼
                                  CogVideoX-5B diffusion
                                         │
                                         ▼
                                     pixel video
```

- **9-dim latent state**：位置 (x,y)、速度 (vx,vy)、旋轉 (θ, ω)、形狀 (shortest dim s, longest dim l)、有效面積 a。涵蓋平移 + 旋轉 + 變形的最小集合。
- **Hybrid ODE**：學 4 個 scalar (az, bz, cz, dz) 描述 linear 二階動力學，再加一個 MLP 處理 nonlinear residual。這正是 ontology v1.1 review 提的 `architecture-bias-soft`：有物理 inductive bias，但**不保證**守恆律。
- **12 種動作評測**：uniform / acceleration / deceleration / parabolic / 3D / slope sliding / circular / rotation / parabolic+rotation / damped oscillation / size changing / deformation。
- **限制（作者承認）**：multi-object interaction（collision, coalescence）效果差，因為連續 ODE 不擅長 event-driven discontinuity。

### 2.3 兩者放在 ontology 上的差別

```
                injection (Axis 2)              control (Axis 3)            temporal (Axis 4)
                ─────────────────              ──────────────────            ─────────────────
Force Prompting   data-only +                  force + (text via base)      joint-rollout
                  conditioning channel
NewtonGen         architecture-bias-soft       param (state) + text         joint-rollout
                  (neural ODE prior)                                          但 ODE 是 streaming
```

> Per v1.1 review 的「tag by *physics mechanism*, not loss family」原則，NewtonGen 不是 `hard-constraint`（沒有 exact 守恆），也不是 pure `data-only`（架構顯式有牛頓 ODE 形式）。`architecture-bias-soft`（v1.1 提案值）最貼。Force Prompting 本質是 `data-only` + 第三軸多了個 force channel。

---

## 3. 五軸定位 + 同軸對手

| Method | Output | Injection | Control | Temporal | Domain |
|---|---|---|---|---|---|
| **Force Prompting** | pixel-video | data-only + cond-channel | text+**force** | joint-rollout | rigid+generalist |
| **NewtonGen** | pixel-video | architecture-bias-soft | text+**param**(state) | joint-rollout | rigid+generalist |
| Sora (→ [sora.md](../video-world-models/sora.md)) | pixel-video | data-only | text | joint-rollout | generalist |
| Cosmos-Predict (→ [cosmos-wfm.md](../foundation-physics-models/cosmos-wfm.md)) | pixel-video | data-only + sim-data | text+image+camera | clip-parallel | generalist |
| PhysGen (→ `./physgen.md`) | pixel-video | sim-in-loop-train (rigid-body sim) | text+image | single-frame→render | rigid |
| PhysDiff (→ `./physdiff.md`) | motion | sim-in-loop-infer (projection) | text+contact | autoregressive | rigid+soft (human) |
| ContactGen | particle/mesh | constraint-loss | contact | clip-parallel | rigid |

**同軸對手分析**：

- **vs Sora text-only**：Sora 拒絕接受 "5N 往右" 這種 token；只能 "shake the apple"，且不可微調力的方向/量。Force Prompting 把這條 dial 接出來。
- **vs Cosmos-Predict (text+image+camera)**：Cosmos 控制軸沒有 force/param；走 camera trajectory 條件。兩者實際 **正交**，可以疊（Cosmos 出基底 + force prompt 局部干預），未來 Cosmos-Force fork 可能出現。
- **vs PhysGen (hard-sim)**：PhysGen 物理對但需要 image → 3D mesh → 剛體 sim pipeline，asset prep 痛苦；Force Prompting 完全跳過 3D，但物理只是 plausible，不是守恆。trade-off 在 fidelity-vs-controllability 邊界的兩側（→ [`crossing/controllability-vs-fidelity/`](../../crossing/controllability-vs-fidelity/)）。
- **vs PhysDiff (motion + score-conditioned projection)**：PhysDiff 在 denoising loop 裡丟 simulator projection 修正人體 motion；input 是文字+contact。和本篇兩者比，PhysDiff 更接近「事後物理糾錯」，本篇兩者更接近「事前物理指定」。
- **vs ContactGen**：control 軸的 sibling（接觸 vs 力）；ContactGen 給 contact map，本篇給 force vector / Newtonian state。物理層級上 contact ⊂ force（force = ∫ pressure dA over contact area），但操作上 contact 更容易標。

---

## 4. ⚡ shines / ❌ breaks

### ⚡ 真正領先的 regime

- **顯式 dial 換可預測物理輸出**：用戶在 image 上戳一個力 + 角度 + 量值 → 視頻 plausible 地按指令反應；產品上首次拿到「我給力，視頻改變」的 closed loop（before：只能 reroll prompt）。
- **互動式編輯**：Force Prompting 的 demo 顯示同一張圖，改力的方向就能得到不同的 plant bend / fabric ripple；對 creative tooling 是 game changer。
- **NewtonGen 的軌跡一致性**：因 ODE 解算的軌跡是內嵌的，object 不會中途憑空變速、反向加速、漂浮——這幾類是 Sora-class baseline 的常見崩潰模式。
- **小資料 generalize**：Force Prompting 用 ~15k–23k 純合成樣本 + 一天四張 A100，能 generalize 到不同材質/物體——說明 force 這個 conditioning 軸對 pretrained backbone 是相對輕量的擴充。

### ❌ Known failure modes

| Method | Failure | 根因 |
|---|---|---|
| Force Prompting | OOD force（極大 / 物理不可能） | 訓練分布有界，外推就漂 |
| Force Prompting | Base model 沒看過的物理（人推犁、放風箏） | pretrained visual prior 不夠 |
| Force Prompting | 力 ≠ 軌跡，鏈式效應沒建模（推第一張骨牌） | 沒有 dynamics solver，靠 backbone 隱式湊 |
| NewtonGen | Multi-object collision / coalescence | continuous ODE 不擅 event |
| NewtonGen | 形狀劇變 / topology change | 9-dim state 不含 topology |
| 共同 | force-paired video 資料稀缺 | 真實世界沒有 force GT，全靠合成→generalize 賭 pretrained prior |
| 共同 | force vocabulary 有限 | local point force + global wind field 是 Force Prompting 兩種；NND 是 12 種 motion family——超出就 OOD |

---

## 5. Reproduction notes

### Force Prompting
- **Code**: https://github.com/brown-palm/force-prompting (NeurIPS 2025 official)
- **Project page**: https://force-prompting.github.io/
- **Compute**: 4× A100, ~1 day（按論文 claim）
- **Data**: Blender 合成腳本應已於 repo 提供；local force (~11k plant + ~12k ball) + global wind (~15k fabric)
- **Base model**: project page 未明說（[TBD: 確認 backbone — 推測是 CogVideoX 或 SVD 系列，需查 repo `train.py`]）
- **典型踩坑**: force vector 的座標系（image-space vs world-space）容易 mismatch；wind field 的 magnitude scale 須跟訓練分布對齊

### NewtonGen
- **Code**: https://github.com/pandayuanyu/NewtonGen (ICLR 2026)
- **Project page**: https://yuyuanspace.com/NewtonGen/
- **Base**: CogVideoX-5B + Go-with-the-Flow LoRA（依賴 optical-flow noise warping）
- **Env**: CUDA 12.6 / Python 3.10 / PyTorch 2.5.1
- **Compute**: [TBD: 未明說 GPU memory；CogVideoX-5B inference 一般需 ~24GB+ VRAM]
- **設定範圍**: 用戶可設 `(x, y, vx, vy, θ, ω, s, l, a)` 初始值 + `DT` + `METER_PER_PX`
- **典型踩坑**: `METER_PER_PX` 沒校準會讓物理直觀完全對不上像素尺度；multi-object 不要嘗試碰撞

---

## 6. Cross-line synthesis

兩篇放在四條技術路線的座標系上：

```
                  fidelity ▲
                          │     hard-sim (PhysGen)
                          │     │ 物理對但脆
                          │     │
                          │     ●  ← PhysGen
                          │
                          │              ● NewtonGen  ← ODE inside; 軌跡一致
                          │           ●               ← Force Prompting; 力可控
                          │
                          │     ● Sora / Cosmos       ← fidelity 高但 zero force dial
                          │
                          │     ● PhysDiff (motion only, 後驗修正)
                          │
                          └──────────────────────────────▶ controllability (force/param)
```

- **vs hard-sim (PhysGen → `./physgen.md`)**：PhysGen 走 image→3D→sim→render，物理 exact 但管線脆；Force Prompting/NewtonGen 跳過 3D，物理只是 plausible，但部署成本低一兩個數量級。短期 hard-sim 仍是「我需要這顆球真的等加速度 9.8」的選擇，Force/Newton 是「我要創作 + 大致對」的選擇。
- **vs score-conditioned 後驗 (PhysDiff → `./physdiff.md`)**：PhysDiff 在 denoising loop 投影到 physics-valid 流形；本篇兩者在 conditioning 端注入。兩者**可疊**：未來 NewtonGen 出 trajectory → PhysDiff-style projection 守住每幀的接觸合法性。
- **vs constraint-loss (PINN → `./pinn.md`)**：PINN 把 PDE residual 當 aux loss；NewtonGen 走得更深，把 ODE 整段塞進架構（不是 loss）。所以 NewtonGen 是 `architecture-bias-soft`，不是 `aux-loss`。PINN 線今天在 video 還沒打通，主要因 video 沒有逐 pixel 的 PDE GT。
- **Action ladder 的位置**（指向 [`crossing/text-action-trajectory-spectrum/`](../../crossing/text-action-trajectory-spectrum/)）：text < action < trajectory < **force** < contact + param。Force/NewtonGen 把 video gen 條件梯往上推了一格。下一步若有 multi-object force network、contact graph、constraint topology，會繼續往「全 Newtonian scene specification」靠攏。
- **同 control 軸 sibling**：ContactGen — 接觸圖 vs 力向量。物理上 force = ∫ pressure dA over contact area，但人類標 contact 比標 force 容易（手指碰桌面 vs 「以 2N 往右下推」）。長期可能 contact-first 較易拿資料、force-first 較準。

---

## 7. References

### Canonical

- **Force Prompting**: Gillman, N., Herrmann, C., Freeman, M., Aggarwal, D., Luo, E., Sun, D., Sun, C. *Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals.* arxiv **2505.19386**, NeurIPS 2025. Brown University × Google DeepMind. — https://arxiv.org/abs/2505.19386 · https://force-prompting.github.io/ · https://github.com/brown-palm/force-prompting
- **NewtonGen**: Yuan, Y., Wang, X., Wickremasinghe, T., Nadir, Z., Ma, B., Chan, S.H. *NewtonGen: Physics-Consistent and Controllable Text-to-Video Generation via Neural Newtonian Dynamics.* arxiv **2509.21309**, ICLR 2026. Purdue（[TBD: 確認 affiliation —— Stanley Chan 是 Purdue ECE，餘作者推測同 lab]）。— https://arxiv.org/abs/2509.21309 · https://yuyuanspace.com/NewtonGen/ · https://github.com/pandayuanyu/NewtonGen

### Related (同 zone 對手)

- ContactGen — 接觸圖條件，control 軸 sibling
- MotionDirector — text/trajectory 軸的 motion conditioning，沒 force
- ControlNet-video — 通用 conditioning branch 機制；Force Prompting 的 conditioning 注入借鑑這條
- Go-with-the-Flow — NewtonGen 的 noise-warping 底座，optical-flow controlled video
- PhysGen (2409.18964) — hard-sim 對照（→ `./physgen.md`）
- PhysDiff (2212.02500) — score-conditioned 對照（→ `./physdiff.md`）
- Cosmos-Predict (2501.03575) — text/image/camera 條件對照（→ [cosmos-wfm.md](../foundation-physics-models/cosmos-wfm.md)）

### Ontology cross-ref

- v1.1 review 把 Force Prompting 標為 `data-only` 的 force-conditioning channel，NewtonGen 標為新增的 `architecture-bias-soft` —— 詳見 [`ontology-v1.1-review.md`](../../cheat-sheet/ontology-v1.1-review.md) Axis 2 anchors。

---

## 8. §8 Pitfall log

> Source 標註：📄=原論文 / 🌐=project page / 🐙=GitHub issue / 🧪=本倉實測 / 💬=作者 talk。

### §8.1 Force-vector specification ambiguity （Force Prompting）
- **Source**: 🌐 force-prompting.github.io demo + 📄 §appendix
- **Severity**: medium
- **Problem**: 用戶在 image 上指 force = (location, angle, magnitude)，但「magnitude」單位未明確物理量（牛頓？任意 unit？）。論文是相對 scale，跨場景不可比。
- **Workaround**: 在工程使用上把 magnitude 看作 normalized intensity；不要相信 1.0 = 1N。

### §8.2 Force scale OOD （Force Prompting）
- **Source**: 📄 limitations
- **Severity**: high
- **Problem**: 訓練分布的 force magnitude 有上下界；超出（極大 force / 接近零）會崩成 plausible-looking-but-physically-trivial motion。
- **Workaround**: 用 binary search 找模型的 sane range；產品 UI 加 slider clamp。

### §8.3 力 ≠ 軌跡，鏈式效應未建模 （Force Prompting）
- **Source**: 🌐 demo failures（kite, plow）
- **Severity**: high
- **Problem**: 給第一張骨牌一個力，模型不會自動 propagate 到後面所有骨牌；缺 dynamics solver，鏈式 / 多物體傳遞靠 backbone 隱式湊。
- **Workaround**: 把場景拆成多步：每步指一個 force；或改用 NewtonGen（但 NewtonGen 自己也不擅 collision—— §8.5）。

### §8.4 OOD physics that base model never saw （Force Prompting）
- **Source**: 📄 §limitations（人推犁、放風箏）
- **Severity**: medium
- **Problem**: pretrained backbone 沒見過這類動作 → 即使 force 條件對，視覺先驗也回不來。
- **Workaround**: 用更大 / 更廣的 base model；或專門蒐集少量該場景的 force-paired data 做 LoRA。

### §8.5 Multi-object collision/coalescence（NewtonGen）
- **Source**: 📄 §limitations（作者明說）
- **Severity**: high
- **Problem**: NND 是 continuous ODE，不擅 event-driven discontinuity；碰撞 / 合併 / 拓撲變化會生成糊掉的中間態。
- **Workaround**: 限制到單物體 12 motion family；多物體場景考慮 hybrid（NewtonGen + 一個 collision detector → re-init state）。

### §8.6 9-dim state 不含 topology / material（NewtonGen）
- **Source**: 📄 state definition
- **Severity**: medium
- **Problem**: `[x, y, vx, vy, θ, ω, s, l, a]` 沒有質量、不含 material elasticity、不含 topology。所以「重的箱子 vs 輕的箱子掉」這種純由 mass 區分的指令給不出來。
- **Workaround**: 走 Force Prompting 線（它的 ~15k 訓練樣本實測有「some initial hints at mass understanding」），或把 mass 編進 text prompt 讓 backbone 補。

### §8.7 METER_PER_PX 校準誤差 （NewtonGen）
- **Source**: 🐙 repo config
- **Severity**: medium
- **Problem**: 用戶設定 `METER_PER_PX` 不準 → 物理直觀 (9.8 m/s²) 對不上像素 (px/frame²) → 視覺速度 plausibility 崩。
- **Workaround**: 對標 reference scene 校準；提供 calibration UI / preset。

### §8.8 訓練資料稀缺 — force-paired video 全靠合成（共同）
- **Source**: 📄 Force Prompting §data; NewtonGen 依賴 Go-with-the-Flow 的 optical-flow paired
- **Severity**: structural
- **Problem**: 真實世界沒有 force ground truth；全靠 Blender / sim 生成 paired data → real-world generalization 賭的是 pretrained visual backbone 的內插能力。長期 scaling 不像 text-video 那樣有 internet-scale corpus。
- **Workaround**: 投資 sim2real 增強；或從 robot teleop / haptic glove 抓 force-real-video pair（產業視角是 robotics community 的機會）。

---

*Last updated: 2026-05-25 · Sources verified via WebSearch + WebFetch on arxiv abs/html, force-prompting.github.io, yuyuanspace.com/NewtonGen, github.com/brown-palm/force-prompting, github.com/pandayuanyu/NewtonGen.*
