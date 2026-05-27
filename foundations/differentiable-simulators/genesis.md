<!-- ontology-5axis
output=N/A (state/contact/RGB via render)
injection=sim-in-loop-train | hard-constraint (MPM)
control=action | trajectory | force | contact | param
temporal=streaming
domain=robotics | rigid | soft | fluid | granular
ref: ../../cheat-sheet/ontology.md §differentiable-sim
-->

# Genesis 解構（Genesis-Embodied-AI Universal Simulator）

> **發布時間**：2024-12-19 · GitHub release（無正式 arXiv paper）
> **項目**：*Genesis: A Universal and Generative Physics Engine for Robotics and Beyond*
> **主導**：CMU 牽頭，20+ 實驗室合作（含 MIT、Stanford、Tsinghua、UMD 等）；Apache 2.0
> **核心定位**：第一個把 rigid + articulated + MPM + SPH + FEM + PBD + Stable-Fluid **塞進同一個 Taichi-backed Python 引擎**的開源 sim；定位「世界最快」 + 「生成式 4D 世界引擎」 — 但 marketing 宣稱的 430,000× realtime 在 contact-rich + render-on 工況實測掉到 **~10× realtime**，speed claim 與真實 envelope 落差 ~150×。

**Status:** v0.5 — 解構基於 README + GitHub issue #181 + Stone Tao substack + MuJoCo discussion #2303 + 個人 install 跑過 examples。完整 rigid-body autodiff GA 時程、Taichi MPM 數值穩定性曲線、AMD ROCm 成熟度待維護者升 v1。

**TL;DR:** Genesis 做四件事：① 用 **Taichi DSL** 把多 solver（rigid/MPM/SPH/PBD/FEM/Stable-Fluid）統一到單一 Python API，零 C++ binding；② 把 rigid-MPM / rigid-SPH / rigid-PBD **三條 coupling layer** 公開，這是 MJX/Brax/Warp 都缺的；③ 自帶 LuisaRender + 「text→4D scene」生成式資料管道（design 賣點，code 完整度未確認）；④ 行銷宣稱 4090 跑 Franka 43M FPS / 430,000× realtime — **但 Stone Tao 獨立實測證明開 self-collision + multi-substep + render 後實質 ~10× realtime，落差 150×**。對本 handbook 讀者的關鍵事實：marketing 430,000× → 真實基準 **~10× realtime**（render on, contact-rich）。

**X-Ray.** Genesis 是 **DiffTaichi (2019) → Taichi runtime (2020) → cross-material 統一 sim** 這條線最大膽的一次封裝賭注 —— 把 Yuanming Hu 的 DiffTaichi 物種放大到「全 material + robotics 整合 + photoreal render + 生成式 pipeline」。它真正的、與行銷無關的 contribution 是 **§1.2 那個 cross-material coupling 抽象**：rigid arm × soft dough × granular sand 同場演化在 MJX/Warp 公開版仍要自己拼。但 release 後 Stone Tao（ManiSkill author）公開 benchmark 解構（**issue #181** + substack），把「43M FPS」拆成「self-collision 關 + 999 步無動作 + render 關」的合成基準，這成為整個 diff-sim 領域的信任危機案例 —— 它讓社群第一次系統性問「**marketing claim vs operational envelope**」這個問題。對 Physics-Gen handbook 讀者意義：(a) 當你的 generation 任務 *必須* 同時涵蓋 rigid+soft+fluid，Genesis 仍是目前唯一的開源單 API oracle；(b) 但選型決策不能讀 README 數字 — 必須區分 design contribution 與 perf marketing，本篇就是這個拆解。打不開的 envelope：純 rigid manipulation（MJX-Warp 已反超）、需要完整 rigid `.grad` 的 diff-MPC（rigid autodiff 從 2024-12 拖到 2026-05 仍未 GA）、需要穩定 photoreal RGB throughput 的 video-WM 數據工廠（render-on 掉 4 個數量級）。

## 📍 研究全景時間線

```ascii
   2019              2020              2024-12              2025              2026-05?
   DiffTaichi ────► Taichi ───────► YOU ARE HERE ──────► MJX-Warp ──────► MuJoCo-Warp + Genesis-2
   ICLR 2020        SIGGRAPH 2020   Genesis release      rigid 反超       rigid GA + render fix?
   PDE-first        DSL runtime     cross-material        Genesis rigid    cross-material 
   single-domain    backend-agnostic 統一 ★               on rigid         + diff-rigid 整合?
   research toy     building block  全 material + 生成   only domain       
   └─ diff-sim 路線 ──────► 工具化 ─────► Genesis 大封裝 ────► 反思期 ─────► 整合期 ──►
                                          (含 marketing 偏差)
```

★ = 主要新點：**cross-material coupling + Taichi 多 backend + 生成式 data engine + 開源 Apache 2.0**。**仍未解：rigid-body fully differentiable（一年多 coming soon）、render-on throughput、contact gradient noise、reproducible marketing benchmark**。

---

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 同軸對手

| 維度 | DiffTaichi (2019) | MJX / Warp (2024-) | **Genesis** |
|---|---|---|---|
| Domain coverage | PDE only（soft / fluid） | rigid（MJX）/ rigid+soft via SDK（Warp） | **rigid + articulated + MPM + SPH + FEM + PBD + Stable-Fluid 同 API** |
| Cross-material coupling | ❌ 單 solver | 需自己拼 | **rigid-MPM / rigid-SPH / rigid-PBD 三條原生 coupling** |
| Language stack | Taichi DSL + Python | C++ + JAX/Python binding | **Python 100% + Taichi kernel** |
| Backend | CUDA | CUDA only (MJX) / CUDA (Warp) | **CUDA + AMD + Apple Metal**（Taichi runtime） |
| Differentiability | ✅ full（PDE） | ✅ rigid (MJX-JAX) | **部分**：MPM + Tool Solver only；rigid「coming soon」 |
| Photoreal render | ❌ | ❌（外掛 Omniverse） | **內建 LuisaRender / OptiX** |
| Generative data pipeline | ❌ | ❌ | **text → 4D scene**（design 賣點，code 完整度未確認） |
| License | MIT | Apache 2.0 / BSD | **Apache 2.0** |
| Speed marketing | 學術 paper FLOPs | 內部 benchmark | **「430,000× realtime」（已被 #181 質疑）** |

### 1.2 ⚡ Eureka Moment

> **核心 trick 一句話** —— 用 Taichi DSL 把「多 solver + coupling layer」抽象成統一 spatial-hash + broad-phase，所有 material 共用 contact 介面，Python 開發者第一次能在單一 file 同場跑 rigid arm × soft dough × granular sand × liquid。

直覺：MJX 強在 rigid contact LCP，Warp 強在 PDE kernel，但 cross-material coupling **沒有單一引擎做過開源**。Genesis 賭的是「coupling abstraction」這個 design 比「單 domain 最快」更重要 —— 即使單 domain 它不是最快（事實上在 rigid 上已被 MJX-Warp 反超），它的位置仍獨佔。**這個 contribution 獨立於速度爭議，是真實的工程貢獻**。

### 1.3 信息流（架構圖）

```ascii
                  Genesis Scene (Taichi-kernel-on-GPU)
   ┌───────────────────────────────────────────────────────────┐
   │                                                            │
   │   text/URDF/USD ─┐                                         │
   │                  ▼                                         │
   │           ┌─Rigid Solver─┐   ┌─MPM Solver─┐                │
   │           │  articulated │   │  soft /    │                │
   │           │  contact LCP │   │  granular  │ ◄── ✅ diff   │
   │           └──────┬───────┘   └─────┬──────┘                │
   │                  │                 │                       │
   │           ┌──────┴─── coupling layer ─────┐                │
   │           │  rigid↔MPM ↔ SPH ↔ PBD ↔ FEM  │  ★ USP        │
   │           └──────┬─────────────────────────┘                │
   │                  │                                          │
   │           ┌─SPH─┴┐ ┌─PBD ─┐ ┌─FEM ─┐ ┌─Stable-Fluid─┐      │
   │           │ liq │ │ cloth│ │ defor│ │  smoke/gas    │      │
   │           └─────┘ └──────┘ └──────┘ └───────────────┘      │
   │                                                            │
   │   ⬇  Tool Solver  (✅ differentiable)                      │
   │   ⬇  LuisaRender / OptiX  (photoreal RGB)                  │
   │   ⬇  Generative data engine  (text → 4D)                   │
   └───────────────────────────────────────────────────────────┘
                              │
       ─────────────────────► state / contact / force / RGB
       (sim-in-loop oracle for video-WM, VLA, surrogate）
```

對比 prior：DiffTaichi 只有最下游兩條（PDE solver + diff），MJX 只有 rigid + diff，Warp 是 kernel-level building block 沒整合 robotics。Genesis 第一次把這整個 stack 封裝在 Python import 之內。

---

## §2 · 數學層

### 📌 Napkin Formula

```
   Taichi Source-to-Source Transformation (SCT) for differentiation:
   
   forward kernel  k(x):  Taichi AST ──► CUDA kernel
                                │
                                ▼ source-level reverse-mode
   adjoint kernel  k̄(x): Taichi AST ──► CUDA kernel (auto-generated)
   
   ∂L/∂θ = Σ_t  k̄_t( state_t, adjoint_{t+1} )       ← chain through time
   
   Cost: O(T · |kernel|)   memory: checkpointed O(√T · |state|)
         vs PyTorch tape:  O(T · |state|) memory ← Taichi 比 tape 省
```

**直覺**：Taichi 的 differentiation 不是 record-and-replay tape（PyTorch / JAX）— 它在 **kernel AST 層級做 source code transformation**，反向 kernel 是「生成」出來而不是「追蹤」出來。這對 PDE-style 巨量元素 simulation 是關鍵：tape memory ∝ state size × T 會直接 OOM，SCT 只需要保留 checkpoint。

**但**：SCT 自動微分對 **contact discontinuity** 仍無解 — 接觸/分離瞬間 gradient 噪聲是 hard-contact sim 通病，Genesis 沒提出新 contact model（不像 arXiv 2506.14186 那類「soft gradient for hard contact」專門方案）。所以即使 rigid `.grad` 將來 GA，diff-MPC 在 contact-rich 任務仍會 noisy。

### 2.x Coupling 損失（design level）

cross-material coupling 用 unified contact constraint：

```
   ∀ pair (rigid_i, MPM_particle_j):
      penalty_λ · max(0, -signed_distance(i,j))²  + friction(μ)
```

具體 λ / friction model / restitution `UNVERIFIED` 待維護者讀 source 補。

---

## §3 · 數據層 / 訓練 scale

Genesis **本身不是訓練的模型**，它是 **sim-in-loop oracle + 數據工廠**。所以「訓練 scale」這節改問：作為 oracle 它能餵多大規模？

| 用途 | 規模上限（實測） | 瓶頸 |
|---|---|---|
| 單環境 rigid (Franka) | 4090 上理論 43M FPS（marketing）/ 實測 0.29M FPS（self-collision + 連續 action） | 見 §8.1 / §8.2 |
| 多環境 parallel rigid | 推薦 A100 / 4090 ×多卡 | broad-phase scaling `UNVERIFIED` |
| Cross-material (rigid + MPM + SPH 同場) | 單 4090 約數萬 FPS 量級 `UNVERIFIED` | MPM grid resolution |
| Render-on (LuisaRender) | ~10× realtime（Stone Tao 實測） | render pipeline 是 §8.5 卡點 |
| Generative text→4D scene | demo 多，公開 code 完整度未確認 | §8 [TBD: verify] |

**關鍵事實**：對「用 Genesis 當 video-WM 訓練資料工廠」這個用途，render-on throughput 是直接瓶頸（§8.5）— Isaac Lab / ManiSkill 同任務 render-on 約 1,000× realtime。**所以 Genesis 餵 pixel-WM 不如 Isaac，餵 latent-WM / state-WM 才合理**。

---

## §4 · 代碼層

| 項 | 狀態 |
|---|---|
| Repo | https://github.com/Genesis-Embodied-AI/Genesis |
| Project page | https://genesis-embodied-ai.github.io/ |
| Docs | https://genesis-world.readthedocs.io/ |
| License | **Apache 2.0**（已查 LICENSE 檔） |
| Install | `pip install genesis-world` 或 `git clone + pip install -e .` |
| Min GPU | RTX 3060 12GB（單環境 rigid+MPM）；推薦 4090/A100 (parallel) |
| Backend | CUDA / AMD ROCm / Apple Metal（後兩者社群實測案例少 — [TBD: verify]） |
| Differentiability | MPM + Tool Solver ✅；rigid ❌「coming soon」一年 |
| Streaming（state output） | ✅ |
| Photoreal render | ✅（LuisaRender / OptiX 內建） |
| 正式 paper | ❌（release 走 GitHub + project page；子系統 ThinShellLab / DiffTactile 有 ICLR 2024 paper） |
| Maintainer 響應 | issue #181 後出修正 benchmark，承認部分問題；rigid `.grad` GA 時程仍未公開 |

典型踩坑：
- `gs.init(backend=gs.cuda | gs.metal | gs.cpu)` numerical 不 bit-exact
- MPM `grid_size` 預設偏粗 → soft demo「沒物理感」常是這個沒調
- **不要直接信 README benchmark** → 跑 `examples/speed_benchmark.py` 並**開 self-collision + 多 substep + 連續 random action**
- rigid `.grad` 還沒上 → diff-MPC 工作流目前要降級到 MPM-only

---

## §5 · 評測 / Benchmark

### 5.1 已知數字（marketing vs 實測）

| Benchmark / 工況 | Marketing | Stone Tao 實測 | Δ | 來源 |
|---|---|---|---|---|
| Franka 4090, self-collision OFF, 999 idle steps | **43M FPS / 430,000× realtime** | reproducible（但 setup 不真實） | — | README + #181 |
| Franka 4090, self-collision ON, random action | — | **0.29M FPS** | **~150× 落差** | Stone Tao substack |
| Contact-rich manipulation vs ManiSkill/SAPIEN | 「最快」 | **3-10× slower** | — | Stone Tao |
| Render ON, video-WM data generation | 「最快」 | **~10× realtime** | Isaac Lab/ManiSkill ~1,000× | Stone Tao |
| Rigid-only vs MJX-Warp (2025+) | 「10-80× MJX」 | **MJX-Warp 已追上或反超** | — | mujoco discussion #2303 |

### 5.2 解讀

- **真實 capability**：cross-material 同場（rigid + soft + fluid 共存）— 這個沒有對手，是 design contribution。
- **Goodhart / benchmark gaming**：43M FPS 數字是 (a) self-collision 關 (b) 999 步無動作 → rigid solver early-exit (c) render 關 三條合成出來的；單獨 (b) 就會讓任何 sim 的 throughput 飆。**這不是 capability 數字，是「最寬鬆配置」數字。**
- **Genesis team 後續修正**：#181 之後放出修正 benchmark，**保留 43M FPS (self-collision-only) 但承認 random-action 場景掉到 27M FPS**。仍未對應 Stone Tao 0.29M 的 contact-rich 場景。

---

## §6 · Issues & Limitations

### 6.1 官方自述（或社群拷問後承認的）limitations

- **Differentiability 表面承諾 > 實際覆蓋**：docs 一年多反覆「coming soon」對 rigid
- **Constraint solver 調參敏感**：作者承認 demo video 中 cube 穿過 gripper / 漂浮是「poorly tuned constraint solver configurations」
- **Camera + render throughput 崩**：未在 README 標明，需用戶自己踩
- **缺正式 peer-reviewed paper**：學術引用 + 工程 baseline 兩用都不穩

### 6.2 Hidden Assumptions（推斷）

- **Taichi runtime stable across backend**：CUDA / Metal / ROCm numerical 不完全 bit-exact；CI/regression 怎麼跨 backend `UNVERIFIED`
- **Cross-material coupling 對 stress-test 場景的穩定性**：rigid×MPM coupling 在劇烈接觸（爆炸/破裂）下 numerical stability `UNVERIFIED`
- **Generative data engine code completeness**：「text → 4D scene」demo video 多，code 完整度 [TBD: verify]
- **Multi-GPU parallel scaling**：broad-phase 在多卡 GPU 下 scaling 曲線未公開
- **AMD ROCm + Apple Metal 在生產用例的成熟度**：社群實測案例少，[TBD: verify]

### 6.x GitHub-validated 失敗模式

| 失敗 / 問題 | GitHub evidence | 嚴重度 |
|---|---|---|
| Marketing benchmark 設定不真實 | [#181 StoneT2000](https://github.com/Genesis-Embodied-AI/Genesis/issues/181): 「benchmark used fastest physics setting; one action followed by 999 idle steps; self-collision off by default」 | 🔴 影響選型決策 |
| Rigid `.grad` 拖延 | docs 0.4.x: 「differentiability for other solvers being added soon (starting with rigid-body)」 — 2024-12 → 2026-05 仍未 GA | 🔴 硬限制 diff-MPC |
| 與 MJX-Warp 相對位置反轉 | [mujoco discussion #2303](https://github.com/google-deepmind/mujoco/discussions/2303): rigid-only MJX-Warp 已追上或反超 | 🟠 純 rigid 改用 MJX |
| Demo video glitch | 作者公開承認 cube glitch 來自 constraint solver 調參 | 🟠 預設值不可信 |
| Render-on throughput | Stone Tao 量到 render on 掉到 ~10× realtime | 🔴 卡 video-WM data gen |
| 缺正式 paper | release 走 GitHub + 部落格 | 🟡 學術引用不穩 |
| Contact discontinuity 未解 | Genesis 沒提新 contact model | 🟠 contact-rich diff-MPC noisy |
| Stone Tao 二手實測 substack | 「150× off」「contact-rich 3-10× slower」 | 🔴 publication-tier critique |

**Maintainer 響應度**：#181 有回應並出修正 benchmark；但 rigid `.grad` GA 時程、ROCm/Metal 成熟度、generative engine code 完整度，問題仍懸（2026-05 時點）。

---

## §7 · 比較 & 面試 Tip

| 同軸對手 | Injection | Diff? | Cross-material? | Render | Open? | Status (2026-05) |
|---|---|---|---|---|---|---|
| **Genesis** | sim-in-loop + hard-constraint(MPM) | MPM+Tool only | ✅ rigid+MPM+SPH+PBD+FEM | ✅ LuisaRender | ✅ Apache 2.0 | shipped 0.4.x, rigid grad pending |
| MuJoCo MJX | sim-in-loop | ✅ rigid (JAX) | ❌ rigid only | ❌ external | ✅ Apache 2.0 | shipped, MJX-Warp 反超 rigid |
| NVIDIA Warp + Isaac Lab | sim-in-loop | partial | rigid+soft via SDK | ✅ Omniverse | ✅ open core | shipped, Cosmos 整合最強 |
| Brax | sim-in-loop | ✅ (JAX) | ❌ rigid only | ❌ | ✅ Apache 2.0 | shipped, locomotion RL 標配 |
| DiffTaichi (2019) | sim-in-loop (PDE) | ✅ full | PDE only | ❌ | ✅ MIT | research baseline (parent line) |
| Aerial Gym | sim-in-loop | partial | rigid+aerial | ❌ | ✅ MIT | aerial-only specialized |

> **🎤 Interview Tip.** 「Genesis 比 MuJoCo MJX 快 150 倍，我們該換嗎？」**正確答**：「**看實測，不看 marketing**。43M FPS / 430,000× 是 self-collision 關 + 999 idle steps + render 關 三條合成出來的；Stone Tao 實測 contact-rich + render on 後是 ~10× realtime，比 ManiSkill/SAPIEN **慢 3-10×**，比 MJX-Warp 也已被反超。所以 (a) 如果你做純 rigid manipulation / VLA 評測 → 用 MJX（生態 + reproducibility 都贏）；(b) 如果你做 cross-material 同場（rigid+soft+fluid）→ Genesis 仍是唯一開源選項；(c) 如果你做 diff-MPC → rigid `.grad` 還沒 GA，用 MJX-JAX。」**錯答**：「Genesis 號稱最快所以換」— 沒分清 marketing benchmark 配置 vs operational envelope，是 diff-sim 領域 2024-2026 信任危機的核心 case study。

### 7.1 Falsifiable predictions

1. **2026-12 前**：Genesis 0.5.x ship rigid-body fully differentiable GA（已拖一年半，社群壓力 + MJX-JAX 對比下大概率 ship；若再拖則「coming soon」承諾失信，diff-sim 選型主流轉 MJX）。
2. **2027-06 前**：Genesis render-on throughput 不會追上 Isaac Lab/ManiSkill（~1,000× realtime）— LuisaRender 整合是工程 debt，且這不是 Genesis 團隊的核心 priority；想要 photoreal 大量訓練資料的 video-WM 工作流仍會留在 Omniverse / Isaac Lab。
3. **2027-12 前不會發生**：Genesis 取代 MJX 成為 VLA 評測事實標準 — rigid-only 場景 MJX 生態（OpenVLA / RT-2 / Octo 都用 MJX）已深；Genesis 的位置是「cross-material 同場」的補集，不是 rigid 標準的替代。

---

## §8 · For the Reader（按 persona 分流）

- **VLA / robot policy 工程師** —— 純 rigid manipulation 繼續用 MJX/ManiSkill，生態深、reproducibility 好。Genesis 留給 *必須* cross-material（攪粥、軟體機器人、granular pile）的任務。**不要信 README 速度數字**，自己跑 self-collision-on benchmark。
- **影片生成工程師（video-WM）** —— 想用 Genesis 當 photoreal 數據工廠？render-on 是 ~10× realtime，Isaac Lab 是 ~1,000× — **拆 pipeline**：state-only fast rollout 在 Genesis，photoreal render 走 Isaac/Omniverse 或 offline batch。或直接 latent-WM 路線繞過 RGB throughput。
- **神經 surrogate 研究者** —— Genesis 的 MPM/SPH ground truth 是 FNO / MeshGraphNet 訓練 oracle 的好選擇 — cross-material coverage 比 DiffTaichi 廣，這是 surrogate 路線的自然擴展。MPM `.grad` 已 GA，可以做 diff-aware surrogate。
- **Diff-MPC / sim-in-loop training 研究者** —— rigid `.grad` 還沒 GA，**繞道**：(a) rigid 用 MJX-JAX；(b) MPM/soft 用 Genesis 的 Tool Solver 的 diff；(c) hybrid 用 MJX rigid + Genesis MPM 各自獨立 backward 後拼。
- **物理 conditioning 研究者（pixel-WM × diff-sim）** —— Genesis 的 cross-material state 當「物理正確」 reference，餵 Cosmos-Predict / Sora 類 WM 做 physics-fidelity loss — 這個 composition 在其他 sim 上拼不出來（Isaac Lab 缺 fluid，MJX 缺 soft），是 Genesis 真正獨佔的 generation 接點。
- **Research 學生** —— Genesis 是 *讀懂 diff-sim 路線* 的最好教材（Python 100%，Taichi DSL 透明），但要批判性看 marketing。把 §8 pitfall log 當必讀附錄。注意 §7.1 三條預測，rigid `.grad` 與 render throughput 是兩個值得追蹤的 KPI。

---

## §8 · Pitfall Log（8 條，編號保留 v1 對齊）

| # | Issue / 來源 | 原文摘錄 / 數據 | Severity | Workaround |
|---|---|---|---|---|
| 8.1 | [Issue #181](https://github.com/Genesis-Embodied-AI/Genesis/issues/181) by StoneT2000 | (a) benchmark 用「fastest physics setting」其他 tutorial 不用；(b) 「one action followed by 999 steps of no actions」→ rigid solver early-exit；(c) self-collision 預設關 | 🔴 **Critical** | 自己重跑：開 self-collision + multi-substep + 連續 random action。team 後續放修正 benchmark，**保留 43M FPS (self-collision-only) 但承認 random-action 場景掉到 27M FPS** |
| 8.2 | Stone Tao substack 二手實測 | 「publicly reported numbers are off by **150×**」「3-10× slower than ManiSkill/SAPIEN on collision-rich manipulation」「rendering on: drops to ~10× realtime vs Isaac Lab/ManiSkill ~1,000×」 | 🔴 **High** | 純 rigid 用 MJX/ManiSkill；Genesis 留給 cross-material 任務 |
| 8.3 | Differentiability 覆蓋不完整 | docs (0.4.x)：「MPM solver and Tool Solver currently differentiable, differentiability for other solvers being added soon (starting with rigid-body)」— rigid `.grad` 從 2024-12 拖到 2026-05 仍未 GA | 🔴 **High** | rigid diff 改用 MJX-JAX / Brax；Genesis 只在 MPM/soft 用其 autodiff |
| 8.4 | Constraint solver 調參敏感 | 作者公開承認「cubes glitching stems from poorly tuned constraint solver configurations」 | 🟠 **Medium** | Genesis Discord / Discussions 對齊「known good」config；不信 example 預設值 |
| 8.5 | Camera + render throughput | Stone Tao：「render on → 430,000× → ~10× realtime」 | 🔴 **High** | 拆 pipeline：state-only fast rollout 在 Genesis；photoreal render 走 Isaac/Omniverse 或 offline batch |
| 8.6 | 缺正式 paper / peer review | release 走 GitHub + project page；社群是主要驗證渠道 | 🟡 **Medium** | 引用時標 commit SHA + 自跑 benchmark；不引「世界最快」表述 |
| 8.7 | Contact discontinuity 老問題未解 | Genesis 沒提新 contact model（vs arXiv 2506.14186）；hard-contact gradient 仍 noisy | 🟠 **Medium** | MPM 軟接觸近似；或外掛 implicit-diff contact scheme |
| 8.8 | 與 MJX-Warp 的相對位置 | [mujoco discussion #2303](https://github.com/google-deepmind/mujoco/discussions/2303) 社群共識：rigid-only 場景 MJX-Warp 已追上或反超 | 🟠 **Medium**（選型 default 仍應是 MJX） | rigid VLA 選 MJX；Genesis 只在 cross-material 同場時 first choice |

---

## References

- **Genesis** — Genesis-Embodied-AI · 2024-12-19 GitHub release · [repo](https://github.com/Genesis-Embodied-AI/Genesis) · [project page](https://genesis-embodied-ai.github.io/) · [docs](https://genesis-world.readthedocs.io/) · Apache 2.0
- **DiffTaichi (parent line)** — Hu et al. *ICLR 2020* · [arXiv:1910.00935](https://arxiv.org/abs/1910.00935)（diff-sim 路線源頭）
- **Taichi runtime** — Hu et al. *SIGGRAPH 2020* · DSL + multi-backend
- **MuJoCo MJX** — Google DeepMind · [repo](https://github.com/google-deepmind/mujoco)（同軸 rigid 對手）
- **NVIDIA Warp** — [repo](https://github.com/NVIDIA/warp)（同軸 kernel-level）
- **Stone Tao critique** — *"How fast is the new hyped Genesis simulator?"* [substack](https://stoneztao.substack.com/p/the-new-hyped-genesis-simulator-is) · [benchmark repo](https://github.com/zhouxian/genesis-speed-benchmark) — **最關鍵的二手分析**
- **GitHub Issue #181** — StoneT2000 · [link](https://github.com/Genesis-Embodied-AI/Genesis/issues/181)
- **MuJoCo Discussion #2303** — 「Genesis claims 10-80× MJX, true?」 · [link](https://github.com/google-deepmind/mujoco/discussions/2303)
- **Silicon Valley Robotics Center 2026 RL sim comparison** — [link](https://www.roboticscenter.ai/rl-environments/best-2026)
- **Marketing-side（批判性讀）**：The Decoder「430,000× faster than reality」· MarkTechPost 2024-12-19 release writeup

---

## Boundary

- **Parent diff-sim line（DiffTaichi 完整解構）** → [`./difftaichi.md`](./difftaichi.md)
- **同軸 rigid 對手（MJX 完整解構）** → [`./mujoco-mjx.md`](./mujoco-mjx.md)
- **同軸 kernel-level（Warp 完整解構）** → [`./nvidia-warp.md`](./nvidia-warp.md)
- **aerial 專用 sim（cross-ref）** → [`./aerial-gym.md`](./aerial-gym.md)
- **與 video-WM 組合（physics-fidelity loss）** → [`../foundation-physics-models/cosmos-wfm.md`](../foundation-physics-models/cosmos-wfm.md)
- **與 latent-WM 組合（reconstruction target）** → [`../latent-world-models/dreamer-v4.md`](../latent-world-models/dreamer-v4.md)
- **與 neural surrogate 組合（FNO oracle）** → [`../neural-surrogates/fno.md`](../neural-surrogates/fno.md)
- **5 axis 全景** → [`../../cheat-sheet/ontology.md`](../../cheat-sheet/ontology.md)

---

## ✍️ 維護者註（v0.5 → v1 升級清單）

本 v0.5 基於 README + #181 + Stone Tao substack + MJX discussion #2303 + 個人 install 跑 examples。下次升 v1 時補：

1. ⏳ Rigid-body `.grad` GA 狀態（2026-12 前 falsifiable check）
2. ⏳ AMD ROCm + Apple Metal backend 在生產用例的成熟度（社群 case studies）
3. ⏳ 「Generative data engine (text → 4D scene)」具體 code 完整度 — release demo 多，公開 code 是否完整未確認
4. ⏳ Taichi SCT 在 cross-backend 下的 numerical 一致性 / CI regression 機制
5. ⏳ Multi-GPU parallel scaling 曲線（單卡 → 多卡 broad-phase）
6. ⏳ MPM grid_size / SPH particle density 對 cross-material coupling 穩定性的曲線
7. ⏳ 任何 2026-Q3+ 出來的獨立第三方 reproduction benchmark
8. ⏳ Status v0.5 → v1，刪本節

---

[← Back to Differentiable Simulators](./overview.md)

Sources:
- [Genesis GitHub](https://github.com/Genesis-Embodied-AI/Genesis)
- [Genesis Project Page](https://genesis-embodied-ai.github.io/)
- [Issue #181 — Speed benchmark critique](https://github.com/Genesis-Embodied-AI/Genesis/issues/181)
- [Stone Tao — substack critique](https://stoneztao.substack.com/p/the-new-hyped-genesis-simulator-is)
- [MuJoCo Discussion #2303](https://github.com/google-deepmind/mujoco/discussions/2303)
- [Silicon Valley Robotics Center 2026 comparison](https://www.roboticscenter.ai/rl-environments/best-2026)
