<!-- ontology-5axis output=field|particle injection=aux-loss control=param|3d-init temporal=autoregressive domain=fluid|rigid|soft -->

# PINN 解構（Physics-Informed Neural Networks）

> **發布時間**：2017-11 arXiv（[1711.10561](https://arxiv.org/abs/1711.10561) Part I + [1711.10566](https://arxiv.org/abs/1711.10566) Part II）· 2019 J. Comput. Phys. 378 686-707（合併正式版）
> **論文**：*Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations*
> **作者**：Maziar Raissi, Paris Perdikaris, George Em Karniadakis（Brown University，Karniadakis 是 spectral/hp element 與 scientific ML 元老）
> **核心定位**：本 handbook **`aux-loss` injection 線的鼻祖 + canonical anchor**。整個 physics-conditioning USP zone 對它都有 strong dependence —— PhysGen / PhysDiff / Force Prompting 的 aux-loss 改良版全是 PINN 思想再包裝。但是要從 v0.5 升 v1 必須讀 Krishnapriyan 2021 + Wang NTK 2022 揭露的 failure modes，不是讀原 paper 的 Burgers demo。

**Status:** v0.5 — 解構基於 Raissi 2017-2019 三篇原文 + Krishnapriyan NeurIPS 2021 + Wang NTK JCP 2022 + DeepXDE 8 條 GitHub issue 採樣。完整 SOTA 變體 benchmark 表（SA-PINN / Causal PINN / XPINN / hash-encoded PINN）待維護者升 v1。
**TL;DR:** PINN 做的事可以一句話講完 —— **用 autograd 對 NN 輸出取偏導數，把 PDE residual `‖N[u_θ]‖²` 當 auxiliary loss 加進總 loss**，於是 forward / inverse PDE 都變成同一套 optimization。這個 trick 從 2017 流傳到今天仍是 aux-loss baseline；但**真正該記得的事實**：原 paper 在 Burgers / Schrödinger demo 漂亮，Krishnapriyan 2021 把 convection coefficient 從 small 調大，vanilla PINN error 直接從 10⁻³ 跳到 O(1) —— **NTK 失效分析（Wang 2022）證明 loss term 之間 gradient 量級差 2-4 個數量級，這個理論根因 2022 才補上**，前面 5 年所有人都在瞎調 λ。

**X-Ray.** PINN 是這本 handbook 整個 `physics-conditioning` zone 的**祖父節點**：v2 ontology axis 2 的 `aux-loss` 就是它定義出來的 idiom。所有後續 video / scene / NeRF + physics loss（PhysGen 的 rigid-body ODE loss、PhysDiff 的 score guidance、Force Prompting 的 attention conditioning）都是把「**PDE 寫成可微 residual + 加進 total loss**」這個模式換 domain 包裝再用一次。**但讀者更該知道的是 failure mode 而非 demo**：Krishnapriyan NeurIPS 2021（[2109.01050](https://arxiv.org/abs/2109.01050)）系統性證明 vanilla PINN 在 convection / reaction / diffusion 三類 operator 上稍微偏離 trivial regime 就崩；Wang Yu Perdikaris JCP 2022（[2007.14527](https://arxiv.org/abs/2007.14527)）用 Neural Tangent Kernel 解釋了**為什麼 loss-weight 怎麼調都不對** —— `L_pde` 與 `L_bc` 的 gradient eigenvalue 跨 10²-10⁴ 量級，主導項把 NN 訓往單一方向、另一項根本沒在學。對 **video / 大 scale generation** 而言，直接把 PDE residual 套到 video diffusion 完全不 work（score gradient 與 PDE residual gradient 方向不相容），但 **PhysGen 走的「physics loss 只在中介 latent 加、pixel 端用 generative diffusion 渲染」是 PINN 思想的活路**。這就是為什麼這篇要寫成 anchor：理解 PINN failure → 理解整個 zone 的 lower bound。

## 📍 研究全景時間線

```ascii
   1990s         2017-11           2019            2021-09           2022                YOU ARE HERE 2024-25
   autodiff ──► PINN arxiv ────► PINN JCP ────► Krishnapriyan ───► Wang NTK ─────► PhysGen / PhysDiff /
   (Griewank)   (Raissi)         (canonical)    NeurIPS failure   JCP theory       Force Prompting
                                                modes              of why          aux-loss 後代
                                                                                    ↘
                                                                  ★ injection idiom 從 PDE-MLP 擴張到 diffusion-video
   ☆ enabling   ★ paradigm        ★ headline    ★ "demo OK,        ★ ill-conditioned
     tech       shift             demos         production fails"  optimization

   └─ autograd 可算高階偏導 ─► PDE as soft loss ─► failure exposed ─► theory closes ─► generative descendants ─►
```

★ = 主要新點。**仍未解（留給下一代）**：(a) hard-constraint 嚴格滿足 Neumann / 守恆律的 clean 形式；(b) chaotic / turbulent regime 的 long-horizon rollout；(c) PDE residual 與 diffusion score 兩種 gradient 方向相容的統一框架。

---

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 純資料驅動 NN（pre-2017） | Mesh-based FEM / FD solver | **PINN（Raissi 2017）** |
|---|---|---|---|
| **PDE 知識的注入點** | 完全不用 PDE，靠 supervised pair `(x, u_gt)` | PDE 寫成離散 stencil，**solver 端硬解** | **autograd 算 `N[u_θ]`，PDE 寫成 loss term** |
| **Mesh / discretization** | 不需要 mesh | 必須生成 mesh + 邊界離散 | **mesh-free**，隨機取 collocation 點 |
| **Inverse problem（反推係數）** | 需要單獨設計 | 需要 adjoint method 寫一遍 | **PDE 係數設成 trainable variable 就能跟 θ 一起學** |
| **OOD 外推** | 完全失效 | solver 在新邊界仍可解 | 在 PDE 形式不變的前提下優於純資料驅動 |
| **稀疏 sensor data 融合** | 需要 imputation | 不自然 | **L_data + L_pde 同一個 loss，soft 融合天然** |
| **高維 PDE (>6D)** | scalable | mesh 爆炸 | collocation 點數仍可行 |
| **新 PDE prototyping 時間** | 不適用 | 寫 solver + mesh = 數週 | **改個 lambda function 即可**，數小時 |

### 1.2 ⚡ Eureka Moment

> **「autograd 算 PDE residual → 加進 loss → 不需要 mesh、不需要 FEM solver」** —— 神經網路本來就用 autograd 算 `∂L/∂θ`；既然能對參數取偏導，那當然也能對 **input** 取偏導。`∂u_θ/∂x`、`∂²u_θ/∂x²` 都是 autograd graph 上自然能拿到的東西。把它們組合成 PDE 算子（如 `u_t + u·u_x - ν·u_xx`），residual 就是這個算子作用在 NN 輸出上的值；residual 的 L2 norm 就是 loss。

這個 trick 在 1990s automatic differentiation（Griewank）就有理論基礎，但要等到 2017 framework（TF/PyTorch）把高階 autograd 變成一行 API，PINN 才能成為可重複的 baseline。**整個 idiom 的價值在「把 PDE 從 solver 搬到 optimization」這個 paradigm shift，而不在某個具體 demo**。

### 1.3 信息流（架構圖）

```ascii
           Pre-PINN: data-driven NN                        PINN (Raissi 2017)
   ─────────────────────────────────────         ─────────────────────────────────────

   (x, t) ──► MLP ──► û                           (x, t) ──► MLP u_θ ──► û
              │                                              │
              ▼                                              ├──autograd──► ∂û/∂t, ∂û/∂x, ∂²û/∂x², ...
   L = ‖û - u_gt‖²                                          │                       │
   (需要密集標註)                                            │                       ▼
                                                            │             r = N[û]  ← PDE residual
                                                            │             (e.g. u_t + u·u_x - ν·u_xx)
                                                            │                       │
                                                            ▼                       ▼
                                                  L_data  L_bc  L_ic       L_pde = ‖r‖²
                                                    │       │     │              │
                                                    └───────┴─────┴──────────────┘
                                                                ▼
                                                  L_total = λ_data·L_data + λ_pde·L_pde
                                                          + λ_bc·L_bc + λ_ic·L_ic
                                                                ▼
                                                  Adam (~10⁴-10⁵ iter) → LBFGS (~10³ iter)
                                                  ★ LBFGS 不接，paper headline 數字復現不出
```

---

## §2 · 數學層

### 📌 Napkin Formula

```
   給 PDE   N[u](x,t) = 0  on Ω×[0,T]
       BC   B[u](x,t) = g  on ∂Ω
       IC   u(x,0)    = u_0

   PINN 訓練：u_θ(x,t) := MLP，autograd 對 (x,t) 取偏導

   L_total(θ) = λ_pde · 𝔼_{(x,t)∈Ω_c} ‖N[u_θ](x,t)‖²        ← collocation residual
              + λ_bc  · 𝔼_{(x,t)∈∂Ω_c} ‖B[u_θ] - g‖²         ← soft boundary
              + λ_ic  · 𝔼_x ‖u_θ(x,0) - u_0(x)‖²            ← soft initial
              + λ_data· 𝔼_obs ‖u_θ - u_obs‖²                ← (optional) sparse sensor

   Cost: O(N_collocation · D_high_order_grad)  per iteration
         vs FEM:  O(N_mesh · solver_iter)      per time step
```

**直覺**：PDE 從「solver 端 hard solve」搬到「optimization 端 soft penalty」。三件事被一鍋打包：(1) **forward** — 給 PDE + BC/IC，解 u；(2) **inverse** — 把 ν 也設成 `trainable variable`，給稀疏 u_obs 反推 ν；(3) **assimilation** — L_data 與 L_pde 同層 soft 融合。代價 = 在 collocation 點計算高階偏導，**而且不同 loss term 的 gradient 量級沒人保證匹配**（這就是 §6 + §8 整個 failure 主軸的入口）。

### 2.1 Loss / 訓練細節

實作上 Adam warm-up 之後幾乎都要接 **LBFGS**：Adam 把 loss 從 O(1) 拉到 O(10⁻²)，LBFGS 才能再壓到 O(10⁻⁵)；少了 LBFGS 這步，幾乎所有 paper 的 headline 數字都復現不出來。`λ` 權重組合 = grid search / NTK-weighted（Wang 2022 §4 給公式）/ SA-PINN（self-adaptive，McClenny 2020）/ GradNorm 四選一，沒人能 free lunch。

### 2.2 自監督 / curriculum 變體

Krishnapriyan 2021 提出 curriculum learning（從小 β 漸增）+ sequence-to-sequence（時間切片逐步 unroll）兩種修法，但都是治標：本質仍是 soft penalty 的 ill-conditioning。Hard-constraint 路線（HNN / LNN）走的是另一條 ontology axis 的解法（見 §7）。

---

## §3 · 數據層 / 訓練 scale

PINN 跟所有後續 aux-loss 後代最大的賣點之一：**訓練資料規模可以很小**。

| 條件 | PINN 需要的資料 | mesh-based FEM 需要的資料 | 純資料驅動 NN 需要的資料 |
|---|---|---|---|
| Forward problem | **只要 PDE 數學式 + BC/IC** | PDE + mesh | 密集 supervised pair |
| Inverse problem | PDE form + 幾個 sensor 點 | adjoint solver + sensor | 大量 paired observations |
| 新 PDE 第一次嘗試 | 1 個午後 | 數週寫 solver | 需要先生成 dataset |

**Collocation 點數實務值**：1D 約 10k、2D 約 50k、3D 容易爆 GPU memory；這個尺度跟「video diffusion 訓練要看 100M+ frames」是不同數量級。**這也是為什麼 PINN 主場仍是 scientific ML niche，而不是大 scale generation**。

---

## §4 · 代碼層

| 項 | 狀態 |
|---|---|
| Canonical repo | [Raissi 原作 maziarraissi/PINNs](https://github.com/maziarraissi/PINNs)（TF 1.x，今天已不維護） |
| 事實標準 lib | **DeepXDE**（[lululxvi/deepxde](https://github.com/lululxvi/deepxde)）— Lu et al SIAM Review 2021；TF / PyTorch / JAX / PaddlePaddle 多 backend |
| 工業級 lib | NVIDIA **Modulus / PhysicsNeMo**（[github.com/NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo)，改名遷移中）— multi-GPU、整合 FNO + PINN + DeepONet |
| 教學級 lib | SciANN — Keras 包裝，社群活躍度遜於前兩者 |
| License | DeepXDE LGPL-2.1；Modulus Apache-2.0 |
| Inference GPU | 1D Burgers V100 5-15 分鐘到 paper-level error |
| Streaming | ❌（必須 batch optimize；inference 是「query 任意 (x,t) 拿 u_θ」） |
| Metric scale | ✅（PDE 用真實單位寫，但要 non-dimensionalize 才訓得動） |
| 典型 setup | MLP 4-8 層 / 寬度 50-100 / tanh；Adam lr=1e-3, ~20k iter + LBFGS-B ~1-5k iter |

---

## §5 · 評測 / Benchmark

| Benchmark | Metric | 純資料 NN | **vanilla PINN** | Krishnapriyan β-stress test |
|---|---|---|---|---|
| 1D Burgers (ν=0.01/π) | L2 rel error | 不收斂 | **6.7e-4** ★ | n/a |
| Schrödinger 1D | L2 rel error | 不收斂 | **1.97e-3** ★ | n/a |
| 2D NS (low-Re) cylinder | viscosity ν 反推 | 不適用 | **1.5% 誤差** ★ | n/a |
| Convection (β=1) | L2 rel error | n/a | ~10⁻³ | OK |
| Convection (β=30) | L2 rel error | n/a | n/a | **O(1) 完全崩** 🔴 |
| Reaction (ρ=5) | L2 rel error | n/a | n/a | **O(1) 完全崩** 🔴 |
| Lorenz / KS chaotic | long-horizon | n/a | 失效 | （Wang 2022 causal-PINN 才解） |

★ = 原 paper headline 數字（Raissi 2019）。

**解讀**：原 paper 的 headline 數字是真的，**但只在 trivial regime**。Krishnapriyan 2021 的 β-stress test 是這個方法落地最重要的 reality check —— **demo 跑得動 ≠ production 撐得住**，這個鴻溝是接下來五年所有 PINN 變體（causal / self-adaptive / hard-constraint / XPINN / Fourier feature）想填的。Benchmark Goodhart 在 PINN community 也存在：很多 paper 只報 Burgers/Schrödinger 不報 convection-large-β，這是 reader 該警覺的 selection。

---

## §6 · Issues & Limitations

### 6.1 論文自述 limitations

- 原 paper 對「Adam 卡 local min → 必須接 LBFGS」描述偏輕；後續社群實證 LBFGS 不接 headline 不可復現
- 對 multi-scale / stiff PDE 的處理只給 anecdotal evidence，沒系統性 benchmark
- inverse problem 章節只展示 low-Re NS 與 KdV，sharp gradient / 高 Re / 不連續係數沒測

### 6.2 Hidden Assumptions

- **MLP 對低頻友好（spectral bias）** —— 高頻 / sharp gradient / boundary layer 結構性學不到；對 stiff PDE 是致命
- **Loss term 同尺度** —— 預設 `[1,1,1,1]` 隱含「λ 不重要」這個錯誤先驗
- **Collocation 取樣均勻** —— 對含 shock / boundary layer 的 PDE，均勻取樣對 sharp region 訊號不夠
- **PDE 形式已 normalize** —— ν=1e-6 物理量丟進去 residual 數量級就崩，必須先 non-dimensionalize
- **Initial / boundary 條件夠準** —— soft penalty 對誤差敏感，noisy BC 直接 propagate 到 interior

### 6.3 NTK-validated 失敗模式（Wang Yu Perdikaris JCP 2022）

> **核心理論結果**：在 NTK regime，PINN 的訓練動力學 = `du/dt = -K·(u - u*)`，其中 K 是 stack 所有 loss term 的 NTK matrix。**不同 loss term（L_pde / L_bc / L_ic）的 K 特徵值跨 10²-10⁴ 量級** —— dominant 那一項把 NN 訓往單一方向，其他 term 在訓練早期幾乎不變化。

對應實證症狀：

| 失敗類型 | 根因（NTK 視角） | 對應 Krishnapriyan 實證 |
|---|---|---|
| L_bc 違反 1-5% | `K_bc` eigenvalue ≪ `K_pde` | convection β=30 |
| 訓練 stuck plateau | `K` ill-conditioned | reaction ρ=5 |
| run-to-run variance 大 | NTK 對 init scale 敏感 | DeepXDE #305 |
| Adam→LBFGS 跳變大 | landscape ill-conditioned | 所有 paper headline 需要 LBFGS |

### 6.4 GitHub-validated 失敗模式（DeepXDE atlas 聯動）

| 失敗 / 問題 | GitHub evidence | 嚴重度 |
|---|---|---|
| **Loss-weight imbalance (NTK pathology)** | [#215](https://github.com/lululxvi/deepxde/issues/215)（closed 2021-02）+ [#982](https://github.com/lululxvi/deepxde/issues/982)（open 2022-10）：「Is there a way to make these weights dynamic such that they counteract the issue of gradient imbalance」 | 🔴 universal |
| **Weight init sensitivity / run-to-run variance** | [#305](https://github.com/lululxvi/deepxde/issues/305)：「the order of magnitude of the pdes losses ... varies a lot from run to run」— 同 seed 不同 init 初始 PDE loss 可差 10⁵ 量級，有時 NaN | 🔴 reproducibility |
| **Navier-Stokes 不收斂** | [#80](https://github.com/lululxvi/deepxde/issues/80)：「continuity and x-momentum residuals are at best ~1e-1 regardless of the number of epochs, network size, network architecture」 | 🔴 fluids 主場 |
| **Inverse problem parameter 不收斂** | [#251](https://github.com/lululxvi/deepxde/issues/251)：forward 跑得起來但 inverse 估 unknown velocity 一路發散到 5.24（真值 2.0） | 🔴 scientific ML |
| **Hard boundary 不嚴 / Neumann 難實作** | [#90](https://github.com/lululxvi/deepxde/issues/90) + [#192](https://github.com/lululxvi/deepxde/issues/192) + [#1837](https://github.com/lululxvi/deepxde/issues/1837)：Dirichlet 可用 ansatz `u=g(x)+d(x)·NN(x)`，Neumann 沒乾淨 solution | 🟠 engineering |
| **LBFGS 卡 30 iter (TF backend)** | [#1819](https://github.com/lululxvi/deepxde/issues/1819)：Adam 跑得動，LBFGS-B 恰好 30 iter 停 | 🟠 toolchain |

**Maintainer 響應度**：DeepXDE 仍活躍維護（Lu Lu 個人 repo，2026-05 仍在 commit）；issue 多被回覆但「loss-weight」「NS 不收斂」這類**理論性問題沒人能 close** —— 需要 SA-PINN / Causal PINN / NTK-weighted 等 method-level 變體，不是 lib bugfix 能搞定的。

### 6.5 對 video / 大 scale 的 envelope 限制

- **Score function 與 PDE residual 方向不相容** —— diffusion score 是 denoising 方向，PDE residual 是物理約束方向；天真疊加產生 distribution shift（這就是為什麼**沒有「PI-Loss for video diffusion」成功案例**）
- **Collocation 取樣 → video pixel 不可行** —— video 是 dense field，collocation 點 sparse 取樣的 idiom 不對胃口
- **PhysGen 的活路**：physics loss 只加在中介 latent（rigid-body pose / 接觸 ODE），image 端用 generative diffusion 渲染；**PINN 思想被「腰斬」使用**
- **Force Prompting 的活路**：把 force / wind / 物理 prompt 用 attention conditioning 注入，不算嚴格 aux-loss 但精神同源

---

## §7 · 比較 & 面試 Tip

| 同軸對手 | injection | 守恆? | OOD 寬度 | 訓練資料需求 | 主場 |
|---|---|---|---|---|---|
| **PINN** | `aux-loss` (soft) | ❌（penalty 違反 1-5%） | 中等 | PDE form + 稀疏 sensor | sci-ML / inverse / 新 PDE prototyping |
| HNN / LNN | `hard-constraint`（架構） | ✅ 嚴格 | 窄（only conservative） | trajectories | 剛體 / 經典力學 |
| FNO / MeshGraphNet | `hard-constraint`（spectral bias） | partial | 受訓練分佈限制 | simulator-generated 大量 trajectory | production rollout |
| PhysGen | `aux-loss` on latent（PINN 思想） | rigid-body ODE | 圖→片段 video | image + 物理 prompt | I2V 物理可控 |
| PhysDiff | `guidance-gradient` | classifier-physics-guided | text-conditional | 大規模 video + diffusion | text→video 軟物理 |
| Force Prompting | `aux-loss` + attention conditioning | force-as-prompt | 受 base diffusion 限制 | force-labeled video | 含力場 video gen |

> **🎤 Interview Tip.** 「我們要不要在 video diffusion 加 PINN aux loss？」**正確答**：「**不要直接套**。三個原因：(1) Krishnapriyan 2021 證明 vanilla PINN 在 non-trivial regime 直接崩，video 的 motion field 遠比 1D Burgers 複雜，更不可能 trivial；(2) Wang NTK 2022 證明不同 loss term gradient 量級跨 10²-10⁴，diffusion 的 score 跟 PDE residual 量級沒人保證匹配，會出現 score 主導 → PDE 沒在學 / 或反之 distribution shift；(3) **看 PhysGen 的活路**：他們**沒**把 PDE 整個搬到 video，是把 rigid-body / 接觸 ODE 放在中介 latent 層，image 端走純 diffusion。要在 video 加 physics 結構，**腰斬 PINN 思想用、不是直接套**。」**錯答**：「PINN 是 baseline，加上去總比沒加好」—— 沒讀 NTK paper 的訊號。

### 7.1 Falsifiable predictions

1. **2027-12 前**：會出現一篇 "Score-PDE Compatibility Theorem"（或類似），系統性給出 diffusion score gradient 與 PDE residual gradient 共存的充分條件 —— 主因 NTK 理論已成熟，缺的只是 generative-side 對應公式。
2. **2027-12 前**：第一篇 hash-encoding + PINN 的 stable 工作出來（解掉高階 autograd 對 hash table 的數值不穩定）；當前 [physics-informed NeRF / PI-MLP 流體重建] 仍 unstable。
3. **2027-12 前不會發生**：vanilla PINN（unmodified Raissi 2019 formulation）成為 video diffusion 的 production aux-loss —— 即便有人嘗試，會被 PhysGen-style 中介 latent loss 或 Force Prompting 風格 attention conditioning 取代；**因為 NTK 失效在 video scale 上只會放大不會收斂**。

---

## §8 · For the Reader（按 persona 分流）

- **VLA / robot policy 工程師** —— PINN 直接接 policy 不對胃口；但若你在做 **dynamics learning**（從稀疏軌跡反推 mass / friction），PINN inverse problem 章節仍是 baseline。記得 §6.4 [#251] inverse 不收斂的問題：先 forward 預訓再開 trainable parameter。
- **自駕 closed-loop 工程師** —— 你不直接用 PINN，但你**會吃 PINN 後代的虧**：如果某 vendor 把 vehicle dynamics 寫成 aux-loss 塞進 prediction model，記得問他們調 λ 用哪個 scheme（NTK-weighted? SA-PINN? GradNorm?）—— 預設 `[1,1,1,1]` 是地雷。
- **影片生成工程師** —— **不要把 PINN 直接套到 video diffusion**（§7 interview tip 三條原因）。要加物理結構：走 PhysGen-style（physics loss on intermediate latent）或 Force Prompting-style（force as attention conditioning），這是 PINN 思想的腰斬使用。
- **神經 PDE / surrogate 研究者** —— 你的主場。PINN 是 baseline，但實際 production 八成走 SA-PINN / Causal PINN / XPINN / NTK-weighted。讀 paper 順序：Raissi 2019 → Krishnapriyan 2021 → Wang NTK 2022 → Wang causal 2022（[2203.07404](https://arxiv.org/abs/2203.07404)）。
- **★ 物理 conditioning 研究者** —— 你的世界裡 PINN 是**思想祖父**，但**現代後代**是：(a) **PhysGen** ECCV 2024（rigid-body ODE as latent loss）；(b) **PhysDiff**（physics gradient as score guidance）；(c) **Force Prompting**（force as attention conditioning）。三條路徑都是 PINN 思想在 generative 範式下的折射；**讀本篇 §6 NTK 失效 + §7 interview tip 是 USP zone 的下限知識**。當你設計新 aux-loss 變體，要做的第一件事是檢查它在 NTK regime 下是否會 ill-conditioned —— 這比 demo 跑得動重要。
- **Research 學生** —— 寫 PINN 變體論文時，**至少跑 Krishnapriyan β-stress test**（convection β=1, 10, 30）。只報 Burgers / Schrödinger 是 selection；reviewer 會問。

---

## References

**Canonical**

- **PINN** — Raissi, Perdikaris, Karniadakis · *J. Comput. Phys.* 378 (2019) 686-707 · arXiv [1711.10561](https://arxiv.org/abs/1711.10561) (Part I) + [1711.10566](https://arxiv.org/abs/1711.10566) (Part II)
- **DeepXDE** — Lu, Meng, Mao, Karniadakis · *SIAM Review* 63(1) (2021) 208-228 · arXiv [1907.04502](https://arxiv.org/abs/1907.04502) · [code](https://github.com/lululxvi/deepxde)

**Failure-mode 文獻（USP zone 必讀）**

- **Krishnapriyan failure modes** — Krishnapriyan, Gholami, Zhe, Kirby, Mahoney · *NeurIPS 2021* · arXiv [2109.01050](https://arxiv.org/abs/2109.01050)
- **Wang NTK theory** — Wang, Yu, Perdikaris · *J. Comput. Phys.* 449 (2022) · arXiv [2007.14527](https://arxiv.org/abs/2007.14527)
- **Causal PINN** — Wang, Sankaran, Perdikaris · arXiv [2203.07404](https://arxiv.org/abs/2203.07404) (2022)

**Modern extensions**

- **SA-PINN** — McClenny, Braga-Neto · *JCP 2022* · arXiv [2009.04544](https://arxiv.org/abs/2009.04544)
- **Causality-Respecting Adaptive Refinement** — arXiv [2410.20212](https://arxiv.org/abs/2410.20212) (2024)
- **NVIDIA PhysicsNeMo** — [github.com/NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo)（Modulus 改名遷移）

**Generative descendants（aux-loss 後代）**

- **PhysGen** — Liu, Ren, Gupta, Wang · *ECCV 2024* · [project page](https://stevenlsw.github.io/physgen/)
- **PhysDiff** — see [`./physdiff.md`](./physdiff.md)
- **Force Prompting** — see [`./force-prompting.md`](./force-prompting.md)

**Third-party reproductions**

- DeepXDE issue cluster: [#80](https://github.com/lululxvi/deepxde/issues/80), [#90](https://github.com/lululxvi/deepxde/issues/90), [#192](https://github.com/lululxvi/deepxde/issues/192), [#215](https://github.com/lululxvi/deepxde/issues/215), [#251](https://github.com/lululxvi/deepxde/issues/251), [#305](https://github.com/lululxvi/deepxde/issues/305), [#982](https://github.com/lululxvi/deepxde/issues/982), [#1819](https://github.com/lululxvi/deepxde/issues/1819), [#1837](https://github.com/lululxvi/deepxde/issues/1837)

---

## Boundary

- **同 zone hard-constraint 對比**（守恆律寫進架構而非 loss）→ [`./hamiltonian-lagrangian-nn.md`](./hamiltonian-lagrangian-nn.md)
- **同 zone aux-loss 後代（rigid-body → video）** → [`./physgen.md`](./physgen.md)
- **同 zone guidance-gradient 對比** → [`./physdiff.md`](./physdiff.md)
- **同 zone force as conditioning prompt** → [`./force-prompting.md`](./force-prompting.md)
- **Neural surrogate（supervised operator learning）對比** → [`../neural-surrogates/fno.md`](../neural-surrogates/fno.md)
- **跨 zone wedge：sim-in-loop vs PINN soft constraint** → [`../../crossing/sim-vs-aux-loss/overview.md`](../../crossing/sim-vs-aux-loss/overview.md)
- **與 sister handbook 接口（Spatial: physics-informed NeRF）** → [`../../bridge-to-spatial/pi-nerf.md`](../../bridge-to-spatial/pi-nerf.md)
- **與 5 axis 全景** → [`../../cheat-sheet/ontology.md`](../../cheat-sheet/ontology.md)

---

## ✍️ 維護者註（v0.5 → v1 升級清單）

本 v0.5 基於 Raissi 2017-2019 三篇 + Krishnapriyan 2021 + Wang NTK 2022 + DeepXDE 8 條 issue 採樣。下次升 v1 時補：

1. ⏳ SA-PINN（McClenny 2022）的完整 mechanism + 跟 NTK-weighted 的 head-to-head
2. ⏳ Causal PINN（Wang 2022 [2203.07404](https://arxiv.org/abs/2203.07404)）的 temporal weight 完整公式 + Lorenz / KS 實驗數字
3. ⏳ XPINN domain decomposition（Jagtap 2020）vs vanilla PINN benchmark
4. ⏳ Hash-encoding + PINN 的高階 autograd 數值穩定性 status（physics-informed NeRF 流體重建 canonical work arXiv ID）
5. ⏳ PhysGen / PhysDiff / Force Prompting 對 PINN 思想腰斬使用的 architecture 細節對比表
6. ⏳ NVIDIA PhysicsNeMo 在大型工業 PDE（航太、半導體蝕刻）的 production case study
7. ⏳ Score-PDE 共存的充分條件理論 paper（§7.1 falsifiable 1 一旦兌現）
8. ⏳ Status v0.5 → v1，刪本節

---

[← Back to Physics-Conditioning](./overview.md)

Sources:
- [Raissi 2019 PINN JCP](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [arXiv 1711.10561 Part I](https://arxiv.org/abs/1711.10561)
- [arXiv 1711.10566 Part II](https://arxiv.org/abs/1711.10566)
- [Krishnapriyan NeurIPS 2021](https://arxiv.org/abs/2109.01050)
- [Wang NTK JCP 2022](https://arxiv.org/abs/2007.14527)
- [DeepXDE GitHub](https://github.com/lululxvi/deepxde)
- [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo)
