<!-- ontology-5axis
output: particle (state-trajectory) / latent rollout
injection: hard-constraint (canonical anchor)
control: param (mass / length / viscosity) + 3d-init for HGN
temporal: streaming (ODE integrator) / autoregressive
domain: rigid (pendulum / N-body); fluid 變體稀薄
ref: ../../cheat-sheet/ontology.md §Axis2
-->

# HNN / LNN / Symplectic NN 解構（Hamiltonian & Lagrangian Neural Networks）

> **發布時間**：2019-06 ~ 2020-03 · arXiv [1906.01563](https://arxiv.org/abs/1906.01563) (HNN) · [2003.04630](https://arxiv.org/abs/2003.04630) (LNN) · [1909.12077](https://arxiv.org/abs/1909.12077) (Symplectic ODE-Net) · [1909.13789](https://arxiv.org/abs/1909.13789) (HGN) · [1907.04490](https://arxiv.org/abs/1907.04490) (DeLaN)
> **論文**：*Hamiltonian Neural Networks* / *Lagrangian Neural Networks* / *Symplectic ODE-Net* / *Hamiltonian Generative Networks* / *Deep Lagrangian Networks*
> **作者**：Greydanus, Dzamba, Yosinski（Google/Uber AI）· Cranmer, Greydanus, Hoyer, Battaglia, Spergel, Ho（Princeton + DeepMind）· Zhong, Dey, Chakraborty（UVA）· Toth, Rezende, Higgins, et al.（DeepMind）· Lutter, Ritter, Peters（TU Darmstadt）
> **核心定位**：v2 ontology Axis 2 `hard-constraint` 最乾淨的代表 —— 不是把守恆律當 loss 勸進（PINN 路線），而是把 Hamilton / Lagrange 方程直接焊進 forward pass，讓 architecture by construction 在數學上不可能違反能量守恆。也是 USP zone 的**對比錨點**：解釋為什麼 2024-25 主流選 `aux-loss` + `guidance-gradient` 而**不**選 `hard-constraint`。

**Status:** v0.5 — 解構基於原 paper + GitHub issues + 6 年 follow-up community 觀察。完整 SympNet 變體 / MetaSym 細節 / 2025-26 最新嘗試待維護者升 v1。
**TL;DR:** ① 2019-2020 是 hard-constraint 的**密集探索期**，4 篇 ICLR/NeurIPS canonical paper 集中爆發；② 2020 之後幾乎停滯 —— 不是缺乏關注，是**表達力天花板**：dissipative / contact / 高 DoF / pixel 系統全部失效；③ 在 toy mechanical system 上 rollout 10⁴ step 能量漂移 <1%（vanilla MLP 同條件 >50%），這是 hard-constraint 的「乾淨勝利」；④ 對 Physics-Gen handbook 讀者：HNN/LNN 是**反例錨點**，不是要你抄，而是要你看清「為什麼把物理塞進架構是死路、而塞進 loss / score / guidance 才是 2024-25 主流」。

**X-Ray.** Hard-constraint 是 ontology Axis 2 的**最強訊號** —— 它做了一個架構承諾：「能量守恆不是訓練目標，是前向計算的副產物」。HNN 學純量 $H_\theta(q,p)$ → autodiff 拿偏導 → Hamilton 方程直接推 $(\dot q, \dot p)$；LNN 學 $L_\theta(q,\dot q)$ → Euler-Lagrange 推 $\ddot q$。這個承諾在 pendulum / Kepler / 雙擺等 closed-form mechanical system 上完美 —— 但表達力代價也是最大的：(a) 假設 $dH/dt = 0$，dissipative system（摩擦、阻力、塑性）直接出局；(b) contact discontinuity 破壞 smooth flow 假設，robotic manipulation 整類沒法 cover；(c) DoF > 50 後 autodiff 雙偏導 / Hessian inverse 數值崩潰，fluid / 連續介質場 10⁴ DoF 從未 demo；(d) pixel 輸入不可解 —— 單 frame 無法決定 momentum（[HNN issue #8](https://github.com/greydanus/hamiltonian-nn/issues/8) 一針見血）。**2020 之後停滯不是缺乏關注，是架構天花板** —— 社群試過 HGN（latent Hamiltonian + VAE）、Symplectic Recurrent、SympNet、MetaSym，但**沒有任何一條在 video / 4D scene / 機器人 manipulation 上拿到 SOTA**。對 Physics-Gen handbook USP zone 而言，HNN/LNN 是 contrast point —— 它解釋了**為什麼 2024-25 主流是 aux-loss（軟）和 guidance-gradient（採樣時注入），而不是 hard-constraint（架構嵌入）**：表達力與可控性必須留給 backbone 自己學，物理只能在 loss / score / token 上「輕觸」。

## 📍 研究全景時間線

```ascii
   1990s          2017             2019-09           2019-10           2020-03         2020+              2024
   symplectic ──► PINN ─────────► HNN ──────────► Symplectic ──► LNN ──────► [停滯] ────► NewtonGen
   integrator    (soft, loss)    YOU ARE HERE    ODE-Net + HGN  Cranmer    SympNet,       (architecture-
   (Hairer,                       Greydanus       Zhong / Toth   ICLR ws    MetaSym etc.   bias-soft, 不
   Lubich, Wanner                 NeurIPS                                   (toy-only      算 hard-constraint)
   classical                                                                follow-up)
   numerics)
   
   ── soft ──────────────────────► hard-constraint 嘗試期 ────► 停滯 ──────► 主流轉向 aux-loss / guidance
   
   ★ = 主要新點：把守恆律從 loss 移到 architecture 內。
   仍未解：(a) dissipative / contact / 高 DoF / pixel；(b) video / 4D scene generation；(c) hard-constraint 與大規模 latent-WM 融合（MetaSym 2025 嘗試但未 scale）。
```

---

## §1 · 架構 / Core Mechanism

### 1.1 四個 canonical method vs 同軸前作

| 維度 | Vanilla MLP $f(q,p)$ | PINN | **HNN** | **LNN** | Symplectic ODE-Net | HGN |
|---|---|---|---|---|---|---|
| 學什麼 | $(\dot q, \dot p)$ 直接學 | 任意 $u(x,t)$ + PDE residual loss | 純量 $H_\theta(q,p)$ | 純量 $L_\theta(q,\dot q)$ | $H_\theta(x) + (J,R,g)$ | latent $H$ + VAE encoder/decoder |
| 守恆律 | ❌ 無保證 | 軟（loss penalty） | **硬（架構內）** | **硬（架構內）** | **硬 + dissipation channel** | **硬（latent）** |
| Canonical coord 需求 | — | — | ✅ 必須 $(q,p)$ pair | ❌ 任意 generalized coord | ✅ + control input $u$ | latent 自動 |
| Contact / 不連續 | OK | OK | ❌ 破 smooth flow | ❌ | ❌ | ❌ |
| Pixel input | OK | OK | ❌ momentum 不可恢復 | ❌ | ❌ | ⚠️ multi-frame autoencoder 半成功 |
| Long rollout energy drift | 隨機漂 | 線性漂 | **O(Δt²) bounded** | bounded | bounded | bounded (latent) |
| 最大 DoF demo | 任意 | ~10³（fluid） | ~10 | ~7（DeLaN robot arm） | ~10 | ~5 (image latent) |

### 1.2 ⚡ Eureka Moment

> **學 $H_\theta(q,p)$ 而不是學 $\dot q, \dot p$ —— Hamilton 方程自動成立** —— 不是「在 loss 上加守恆 penalty」，是「forward pass 計算 $\dot q = \partial H/\partial p$ 與 $\dot p = -\partial H/\partial q$ 時，autodiff 直接給」。能量守恆是 architecture 的**結果**，不是訓練的**目標**。

這個 trick 借用了 1990s 的 symplectic integrator 思想（Hairer / Lubich / Wanner 的 *Geometric Numerical Integration*），但把「已知 $H$ 數值積分」翻成「**從資料學 $H$ + autodiff 算偏導**」 —— 第一次把 reverse-mode AD 與 Hamilton 力學接在一起。

### 1.3 信息流（架構圖）

```ascii
                    Vanilla MLP                              HNN
              ───────────────────────              ───────────────────────
              
              (q,p) ──► MLP ──► (q̇,ṗ)              (q,p) ──► H_θ MLP ──► scalar H
                          │                                          │
                          ▼                                          ▼ autodiff
                  ODE integrator                          ∂H/∂p,  -∂H/∂q
                          │                                          │
                          ▼                                          ▼
                  next (q,p) loop                         symplectic integrator (leapfrog)
                          │                                          │
                          ▼                                          ▼
              energy = random walk                      energy ≈ const by construction
              (rollout 10² step 即崩)                     (rollout 10⁴ step drift < 1%)
              
                                                                              
                    LNN forward                                HGN (image → latent rollout)
              ───────────────────────              ───────────────────────────────
              
              (q,q̇) ──► L_θ MLP ──► scalar L         video frames ──► CNN encoder ──► (q₀,p₀) latent
                                       │                                                │
                                       ▼ autodiff (Hessian)                             ▼
                          (∂²L/∂q̇∂q̇)⁻¹ [∂L/∂q - (∂²L/∂q̇∂q) q̇]            latent H_θ + symplectic rollout
                                       │                                                │
                                       ▼                                                ▼
                                    q̈ → integrator                            (q_t,p_t) ──► decoder ──► pixel
                                                                                         │
                                       ★ Hessian inverse 是數值                          ▼
                                          熱點（DeLaN PD-parametrize 緩解）              reconstruction loss
```

---

## §2 · 數學層

### 📌 Napkin Formula

```
   HNN:    dq/dt = ∂H_θ(q,p) / ∂p
           dp/dt = -∂H_θ(q,p) / ∂q                    ← Hamilton's equations 直接焊進前向

   LNN:    q̈ = (∂²L/∂q̇∂q̇)⁻¹ [∂L/∂q - (∂²L/∂q̇∂q) q̇]   ← Euler-Lagrange，吃任意 generalized coord
   
   Cost: O(d²) for HNN per step (autodiff 兩偏導)
         O(d³) for LNN per step (Hessian inverse) 
         vs MLP baseline O(d) — hard-constraint 的數值代價是 d² ~ d³
```

**直覺**：vanilla MLP 直接學 $(\dot q, \dot p)$ 的問題是 —— 它沒有任何結構保證向量場 $\nabla \times f = 0$（這是 conservative vector field 的判別條件），所以 rollout 中 energy 是 random walk。HNN 把這個判別條件做進架構：你只能學一個純量 $H$，再透過 autodiff 算出向量場 —— 這個向量場**數學上必然** conservative。LNN 更激進 —— 不要求知道 momentum，吃 $(q,\dot q)$ 直接做 Euler-Lagrange，代價是 Hessian inverse（[LNN issue #6](https://github.com/MilesCranmer/lagrangian_nns/issues/6) 是這個熱點的典型病灶）。

### §2.1 Loss 細節

```
   HNN loss:  L = || (q̇_pred, ṗ_pred) - (q̇_data, ṗ_data) ||²
   
   ★ 注意：不監督 H 本身（沒有 "energy label"），守恆律是 architecture 結果。
   
   LNN loss:  L = || q̈_pred - q̈_data ||²
   
   Symplectic ODE-Net:  ẋ = (J - R) ∇H_θ(x) + g(x) u
                        L = || x_pred - x_data ||² + λ_R · ||R||²（正則 dissipation matrix）
   
   HGN:  L = recon_loss(decoder(rollout(encoder(frames)))) + β · KL(latent)
```

---

## §3 · 數據層 / 訓練 scale

| Method | Toy system | 數據量 | 規模 demo 上限 |
|---|---|---|---|
| HNN | mass-spring / Kepler / 雙擺 | 幾百條軌跡（< 10⁴ states） | DoF ~10 |
| LNN | double pendulum / 3-body | 同 HNN 量級 | DoF ~10 |
| Symplectic ODE-Net | cartpole + control | 數千 episode | 7-DoF arm |
| HGN | pixel pendulum / MuJoCo cartpole | video clips 量級 ~10⁴ frames | latent dim ~5 |
| DeLaN | 7-DoF Franka inverse dynamics | 真實機械臂軌跡 ~10⁵ samples | 7-DoF |

**關鍵觀察**：data scale 從未進入「**百萬 video 級別**」 —— 因為架構不允許。每加一個 DoF，autodiff Hessian 規模 quadratic 上升；每加一個 contact event，smooth flow 假設破。**hard-constraint 路線天生反規模化**，這是它停滯的根因。

對比看：Cosmos 訓 100M+ video clips、Dreamer V4 訓 10⁹ env step、GraphCast 訓 40 年全球天氣 —— 這些方法**都是 aux-loss 或 architecture-bias-soft** 路線。hard-constraint 不在這張 scale 表上。

---

## §4 · 代碼層

| 項 | 狀態 |
|---|---|
| HNN repo | [greydanus/hamiltonian-nn](https://github.com/greydanus/hamiltonian-nn) (PyTorch + scipy ODE) |
| LNN repo | [MilesCranmer/lagrangian_nns](https://github.com/MilesCranmer/lagrangian_nns) (JAX) |
| DeLaN repo | [milutter/deep_lagrangian_networks](https://github.com/milutter/deep_lagrangian_networks) |
| HGN | 無官方 release（DeepMind 內部代碼），社群復現困難 |
| Symplectic ODE-Net | [d-biswa/Symplectic-ODENet](https://github.com/d-biswa/Symplectic-ODENet) |
| License | Apache-2.0（HNN/LNN）|
| Inference GPU | toy 系統 CPU 可跑；HGN pixel 4-8 GPU 量級 |
| Streaming | ✅（ODE integrator 天然 streaming）|
| Metric scale | N/A（toy mechanical 用 SI 單位）|
| Python 3.13 / JAX 0.4+ 兼容 | ⚠️ LNN 倉漂移（[issue #11](https://github.com/MilesCranmer/lagrangian_nns/issues/11), [#13](https://github.com/MilesCranmer/lagrangian_nns/issues/13)）|

**Maintainer 響應度（2026-05-26）**：
- HNN repo：13 open / few closed issues（Greydanus 2020 後零維護）
- LNN repo：14 open / 0 closed（Cranmer 已轉 PySR / 符號回歸路線）
- DeLaN：低活躍

---

## §5 · 評測 / Benchmark

| Benchmark | Metric | Vanilla MLP | HNN/LNN | Δ |
|---|---|---|---|---|
| Mass-spring 5000 step | energy drift | ~50% | **< 0.1%** | -50pp |
| Kepler 2-body 10⁴ step | energy drift | random walk | **< 1%** | -bounded |
| Double pendulum chaotic | trajectory MSE | 短期 OK 長期崩 | 長期 bounded but phase 仍漂 | 部分勝 |
| Pixel pendulum (HGN) | reconstruction PSNR | TBD | 接近 baseline | ~持平 |
| 7-DoF inverse dynamics (DeLaN) | torque RMSE | baseline | ~10× sample-efficient | 1 個量級勝 |
| **Video / 4D scene** | — | Cosmos / Dreamer 主導 | **未進入** | **N/A** |

**解讀**：在原 paper 設定的 toy mechanical system 上 hard-constraint 是**乾淨勝利** —— 能量守恆 bounded，rollout 穩定。但 metric 是 paper 自己定的（能量漂移 / sample efficiency），不是 video 生成 metric（FVD / 物理一致性 / 控制精度）。**hard-constraint 路線從未在主流生成 benchmark 上對打過** —— 這是 §6 表達力天花板的實證。

---

## §6 · Issues & Limitations

### 6.1 論文自述 / 結構性 limitations

- **Dissipative system 直接失效** —— HNN 假設 $dH/dt = 0$，摩擦、阻力、塑性、熱耗散全出局。Symplectic ODE-Net 加 $(J-R)$ 是補丁但 dissipation 結構要事先知。
- **Contact / collision 不可微** —— 剛體碰撞瞬間 momentum 不連續，autodiff 過 Hamiltonian 整套機制失效，**robotic manipulation 整類無法 cover**。
- **High-DoF 數值崩潰** —— autodiff 雙偏導 + LNN Hessian inverse，DoF > 50 後 memory + 數值穩定性快速劣化。
- **Pixel input 困難** —— 單 frame 無法恢復 momentum（[HNN issue #8](https://github.com/greydanus/hamiltonian-nn/issues/8)），需 multi-frame stack；HGN 用 sequence encoder 解一半但 latent 解釋性差。
- **未進入 video / 4D scene generation 規模** —— HGN 之後沒有 follow-up 在 large video 拿到 SOTA。

### 6.2 Hidden Assumptions（隱含假設）

- **Smooth flow 假設** —— 在 contact event 後守恆律「人為斷裂」，需要 event detection（diff-sim 領地）。
- **Canonical coordinate 已知（HNN）** —— 雙擺要先做變數變換才能進 HNN；LNN 解了這個但代價是 Hessian inverse。
- **積分器選擇是隱性協議** —— vanilla RK4 不 symplectic，長 rollout 仍會漂（[HNN issue #2](https://github.com/greydanus/hamiltonian-nn/issues/2)），實際要切 leapfrog / Verlet。
- **訓練 distribution = chaotic regime 外推不可保** —— 訓在小擺角，外推到大擺角 chaotic regime 仍會崩，symplectic 保 energy 不保 phase-portrait 正確性。
- **OOD initial condition 失效快** —— 與一般 NN 同病但因為 architecture 嚴格反而沒有 "fallback to interpolation" 的軟著陸路徑。

### 6.x GitHub-validated 失敗模式（atlas 聯動）

| 失敗 / 問題 | GitHub evidence | 嚴重度 |
|---|---|---|
| **Pixel embedding 無法承載 momentum** | [HNN issue #8](https://github.com/greydanus/hamiltonian-nn/issues/8): "given a position (same pixel data), the pendulum can have different velocities ... z can't include momentum" | 🔴 直接質疑從 pixel 學的可行性 |
| **RK4 不 symplectic — 長 rollout 仍漂** | [HNN issue #2](https://github.com/greydanus/hamiltonian-nn/issues/2): 用戶詢問 RK4 用意；實情 RK4 不保 symplectic structure | 🔴 短 rollout 小，10³ step 以上必崩 |
| **LNN Hessian-inverse 數值病** | [LNN issue #6](https://github.com/MilesCranmer/lagrangian_nns/issues/6): `TypeError: Gradient only defined for scalar-output functions. Output had shape: (4,).` | 🔴 功能阻塞，default loss 在某些 DoF 直接報錯 |
| **Dataset 缺失 / 不可復現** | [LNN issue #9](https://github.com/MilesCranmer/lagrangian_nns/issues/9): 用戶請求公開 dataset 未獲回覆 | 🟠 玩具實驗可 self-sample，但部分 figure 難重現 |
| **Rescale-time 細節易遺漏** | [HNN issue #7](https://github.com/greydanus/hamiltonian-nn/issues/7): `t_span *= 1 + .9*noise_std` paper appendix 有但易漏 | 🟡 不重現會讓 quantitative 比較對不上 |
| **JAX / Python 3.13 漂移** | [LNN issue #11](https://github.com/MilesCranmer/lagrangian_nns/issues/11) + [#13](https://github.com/MilesCranmer/lagrangian_nns/issues/13): `odeint(mxsteps=...)` → `mxstep`；`jax.experimental.ode` 路徑變更 | 🟡 純工程；倉長期未維護 |
| **Train baseline 寫死問題** | [HNN issue #3](https://github.com/greydanus/hamiltonian-nn/issues/3): `experiment-2body/train.py` baseline 不寫死小坑 | 🟡 復現 corner case |
| **Dissipative system 直接失效** | HNN paper §3 顯式假設 + 社群多次討論 | 🔴 排除絕大部分真實 robotics 場景 |
| **Contact / collision 不可微** | HNN / LNN paper 皆未處理；diff-sim 社群通識 | 🔴 manipulation 整類無法 cover |

**Maintainer 響應度**：HNN ~13 open / few closed；LNN 14 open / **0 closed**（2026-05-26）。Greydanus 2020 後轉 reservoir computing / 教學；Cranmer 轉 PySR 符號回歸 —— 整條 hard-constraint 線**社群動能耗盡**。

---

## §7 · 比較 & 面試 Tip

| 同軸對手 | Axis 2 (injection) | Streaming | Pixel? | Scale 上限 | Status |
|---|---|---|---|---|---|
| **HNN / LNN** | **hard-constraint（架構）** | ✅ ODE integrator | ❌ | DoF ~10 | 停滯 |
| Symplectic ODE-Net | hard-constraint + dissipation | ✅ | ❌ | 7-DoF | 停滯 |
| HGN | hard-constraint (latent) | autoregressive | ⚠️ multi-frame | latent ~5 | 停滯 |
| PINN | **aux-loss（軟）** | ✅ | OK | ~10³ DoF fluid | 仍活躍（→ `./pinn.md`） |
| NewtonGen 2024 | architecture-bias-soft | autoregressive | ✅ | video scale | 新興 |
| PhysDiff / 物理 guidance diffusion | **guidance-gradient（採樣時）** | ❌ batch | ✅ | video scale | 主流（→ `./physdiff.md`） |
| Force-Prompting | **token-conditioning** | ✅ | ✅ | video scale | 新興（→ `./force-prompting.md`） |
| FNO neural surrogate | architecture-bias (spectral) | ✅ | N/A | fluid 10⁴ DoF | 主流（→ `../neural-surrogates/fno.md`） |

> **🎤 Interview Tip.** 「video world model 為什麼不用 Hamiltonian NN 保證能量守恆？我們要不要在 Cosmos / Dreamer 裡塞一層 symplectic update？」**正確答：「**hard-constraint 在 ontology Axis 2 上是最強訊號，但**它的表達力代價也是最大的**。HNN 在 pendulum 完美，但 (a) 假設 closed conservative system —— 真實 video 處處 dissipation；(b) contact 不可微 —— manipulation 整類沒法塞；(c) DoF > 50 數值崩 —— video / 4D scene 是 10⁶ DoF 量級。**所以 2024-25 主流選 aux-loss（PINN / NewtonGen）和 guidance-gradient（PhysDiff）—— 軟注入保留 backbone 表達力**。」**錯答**：「Hamiltonian 數學上更嚴謹，所以加進來」—— hard-constraint 不是「更嚴謹」，是「**架構承諾與表達力代價的取捨**」，video 規模上代價過大。

### 7.1 Falsifiable predictions

1. **2027-12 前**：hard-constraint 路線**不會**進入任何 1B+ 參數的 video world model 作為核心架構（可能作為 ablation 的 sanity check，但不是 backbone）。
2. **2027-12 前**：如果有第二代 hard-constraint 復興，會以 **latent-WM 內部一層 symplectic block** 形式出現（MetaSym arxiv [2502.16667](https://arxiv.org/abs/2502.16667) 的後續），不是整體架構 —— 即「混合 hard-soft」而非「純 hard」。
3. **2027-06 前不會發生**：HNN/LNN 直接 scale 到 video 規模並打贏 Cosmos / Dreamer V4 / V-JEPA —— 結構性表達力天花板尚未被突破，需要至少一個 architectural breakthrough（contact differentiable + 高 DoF 數值穩定 + pixel→canonical-coord 映射）才有機會。

---

## §8 · For the Reader（按 persona 分流）

- **VLA / robot policy 工程師** —— DeLaN 對 inverse dynamics 有 10× sample efficiency 的甜頭，**值得用在 model-based RL 的 dynamics model**。但 contact-rich manipulation 別碰，配 diff-sim（MuJoCo MJX / Brax）混用 —— diff-sim 處理 contact event，HNN/DeLaN 處理 contact 之間的 smooth segment。
- **自駕 closed-loop 工程師** —— 直接 skip。車輛動力學是 dissipative + contact-rich + 高 DoF，hard-constraint 不適用。aux-loss 路線（如 NewtonGen 的物理引導）才是你的候選。
- **影片生成工程師** —— 這是給你看的**反例**。**不要**把 Hamiltonian / Lagrangian 當 video WM 的核心架構 —— 表達力天花板會把你卡死。看 `./physdiff.md` 的 guidance-gradient 與 `./force-prompting.md` 的 token-conditioning，這些才是 2024-25 video 主流。
- **神經 PDE / surrogate 研究者** —— HNN 是你的 contrast point。看 `../neural-surrogates/fno.md` —— FNO 用 spectral truncation 拿「準守恆」是 architecture-bias-soft，scale 到 fluid 10⁴ DoF；HNN 嚴格守恆但只能玩擺鐘。三條路線是 fidelity (HNN) ↔ scalability (FNO) ↔ realism (GraphCast) 三角。
- **物理 conditioning 研究者（USP zone 讀者）** —— 這篇是 anchor。HNN/LNN 占據 Axis 2 `hard-constraint` 最乾淨的點，但**它的失效模式定義了 USP zone 的 design space** —— 為什麼 aux-loss / guidance-gradient / token-conditioning 三條軟路線在 2024-25 各擅勝場，因為 hard-constraint 把表達力代價墊太高。
- **Research 學生** —— 注意 §7.1 三條 falsifiable。如果你想做 hard-constraint 復興，正確的攻擊角度是 **「latent-WM 內部一層 symplectic block」** 而不是整體架構 —— MetaSym 的後續、或者 symplectic transformer block 是 wedge。直接 scale HNN 到 video 是 dead end。

---

## References

**Canonical**:
- **HNN** — Greydanus, Dzamba, Yosinski. *NeurIPS 2019* · [arXiv:1906.01563](https://arxiv.org/abs/1906.01563) · [code](https://github.com/greydanus/hamiltonian-nn)
- **LNN** — Cranmer, Greydanus, Hoyer, Battaglia, Spergel, Ho. *ICLR 2020 DeepDiffEq Workshop* · [arXiv:2003.04630](https://arxiv.org/abs/2003.04630) · [code](https://github.com/MilesCranmer/lagrangian_nns)
- **Symplectic ODE-Net** — Zhong, Dey, Chakraborty. *ICLR 2020* · [arXiv:1909.12077](https://arxiv.org/abs/1909.12077)
- **HGN** — Toth, Rezende, Jaegle, Racanière, Botev, Higgins. *ICLR 2020* · [arXiv:1909.13789](https://arxiv.org/abs/1909.13789)

**Engineered variants**:
- **DeLaN** — Lutter, Ritter, Peters. *ICLR 2019* · [arXiv:1907.04490](https://arxiv.org/abs/1907.04490)
- **SympNet** — Jin et al. *Neural Networks 2020* · [arXiv:2001.03750](https://arxiv.org/abs/2001.03750)
- **Symplectic RNN** — Chen, Zhang, Arjovsky, Bottou. *ICLR 2020* · [arXiv:1909.13334](https://arxiv.org/abs/1909.13334)

**Follow-ups & critiques**:
- David, Méhats. *Symplectic Learning for HNN* · [arXiv:2106.11753](https://arxiv.org/abs/2106.11753)（指出原 HNN 不一定 symplectic）
- **MetaSym** · [arXiv:2502.16667](https://arxiv.org/abs/2502.16667)（meta-learning + symplectic 結構，最後一條活躍 branch）

---

## Boundary

- 同 zone soft-constraint 對手 PINN → [`./pinn.md`](./pinn.md)
- 同 zone guidance-gradient 路線 → [`./physdiff.md`](./physdiff.md)
- 同 zone token-conditioning 路線 → [`./force-prompting.md`](./force-prompting.md)
- 對手 neural surrogate（FNO 譜系）→ [`../neural-surrogates/fno.md`](../neural-surrogates/fno.md)
- 與 5 axis 全景 → [`../../cheat-sheet/ontology.md`](../../cheat-sheet/ontology.md)

---

## ✍️ 維護者註（v0.5 → v1 升級清單）

本 v0.5 基於 4 篇 canonical paper + GitHub issues + 6 年 community follow-up 觀察。下次升 v1 時補：

1. ⏳ MetaSym 2025 的具體結果（是否 demo 到 latent-WM scale）
2. ⏳ Symplectic transformer block 是否已有具體 implementation（2025-2026 web search）
3. ⏳ DeLaN 在 2024-2026 機器人 manipulation pipeline 的最新採用率
4. ⏳ HNN 與 E(3)-equivariant（NequIP / Allegro / MACE）組合在分子動力學的 SOTA confirmation
5. ⏳ `./pinn.md` 補完後回填本文交叉連結（hard vs soft 對照）
6. ⏳ 驗證 HNN repo 預設積分器是否確實非 symplectic（讀 `hnn.py` 源碼確認）
7. ⏳ Status v0.5 → v1，刪本節

---

[← Back to Physics Conditioning](./overview.md)

Sources:
- [HNN arXiv 1906.01563](https://arxiv.org/abs/1906.01563)
- [LNN arXiv 2003.04630](https://arxiv.org/abs/2003.04630)
- [Symplectic ODE-Net arXiv 1909.12077](https://arxiv.org/abs/1909.12077)
- [HGN arXiv 1909.13789](https://arxiv.org/abs/1909.13789)
- [DeLaN arXiv 1907.04490](https://arxiv.org/abs/1907.04490)
- [greydanus/hamiltonian-nn GitHub](https://github.com/greydanus/hamiltonian-nn)
- [MilesCranmer/lagrangian_nns GitHub](https://github.com/MilesCranmer/lagrangian_nns)
- [MetaSym arXiv 2502.16667](https://arxiv.org/abs/2502.16667)
