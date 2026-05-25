<!-- ontology-5axis output=field|particle injection=constraint-loss control=physical-param|3d-prompt temporal=autoregressive domain=fluid|rigid|soft -->

# Physics-Informed Neural Networks (PINN)

> Canonical: Raissi, Perdikaris, Karniadakis, *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*, J. Comput. Phys. 378 (2019) 686-707. Preprints: arxiv [1711.10561](https://arxiv.org/abs/1711.10561)（Part I — Data-driven Solutions, 2017-11-28）+ [1711.10566](https://arxiv.org/abs/1711.10566)（Part II — Data-driven Discovery, 2017-11-28）。Library: Lu et al, *DeepXDE*, SIAM Review 63(1) 2021, arxiv [1907.04502](https://arxiv.org/abs/1907.04502)。

## 1. One-paragraph TL;DR

純資料驅動的 NN（Sobel-only / pixel L2）對 PDE 系統的兩個致命傷：(a) **資料稀疏**就學不動 — 真實實驗只有 boundary / sensor 幾個點；(b) **OOD extrapolation 失效** — 沒看過的 viscosity / boundary 一外推就崩。PINN 把 PDE residual 本身寫成 auxiliary loss，用 autograd 對網路輸出取 ∂/∂x、∂²/∂x²，硬塞進 total loss。**Prior gap**：把 PDE 從「solver 端硬解」搬到「optimization 端 soft penalty」，於是 inverse problem（從觀測反推未知 PDE 係數）變成跟 forward problem 同一套機制。**為什麼這篇 handbook 把 PINN 當 `constraint-loss` 的標準參考**：所有後續 video / scene / NeRF + physics loss 的論文，基本都是 PINN 把 PDE 換成自家領域守恆律的再包裝；理解 PINN 的 failure mode（多尺度、stiff、loss-weight Pareto）就是理解這條 injection 線的下限。**遺留問題**：原 paper 在 Burgers / Schrödinger / Navier-Stokes（low-Re）漂亮 demo，但 Krishnapriyan et al NeurIPS 2021 證明 convection / reaction / diffusion 稍微把參數調出 trivial regime，vanilla PINN 就直接失效。這個「demo 跑得動但 production 撐不住」的鴻溝，是接下來五年所有 PINN 變體（causal / self-adaptive / hard-constraint）想填的。

## 2. Core mechanism

給 PDE $\mathcal{N}[u](x,t) = 0$，邊界算子 $\mathcal{B}[u] = g$，初始條件 $u(x,0) = u_0$，PINN 用 MLP $u_\theta(x,t)$ 同時拟合所有條件：

```
        x, t ──┐
               ├──► MLP u_θ(x,t) ──► û
        (時空)  │           │
               │           ├──autograd──► ∂û/∂t, ∂û/∂x, ∂²û/∂x², ...
               │           │                       │
               │           │                       ▼
               │           │             residual r = N[û] (e.g. u_t + u·u_x - ν·u_xx)
               │           │                       │
               │           │                       ▼
               │           │             L_pde = ‖r‖²  (collocation 點)
               │           ├──► L_bc  = ‖B[û] - g‖²  (boundary 點)
               │           ├──► L_ic  = ‖û(x,0) - u_0‖²  (initial 點)
               │           └──► L_data= ‖û - u_obs‖²  (sparse sensor，可選)
               │                       │
               ▼                       ▼
       total: L = λ_pde·L_pde + λ_bc·L_bc + λ_ic·L_ic + λ_data·L_data
              └─► ∇_θ L → Adam (~10⁴-10⁵ iter) → LBFGS finetune (~10³ iter)
```

關鍵在三點：(1) PDE residual 是 **網路本身對 input 的 autograd 算出來**，不需要任何 mesh 或 finite-difference stencil；(2) 整個 loss 在隨機取樣的 collocation 點上計算，可以 mesh-free；(3) inverse problem 只要把 PDE 係數 ν 也設成 `trainable variable`，就能跟 θ 一起學 — 這是 PINN 在 scientific ML 最大的賣點。

實作上 Adam warm-up 之後幾乎都要接 LBFGS：Adam 把 loss 從 O(1) 拉到 O(10⁻²)，LBFGS 才能再壓到 O(10⁻⁵)；少了 LBFGS 這步，幾乎所有 paper 的 headline 數字都復現不出來。

## 3. 五軸定位 + 同軸對手

| 軸 | PINN | Hamiltonian/Lagrangian NN | FNO/MeshGraphNet | PhysDiff |
|---|---|---|---|---|
| output | `field`（u(x,t) 連續場） | `field`（particle q,p over t） | `field` | `pixel-video` |
| injection | **`constraint-loss`**（PDE residual as soft penalty） | **`hard-PDE`**（symplectic 架構 by-construction） | `hard-PDE`（spectral inductive bias） | `score-conditioned`（physics gradient 加在 score） |
| control | `physical-param`（PDE coeff, BC, IC） | `physical-param`（initial state） | `physical-param` | `text` + `physical-param` |
| temporal | `autoregressive`（在 (x,t) 空間 query 任意 t） | `streaming`（ODE 解算） | `autoregressive` | `joint-rollout` |
| domain | `fluid` / `rigid` / `soft`（generalist PDE） | `rigid`（conservative system） | `fluid` 為主 | `rigid` |

**同軸對手**：

- **Hamiltonian NN / Lagrangian NN**（[link](./hamiltonian-lagrangian-nn.md)）— 把守恆律寫到**架構**裡而非 loss 裡。優勢 = 能量嚴格守恆、long rollout 不漂；劣勢 = 只適用 conservative system，dissipation / contact / forcing 一律不行。**取捨**：PINN = 通用但 soft，Hamiltonian = 嚴格但窄。
- **FNO / MeshGraphNet**（[FNO](../neural-surrogates/fno.md)、[GraphCast](../neural-surrogates/graphcast.md)）— supervised operator learning。優勢 = 一旦訓好 inference 是 millisecond；劣勢 = 需要 simulator 預先產 trajectory 資料集（PINN 只要 PDE 數學式）。**取捨**：PINN 在 inverse / 稀疏資料 / 新 PDE 第一次嘗試最強；FNO 在 production rollout 最強。
- **PhysDiff**（[link](./physdiff.md)）— diffusion score 加 physics gradient guidance。同樣是 soft constraint 但載體是 diffusion 而非 MLP；對「生成可控 video」是更直接的路。**取捨**：PINN 學的是「解 u(x,t)」；PhysDiff 學的是「採樣符合物理的 pixel」— output space 根本不同，是兩條互不取代的線。

簡言之：PINN 在「injection 強度／generalization 寬度」這軸上是中等偏弱的 baseline — 是所有比較的原點，不是冠軍。

## 4. ⚡ Shines / ❌ Breaks

### ⚡ Shines

- **Inverse problem with sparse sensor data**：給定幾個點的 u 觀測，反推 PDE 係數（如未知 viscosity、reaction rate）。原 paper Part II 在 Burgers / KdV / NS 的 viscosity discovery 是教科書級 demo。
- **新 PDE 第一次 prototyping**：不需要寫 solver、不需要 mesh、不需要 boundary discretization；改個 lambda function 就能換 PDE。
- **不規則 / 高維 domain**：6D Fokker-Planck、複雜幾何邊界，傳統 solver 要重寫 mesh，PINN 重新取樣 collocation 點即可。
- **Soft constraint + data fusion**：可以同時拟合 PDE + 雜訊感測器資料，自然處理 ill-posed inverse problem。

### ❌ Breaks

- **Krishnapriyan et al NeurIPS 2021（arxiv [2109.01050](https://arxiv.org/abs/2109.01050)）系統性 failure mode**：作者在 convection / reaction / diffusion 三類 operator 上，**只把 convection coefficient β 從 small 調大**，vanilla PINN error 就從 ~10⁻³ 跳到 O(1)。根本原因 = soft regularization 讓 optimization landscape **ill-conditioned**，Adam/LBFGS 都卡在 local minimum。論文提出 curriculum + sequence-to-sequence 兩種修法但都治標。
- **Wang, Yu, Perdikaris JCP 2022（NTK 視角，arxiv [2007.14527](https://arxiv.org/abs/2007.14527)）**：用 Neural Tangent Kernel 解釋為什麼 — `L_pde` 與 `L_bc` 的 gradient 量級可差 2-4 個數量級，dominant 的那一項把 NN 訓往單一方向，另一項根本沒在學。**這是「loss weight tuning instability」的理論根因**。
- **Multi-scale / stiff PDE**：頻率成份跨好幾個量級時，MLP 偏好低頻、高頻被壓制（spectral bias），導致 sharp gradient / boundary layer 學不到。
- **Long-horizon propagation**：PDE 在大時間域上 PINN 容易 fit「全 t 平均」而違反 causality — Wang et al 2022（arxiv [2203.07404](https://arxiv.org/abs/2203.07404)）證明 vanilla PINN 在 Lorenz / Kuramoto-Sivashinsky / 高 Re NS 完全失效。
- **Curse of dimensionality**：當 PDE 維度 > 6-8 時 collocation 取樣成本爆炸；雖然比傳統 mesh-based solver 緩，但仍然輸給有結構利用的方法（如 Fokker-Planck 的 tensor train）。
- **Hard boundary 不嚴**：soft penalty 是「儘量滿足」不是「強制滿足」— DeepXDE issues [#90](https://github.com/lululxvi/deepxde/issues/90)、[#192](https://github.com/lululxvi/deepxde/issues/192) 反復出現 BC 違反 1-5% 的回報。

## 5. Reproduction notes

- **Libraries（按成熟度排序）**：
  1. **DeepXDE**（[lululxvi/deepxde](https://github.com/lululxvi/deepxde)）— Lu et al, SIAM Review 2021；TF / PyTorch / JAX / PaddlePaddle 多 backend；社群事實標準。
  2. **NVIDIA Modulus / PhysicsNeMo**（[NVIDIA/modulus](https://github.com/nvidia/modulus)，已改名 PhysicsNeMo）— 工業級、multi-GPU、整合 FNO + PINN + DeepONet。適合大規模 / 真實工程 PDE。
  3. **SciANN** — Keras 包裝，教學友好但社群活躍度遜於前兩者。
- **典型 setup**：MLP 4-8 層 / 寬度 50-100 / tanh 啟用；Adam lr=1e-3, ~20k iter；接 LBFGS-B finetune ~1-5k iter。1D Burgers 在 V100 約 5-15 分鐘到 paper 級 error。
- **常見踩坑**：
  1. **Adam 跑完忘了接 LBFGS** — paper 的 headline error 幾乎都靠 LBFGS finetune；只跑 Adam 的人會以為 PINN 是騙子。
  2. **`loss_weights` 設成 [1,1,1,1]** — 不同 term 的 gradient 量級差幾個數量級（NTK 已證），需要 grid search 或用 adaptive scheme（如 SA-PINN、NTK-weighted、GradNorm）。
  3. **Collocation 點數太少** — 1D 至少 ~10k，2D ~50k，3D 容易爆 GPU memory。
  4. **PDE 形式沒 normalize** — 把 PDE 寫成 $u_t + u u_x = \nu u_{xx}$ 時，如果 ν=1e-6（真實低粘性）會讓 residual 數量級失衡，先 non-dimensionalize 再丟給 PINN。
  5. **LBFGS 在 TF backend 卡在 30 iter** — DeepXDE issue [#1819](https://github.com/lululxvi/deepxde/issues/1819) 報過；換 PyTorch backend 或升 tf-probability 版本。
  6. **Hard constraint 不會自動處理 Neumann** — Dirichlet 可以用 ansatz `u = g(x) + N(x)·NN(x)`，但 Neumann 需要更巧的 output transform（DeepXDE issue [#1837](https://github.com/lululxvi/deepxde/issues/1837)）。

## 6. Cross-line synthesis

- **PINN 餘味在 video diffusion**：PhysGen（Liu et al, ECCV 2024，[link](./physgen.md)）並非把 PINN 整個搬到 video，而是把 PINN 的「rigid body / contact ODE 寫進 loss」的精神拆出來 — image 端用 generative diffusion 渲染，simulation 端用真 solver，**physics loss 只在中介 latent 上加**。直接把 PDE residual 丟到 video diffusion 的天真做法（PI-Loss for video diffusion）目前沒有成功案例 — score function 是 denoising 方向梯度，PDE residual 是物理約束梯度，**兩者方向不一定相容**，疊加會產生 distribution shift。這條路在 `crossing/controllability-vs-fidelity/` 是 open Pareto。
- **PINN-meets-NeRF**：NeRF 本質也是「空間座標 → 場值」的 coord-based MLP，跟 PINN 形式上同構。社群已有把 PDE residual 當 aux loss 加在 NeRF 上的嘗試（physics-informed NeRF for fluid reconstruction、PI-MLP for hash-encoded fields），核心難題仍是 spectral bias — NeRF 的 hash encoding 對高頻友好，但 PINN 的 autograd 高階導數在 hash encoding 上數值不穩。[TBD: verify exact arxiv ID for physics-informed NeRF 流體重建 canonical work]。
- **vs sim-in-loop**：當你有可微 simulator（Genesis、PhysX-diff），sim-in-loop 直接給 ground-truth trajectory 訊號，比 PINN 的 soft PDE penalty 強得多。PINN 主場仍是「**沒 simulator、只有 PDE 數學式 + 稀疏觀測**」這個 niche。
- **vs `score-conditioned` 線**：PhysDiff / classifier-physics-guided diffusion 是把物理梯度加在 score function 上，本質仍是 soft constraint，但載體是 diffusion 採樣而非 MLP 拟合。可以視為 PINN 在 generative 範式下的近親；但兩者的 generalization 來源不同（PINN 來自 PDE 數學式，PhysDiff 來自大規模 video 預訓 + guidance）。

## 7. References

**Canonical**

1. Raissi, M., Perdikaris, P., Karniadakis, G. E. *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* J. Comput. Phys. 378 (2019) 686-707. [arxiv 1711.10561](https://arxiv.org/abs/1711.10561) (Part I) + [arxiv 1711.10566](https://arxiv.org/abs/1711.10566) (Part II).
2. Lu, L., Meng, X., Mao, Z., Karniadakis, G. E. *DeepXDE: A deep learning library for solving differential equations.* SIAM Review 63(1) (2021) 208-228. [arxiv 1907.04502](https://arxiv.org/abs/1907.04502).

**Failure-mode 文獻**

3. Krishnapriyan, A., Gholami, A., Zhe, S., Kirby, R., Mahoney, M. *Characterizing possible failure modes in physics-informed neural networks.* NeurIPS 2021. [arxiv 2109.01050](https://arxiv.org/abs/2109.01050).
4. Wang, S., Yu, X., Perdikaris, P. *When and why PINNs fail to train: A neural tangent kernel perspective.* J. Comput. Phys. 449 (2022). [arxiv 2007.14527](https://arxiv.org/abs/2007.14527).

**Modern extensions**

5. McClenny, L., Braga-Neto, U. *Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism.* JCP 2022. [arxiv 2009.04544](https://arxiv.org/abs/2009.04544).
6. Wang, S., Sankaran, S., Perdikaris, P. *Respecting causality is all you need for training physics-informed neural networks.* [arxiv 2203.07404](https://arxiv.org/abs/2203.07404) (2022).

**Generative / cross-line**

7. Liu, S., Ren, Z., Gupta, S., Wang, S. *PhysGen: Rigid-Body Physics-Grounded Image-to-Video Generation.* ECCV 2024. [project page](https://stevenlsw.github.io/physgen/).
8. NVIDIA Modulus / PhysicsNeMo: [github.com/NVIDIA/modulus](https://github.com/nvidia/modulus)（已改名 [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo)，遷移過渡中）。

## 8. §8.x Pitfall log

- **§8.1 Loss-weight imbalance（NTK pathology）**（severity: high）。DeepXDE issue [#215](https://github.com/lululxvi/deepxde/issues/215)（closed, 2021-02）+ [#982](https://github.com/lululxvi/deepxde/issues/982)（opened 2022-10）。原文：「Is there a way to make these weights dynamic such that they counteract the issue of gradient imbalance between different magnitudes of gradients of different loss terms?」理論根因見 Wang et al 2022 NTK 論文：不同 loss term 的 gradient eigenvalue 跨數量級，dominant term 把訓練拖向自己方向，其他 term 形同沒在學。Workaround：(a) GradNorm；(b) NTK-weighted（Wang 2022 §4 直接給公式）；(c) Self-Adaptive PINN（McClenny 2020）；(d) Sobol-sampled collocation + λ grid search。
- **§8.2 Weight initialization sensitivity / 訓練 run-to-run variance**（severity: high）。DeepXDE issue [#305](https://github.com/lululxvi/deepxde/issues/305)（closed）。原文：「the order of magnitude of the pdes losses when they are started to be trained varies a lot from run to run」— 同 seed 不同 init 下，初始 PDE loss 可以差 10⁵ 量級，有時直接 NaN。Workaround：(a) 固定 Xavier/Glorot scale；(b) 跑 N=5+ seed 取 median；(c) PDE non-dimensionalize 把 residual 量級先正規化。
- **§8.3 Navier-Stokes 不收斂**（severity: high for fluids）。DeepXDE issue [#80](https://github.com/lululxvi/deepxde/issues/80)。原文：「continuity and x-momentum residuals are at best ~1e-1 regardless of the number of epochs, network size, network architecture」— 標準 incompressible NS 不論怎麼調都壓不下殘差。對應 Krishnapriyan 2021 的 convection failure mode。Workaround：(a) 從低 Re 退到 trivial regime；(b) 換 vorticity-streamfunction 形式（自動滿足 continuity）；(c) curriculum on Re。
- **§8.4 Inverse problem parameter 不收斂**（severity: high for scientific ML）。DeepXDE issue [#251](https://github.com/lululxvi/deepxde/issues/251)。Forward 跑得起來但 inverse 估 unknown velocity 一路發散到 5.24（真值 2.0）。Workaround：(a) 預訓 forward 再開 trainable parameter；(b) parameter 加先驗 / log-domain reparametrize；(c) 增加 sensor data 點密度。
- **§8.5 Hard boundary 不嚴 / Neumann 難實作**（severity: medium-high for engineering）。DeepXDE issues [#90](https://github.com/lululxvi/deepxde/issues/90)、[#192](https://github.com/lululxvi/deepxde/issues/192)、[#1837](https://github.com/lululxvi/deepxde/issues/1837)。Dirichlet 可以用 ansatz `u(x) = g(x) + d(x)·NN(x)`（d(x) 在邊界為 0）達 hard constraint；但 Neumann（∂u/∂n=0）需要更精巧的 output transform，社群長期沒有 clean solution。Workaround：(a) 用 ghost-point penalty；(b) DeepXDE 1.10+ 的 `output_transform` 自訂；(c) 直接放棄 hard、加大 BC loss weight 50-100×。
- **§8.6 LBFGS 在 TF backend 卡 30 iter**（severity: medium）。DeepXDE issue [#1819](https://github.com/lululxvi/deepxde/issues/1819)。Adam 跑得動，切到 LBFGS-B 後恰好 30 iter 停。Workaround：(a) 換 PyTorch backend；(b) 升 tensorflow-probability ≥0.20；(c) 改用 SciPy LBFGS-B 包外接（DeepXDE 提供 `model.train(optimizer="L-BFGS")` 但底層仍是 tfp）。
- **§8.7 Multi-scale spectral bias**（severity: high for stiff PDE）。MLP 對低頻友好、高頻被壓 — 經典 spectral bias 在 PINN 上加倍嚴重，因為 PDE residual 高階導數放大高頻誤差。Krishnapriyan 2021 + Wang 2022 都實證 sharp gradient / boundary layer / shock front 無法被 vanilla PINN 學到。Workaround：(a) Fourier feature embedding（Tancik 2020）；(b) multi-scale PINN（Liu 2020 分 frequency band）；(c) domain decomposition（XPINN, Jagtap 2020）；(d) hash encoding（要小心高階 autograd 數值穩定性）。
- **§8.8 Causality violation in long-horizon rollout**（severity: high for chaotic / turbulent）。Wang, Sankaran, Perdikaris 2022（arxiv [2203.07404](https://arxiv.org/abs/2203.07404)）證明 vanilla PINN 對 t 全域同等 loss → 模型可能先 fit t=T 把全域平均拉低，違反「t=0 解錯則 t=T 不該對」的因果。在 Lorenz / Kuramoto-Sivashinsky chaotic regime / 高 Re NS 直接失效。Workaround：(a) Causal PINN — 殘差加 temporal weight，前面點沒收斂前下游 weight 自動為零；(b) time-marching / sequential training；(c) Causality-Respecting Adaptive Refinement（[arxiv 2410.20212](https://arxiv.org/abs/2410.20212), 2024）。

---

**衍生 dissection 候選**：Hamiltonian/Lagrangian NN（同 zone hard-PDE 對比）、PhysGen（cross-line 到 video generation）、Self-Adaptive PINN deep-dive（PINN 變體獨立一篇）、PhysDiff（同 zone score-conditioned 對比）。
