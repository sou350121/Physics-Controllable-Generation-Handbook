<!-- ontology-5axis output=field injection=hard-PDE control=physical-param temporal=autoregressive domain=fluid -->

# Fourier Neural Operator (FNO) + PDE-Refiner

> Canonical: Li et al, *Fourier Neural Operator for Parametric Partial Differential Equations*, arxiv [2010.08895](https://arxiv.org/abs/2010.08895)（v1 2020-10-18, v3 2021-05-17, ICLR 2021）。Extension: Lippe et al, *PDE-Refiner*, arxiv [2308.05732](https://arxiv.org/abs/2308.05732)（NeurIPS 2023）。

## 1. One-paragraph TL;DR

FNO 把 PDE solver 抽象成「函數到函數的映射」 — 不是學「給網格 → 給網格」，而是學 operator $\mathcal{G}: a(x) \mapsto u(x)$。實作上把每層 kernel integral 換成 FFT → 在低頻 mode 上做 learned linear transform → IFFT。這個 spectral parametrization 帶來兩個 claim：(a) 對 grid resolution 不變（同樣的權重可以套到更密的 grid 推理）；(b) 對 Burgers/Darcy/Navier-Stokes 比 ResNet/U-Net baseline 快 1-3 個數量級且更準。**Prior gap**：DeepONet 把 branch/trunk net 拼起來學 operator 但沒利用 PDE 的 spectral 結構；CNN-based surrogate 是 mesh-dependent。FNO 用 spectral parametrization 把這兩端打平。**遺留問題**：autoregressive 長 rollout 仍會 drift —— PDE-Refiner（2023）用 diffusion-style 多步 refinement 修高頻分量，把 stable rollout horizon 拉長一個量級。

## 2. Core mechanism

每個 FNO layer 做的事：

```
input v(x) ──FFT──> v̂(k) ──truncate to k_max modes──> R_θ · v̂(k) ──IFFT──> back to x-space
                                                          │
                                                          └─── + W·v(x) (skip, pointwise)
                                                                       │
                                                                       └── σ(·) → next layer
```

- $R_\theta \in \mathbb{C}^{k_{max} \times d_v \times d_v}$ 是 learnable complex tensor，只在 low-frequency truncation 後的 mode 上作用。
- 因為 truncation 是 fixed `k_max`，而 FFT 對任意 grid 都能算，**同一組 $R_\theta$ 可以套到不同 resolution** —— zero-shot super-resolution claim 由此而來。
- Loss：典型 relative L2，supervised on simulator-generated trajectories。
- Computational cost：$O(N \log N)$ per layer（N=grid points），對比 attention $O(N^2)$。

PDE-Refiner 的擴充：把 single-step prediction 改成 K-step iterative refinement，每步加入越來越小的 Gaussian noise 然後 denoise —— 等價於 diffusion model with very few steps（典型 K=3-4）。關鍵 insight：MSE loss 會優先 fit dominant low-frequency mode，high-frequency error 累積導致 rollout drift；多步 refinement + noise schedule 強迫模型在每個 frequency band 都收斂。

## 3. 五軸定位 + 同軸對手

| 軸 | FNO | PDE-Refiner |
|---|---|---|
| output | `field`（連續速度/壓力場） | `field` |
| injection | `hard-PDE`（spectral truncation 隱含 smoothness inductive bias；但**並非嚴格守恆**，見 §8） | `hard-PDE` + diffusion-style refinement |
| control | `physical-param`（viscosity, IC, BC 作為 input function $a$） | 同 |
| temporal | `autoregressive`（多步 rollout 是 chain prediction） | `autoregressive` with K-step inner refinement |
| domain | `fluid`（Burgers / NS / Darcy 為主） | `fluid` |

**同軸對手**：

- **DeepONet**（Lu et al 2021）：用 branch（input function sampled）+ trunk（query coordinates）兩支 MLP 學 operator。優勢 = mesh-free（任意 query point），劣勢 = 不利用 spectral 結構，turbulence 表現遜於 FNO（原 FNO paper Table 1 直接比較）。
- **MeshGraphNet**（Pfaff et al 2021）：GNN over irregular mesh，message passing 學 local dynamics。優勢 = 任意 geometry，劣勢 = 不能 zero-shot 換 resolution，且 long-rollout drift 比 FNO 更嚴重（local-only inductive bias）。
- **U-Net baseline**：CNN on regular grid，是 FNO 原 paper 的主對手。FNO 在同等參數下準確度高 1 個量級。
- **PDE-Refiner**：與 FNO 互補 —— refinement 框架可套在 FNO/U-Net/任何 backbone 上，主要打的是「long rollout horizon」這個 FNO 自家失效的指標。

簡言之：spectral（FNO）vs spatial (MeshGraphNet) vs decoupled-MLP (DeepONet) vs refinement (PDE-Refiner) 是這條線的四向分裂。

## 4. Where it shines / where it breaks

### ⚡ Shines

- **Burgers 1D / Darcy 2D / NS 2D（vorticity formulation, Re≤10000）**：原 paper Table 1，FNO 比 U-Net 低 1 個量級 error，比 traditional pseudo-spectral solver 快 ~1000×。
- **Zero-shot super-resolution**：64×64 訓練的權重可以在 256×256 推理 — 對 multi-scale physics 是巨大優勢。
- **Periodic boundary**：FFT 天然假設 periodic，這個 regime FNO 幾乎沒對手。
- **Single-step / short-rollout 推理速度**：millisecond 量級對比 solver 的 minute 量級。

### ❌ Breaks

- **Long-rollout drift**：autoregressive 多步預測時 error 指數累積。原 paper 已坦承 rollout 不穩定；社群實測（PDEBench, Takamoto et al 2022）顯示 FNO 在 t > 50 step 後 NS turbulence 失效。**PDE-Refiner 主要打的就是這個點** —— 在 Kolmogorov flow 上把 stable horizon 從 ~50 拉到 ~400 step。
- **Irregular geometry**：FFT 要 regular grid + periodic BC，工程上常見的不規則邊界（airfoil、複雜管道）要走 Geo-FNO / F-FNO / domain decomposition 等變體，性能折扣明顯。
- **Hard boundary condition enforcement**：spectral truncation 不保證 Dirichlet / Neumann BC 嚴格滿足，需要 penalty loss 或 post-processing。
- **守恆律不嚴格**：雖然 `injection=hard-PDE` 標籤，但 FNO 不像 Hamiltonian NN / E(3)-equivariant 那樣 by-construction 保守 —— 質量/能量在 long rollout 會漂移。
- **High-Reynolds turbulence**：Re > 10^5 的真實 turbulence FNO 仍力不從心；DeepMind GenCast / GraphCast 的 graph approach 在 weather scale 更穩。
- **OOD viscosity / IC**：訓練時 viscosity 範圍外的 zero-shot generalization 差，這在原 paper 沒充分驗證。

## 5. Reproduction notes

- **Codebase**：[`neuraloperator/neuraloperator`](https://github.com/neuraloperator/neuraloperator)（PyTorch Ecosystem，Caltech / Anandkumar group 維護）。Pip 安裝 `neuraloperator`。
- **Benchmark**：[PDEBench](https://github.com/pdebench/PDEBench)（Takamoto et al 2022, NeurIPS D&B track）是社群事實標準，cover Burgers / NS / Darcy / SWE / DiffReact。
- **GPU 預算**：Burgers 1D 在單張 A100 約 30 分鐘到收斂；NS 2D 64×64 約 2-4 小時；NS 256×256 直接訓而非 super-resolution 推理則需要 8×A100 一天起跳。
- **典型踩坑**：
  1. `n_modes` 設太低（如 8）會 underfit dominant frequency；太高（>32 for 64-grid）會 overfit + 浪費參數。社群 rule of thumb：`n_modes ≈ grid/4`。
  2. Normalization：input/output 都要 per-channel standardize，否則 spectral coefficient 量級爆炸。
  3. Mixed precision：repo issue [#322](https://github.com/neuraloperator/neuraloperator/issues/322) 報 FP16 訓練不穩，bf16 較可行。
  4. Non-periodic BC：直接用 vanilla FNO 會在邊界出 artifact，要加 padding 或用 [Geo-FNO](https://arxiv.org/abs/2207.05209) 變體。
  5. Rollout evaluation：別只看 single-step MSE，要看 t=10/50/100 step 的 energy spectrum drift。

PDE-Refiner reference 實作見 [phlippe/PDE-Refiner](https://github.com/phlippe/PDE-Refiner)（作者倉）。

## 6. Cross-line synthesis

- **vs [GraphCast](./graphcast.md)**（graph-based surrogate）：GraphCast 用 icosahedral mesh GNN —— 結構天然處理球面非歐 geometry，在 weather production（ECMWF AIFS pipeline）裏壓 FNO 一頭。但 GraphCast 不能 zero-shot resolution，且訓練 cost 更高。**取捨**：規則 grid + periodic → FNO；不規則球面/邊界 → GraphCast。
- **vs sim-in-loop ([Genesis](../differentiable-simulators/genesis.md)-train, [Cosmos](../foundation-physics-models/cosmos-wfm.md)-rollout)**：sim-in-loop 用真實 solver 提供 ground truth 或 reward；FNO 是離線 supervised。Compose 方向：用 FNO 做 fast rollout，每 K step 用真 solver 校正（hybrid solver-surrogate scheme）—— PDEBench / NVIDIA Modulus 已有先例。
- **與 diffusion-based generation 結合**：PDE-Refiner 本身就是「FNO/U-Net backbone + diffusion-style K-step refinement」的典範。延伸方向 —— 把 score-based diffusion guidance（physics gradient）疊在 FNO 上做可控 generation，與本倉 `score-conditioned` injection axis 接上。
- **與 video WM 互補**：FNO 生成「場」（velocity / pressure），video WM 生成「像素」。Compose：FNO 算 fluid field → renderer 轉 pixel → video WM 拿來條件生成 visual 觀察。這條路在 `bridge-to-VLA/` 是 robotic manipulation with deformables 的潛在組合。
- **與 latent WM 對比**：[DreamerV4](../latent-world-models/dreamer-v4.md) 在 latent space rollout，FNO 在 physical field space rollout。前者 task-driven、後者 physics-driven —— 不直接競爭，但對 long-horizon 預測誰更穩仍是開放問題。

## 7. References

**Canonical**

1. Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A. *Fourier Neural Operator for Parametric Partial Differential Equations.* arxiv [2010.08895](https://arxiv.org/abs/2010.08895), ICLR 2021.
2. Lippe, P., Veeling, B. S., Perdikaris, P., Turner, R. E., Brandstetter, J. *PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers.* arxiv [2308.05732](https://arxiv.org/abs/2308.05732), NeurIPS 2023.

**Secondary / benchmark**

3. Takamoto, M. et al. *PDEBench: An Extensive Benchmark for Scientific Machine Learning.* NeurIPS 2022 D&B [arxiv 2210.07182](https://arxiv.org/abs/2210.07182).
4. Lu, L. et al. *DeepONet*: Nat. Mach. Intell. 2021 — operator-learning 同期對手。
5. Li, Z. et al. *Geo-FNO: Fourier Neural Operator with Learned Deformations for PDEs on General Geometries.* arxiv [2207.05209](https://arxiv.org/abs/2207.05209) — irregular geometry 補丁。
6. Pfaff, T. et al. *Learning Mesh-Based Simulation with Graph Networks.* ICLR 2021 — MeshGraphNet 同軸對手。
7. [TBD: verify GenCast 2024 DeepMind ensemble paper exact arxiv ID for cross-ref].

## 8. §8.x Pitfall log

- **§8.1 Long-rollout drift**（severity: high）。原文 §5 已坦承；PDEBench 系統性量測證實；PDE-Refiner motivation 即為此。Workaround：(a) 套 PDE-Refiner refinement; (b) hybrid solver-in-the-loop 校正; (c) ensemble 平均；(d) energy spectrum regularization loss。
- **§8.2 Non-periodic BC artifact**（severity: high for engineering apps）。FFT 假設 periodic，Dirichlet/Neumann 邊界在 vanilla FNO 會洩漏 high-freq 振盪到邊界。Workaround：Geo-FNO（learnable deformation map）或 zero-padding + masked loss。社群普遍承認此限制；neuraloperator 官方 docs 建議 non-periodic 場景優先用 U-FNO / F-FNO 變體。
- **§8.3 Resolution-invariance 有限**（severity: medium）。Zero-shot super-resolution claim 在 paper 是 Burgers / Darcy 的 mild upscale；社群測試（多篇 PDEBench follow-up）顯示 train 64 → test 1024 在 high-Re NS 上仍有顯著 error 增加 —— **claim 真實但被誇大**。
- **§8.4 Mixed precision instability**（severity: medium）。GitHub issue [#322](https://github.com/neuraloperator/neuraloperator/issues/322)：FP16 訓練 spectral conv 的複數運算數值不穩。Workaround：用 bf16 或 FP32 spectral block + FP16 其他部分。
- **§8.5 守恆律不嚴**（severity: high for scientific apps）。FNO 不是 by-construction 守恆架構，long rollout 質量/能量會漂移。對需要嚴格守恆的 application（如 climate ensemble）必須補 conservation loss 或後處理投影。**這是 `injection=hard-PDE` 標籤的 caveat** —— 比 Hamiltonian NN 弱。
- **§8.6 3D extension 不成熟**（severity: medium）。GitHub issue [#704, #697](https://github.com/neuraloperator/neuraloperator/issues) 報 3D FNO 內存爆炸 + 訓練不收斂。3D NS turbulence 仍是 open problem。
- **§8.7 Dataset / checkpoint reproducibility issues**（severity: low-medium）。Issue [#558, #691, #657](https://github.com/neuraloperator/neuraloperator/issues) 報 checkpoint 加載失敗、dataset 缺失。社群正在搬遷到 PDEBench standard，過渡期踩坑頻發。
- **§8.8 PDE-Refiner 對 backbone 的依賴**（severity: medium）。Refiner 不是「對任何 backbone 都等量提升」；原 paper appendix 顯示在 U-Net 上效果最強，FNO backbone 增益較小（推測因 FNO 已 frequency-aware，refinement 邊際遞減）。實作時別預期搭 FNO 就拿 paper headline number。

---

**衍生 dissection 候選**：GraphCast（同 zone）、Geo-FNO（FNO 變體）、Hamiltonian Neural Network（more-hard-PDE 對比）。
