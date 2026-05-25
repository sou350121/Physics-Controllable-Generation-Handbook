<!-- ontology-5axis output=field|particle injection=architecture-bias-soft|aux-loss control=param|3d-init temporal=autoregressive domain=fluid|rigid|soft -->

# MeshGraphNet

> Canonical: Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., Battaglia, P.W. *Learning Mesh-Based Simulation with Graph Networks.* arxiv [2010.03409](https://arxiv.org/abs/2010.03409), ICLR 2021 (outstanding paper / spotlight). 全員 DeepMind London。Code: [`google-deepmind/deepmind-research/meshgraphnets`](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) (Apache-2.0, TensorFlow 1.x)。

## 1. One-paragraph TL;DR

MeshGraphNet 是 DeepMind 2020 年 10 月放出、2021 ICLR 入選的 **mesh-based physics surrogate 鼻祖**：把有限元分析的 mesh 當作 graph，用 encode-process-decode GNN 學 next-step dynamics，並可在 forward simulation 中**自適應 remesh**。在六個跨物理域（cloth、incompressible CFD、compressible 空氣動力、結構力學）達到 1-2 個量級的 wall-clock 加速 vs 訓練用的 traditional solver。對本 handbook 的意義：它是 v2 ontology `injection=architecture-bias-soft` 軸的 **canonical anchor**（[ontology v2 §Axis 2](../../cheat-sheet/ontology.md#axis-2--physics-injection) line 56 明示）——「architecture 帶物理風味的 inductive bias（local message passing 對應 local PDE stencil；relative encoding 對應平移不變性），但**不**保證守恆律」，與 [Hamiltonian NN](../physics-conditioning/hamiltonian-lagrangian-nn.md)（hard-constraint）和 [PINN](../physics-conditioning/pinn.md)（aux-loss）三足鼎立。GraphCast 是它在球面氣象 domain 的後繼放大版（multi-mesh + 全球邊）。

## 2. Core mechanism

Encode-Process-Decode，three-block 標準 GN 結構。L=15 message passing layers 是論文 sweet spot。

```
FEM mesh state vᵢ at t                world-space proximity edges (collision)
   │                                       │
   ▼                                       ▼
┌──────────────────── Encoder ────────────────────┐
│  node embed:  vᵢ ← MLP(node_feat_i)             │
│  mesh edge:   e^M_{ij} ← MLP(x_i-x_j in u-space)│  ← relative mesh coord
│  world edge:  e^W_{ij} ← MLP(x_i-x_j in 3D)     │  ← relative world coord
└──────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────── Processor: L=15 GN blocks ───────────┐
│  ∀ block ℓ:                                     │
│    e^{M'}_{ij} ← f^M(e^M_{ij}, v_i, v_j)        │  mesh-edge update
│    e^{W'}_{ij} ← f^W(e^W_{ij}, v_i, v_j)        │  world-edge update
│    v'_i      ← f^V(v_i, Σ_j e^{M'}_{ij},        │
│                        Σ_j e^{W'}_{ij})         │
│   (residual connections; LayerNorm)             │
└──────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────── Decoder ────────────────────┐
│  acceleration / first-derivative per node:      │
│     pᵢ ← MLP(vᵢ)                                 │
│  Integrator (Euler) → state at t+Δt             │
└──────────────────────────────────────────────────┘
                      │
                      ▼
Optional remesher：sizing field → adaptive split/collapse
```

關鍵設計：

- **雙 edge 集合**：`mesh-space edges` 沿 FEM 拓撲（近似 PDE 內部 stencil）+ `world-space edges` 由 3D 距離閾值動態構造（補捕捉接觸/碰撞——這在 cloth-sphere、self-collision 必需）。這個拆分讓「PDE 內部動力學」與「接觸外力」分屬不同 MLP，效果上類似把 reaction term 與 contact term 解耦。
- **Relative encoding**：edge feature 用 $x_i - x_j$（mesh-space 與 world-space 各一份），node feature 不直接吃絕對座標——這給 translation invariance（DeepMind 主打 generalization 論據），但**非 rotation invariance**（旋轉一致性靠 data augmentation）。
- **Training noise injection**：訓練時對最近一步輸入加 Gaussian noise，並把 target 對應修正——逼模型學「從擾動狀態回到正確軌跡」，這是 long-rollout 穩定的關鍵 trick（一階系統直接 noise；二階 cloth 用 γ=0.1 加權兩種校正策略）。沒這招，autoregressive rollout 在 30-50 步就爆。
- **Adaptive remeshing**：另訓一個 sizing-field predictor（同款 GN backbone），輸出每節點期望邊長 → 用傳統 remesher split/collapse edge。Adaptive case 的訓練資料來自 ArcSim / COMSOL 的真實 adaptive simulation。
- **One-step supervision**：train 只看 t→t+Δt，rollout 純 inference 時 iterate。Noise injection 是唯一補 covariate shift 的機制。

L=15 是 paper Table 5 報的「accuracy / cost 折衷點」；社群復現（NVIDIA PhysicsNeMo、`echowve/meshGraphNets_pytorch`）多沿用。

## 3. 五軸定位 + 同軸對手

| Axis | MeshGraphNet | [GraphCast](./graphcast.md) | [FNO](./fno.md) | Pangu-Weather (transformer 線) | [PINN](../physics-conditioning/pinn.md) |
|---|---|---|---|---|---|
| **Output** | `field` (velocity/pressure/stress on mesh)；可視為 `particle` for cloth nodes | `field` (atmos state on icosahedral mesh) | `field` (regular grid) | `field` (regular grid + 4 vertical levels) | `field` (continuous via MLP) |
| **Injection** | `architecture-bias-soft` (GNN message passing → local PDE stencil；不保證守恆) + 部分 `aux-loss` for remesher | `architecture-bias-soft` + `aux-loss` (multi-step rollout loss) | `architecture-bias-soft` (spectral truncation → smoothness prior；但實際不嚴格守恆，被 v1 誤標 hard-constraint，v2 已修) | `data-only` (3D Earth-Specific Transformer, 純 ERA5 supervised) | `aux-loss` (PDE residual in loss) |
| **Control** | `param` (BC, material) + `3d-init` (initial mesh + IC) | `param` (initial atmos state) | `param` (IC, viscosity, BC as input function) | `param` (initial state) | `param` (PDE coeffs, BC) |
| **Temporal** | `autoregressive` (1-step train, multi-step rollout) | `autoregressive` (6h step) | `autoregressive` | `hierarchical autoregressive` (1/3/6/24h cascade) | non-temporal or autoregressive |
| **Domain** | `fluid \| rigid \| soft` (cross-physics！) | `weather` only | `fluid` 主 | `weather` only | PDE 通用 |

**同軸對手**（neural surrogate 軸）：

- **[GraphCast](./graphcast.md)**：MeshGraphNet 的「球面氣象大規模」後繼。從 single-mesh 升級到 multi-mesh（icosahedral 細分共存），把局部 stencil 拓寬到全球 teleconnection。Domain 完全不同（weather vs lab physics），但 architectural lineage 直系。GraphCast 在 ECMWF 進 prod 就是 MeshGraphNet 線「scale-up + 對 domain 做 inductive bias 補完」後的果。
- **[FNO](./fno.md)（spectral）**：跟 MeshGraphNet 構成「spectral vs spatial」二元。FNO 在 regular grid + periodic BC regime 強（zero-shot resolution）；MeshGraphNet 在不規則 geometry + 接觸碰撞強。**取捨**：規則 grid → FNO；複雜邊界 / 自適應 mesh / 跨物理域 → MeshGraphNet。Paper §5 直接拿 FNO/U-Net 當 baseline，MeshGraphNet 在 CylinderFlow 全程低 1 個量級 error。
- **Pangu-Weather**（[TBD: anchor 待補 `pangu-weather.md`](./graphcast.md#3-五軸定位--同軸對手)）：純 transformer + 3D Earth-Specific position encoding，inductive bias 走「encoding」而非「graph topology」。對 MeshGraphNet 的反命題——「GNN 不是必要，attention + 好 PE 也行」。但只在 weather 證明，跨 cloth/rigid/soft 沒人試過 transformer-only。
- **[PINN](../physics-conditioning/pinn.md)（aux-loss）**：injection 軸 orthogonal 對手。PINN 把 PDE 殘差塞 loss、架構保持普通 MLP；MeshGraphNet 把 PDE 結構塞架構（local message passing）、loss 純 supervised。兩者都「不 hard-constraint」但 mechanism 互補——理論上可 compose（MeshGraphNet 架構 + PINN PDE residual loss），實務上少見因為訓練資料 from solver 已含 PDE 約束。

## 4. Where it shines / where it breaks

### ⚡ Shines

- **跨物理域同一架構**：同一 `core_model.py` cover incompressible CFD (`CylinderFlow`)、compressible aero (`Airfoil`)、彈性結構 (`DeformingPlate`)、cloth (`FlagSimple/Dynamic`, `SphereDynamic`)。這是 paper 主賣點——**cross-material generalization without architectural changes**。後續工業界 ECMWF / NVIDIA / Siemens 採用都基於這個「one architecture, many physics」的 portability。
- **Adaptive resolution**：sizing-field predictor 學 ArcSim/COMSOL 自適應 remesh 策略，inference 時自己決定哪裡細哪裡粗。FlagDynamic 平均 ~2,767 nodes adaptively remeshed over 250 timesteps。其他 surrogate（FNO / Pangu）都是 fixed grid。
- **Speedup**：對訓練用 solver 加速 1-2 量級。Paper Table 表：FlagSimple 214.7×、FlagDynamic 31.3×、SphereDynamic 11.5×、DeformingPlate 89.0×、CylinderFlow 35.3×、Airfoil 289.1× on GPU。
- **Long rollout 在 in-distribution 穩**：訓 400 步，FlagSimple 推到 40,000 步仍視覺合理（noise injection 立功）。
- **Foundational citation**：截至 2026-05 是 mesh-based neural simulation 引用最高的 paper，幾乎所有後續 mesh GNN surrogate（X-MeshGraphNet、MeshGraphNetRP、PhysicsNeMo MGN）都以此為起點。

### ❌ Breaks

- **OOD mesh topology / 幾何 generalization 弱**：[Bonnet et al. 2024 arxiv 2408.06101](https://arxiv.org/abs/2408.06101) 系統測 MGN 跨 obstacle 形狀：訓 single cylinder → 測 `mixed_all`（多 obstacle + 異形）velocity all-steps error 從 89.19±17.2 升到 115.44±11.12（×10⁻³），pressure error 近乎翻倍（10.3 → 16.6 ×10⁻²）。直白引：「DeepMind claims that their MGN has strong generalization capabilities due to using a relative encoding on the mesh graphs, however, they report no experimental evidence for this」——relative encoding 給平移不變性，但**不給 vortex pattern 跨幾何 transfer**。
- **不保證守恆**：架構是 local message passing，質量 / 動量 / 能量在 long rollout 漂移。這正是 `injection=architecture-bias-soft` 而非 `hard-constraint` 的核心理由。對需要嚴格守恆的科學模擬（climate ensemble、long-term cardiovascular），MeshGraphNet 必須補 conservation loss 或 post-projection。
- **L=15 限制資訊傳播距離**：固定 15 hops → 一次 forward 內 node 只能看到 15-step 內鄰居。對需要 long-range 全局耦合的工況（高 Re turbulence、大形變 buckling）這就是硬上限。X-MeshGraphNet、multi-scale MGN 等後續就是補這個。GraphCast 的 multi-mesh 同樣是對症下藥。
- **Compute scales O(E)**：邊數線性，但 world-space edges 在密接觸（cloth pile）下會爆炸。20k nodes 的大 cylinder 訓練在 paper appendix 已標「需 multi-GPU」。
- **Remesher 依賴 ground-truth adaptive sim**：sizing-field 學自 ArcSim/COMSOL 的 adaptive output——若 ground-truth solver 本身就不會自適應（純結構 FEM），MGN 也學不會自適應 remesh。
- **TensorFlow 1.x rot**：原 repo 依賴 TF1.15 + sonnet v1，2026 環境近乎不可裝；社群一律 fork 到 PyTorch（`echowve/meshGraphNets_pytorch`、NVIDIA PhysicsNeMo）。

## 5. Reproduction notes

- **Repo**：[`google-deepmind/deepmind-research/meshgraphnets`](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)（TF1.x + sonnet v1，Apache-2.0）。PyTorch fork 推薦 [`echowve/meshGraphNets_pytorch`](https://github.com/echowve/meshGraphNets_pytorch)；商用 / 多 GPU 上 [NVIDIA PhysicsNeMo MGN tutorial](https://docs.nvidia.com/physicsnemo/latest/user-guide/model_architecture/meshgraphnet.html)。
- **Datasets**：`download_dataset.sh` 抓 `airfoil` / `cylinder_flow` / `deforming_plate` / `flag_simple` / `flag_dynamic` / `flag_dynamic_sizing` / `sphere_simple` / `sphere_dynamic` / `sphere_dynamic_sizing` / `flag_minimal`（test smoke）。`*_sizing` 變體是 pre-remesh + target sizing field。Hosted on Google Cloud public bucket。
- **GPU 預算**：CylinderFlow 在單 V100 約 1-2 天到收斂；FlagSimple 約 1 天；DeformingPlate / Airfoil 類似量級。Long-rollout evaluation 推 40k step 約幾分鐘。
- **典型踩坑**：
  1. **TF1.x 環境**：原 repo 在 TF 2.x 直接死，要么改 PyTorch fork 要么裝 Python 3.7 + TF1.15 容器。
  2. **Noise injection scale**：對不同物理域 noise std 需 tune（cloth 用 ~3e-3 m，CFD 用 velocity 的 ~2e-2）。配錯則 long rollout 早爆。
  3. **World-edge radius**：collision proximity 閾值對 cloth-cloth self-contact 敏感。Paper 用 ~3e-2 m，小了漏接觸大了爆 O(E²)。
  4. **L=15 別輕易調**：social fork 試過 L=10 大幅掉精度，L=20 邊際遞減 + memory 爆。
  5. **Adaptive remesh 推理時序**：先預測 sizing field → 跑外部 remesher → 再預測下一步。Pipeline 易卡在 remesher（C++）。

## 6. Cross-line synthesis

- **vs [FNO](./fno.md)（同 zone, spectral 對手）**：MeshGraphNet 是 spatial / local message passing，FNO 是 spectral / global frequency mixing。一句話：**MeshGraphNet 處理 geometry，FNO 處理 frequency**。MeshGraphNet 的 long rollout drift 比 FNO 更早出現（local-only inductive bias），但 OOD geometry 反而有些優勢（FNO 死在 non-periodic BC）。Compose 思路：FNO 做 bulk fluid + MeshGraphNet 做 boundary layer / 接觸——目前 NVIDIA PhysicsNeMo 已開始嘗試這種 hybrid。
- **vs [GraphCast](./graphcast.md)（同 zone, scale-up）**：直系後繼。Multi-mesh 補 long-range 耦合（解 L=15 信息距離限制），weather supervision 替換 lab physics solver supervision。**意義**：MeshGraphNet 證明 GNN 可學 PDE，GraphCast 證明這個 lineage 能進業務。
- **vs Hamiltonian / Lagrangian NN（同 inject 軸對位）**：HNN/LNN 是 `hard-constraint`（symplectic structure by construction → energy 守恆嚴格）；MeshGraphNet 是 `architecture-bias-soft`。**v2 Pareto sweet spot 論證**（[ontology v2 §Axis 2 圖 2](../../cheat-sheet/ontology.md#axis-2--physics-injection)）：HNN 嚴格但 narrow（單系統能量），MeshGraphNet 廣譜但不嚴格（多物理域 + 軟 bias）。實務上 MeshGraphNet 的 fidelity / expressivity 折衷更實用——這是 architecture-bias-soft 軸在 2020-2026 爆發的根本原因。
- **vs [PINN](../physics-conditioning/pinn.md)（同 inject 軸對位）**：兩條路都「不 hard-constraint」但 mechanism 對立——PINN 把 physics 塞 loss、MeshGraphNet 塞 architecture。Compose 在理論上可行（GNN backbone + PDE residual aux loss），但 supervised data 從 solver 來時 PDE residual 已隱含 → 邊際遞減。
- **與 diff-sim 接（[Genesis](../differentiable-simulators/genesis.md) / MJX）**：MeshGraphNet 一旦訓好就是 differentiable fast surrogate，可塞 RL / inverse design loop 當代理 simulator（比 Genesis 快數量級，但精度差）。NVIDIA Modulus 已有此 pattern——粗 design space 用 MGN 搜，精篩用 真 solver。
- **與 video WM / VLA 接**：MeshGraphNet 輸出 `field` 不直接生 pixel，但 cloth / fluid simulation 渲染成 video 後可給下游 video WM 當 ground truth、或給 [VLA-Handbook](https://github.com/sou350121/VLA-Handbook) 的 manipulation policy 提供 deformable object dynamics 訓練資料（dish-towel folding、soft-grasp training）。這是 `bridge-to-vla/` 的 potential composition。

## 7. References

**Canonical**：

1. Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., Battaglia, P.W. *Learning Mesh-Based Simulation with Graph Networks.* arxiv [2010.03409](https://arxiv.org/abs/2010.03409), ICLR 2021 (outstanding paper / spotlight). DeepMind London.

**Secondary / 二手實測**：

2. Bonnet, F., et al. *Generalization capabilities of MeshGraphNets to unseen geometries for fluid dynamics.* arxiv [2408.06101](https://arxiv.org/abs/2408.06101) (Aug 2024) — 系統量 OOD geometry 失效。
3. *X-MeshGraphNet: Scalable Multi-Scale Graph Neural Networks for Physics Simulation.* arxiv [2411.17164](https://arxiv.org/abs/2411.17164) (Nov 2024) — 補 L=15 信息距離限制。
4. *MeshGraphNetRP: Improving Generalization of GNN-based Cloth Simulation.* ACM SIGGRAPH MIG 2023 — cloth generalization 後續。
5. Battaglia, P.W., et al. *Relational inductive biases, deep learning, and graph networks.* arxiv [1806.01261](https://arxiv.org/abs/1806.01261) (2018) — GN block 抽象定義，是 MGN 的 prior work。
6. NVIDIA PhysicsNeMo [MeshGraphNet tutorial](https://docs.nvidia.com/physicsnemo/latest/user-guide/model_architecture/meshgraphnet.html) — 工業界復現參考。
7. Sanchez-Gonzalez, A., et al. *Learning to Simulate Complex Physics with Graph Networks (GNS).* arxiv [2002.09405](https://arxiv.org/abs/2002.09405), ICML 2020 — MeshGraphNet 的粒子版前作，同組同年早幾個月。

## 8. §8.x Pitfall log

| § | Source | Issue / Observation | Severity | Workaround |
|---|---|---|---|---|
| §8.1 | Bonnet et al. 2024 arxiv 2408.06101 | 訓 single cylinder → 測異形 / 多 obstacle，velocity all-steps error +29%、pressure error 翻倍 | High | 訓練集做 geometry augmentation；切 `mixed_all` 訓而非單一形狀 |
| §8.2 | Bonnet et al. 2024 §4 | 「DeepMind 主打的 relative encoding 跨幾何 generalization claim 無實驗證據」 | High | 不可信 paper §3 implicit claim，務必自測 OOD |
| §8.3 | 多篇社群復現 + paper §S2 | TF1.x + sonnet v1 環境在 2024 後幾乎不可重建（Python 3.7、CUDA driver 老化） | High | 用 [`echowve/meshGraphNets_pytorch`](https://github.com/echowve/meshGraphNets_pytorch) 或 NVIDIA PhysicsNeMo |
| §8.4 | Paper §4 + community | Long-rollout（>1k steps）若沒做 noise injection 訓練則必爆 | High | 一定要做 noise injection；對二階系統用 γ=0.1 加權校正 |
| §8.5 | Paper §S3 + X-MeshGraphNet motivation | L=15 message passing → 信息傳播只 15 hops，跨遠距 coupling 工況（高 Re vortex、buckling）力不從心 | High | 用 multi-scale GNN（X-MeshGraphNet）或 multi-mesh（GraphCast 思路） |
| §8.6 | Ontology v2 §Axis 2 | 不保證守恆律——稱 `architecture-bias-soft` 而非 `hard-constraint` 的根本理由 | Med | 補 conservation loss / post-projection；或上 HNN/LNN |
| §8.7 | World-edge 設計 paper §3 | Self-collision dense contact（cloth pile）world-edge 數 O(N²) 爆 memory | Med | 提高 proximity threshold + spatial hash；或 attention sparsification |
| §8.8 | Paper §S2 + adaptive remesh 章 | Sizing-field predictor 依賴 ground-truth adaptive sim（ArcSim/COMSOL）；若 solver 本身 fixed mesh，學不會自適應 | Med | 提供 adaptive solver baseline，或合成 sizing-field GT |
| §8.9 | deepmind-research README + 多 fork issue | 原 repo 自 2021 後幾乎無維護；issue 累積不回覆 | Low | 接受「論文配套 release」非 production code；用 NVIDIA PhysicsNeMo 是 SLA 路線 |

---

**TBD checklist for next pass**:

- 補 `pangu-weather.md` anchor 後修正 §3 cross-ref 路徑。
- 拉 `deepmind-research` GitHub issue tracker 具體編號（README 並無顯式 known-limitations issue thread，截至 2026-05 該 repo 無活躍 issue 標籤系統用 review）——目前只能引論文 §S2 / Bonnet 2024 / X-MeshGraphNet motivation 作二手證據，**未在 §8 直接引 deepmind-research 編號 issue**。
- 確認 paper 報的 speedup 表是否在 v4（ICLR camera ready）有變動；目前用 ar5iv mirror 的數字。
- 補 `bridge-to-vla/` 中 deformable manipulation 對 MeshGraphNet 的具體 compose 案例（若有）。
