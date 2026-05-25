<!-- ontology-5axis output=particle injection=architecture-bias-soft|aux-loss control=param temporal=autoregressive domain=fluid|granular|soft -->

# NeuralMPM (+ GNS / Inverse-GNS)

## 1. One-paragraph TL;DR

NeuralMPM (Rochman Sharabi, Lewin, Louppe, arXiv 2408.15753, latest v3 Feb 2025) 是 University of Liège 把 **Material Point Method 的 grid-particle 雙重表徵搬進 neural emulator** 的工作：粒子狀態 P2G 插值到 fixed-size voxel grid → 用 **image-to-image CNN（U-Net 系）算下一步 grid 更新** → G2P 插值回粒子。它要填的 prior gap 不在「神經模擬粒子」本身（**GNS** Sanchez-Gonzalez et al. ICML 2020 / arXiv 2002.09405 早就證明 GNN 可學 fluid/rigid/deformable），而是 **GNS 的 cost wall**——GNS 訓練 10 天、推理慢、memory 隨粒子數線性炸。NeuralMPM 把 message-passing 換成 grid-CNN，**訓練 10 天 → 15 小時、memory 降 10×–100×、推理快 5×–10×**，long-rollout accuracy 不退（作者聲稱對比 GNS / DMCF 持平或更好）。對 handbook 的意義：它把 `output=particle` × `injection=architecture-bias-soft` 這格從 GNS 一個 anchor 擴成 grid-vs-graph 兩條路線，並把 **Inverse-GNS**（Choi & Kumar, arXiv 2401.13695，granular landslide 反問題）這類 differentiable-surrogate-for-design loop 從學術小眾搬上日程——它證明 particle surrogate 已經能進 `sim-in-loop-train` 的 inverse-design pipeline，不只做 forward forecast。

## 2. Core mechanism

NeuralMPM 一個 timestep 的資料流：

```
particle state at t:  positions x_i ∈ R^3, velocities v_i, type id
                              │
                              │  P2G interpolation (MPM kernel, e.g. cubic B-spline)
                              ▼
                  fixed-size voxel grid G_t  (C × H × W × D)
                              │
                              │  image-to-image net  f_θ : G_t  →  ΔG_t
                              │  (U-Net / ConvNeXt-style, K residual blocks)
                              ▼
                       grid update ΔG_t
                              │
                              │  G2P interpolation (same kernel, transposed)
                              ▼
particle state at t+1:  x_i ← x_i + Δt·v_i,   v_i ← v_i + Δt·a_i(ΔG_t)
```

對比 **GNS** (2002.09405) 同樣的 timestep：

```
particles → kNN graph (radius r)
          → 10 layers of edge/node message passing  ← 這裡爆 memory / 慢
          → per-particle Δv prediction
          → semi-implicit Euler integration
```

關鍵設計差異：

- **Spatial mixing 機制**：GNS 用 GNN message passing 在不規則粒子鄰域聚合；NeuralMPM 把粒子 splat 到規則 grid，用 CNN 在 grid 上做 mixing → 同樣的 receptive field，但 CNN 的 throughput 遠勝動態 graph。
- **Memory**：GNS 的 graph edges 隨粒子數 N 線性增長且每步重建（kNN 在 GPU 上算力很重）；NeuralMPM 的 grid resolution 固定，memory 與 N **解耦**。論文聲稱 10×–100× memory reduction 主要來自這條。
- **Inductive bias**：兩者皆 `architecture-bias-soft`——CNN 的 translation equivariance + MPM 的 P2G/G2P linear operators 是 soft physics flavor，**不保證**動量 / 質量 / 能量嚴格守恆（與 `hard-constraint` 如 Hamiltonian/Lagrangian NN 區分）。
- **Loss**：one-step MSE on particle accelerations + multi-step rollout fine-tune（與 GNS 相同框架）。**沒有顯式 PDE residual loss**（所以不算 PINN-style `aux-loss` for PDE），但 P2G/G2P 的數值離散一致性可視為弱 aux constraint → header 標 `injection=architecture-bias-soft|aux-loss`。
- **Time integration**：autoregressive 6h-step-like rollout（粒子 sim 一般 Δt 小於毫秒級，但 paradigm 上仍是 autoregressive）。

Inverse-GNS (2401.13695) 的補充機制：把已訓好的 GNS forward model 視為 differentiable simulator，**reverse-mode AD 反傳到輸入參數**（material friction angle / boundary mesh / baffle position），與 finite-difference 對比降 orders of magnitude convergence cost。這把 `injection=architecture-bias-soft` 的 surrogate 接到 `sim-in-loop-train` 的 design loop——同一個權重既當 forward 又當 differentiable optimizer 底層。

## 3. 五軸定位 + 同軸對手

| Axis | NeuralMPM | GNS (Sanchez-Gonzalez 2020) | Inverse-GNS (Choi 2024) | Genesis MPM | DiffTaichi MPM |
|---|---|---|---|---|---|
| **Output** | `particle` | `particle` | `particle` | `particle` (sim, output=N/A) | `particle` (sim, output=N/A) |
| **Injection** | architecture-bias-soft\|aux-loss | architecture-bias-soft | architecture-bias-soft + sim-in-loop-train (inverse) | sim-in-loop-train | sim-in-loop-train |
| **Control** | param (material id, init state) | param | param (friction, BC, baffle) | param | param |
| **Temporal** | autoregressive | autoregressive | autoregressive + inverse-rollout | streaming | streaming |
| **Domain** | fluid\|granular\|soft | fluid\|rigid\|soft | granular | fluid\|granular\|soft\|rigid | fluid\|granular\|soft |

同軸對手摘要：

- **GNS (2002.09405)**：本路線祖師。用 encoder-processor-decoder GNN，10 層 message passing 學 fluid/sand/goop 粒子動力學；single-step 訓練、test-time 推到 thousands of timesteps + 10× particle count 仍穩定。是 NeuralMPM 直接 benchmark 對手。
- **MeshGraphNet** (Pfaff et al. 2010.03409)：GNS 的 mesh-based 表親——同 DeepMind 路線、同 GNN 後端，但走 `output=3d-explicit` (mesh) 不是 `particle`；領域偏軟體 / cloth。Ontology v2 anchor 中 architecture-bias-soft 範例之一。
- **Genesis MPM module**（[Genesis](../differentiable-simulators/genesis.md), 2412.18608）：Genesis 內建 GPU MPM，是 *traditional* MPM（非神經），靠 Taichi/Warp JIT 拿 GPU 速度。與 NeuralMPM 的關係：Genesis 提供 ground-truth dataset 與 sim-in-loop-train gradient，NeuralMPM 提供 inference-time speed-up。兩者互補不互替。
- **DiffTaichi** (Hu et al. ICLR 2020, arXiv 1910.00935)：differentiable MPM via Taichi 自動 AD，scope 是 sim itself（`output=N/A`），不是 emulator；NeuralMPM 對比 DiffTaichi 的價值在「不需要每次重新跑 sim」，但代價是 sim2real gap 多一層。
- **DMCF**（Continuous Convolutions for fluid sim, NeuralMPM paper 對比基準）：另一條 grid-particle 神經路線，NeuralMPM 聲稱在 long-rollout accuracy 上優於。
- **Inverse-GNS** (2401.13695)：把 GNS 包裝成 differentiable simulator 做 granular landslide 反問題（estimate material params、optimize baffle layout）。是 NeuralMPM/GNS 的 *application* 而非 *competitor*——同軸位但跨進 Axis 2 的 sim-in-loop-train。

## 4. Where it shines / where it breaks

### ⚡ Shines

- **Long-rollout stability on Lagrangian fluids**：GNS 本身的招牌 regime（thousands of steps without explosion），NeuralMPM 繼承且 inference 更快。
- **Cross-material transfer within sim family**：同一網絡架構可訓 water / sand / goop / fluid-solid mix（GNS 原論文 + NeuralMPM 的 fluid-solid interaction 測試）——不需要重設計網絡；資料 swap 即可。
- **Memory efficiency**：grid 解耦於粒子數 → 同 GPU 跑 5×–10× 粒子（NeuralMPM paper 聲稱）。
- **Particle-native scenarios**：splash / free-surface / large deformation 這些 mesh-based surrogate（MeshGraphNet / FNO）天生不擅長的場景。
- **Inverse design**（via Inverse-GNS）：對 granular landslide / debris flow 反問題，differentiable surrogate 比 FD-gradient 快 orders of magnitude，是 `sim-in-loop-train` 的少數成功落地之一。

### ❌ Breaks

- **High particle-count scaling**：CNN grid resolution 一旦要解 10M+ 粒子的精細場景，grid 本身的 memory 就爆——NeuralMPM 解掉 graph 的記憶體問題但被 grid 接走；超大尺度 industrial CFD（10^8 粒子）仍是 traditional MPM/SPH 的天下。
- **Distant interaction / long-range forces**：CNN 與 GNN 都靠**很多層 message passing / convolution** 才能傳到遠處——對全域 acoustic / 電磁長程力（不在訓練 distribution）泛化弱。
- **Conservation drift**：架構不嚴格守恆（`architecture-bias-soft`），長 rollout 會慢慢漂質量 / 動量；可疊 post-hoc projection 修正，但這是 retrofit 不是 first-class。
- **Contact discontinuity**：剛體 contact / 摩擦切換 / 拓撲變化（裂紋、撕裂、合併）→ 粒子 emulator 系（GNS / NeuralMPM）共同弱點，與 overview.md §共通 pitfall 對齊。
- **Sim2real gap**：訓練資料來自 Taichi / MPM ground truth，本身就是 sim，real-world granular（沙、土、debris）的粒徑分布 / 含水率 / cohesion 與訓練分布不一致 → Inverse-GNS 的反問題在 sim setting 漂亮，搬到 real landslide 仍有大的 calibration gap [TBD: verify Choi & Kumar 是否有 real-world field data validation]。
- **Multi-material interface**：fluid-solid coupling NeuralMPM 有 demo，但 fluid-fluid（油水）或 multi-phase 含相變的 case，作者未強 claim。
- **No GitHub-issue-validated multi-GPU**：GNS 原始 codebase `deepmind-research/learning_to_simulate` 對多 GPU 訓練支援薄（見 §8）。

## 5. Reproduction notes

- **Repos**：
  - GNS：`github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate`（TensorFlow 1.x；社區有 PyTorch fork e.g. `kks32/learning_to_simulate`）。
  - NeuralMPM：`github.com/OmerRochman/NeuralMPM`（PyTorch；project page `neuralmpm.isach.be`）。
  - Inverse-GNS：[TBD: verify open code release for 2401.13695；論文中未明列 repo URL]。
- **Datasets**：DeepMind GNS 釋出 WaterRamps / WaterDrop / Sand / Goop / SandRamps / MultiMaterial 等 TFRecord（Google Cloud Storage `gs://learning_to_simulate`）；部分 dataset（BoxBath）缺失（見 §8）。
- **GPU budget**：
  - GNS 原文聲明 4× V100 訓練 1 週量級；社區復現多用 1× A100 數天。
  - NeuralMPM 聲稱 15 小時 / 單 GPU（具體型號 [TBD: verify exact GPU in v3 paper]）。
- **典型踩坑**：
  - TensorFlow 1.x → 2.x 遷移痛苦；許多用戶卡在 `tf.contrib` 移除。
  - 資料下載 403（issue #312/#388）需要明確 endpoint。
  - 多 GPU 訓練需自己包 `tf.distribute`（issue #182）。
  - Rendering / animation pipeline 的 matplotlib 版本相容（issue #354）。

## 6. Cross-line synthesis

- **vs [Genesis MPM](../differentiable-simulators/genesis.md) / [MJX](../differentiable-simulators/mujoco-mjx.md)（traditional diff-sim）**：Genesis MPM 與 NeuralMPM 是 **互補替代** 關係——前者「真物理 + 慢」、後者「神經近似 + 快」。Production 路徑通常 Genesis 出 ground-truth → NeuralMPM 蒸餾 → 部署 NeuralMPM 做 RL/MPC inner loop。是 surrogate × diff-sim 經典 compose。
- **vs [FNO](../neural-surrogates/fno.md) / [GraphCast](../neural-surrogates/graphcast.md)（field surrogate）**：FNO/GraphCast 在 `output=field` 的 Eulerian 視角；NeuralMPM 在 `output=particle` 的 Lagrangian 視角。**對 free surface / 大形變**，Lagrangian 是 first principle 上更合適；**對全域場 forecasting**（氣象、湍流統計）Eulerian 更省。兩條路線在 fluid 域分工清楚。
- **vs pixel-WM (Sora / Cosmos-Predict)**：pixel-WM 用 `data-only` 隱式學 fluid 視覺，無顯式粒子；NeuralMPM 拿粒子結構但缺視覺。Compose 路徑：NeuralMPM 算 ground-truth 粒子軌跡 → renderer 生成像素 → 餵 pixel-WM 做 long-tail visual fine-tune（NewtonGen / Force Prompting 方向之一，但對 fluid 還沒有強 anchor [TBD: verify 是否有 paper 明確做 NeuralMPM → video gen 的 pipeline]）。
- **vs VLA**：granular / soft 物質的操作（scooping sand、folding cloth、pouring liquid）正是 VLA 的弱點區，NeuralMPM 作為 inner sim 可給 model-based RL / planning 提供差分梯度；Inverse-GNS 的 baffle optimization 概念可挪到 manipulation gadget design（被動 deformable tools）。
- **與 `sim-in-loop-infer`（PhysDiff / Cosmos-Reason）的對比**：NeuralMPM 本身是 `sim-in-loop-train`（forward train）或 forward inference；不像 PhysDiff 把 sim 塞進 denoising loop。要把 NeuralMPM 升級成 PhysDiff-style infer-time guidance，需要 differentiable rendering + diffusion 結合 [TBD: verify 是否有現成 anchor]。

## 7. References

**Canonical**：

1. Rochman Sharabi, O., Lewin, S., Louppe, G. "A Neural Material Point Method for Particle-based Emulation." arXiv:2408.15753 (v1 Aug 2024, v3 Feb 2025). Project page: `neuralmpm.isach.be`. Code: `github.com/OmerRochman/NeuralMPM`.
2. Sanchez-Gonzalez, A., Godwin, J., Pfaff, T., Ying, R., Leskovec, J., Battaglia, P. W. "Learning to Simulate Complex Physics with Graph Networks." *ICML 2020*. arXiv:2002.09405.
3. Choi, Y., Kumar, K. "Inverse analysis of granular flows using differentiable graph neural network simulator." arXiv:2401.13695 (Jan 2024). [University of Texas at Austin, Maseeh Dept. of Civil/Architectural/Environmental Engineering.]

**Secondary / related**：

4. Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., Battaglia, P. W. "Learning Mesh-Based Simulation with Graph Networks." arXiv:2010.03409. (MeshGraphNet — sister mesh-based GNN simulator, Ontology v2 `architecture-bias-soft` anchor.)
5. Hu, Y., Anderson, L., Li, T.-M., Sun, Q., Carr, N., Ragan-Kelley, J., Durand, F. "DiffTaichi: Differentiable Programming for Physical Simulation." ICLR 2020. arXiv:1910.00935.
6. Ummenhofer, B., Prantl, L., Thuerey, N., Koltun, V. "Lagrangian Fluid Simulation with Continuous Convolutions." ICLR 2020. (DMCF baseline that NeuralMPM benchmarks against.) [TBD: verify exact arXiv ID]
7. Choi, Y., Kumar, K. "Differentiable graph neural network simulator for forward and inverse modeling of multi-layered slope system with multiple material properties." arXiv:2504.15938 (2025). (Inverse-GNS extension to multi-layer slopes.)

## 8. §8 Pitfall log

| § | Source | Issue / Observation | Severity | Workaround |
|---|---|---|---|---|
| §8.1 | `google-deepmind/deepmind-research` issue **#182** (Mar 2021) | GNS reference codebase trains on single GPU only; `_train_distribute=None` 沒設 `tf.distribute` strategy，10× TITAN V 也只跑 GPU0 | High | 自包 `MirroredStrategy` 或直接用社區 PyTorch fork（e.g. `kks32/learning_to_simulate`） |
| §8.2 | `google-deepmind/deepmind-research` issue **#305** (Nov 2021) | **BoxBath dataset** (cube-water interaction) 從未上傳 GCS bucket；想復現 GNS 論文 fluid-solid demo 必卡 | Med | 用 NeuralMPM 提供的 fluid-solid 數據；或自跑 Taichi MPM 生資料 |
| §8.3 | `google-deepmind/deepmind-research` issue **#312** (Dec 2021) | GCS 下載 dataset 出 403 Forbidden，public bucket ACL drift | Low | 用 `gsutil` + 重抓 endpoint；或用 mirror（部分高校 mirror 可用） |
| §8.4 | `google-deepmind/deepmind-research` issue **#354** (Jun 2022) | Rollout 完想 render animation，matplotlib `UserWarning: Animation was deleted without rendering anything`；分析 pipeline 啞掉 | Low | 把 `FuncAnimation` 賦給變量保活；或改用 `imageio` 寫 GIF |
| §8.5 | `google-deepmind/deepmind-research` issue **#388** (Oct 2022) | 後續 dataset 下載連結再次失效；@alvarosg 被指派但 long-running | Med | 等官方修；或自重建 dataset（Taichi MPM ground truth） |
| §8.6 | `google-deepmind/deepmind-research` issue **#528** (May 2024) | `metadata.json` 中 velocity / acceleration statistics 計算方式未文檔化，自製 dataset 時 norm 對不上 → 訓練爆炸 | High | 用論文 supplementary 推導；或從現成 dataset 反算 `mean/std` 拷貝過去 |
| §8.7 | NeuralMPM paper §limitations + 一般 particle-emulator 共病 | **Conservation drift**：架構 soft bias，長 rollout 質量 / 動量緩慢漂移；非物理 smoothing 漸顯 | High | Post-hoc projection（mass/momentum re-normalize）；或縮短 rollout window |
| §8.8 | Choi & Kumar 2401.13695 §experiments | Inverse-GNS 反問題目前**只在 sim 環境驗證**，real landslide field data calibration 缺；sim2real 仍 open [TBD: verify field validation status] | High | 把 inverse 結果當 prior，real-world 上再加 ensemble Kalman update |
| §8.9 | overview.md §共通 pitfall | Cross-material generalization 在 NeuralMPM/GNS 系仍是「資料 swap 才行」，**單一 checkpoint 同時跑沙+水+布**未強 claim | Med | 訓 multi-material 混合 dataset；或多 checkpoint dispatch |

### Cross-axis descriptive notes（per Ontology v2 Check 9b/9c）

- **Output × Injection (9b)**：`particle × architecture-bias-soft` 是 ✓ 合法 cell；`particle × aux-loss` 同 ✓——header 雙標反映 NeuralMPM 的 P2G/G2P 數值一致性可視為弱 aux 結構約束，並非顯式 PDE residual loss。Check 9b 不要求額外解釋。
- **Domain (9c)**：`fluid|granular|soft` 三選一具體 domain，不觸發 generalist 白名單（Cosmos/Sora/Veo only）。
- **Injection × Temporal**：本條目是 `autoregressive` 而非 iterative denoising，所以 `sim-in-loop-infer` 不適用；Inverse-GNS 的 reverse-mode AD 屬於 train-time / design-time 反傳，不是 infer-time guidance。

---

**TBD checklist for next pass**：

- Confirm NeuralMPM v3 paper 中 exact GPU 型號與 15-hour 訓練聲明的 dataset scale。
- Verify Inverse-GNS (2401.13695) 是否有 open code repo（目前 abstract / project page 未明列）。
- Verify Choi & Kumar 是否有 real-world landslide field data validation（§8.8）。
- Confirm DMCF (Ummenhofer et al.) 的 arXiv ID。
- 追加 anchor：NeuralMPM → pixel-video gen 的 compose pipeline（目前 cross-line §6 為 hypothetical）。
- 補充 ContactGen / ContactNets dissection 後，回填本檔對 rigid contact 的 cross-link。
