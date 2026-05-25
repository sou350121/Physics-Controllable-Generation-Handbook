# Ontology v1 — 5-axis taxonomy of Physics-Controllable Generation

**版本**：v1.0 · 2026-05-25 · 初版（待 web-search expert review 升 v1.1）

每篇 dissection 頂部需附：

```html
<!-- ontology-5axis
     output=<value> injection=<value> control=<value>
     temporal=<value> domain=<value> -->
```

`scripts/handbook_audit.py` Check 9 強制 enforce。

> **`N/A` 是合法值**：對於非生成模型的條目（sim like Genesis / MJX / Aerial Gym → output=N/A）以及純評測 benchmark（VBench-Physics → output=N/A injection=N/A）允許 N/A。N/A 表示「此軸不適用於該條目」，不代表「未標」。

---

## Axis 1 — Output space

模型直接生成的「東西」存在哪個空間。

| Value | 描述 | 代表方法 |
|---|---|---|
| `pixel-video` | RGB 像素影片 | Sora, Veo, Cosmos-Predict |
| `latent` | 自編碼後的 latent token / vector（可能含時間維） | V-JEPA, DreamerV4 latent, MuZero-style |
| `3d-scene` | 顯式 3D 表徵（3DGS / mesh / occ / SDF） | World Labs gen-3D, GaussianAnything |
| `mesh` | 三角網格動畫 | MeshDiffusion, neural-mesh-sim |
| `particle` | 粒子 / MPM / SPH | NeuralMPM, ContactGen-particle |
| `field` | 連續場（速度場 / 應力場 / 溫度場） | FNO, GraphCast, MeshGraphNet |
| `action-seq` | 動作序列（state-action token） | Genie-2 action, Decart latent-act |

> 多輸出（如 `pixel-video + action-seq`）允許在 header 寫 `output=pixel-video|action-seq`。

---

## Axis 2 — Physics injection

物理規律怎麼進入模型。**這軸是本 handbook 的核心 USP** — 大多數論文不會明說，需 reverse-engineer。

| Value | 描述 | 強度 | 代表方法 |
|---|---|---|---|
| `implicit-from-data` | 物理規律靠大量真實/模擬影片隱式學會 | 最弱 | Sora, Veo（公開資訊） |
| `constraint-loss` | 訓練時加 PDE / 守恆 / contact 等輔助 loss | 中 | PINN-style, PhysGen |
| `sim-in-loop` | 訓練/推理時掛真實或可微 simulator 提供 ground truth 或 reward | 強 | Genesis-train, Cosmos-Reason rollout eval |
| `energy-based` | 模型輸出能量函數，物理規律以 energy minimum 表達 | 中強 | EBM-physics |
| `score-conditioned` | Diffusion 的 score function 中加入物理梯度（classifier-free 或 guidance） | 中強 | PhysDiff, classifier-guided diffusion |
| `hard-PDE` | 模型結構天然滿足某類 PDE 或守恆律（symmetry-preserving net） | 最強 | E(3)-equivariant, Hamiltonian NN |

設計權衡：強度越高 → fidelity 高、generalization 窄；implicit-from-data 反之。詳見 `crossing/controllability-vs-fidelity/`。

---

## Axis 3 — Controllability input

模型接受的 conditioning 輸入。**多選**（多模態 controllability 是 2026 的主戰場）。

| Value | 描述 | 代表 |
|---|---|---|
| `text` | 文字 prompt | Sora, Veo, Cosmos-Predict |
| `action` | 離散/連續動作 token（agent 視角） | Genie-2, DreamerV4, V-JEPA-2-action |
| `trajectory` | 路徑（2D/3D points over time） | Wayve GAIA-traj, Cosmos-Drive |
| `force` | 力 / 力矩 / 接觸力分布 | ForceGen, ContactGen-force |
| `contact` | 接觸圖 / 接觸時序 | ContactNets, CC-Diff |
| `image-prompt` | 起始幀 / reference image | Stable Video Diffusion, Cosmos-img2vid |
| `3d-prompt` | 起始 3D scene / mesh / 3DGS | World Labs gen-3D conditioning |
| `physical-param` | 顯式物理參數（mass, stiffness, viscosity） | NeuralPhysics, ParamControlNet |
| `multi` | 多種同時（如 text+action+force） | 進階組合 |

> 在 header 用 `|` 分隔，例如 `control=text|action|trajectory`。

---

## Axis 4 — Temporal paradigm

時間維度怎麼處理 — 直接決定 long-horizon 能不能撐住。

| Value | 描述 | 典型問題 |
|---|---|---|
| `single-frame` | 一張圖（無時間） | 不適用 video physics |
| `streaming` | Sim / 連續時間 ODE 解算，無 frame 概念 | Sim2real gap、contact discontinuity |
| `autoregressive` | 一幀一幀往前生成，下一幀 condition 上一幀 | Drift 累積、exposure bias |
| `joint-rollout` | 一次生成整段 clip（如 16/24/48 幀） | 長度受限，跨 clip 銜接難 |
| `latent-rollout` | latent space 中 rollout，最後再 decode 像素 | DreamerV4 主路線；省 compute |
| `hierarchical` | 兩層：高層慢時間規劃、低層快時間細節 | Cosmos-Reason+Predict、TECO |
| `temporal-transformer-rolling` | sliding window transformer + KV cache | Genie-2, Decart 即時 WM |

---

## Axis 5 — Domain coupling

物理「在哪個世界」 — 影響 inductive bias 與 evaluation。

| Value | 描述 |
|---|---|
| `generalist` | 通用世界（沒明顯場域限定） |
| `robotics` | 機器人操作 / 雙臂 / 移動 |
| `driving` | 自駕 / 道路場景 |
| `fluid` | 流體（CFD, weather, ocean） |
| `rigid` | 剛體（contact-rich, mechanism） |
| `soft` | 軟體 / 布料 / 變形 |
| `granular` | 顆粒 / 粉體 / 沙 |
| `bio` | 蛋白質 / 分子動力學 |
| `weather` | 氣象 / 氣候 |

---

## 為什麼是 5 軸不是 4 或 6

- 4 軸（如砍掉 domain）會把 GraphCast 跟 Cosmos 等視為「同類」 — 失去誤差/驗證視角
- 6 軸（如加 training-paradigm: self-supervised / RLHF / curated）會跟 axis 2 重疊
- 5 軸是 sister Spatial-Handbook 同套方法論延伸而來，三冊一致便於 cross-ref

## 與 Spatial Ontology v3.2 的對應

| Physics 軸 | Spatial 對應 |
|---|---|
| Axis 1 Output | 對應 Spatial Axis 2 Representation（生成端視角 vs 感知端視角） |
| Axis 2 Injection | **新增** — Spatial 沒有，是本倉 USP |
| Axis 3 Control | 對應 Spatial Axis 3 Sensor + Axis 4 Paradigm 的子集 |
| Axis 4 Temporal | 對應 Spatial Axis 5 Time |
| Axis 5 Domain | 對應 Spatial Axis 1 Problem 但範圍不同 |

## TODO v1.1

- web-search expert review（4 個 expert × 各補 canonical refs）
- 收 controversies（如 score-conditioned 是不是 constraint-loss 子集？）
- 補 anchor refs（每軸每值至少 1 canonical paper）
