# Physics Conditioning

> **本倉真正的 USP zone**。物理規律「怎麼進」生成模型 —— 對應 [Ontology v2.0](../../cheat-sheet/ontology.md) **Axis 2 (injection)** 的全部 7 個 enumerated value（v1 → v2 升級後從 6 變 7：拆 `sim-in-loop` 為 train/infer + 加 `architecture-bias-soft` + 合 score/constraint 為 `guidance-gradient` + demote `energy-based`）。
>
> 其他 zone 按 **output space** 切（pixel-video / latent-tokens / 3d-explicit / field / particle / motion / mesh）—— 「**輸出長什麼樣**」；
> 本 zone 按 **injection 機制** 橫切 —— 「**規律怎麼進去**」。同一篇方法（如 Cosmos）可能同時被 video-world-models 和本 zone 收錄，但視角完全不同。

## 為什麼這個 zone 不能被其他 zone 取代

學術界的 physics-aware ML 資源被三股力量分裂：
1. **科學計算社群**（SciML / PINN line）— 寫 PDE 殘差 loss，視角是「解方程式」，不在乎 perception 或視覺
2. **CV / 生成社群**（diffusion / video gen line）— 賭 scale，物理 implicit，不寫 PDE
3. **機器人社群**（diff-sim / world-model line）— 寫 contact dynamics，但跟 video / latent 鴻溝大

**本 zone 是把這三股流並排對比的唯一位置** —— 同一個物理規律（如能量守恆 / 動量守恆 / 接觸無穿透），三派各自有不同的「把它編入模型」的工程方案。

---

## 7 種 v2 injection 機制（每篇 dissection 落在某一格）

| Injection (v2) | 訊號強度 | 表達力 | 訓練/推理成本 | Generalization | Anchor dissection |
|---|---|---|---|---|---|
| `data-only` | 最弱 | **最強** | 訓練極貴、推理便宜 | 強（domain 內）；極差（OOD） | [Sora](../video-world-models/sora.md) · [Veo](../video-world-models/veo.md) · [Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md) |
| `aux-loss` | 中 | 強 | 訓練中等、推理便宜 | 中（取決於 loss 權重調參） | **[PINN](./pinn.md)** ★ · [PhysGen aux](./physgen.md) |
| `sim-in-loop-train` | **強** | 強 | 訓練極貴、推理便宜 | 強（sim 覆蓋的）；極差（sim 沒覆蓋的） | [Genesis-train](../differentiable-simulators/genesis.md) · [MJX](../differentiable-simulators/mujoco-mjx.md) |
| `sim-in-loop-infer` | **強** | 強 | 推理時 sim 介入 denoise loop / rollout 校正（PhysDiff 把 sim 塞進每步 diffusion） | 強（sim 覆蓋）；OOD 差 | **[PhysDiff](./physdiff.md)** ★ · Cosmos-Reason rollout eval |
| `guidance-gradient` | 中強 | 強 | 訓練（PINN-式 aux gradient）或推理（CFG）；scale 調參敏感 | 中 | [PhysDiff](./physdiff.md) · [Force Prompting](./force-prompting.md) |
| `architecture-bias-soft` | 中 | 中強 | 訓練輕；架構帶 physics-flavor inductive bias 但**不保證**守恆 | 中 | [Force Prompting + NewtonGen](./force-prompting.md) · [GraphCast](../neural-surrogates/graphcast.md) · [FNO](../neural-surrogates/fno.md) · [MeshGraphNet] |
| `hard-constraint` | **最強** | **最弱** | 架構天然滿足某類 PDE / 守恆律 / 等價性 | 弱（架構假設外即崩） | **[Hamiltonian / Lagrangian NN](./hamiltonian-lagrangian-nn.md)** ★ · E(3)-equivariant |

> **v1 → v2 migration**：`implicit-from-data → data-only` · `constraint-loss → aux-loss` · `sim-in-loop` 拆 train/infer · `score-conditioned → guidance-gradient` · `hard-PDE → hard-constraint`（注意 GraphCast/FNO 因 inductive bias 偏 soft，v2 重歸 `architecture-bias-soft`）· `energy-based` demote（EBM 是 loss family，非物理 mechanism）。
>
> 設計權衡是漸進式的：**訊號強度 ↑ → 表達力 ↓ → OOD 風險 ↑**。沒有 free lunch，但 anchor dissection 揭示每軸的 Pareto 點。

---

## 5 代物理-conditioning 演化（mini-timeline）

```
2017 ─┬─ PINN (Raissi) ─────────────── aux-loss 鼻祖
      │
2019 ─┼─ Hamiltonian NN (Greydanus) ── hard-constraint 鼻祖
      ├─ Symplectic ODE-Net ─────────  ↑
      │
2020 ─┼─ Lagrangian NN (Cranmer) ──── hard-constraint 擴展
      ├─ MeshGraphNet (DeepMind) ────  architecture-bias-soft
      │
2022 ─┼─ PhysDiff (Yuan ICCV 2023) ── guidance-gradient + sim-in-loop-infer
      │
2024 ─┼─ PhysGen (Liu ECCV 2024) ──── sim-in-loop-train + aux-loss hybrid（rigid 2D）
      ├─ PhysDreamer (ECCV 2024) ───  → 3D physical-property estimation
      │
2025 ─┼─ Force Prompting (NeurIPS) ── guidance-gradient (force token)
      ├─ NewtonGen (ICLR'26) ──────── architecture-bias-soft (Newton state token)
      ├─ PhysGen3D (CVPR 2025) ─────  PhysGen 3D 擴展
      └─ Cosmos Reason1 ───────────── sim-in-loop-infer + text，大規模
```

**關鍵觀察**：2024-2025 是 **`aux-loss` + `sim-in-loop`** 混合範式爆發；hard-constraint 線（HNN/LNN）2020 後幾乎停滯 —— 表達力天花板太低，難進 video / 大模型。`architecture-bias-soft`（NewtonGen / MeshGraphNet）是 v2 新增類別 —— 介於 hard-constraint 與 aux-loss 之間的中間派。

---

## Anti-pattern atlas（什麼不該做）

| Anti-pattern | 為什麼錯 | 正確做法 |
|---|---|---|
| 把 `aux-loss` 權重直接設 1.0 | reconstruction loss 會被壓垮 → 物理對但視覺崩 | NTK 分析做 dynamic weighting（Wang et al 2007.14527）/ self-adaptive PINN |
| 用 `sim-in-loop-train` 但 sim 跟 prod distribution 不對齊 | sim2real gap 變主要 noise 來源 | 先做 sim2real bridge（DR / system ID）；不要直接用 sim 出來的 trajectory 當 GT |
| 把 `hard-constraint` 強加到 video generation | 表達力崩 → 生成不出複雜場景；**v2 Check 9b 強制 §8 解釋**（pixel-video × hard-constraint 矩陣為 ✗） | hard-constraint 適合低維 dynamical systems（pendulum / 3-body），不適合 RGB video |
| `guidance-gradient` scale 拉太高 | 採樣崩 → 物理「對」但圖像不合理 | 用 dynamic guidance scheduling（早 step 強 guidance、後 step 弱） |
| 只用 `data-only` 賭 scale | 守恆律違反在 OOD 立即暴露 | 加 `guidance-gradient` 後處理（PhysDiff-style）或 `aux-loss` 微調 |
| 把 `architecture-bias-soft` 當 `hard-constraint` 賣 | NewtonGen / MeshGraphNet 不保證守恆 —— 跨 OOD scenario 違反公示 | v2 已強制歸 `architecture-bias-soft` 不可冒充 hard |
| 多 injection 機制疊加但沒對齊權重 | 互相壓制 → 三輸 | 先用單一機制當 baseline，再增量加 |

---

## 與其他 zone 的 cross-synthesis

| Zone | physics-conditioning 怎麼在那邊用 |
|---|---|
| [`video-world-models/`](../video-world-models/) | Sora/Veo = 純 `data-only`；Cosmos-Predict = `data-only` + `sim-in-loop-infer` 評估；GAIA-2 = `data-only` + `trajectory` cond |
| [`latent-world-models/`](../latent-world-models/) | V-JEPA-2 = `data-only` + masking；DreamerV4 = `data-only` + RL loss 隱式 |
| [`diffusion-physics/`](../diffusion-physics/) | 整個 zone 是 `guidance-gradient` 的子集（diffusion-only 子線） |
| [`differentiable-simulators/`](../differentiable-simulators/) | 整個 zone 是 `sim-in-loop-train` 的 producer side（提供 oracle） |
| [`neural-surrogates/`](../neural-surrogates/) | GraphCast / FNO = `architecture-bias-soft` + `aux-loss` hybrid（v2 重歸後不再算 hard-constraint） |
| [`controllability-mechanisms/`](../controllability-mechanisms/) | force / contact / param / camera / layout 等控制輸入是 injection 機制的「**控制信號接口**」 |
| [`evaluation-physics/`](../evaluation-physics/) | 沒 evaluation 就無法分辨 `data-only` 與 `aux-loss` 在 OOD 的差異 |

---

## 與 [`diffusion-physics/`](../diffusion-physics/) zone 的分工

- `physics-conditioning/` 收 **泛 injection 對比** + PINN + HNN/LNN + 跨範式比較
- `diffusion-physics/` 專收 **score function 加物理梯度 / classifier guidance** 的 diffusion-only 子線

PhysDiff 同時被兩個 zone 收錄 —— 本 zone 視角是「`guidance-gradient` + `sim-in-loop-infer` 的範式」，diffusion-physics zone 視角是「具體 score function 修改方式」。

---

## Anchor dissections（5 篇已落地）

| 篇 | Injection 主軸 (v2) | 代表性 |
|---|---|---|
| **[PINN](./pinn.md)** ★ | `aux-loss` | 鼻祖；science domain；2017→今天仍是 baseline |
| **[Hamiltonian / Lagrangian NN](./hamiltonian-lagrangian-nn.md)** ★ | `hard-constraint` | 架構 baking；低維 dyn 系統 |
| **[PhysGen](./physgen.md)** ★ | `sim-in-loop-train` + `aux-loss` hybrid | 2024 image-to-video 物理範式 |
| **[PhysDiff](./physdiff.md)** ★ | `guidance-gradient` + `sim-in-loop-infer` | 2023 motion diffusion 物理校正；output=`motion` (v2 新值) |
| **[Force Prompting + NewtonGen](./force-prompting.md)** ★ | `guidance-gradient` + `architecture-bias-soft` | 2025 explicit force conditioning |

下一波（Phase 2 wishlist）：
- [ ] E(3) / SE(3)-equivariant 網路的 `hard-constraint` 變體
- [ ] PhysHOI / PhysReact（object interaction physics in motion gen）
- [ ] PINN-NeRF / PI-Loss for video diffusion 的實測
- [ ] MeshGraphNet 獨立 dissection（`architecture-bias-soft` 經典案例）

---

## §8 共通 pitfall（zone-level）

1. **Loss 權重 Pareto** —— `aux-loss` 與主 reconstruction loss 互相壓制；NTK 分析給數學工具但沒給萬靈丹
2. **OOD boundary 崩潰** —— `hard-constraint` 架構在訓練見過 boundary 外失效（contact 拓撲變化、新物體 mass）
3. **Sim2real gap 反噬** —— `sim-in-loop-train` 訓出來的 WM 在真實場景物理感反而下降（sim 跟 prod distribution 不一致）
4. **Guidance scale tuning** —— `guidance-gradient` 全部 sensitive；過大→ sampling 崩，過小→ 物理失效
5. **多 injection 不對齊** —— 同模型用 2+ injection 機制時，loss / guidance scale 互相干擾
6. **物理 vocabulary 不足** —— force / contact / param 的 conditioning 詞彙在訓練資料中稀缺（沒人標 force 給 Sora）
7. **架構 vs loss 混淆** —— `architecture-bias-soft`（NewtonGen / MeshGraphNet）易被誤認為 `hard-constraint`；v2 已強制區分

---

## 為什麼 2025-2026 是這個 zone 的關鍵期

- Sora / Veo / Cosmos 等大模型在 `data-only` 端已飽和 —— **更大 scale 不再線性提升物理感**（PhyWorld 等 benchmark 證實）
- 真實工業需求（robotics-data-gen / driving-sim）必須有 force / contact 級控制 —— 純 text 不夠
- 學術 frontier 已從「物理 implicit 學」轉向「**物理顯式 inject**」—— 5 篇 anchor 中 3 篇是 2024-2025 工作
- 我們的賭注是：未來 2 年最 productive 的 stack 是 `data-only` (大 backbone) + `guidance-gradient` (推理時校正) + `sim-in-loop-infer` (eval/data engine)，而非純 `data-only` 或純 `hard-constraint`
