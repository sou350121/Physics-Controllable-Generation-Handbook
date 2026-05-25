# Physics Conditioning

> **本倉真正的 USP zone**。物理規律「怎麼進」生成模型 —— 對應 ontology [Axis 2 (injection)](../../cheat-sheet/ontology.md) 的全部 6 種值。
>
> 其他 zone 按 **output space** 切（video / latent / 3D / field / particle / mesh）—— 「**輸出長什麼樣**」；
> 本 zone 按 **injection 機制** 橫切 —— 「**規律怎麼進去**」。同一篇方法（如 Cosmos）可能同時被 video-world-models 和本 zone 收錄，但視角完全不同。

## 為什麼這個 zone 不能被其他 zone 取代

學術界的 physics-aware ML 資源被三股力量分裂：
1. **科學計算社群**（SciML / PINN line）— 寫 PDE 殘差 loss，視角是「解方程式」，不在乎 perception 或視覺
2. **CV / 生成社群**（diffusion / video gen line）— 賭 scale，物理 implicit，不寫 PDE
3. **機器人社群**（diff-sim / world-model line）— 寫 contact dynamics，但跟 video / latent 鴻溝大

**本 zone 是把這三股流並排對比的唯一位置** —— 同一個物理規律（如能量守恆 / 動量守恆 / 接觸無穿透），三派各自有不同的「把它編入模型」的工程方案。

---

## 6 種 injection 機制（v1，每篇 dissection 落在某一格）

| Injection | 訊號強度 | 表達力 | 訓練/推理成本 | Generalization | Anchor dissection |
|---|---|---|---|---|---|
| `implicit-from-data` | 最弱 | **最強** | 訓練極貴、推理便宜 | 強（domain 內）；極差（OOD） | [Sora](../video-world-models/sora.md) · [Veo](../video-world-models/veo.md) · [Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md) |
| `constraint-loss` | 中 | 強 | 訓練中等、推理便宜 | 中（取決於 loss 權重調參） | **[PINN](./pinn.md)** ★ · [PhysGen aux](./physgen.md) |
| `sim-in-loop` | **強** | 強 | 訓練/推理皆貴 | 強（sim 覆蓋的）；極差（sim 沒覆蓋的） | [Cosmos-Reason eval](../foundation-physics-models/cosmos-wfm.md) · [Genesis-train](../differentiable-simulators/genesis.md) · [PhysDiff](./physdiff.md) |
| `energy-based` | 中強 | 中 | 採樣慢；訓練不穩 | 中 | EBM-physics（待 anchor） |
| `score-conditioned` | 中強 | 強 | 推理時加 guidance | 中（guidance scale 敏感） | **[PhysDiff](./physdiff.md)** ★ · [Force Prompting](./force-prompting.md) |
| `hard-PDE` (architecturally-built-in) | **最強** | **最弱** | 訓練輕、推理輕 | 弱（架構假設外即崩） | **[Hamiltonian / Lagrangian NN](./hamiltonian-lagrangian-nn.md)** ★ · E(3)-equivariant |

> 設計權衡是漸進式的：**訊號強度 ↑ → 表達力 ↓ → OOD 風險 ↑**。沒有 free lunch，但 anchor dissection 揭示每軸的 Pareto 點。

---

## 5 代物理-conditioning 演化（mini-timeline）

```
2017 ─┬─ PINN (Raissi) ─────────────── constraint-loss 鼻祖
      │
2019 ─┼─ Hamiltonian NN (Greydanus) ── hard-PDE 鼻祖
      ├─ Symplectic ODE-Net ─────────  ↑
      │
2020 ─┼─ Lagrangian NN (Cranmer) ──── hard-PDE 擴展
      │
2022 ─┼─ PhysDiff (Yuan ICCV 2023) ── score-conditioned 鼻祖（motion）
      │
2024 ─┼─ PhysGen (Liu ECCV 2024) ──── sim-in-loop+constraint hybrid（rigid 2D）
      ├─ PhysDreamer (CVPR 2024) ───  → 3D physical-property estimation
      │
2025 ─┼─ Force Prompting ──────────── score-conditioned (force token)
      ├─ NewtonGen ───────────────── score-conditioned (Newton state token)
      ├─ PhysGen3D (CVPR 2025) ─────  PhysGen 3D 擴展
      └─ Cosmos Reason1 ───────────── sim-in-loop+text，大規模
```

**關鍵觀察**：2024-2025 是 **`constraint-loss` + `sim-in-loop`** 混合範式爆發；hard-PDE 線（HNN/LNN）2020 後幾乎停滯 —— 表達力天花板太低，難進 video / 大模型。

---

## Anti-pattern atlas（什麼不該做）

| Anti-pattern | 為什麼錯 | 正確做法 |
|---|---|---|
| 把 `constraint-loss` 權重直接設 1.0 | reconstruction loss 會被壓垮 → 物理對但視覺崩 | NTK 分析做 dynamic weighting（Wang et al 2007.14527）/ self-adaptive PINN |
| 用 `sim-in-loop` 但 sim 跟 prod distribution 不對齊 | sim2real gap 變主要 noise 來源 | 先做 sim2real bridge（DR / system ID）；不要直接用 sim 出來的 trajectory 當 GT |
| 把 `hard-PDE` 強加到 video generation | 表達力崩 → 生成不出複雜場景 | hard-PDE 適合低維 dynamical systems（pendulum / 3-body），不適合 RGB video |
| `score-conditioned` guidance scale 拉太高 | 採樣崩 → 物理「對」但圖像不合理 | 用 dynamic guidance scheduling（早 step 強 guidance、後 step 弱） |
| 只用 `implicit-from-data` 賭 scale | 守恆律違反在 OOD 立即暴露 | 加 score-conditioned 後處理（PhysDiff-style）或 constraint-loss 微調 |
| 多 injection 機制疊加但沒對齊權重 | 互相壓制 → 三輸 | 先用單一機制當 baseline，再增量加 |

---

## 與其他 zone 的 cross-synthesis

| Zone | physics-conditioning 怎麼在那邊用 |
|---|---|
| [`video-world-models/`](../video-world-models/) | Sora/Veo = 純 `implicit-from-data`；Cosmos-Predict = `implicit` + `sim-in-loop` 評估；GAIA-2 = `implicit` + `trajectory cond` |
| [`latent-world-models/`](../latent-world-models/) | V-JEPA-2 = `implicit` + masking；DreamerV4 = `implicit` + RL loss 隱式 |
| [`diffusion-physics/`](../diffusion-physics/) | 整個 zone 是 `score-conditioned` 的子集（diffusion-only 子線） |
| [`differentiable-simulators/`](../differentiable-simulators/) | 整個 zone 是 `sim-in-loop` 的 producer side（提供 oracle） |
| [`neural-surrogates/`](../neural-surrogates/) | GraphCast / FNO = `hard-PDE` + `constraint-loss` hybrid |
| [`controllability-mechanisms/`](../controllability-mechanisms/) | force / contact / param 等控制輸入是 injection 機制的「**控制信號接口**」 |
| [`evaluation-physics/`](../evaluation-physics/) | 沒 evaluation 就無法分辨 `implicit` 與 `constraint-loss` 在 OOD 的差異 |

---

## 與 [`diffusion-physics/`](../diffusion-physics/) zone 的分工

- `physics-conditioning/` 收 **泛 injection 對比** + PINN + HNN/LNN + 跨範式比較
- `diffusion-physics/` 專收 **score function 加物理梯度 / classifier guidance** 的 diffusion-only 子線

PhysDiff 同時被兩個 zone 收錄 —— 本 zone 視角是「score-conditioned 的範式」，diffusion-physics zone 視角是「具體 score function 修改方式」。

---

## Anchor dissections（5 篇已落地）

| 篇 | Injection 主軸 | 代表性 |
|---|---|---|
| **[PINN](./pinn.md)** ★ | `constraint-loss` | 鼻祖；science domain；2017→今天仍是 baseline |
| **[Hamiltonian / Lagrangian NN](./hamiltonian-lagrangian-nn.md)** ★ | `hard-PDE` | 架構 baking；低維 dyn 系統 |
| **[PhysGen](./physgen.md)** ★ | `sim-in-loop` + `constraint-loss` hybrid | 2024 image-to-video 物理範式 |
| **[PhysDiff](./physdiff.md)** ★ | `score-conditioned` + `sim-in-loop` | 2023 motion diffusion 物理校正 |
| **[Force Prompting + NewtonGen](./force-prompting.md)** ★ | `score-conditioned` + force token | 2025 explicit force conditioning |

下一波（Phase 2 wishlist）：
- [ ] EBM-physics（energy-based）—— 需找 anchor paper
- [ ] E(3)-equivariant / SE(3)-equivariant network 的 `hard-PDE` 變體
- [ ] PhysHOI / PhysReact（object interaction physics in motion gen）
- [ ] PINN-NeRF / PI-Loss for video diffusion 的實測

---

## §8 共通 pitfall（zone-level）

1. **Loss 權重 Pareto** —— `constraint-loss` 與主 reconstruction loss 互相壓制；NTK 分析給數學工具但沒給萬靈丹
2. **OOD boundary 崩潰** —— `hard-PDE` 架構在訓練見過 boundary 外失效（contact 拓撲變化、新物體 mass）
3. **Sim2real gap 反噬** —— `sim-in-loop` 訓出來的 WM 在真實場景物理感反而下降（sim 跟 prod distribution 不一致）
4. **Guidance scale tuning** —— `score-conditioned` 全部 sensitive；過大→ sampling 崩，過小→ 物理失效
5. **多 injection 不對齊** —— 同模型用 2+ injection 機制時，loss / guidance scale 互相干擾
6. **物理 vocabulary 不足** —— force / contact / param 的 conditioning 詞彙在訓練資料中稀缺（沒人標 force 給 Sora）

---

## 為什麼 2025-2026 是這個 zone 的關鍵期

- Sora / Veo / Cosmos 等大模型在 `implicit-from-data` 端已飽和 —— **更大 scale 不再線性提升物理感**（PhyWorld 等 benchmark 證實）
- 真實工業需求（robotics-data-gen / driving-sim）必須有 force / contact 級控制 —— 純 text 不夠
- 學術 frontier 已從「物理 implicit 學」轉向「**物理顯式 inject**」—— 5 篇 anchor 中 3 篇是 2024-2025 工作
- 我們的賭注是：未來 2 年最 productive 的 stack 是 `implicit` (大 backbone) + `score-conditioned` (推理時校正) + `sim-in-loop` (eval/data engine)，而非純 `implicit` 或純 `hard-PDE`
