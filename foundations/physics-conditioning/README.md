# Physics Conditioning

> **本倉的 USP zone**。物理規律怎麼「進」生成模型 —— 對應 ontology Axis 2 (injection) 的全部 6 種值。

## 為什麼這個 zone 獨立存在

其他 zone 是按「output space」切的（video / latent / 3D / field / particle / mesh）。
這個 zone 是按「injection 機制」橫切 —— 同一篇方法可能同時被 video-world-models 與本 zone 收錄，但本 zone 的視角是「**規律怎麼進去**」，不是「**輸出長怎樣**」。

## 6 種 injection 對比（v1）

| Injection | Pros | Cons | 代表 |
|---|---|---|---|
| `implicit-from-data` | Scale 紅利；不需 PDE 知識 | 細節易違反守恆律；難 generalize 到 OOD scenario | Sora |
| `constraint-loss` | 簡單可加，不改架構 | Loss 權重難調；不保證 hard constraint | PINN, PhysGen aux loss |
| `sim-in-loop` | Ground truth 級訊號 | 訓練/推理皆貴；sim 自己有 gap | Genesis-train, Cosmos-Reason eval |
| `energy-based` | 自然表達 stability | 採樣難；訓練不穩 | EBM-physics |
| `score-conditioned` | 跟 diffusion 天然契合；推理時可加 guidance | guidance scale 調參敏感；可能 distribution shift | PhysDiff |
| `hard-PDE` (equivariant) | 架構保證符合守恆律 | 表達力受限；難應對複雜 boundary | E(3)-equivariant, HamiltonianNN |

## 與 diffusion-physics zone 的分工

- `physics-conditioning/` 收 **泛 injection 對比**、PINN、EBM、equivariant 等非 diffusion-only 方法
- `diffusion-physics/` 專收 score function 加物理梯度 / classifier guidance 線

## Dissection wishlist (4 篇)

- [ ] PINN 在 video / scene 的擴展嘗試
- [ ] Hamiltonian NN / Lagrangian NN
- [ ] PhysGen aux loss 的有效性實測
- [ ] Sim-in-loop training (Genesis-pulled rollout) 的 cost 與收益

## §8 共通 pitfall

- Constraint-loss 與主 reconstruction loss 互相壓制 —— Pareto 邊界要明確
- Hard-PDE 架構在 OOD boundary 失效（如新 contact 拓撲）
- Sim-in-loop 與真實 distribution gap 反而成為主要 noise
