# Functional Map — 「我有 X 需求 → 看哪條路線」

> Ontology v2.0 對應版（2026-05-26）。19 anchor dissection 已落地後填實。

## 一句話路線分流

| 需求 | 首選技術路線 | foundations 入口 |
|---|---|---|
| 給 VLA pre-train 補大量合成 demo | latent world-model + action conditioning | `latent-world-models/` + `controllability-mechanisms/` |
| 自駕 closed-loop simulation | video world-model + trajectory + camera + long-horizon | `video-world-models/` + `long-horizon-rollout/` |
| 物理科學模擬替代 PDE solver | neural surrogate + `architecture-bias-soft` / `aux-loss` | `neural-surrogates/` + `material-and-dynamics/` |
| 機器人 sim2real 資料工廠 | differentiable simulator + data engine | `differentiable-simulators/` + `data-engine/` |
| 影片內容創作（電影、廣告） | pixel video diffusion + 弱物理 | `video-world-models/` + `diffusion-physics/` |
| 互動式遊戲世界 | latent rolling WM + action + `streaming-cache` | `latent-world-models/` + `long-horizon-rollout/` |
| 無人機 closed-loop sim | diff-sim + sim2real bridge | `differentiable-simulators/` + `use-cases/aerial-sim/` |
| 物理可控影片（force / contact / Newton 條件） | `guidance-gradient` + `architecture-bias-soft` | `physics-conditioning/` ★★ |

## 五軸（v2）對應 quick-lookup

格式：軸值 → 章節 → top-3 代表方法

```
output=pixel-video      → video-world-models       → Sora · Veo · Cosmos-Predict · GAIA-2
output=latent-tokens    → latent-world-models      → V-JEPA-2 · DreamerV4
output=3d-explicit      → 3d-aware-generation      → World Labs · GaussianAnything · Generative-GS
output=3d-implicit      → 3d-aware-generation      → (待 anchor — DeepSDF / NeRF-physics)
output=field            → neural-surrogates        → GraphCast · FNO · Pangu-Weather
output=particle         → material-and-dynamics    → NeuralMPM · ContactGen-particle
output=motion           → physics-conditioning     → PhysDiff (v2 新值，SMPL/骨架 pose 序列)
output=action-seq       → latent-world-models      → Genie-2 · Decart latent-act
```

```
injection=data-only             → video-world-models (scale-pilled)         → Sora · Veo · Cosmos-Predict
injection=aux-loss              → physics-conditioning (PINN class)          → PINN · PhysGen aux
injection=sim-in-loop-train     → differentiable-simulators (gradient source) → Genesis · MJX · Aerial Gym
injection=sim-in-loop-infer     → physics-conditioning (PhysDiff line)       → PhysDiff · Cosmos-Reason eval
injection=guidance-gradient     → diffusion-physics + physics-conditioning   → PhysDiff · Force Prompting · CFG
injection=architecture-bias-soft → physics-conditioning + neural-surrogates  → NewtonGen · GraphCast · FNO · MeshGraphNet
injection=hard-constraint       → physics-conditioning (HNN / equivariant)   → HNN · LNN · E(3)-equivariant
```

> **v1→v2 migration（顯示原名 → 新名，僅供讀者對照舊文獻）**：
> Axis 2：~~implicit-from-data~~ → `data-only` · ~~constraint-loss~~ → `aux-loss` · ~~sim-in-loop~~ 拆 `sim-in-loop-train` / `sim-in-loop-infer` · ~~score-conditioned~~ → `guidance-gradient` · ~~hard-PDE~~ → `hard-constraint`（GraphCast/FNO 重歸 `architecture-bias-soft`）· ~~energy-based~~ demote
> Axis 1：~~mesh~~ → `3d-explicit` · ~~3d-scene~~ → `3d-explicit` / `3d-implicit` · ~~latent~~ → `latent-tokens`（non-decode 用途）
> Axis 3：~~image-prompt~~ → `image-init` · ~~3d-prompt~~ → `3d-init` · ~~physical-param~~ → `param` · ~~multi~~ keyword 刪除（用 `|` 分隔）· +`camera` · +`layout` 新增
> Axis 4：~~joint-rollout~~ → `clip-parallel` · ~~temporal-transformer-rolling~~ → `streaming-cache`

## TODO

- [ ] 加入每條路線的 cost 估計（GPU-day / inference latency / dataset size）
- [ ] 加入 maturity rating（research only / pilot / production-ready）
- [ ] 加入 anchor dissection cross-link（直接 inline）
- [ ] 加入 Axis 3 新增 value 路線（`camera` / `layout` / `motion`）
