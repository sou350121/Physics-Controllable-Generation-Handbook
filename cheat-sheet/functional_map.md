# Functional Map — 「我有 X 需求 → 看哪條路線」

> v0.1 placeholder。等 12 anchor dissections 完成後填實。

## 一句話路線分流

| 需求 | 首選技術路線 | foundations 入口 |
|---|---|---|
| 給 VLA pre-train 補大量合成 demo | latent world-model + action conditioning | `latent-world-models/` + `controllability-mechanisms/` |
| 自駕 closed-loop simulation | video world-model + trajectory conditioning + long-horizon | `video-world-models/` + `long-horizon-rollout/` |
| 物理科學模擬替代 PDE solver | neural surrogate + hard-PDE injection | `neural-surrogates/` + `material-and-dynamics/` |
| 機器人 sim2real 資料工廠 | differentiable simulator + data engine | `differentiable-simulators/` + `data-engine/` |
| 影片內容創作（電影、廣告） | pixel video diffusion + 弱物理 | `video-world-models/` + `diffusion-physics/` |
| 互動式遊戲世界 | latent rolling WM + action conditioning + 即時 | `latent-world-models/` + `long-horizon-rollout/` |

## 五軸對應 quick-lookup（待填）

格式：軸值 → 章節 → top-3 代表方法

```
output=pixel-video        → video-world-models       → Sora · Veo · Cosmos-Predict
output=latent             → latent-world-models      → V-JEPA · DreamerV4 · Genie-2
output=3d-scene           → 3d-aware-generation      → World Labs · GaussianAnything
output=field              → neural-surrogates        → GraphCast · FNO · MeshGraphNet
output=mesh               → material-and-dynamics    → MeshDiffusion · NeuralMPM
output=action-seq         → controllability          → Genie-2-action · Decart latent-act
output=particle           → material-and-dynamics    → NeuralMPM · ContactGen-particle
```

```
injection=implicit          → video-world-models     (scale-pilled)
injection=constraint-loss   → physics-conditioning   (PINN-class)
injection=sim-in-loop       → diff-sim + foundations (生成-評測閉環)
injection=energy-based      → physics-conditioning   (EBM-physics)
injection=score-conditioned → diffusion-physics      (PhysDiff line)
injection=hard-PDE          → neural-surrogates      (equivariant)
```

## TODO

- [ ] 加入每條路線的 cost 估計（GPU-day / inference latency / dataset size）
- [ ] 加入 maturity rating（research only / pilot / production-ready）
- [ ] 加入 anchor dissection cross-link
