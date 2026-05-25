# Bridge: World-Model-as-Policy

> 🚧 **Stub** — waiting on anchor dissections. See [overview.md](overview.md) for design rationale.

## 主題

DreamerV4 / V-JEPA-2 直接當 policy 的範式 — 不是「生成資料訓 policy」，而是「WM rollout 即決策」。

## 預計內容

- **Actor-critic on WM**：DreamerV3/V4 路線
- **MPC on WM**：PETS / TD-MPC / sampling-based plan
- **與 generative data 路線的差別**：前者 WM 是訓練資料源，後者 WM 是決策模組本身
- **失效模式**：WM rollout reliability vs decision quality 的 Pareto

## 引用 anchor

- [`/foundations/latent-world-models/dreamer-v4.md`](../foundations/latent-world-models/dreamer-v4.md)
- [`/foundations/latent-world-models/v-jepa-2.md`](../foundations/latent-world-models/v-jepa-2.md)
- [`/use-cases/embodied-policy-rollout/overview.md`](../use-cases/embodied-policy-rollout/overview.md)
