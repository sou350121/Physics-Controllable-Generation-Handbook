# Bridge: NeRF / 3DGS Meet World Model

> 🚧 **Stub** — waiting on anchor dissections. See [overview.md](overview.md) for design rationale.

## 主題

顯式 3D 表徵的 world model — 把 3DGS / NeRF 當 WM 的 state representation 而不是 pixel / latent。

## 預計內容

- **GS-as-WM**：dynamic 3DGS + transition model 取代 pixel WM
- **NeRF-WM**：scene-conditioned NeRF rollout
- **與 latent WM 的差別**：顯式 3D 結構 vs 抽象 latent
- **與 pixel WM 的差別**：可任意視角 vs camera-locked
- **目前的限制**：訓練資料、長 horizon 穩定性

## 引用 anchor

- [`/foundations/3d-aware-generation/overview.md`](../foundations/3d-aware-generation/overview.md)
- [`/foundations/video-world-models/overview.md`](../foundations/video-world-models/overview.md)
- [`/foundations/latent-world-models/overview.md`](../foundations/latent-world-models/overview.md)
- Spatial-Handbook：`foundations/world-model/`
