# Bridge: Spatial Controllability

> 🚧 **Stub** — waiting on anchor dissections. See [overview.md](overview.md) for design rationale.

## 主題

Spatial structure 作為 conditioning — occupancy / depth / pose / BEV / 6-DoF trajectory 怎麼進生成模型。

## 預計內容

- **9 種 spatial input 的接法**（與 [`/foundations/controllability-mechanisms/`](../foundations/controllability-mechanisms/) 互鏈）：
  - depth-cond / occupancy-cond / pose-cond / BEV-cond / trajectory-3D
- **代表系統**：
  - GAIA-2 trajectory conditioning
  - Cosmos-Drive multi-camera + 3D structure
  - ControlNet-depth in video
- **失效模式**：spatial signal 與 text prompt 衝突時誰勝、cross-camera 不一致

## 引用 anchor

- [`/foundations/controllability-mechanisms/overview.md`](../foundations/controllability-mechanisms/overview.md)
- [`/foundations/video-world-models/gaia-2.md`](../foundations/video-world-models/gaia-2.md)
- Spatial-Handbook：`foundations/feed-forward-3d/`、`embodiments/driving/`
