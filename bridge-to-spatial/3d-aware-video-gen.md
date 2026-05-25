# Bridge: 3D-Aware Video Generation

> 🚧 **Stub** — waiting on anchor dissections. See [overview.md](overview.md) for design rationale.

## 主題

3DGS / NeRF 重建表徵如何條件化影片生成 — generation 端怎麼吃 perception 端的 3D 結構。

## 預計內容

- **3D-as-conditioning**：把 3DGS / NeRF feature 當 ControlNet-style input
- **3D-aware decoder**：generation 模型直接 output 3DGS / NeRF（而非 pixel）
- **與 World Labs 線的對比**：純文字 → 3D vs 3D-conditioned video
- **跨倉互引**：與 Spatial-Handbook `foundations/3dgs-family/` 的差異化視角

## 引用 anchor

- [`/foundations/3d-aware-generation/overview.md`](../foundations/3d-aware-generation/overview.md)
- Spatial-Handbook：`foundations/3dgs-family/`、`foundations/feed-forward-3d/`（VGGT 線）
