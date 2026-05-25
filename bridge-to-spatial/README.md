# Bridge to Spatial-Intelligence-Handbook

> 本倉是 generation 端、Spatial-Handbook 是 perception 端。共同關心 3D 表徵、world-model、driving sim。

## 三篇邊界 essay

| 篇 | 主題 |
|---|---|
| [3d-aware-video-gen.md](3d-aware-video-gen.md) | 3DGS / NeRF 重建表徵如何條件化影片生成 |
| [nerf-3dgs-meet-world-model.md](nerf-3dgs-meet-world-model.md) | 3DGS-based world model（顯式 3D 表徵的 WM） |
| [spatial-controllability.md](spatial-controllability.md) | Spatial structure 作為 conditioning（occupancy / depth / pose 怎麼進生成） |

## 與 Spatial-Handbook 的對應 zone

| 本倉 | Spatial-Handbook 對應 |
|---|---|
| `foundations/3d-aware-generation/` | `foundations/3dgs-family/`, `foundations/feed-forward-3d/` |
| `foundations/video-world-models/` | `foundations/world-model/` |
| `use-cases/autonomous-driving-sim/` | `embodiments/driving/` |
| `companies/wayve.md` | `companies/wayve_world_model.md` |

## 寫作 rule

- 任何在 spatial-handbook 已涵蓋的主題，本倉只寫「generation 端視角差異」
- §6 cross-line synthesis 必須 link 到 spatial 對應 dissection
