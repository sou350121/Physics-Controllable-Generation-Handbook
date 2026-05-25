# 3D-Aware Generation

> 顯式 3D 表徵 + 時間。Spatial-Handbook 的 3DGS/NeRF zone 與本倉的交界。

## 為什麼跟 video-world-models 分開

- output 是顯式 3D（3DGS / mesh / occ / SDF），不是像素
- 可以多視角一致、可以渲染任意視點
- 物理規律可作用於顯式幾何（剛體碰撞 / cloth / fluid 在 3D 結構上）

## 5-axis defaults

- `output=3d-explicit`（v1 `3d-scene`/`mesh` 合併；NeRF/SDF 內部用 → `3d-implicit`）
- `injection=data-only` or `aux-loss`
- `control=text|image-init|3d-init|camera`
- `temporal=clip-parallel` or `hierarchical`
- `domain=rigid`（v2 Check 9c 不允許 `generalist` 給 World Labs 等非 Sora/Veo/Cosmos 系；靜態 3D scene 預設 rigid，動態場景按物理選 fluid/soft 等）

## Anchor methods

| Method | 重點 |
|---|---|
| World Labs gen-3D | 從 image / text 生 explorable 3D scene |
| GaussianAnything | 3DGS 生成 |
| Gaussian-video / 4D-Gaussian | 動態 3DGS |
| DreamFusion → MVDream → Magic3D 線 | text-to-3D 演化 |
| Cosmos 3D variants | NVIDIA 線 3D 補完 |

## 與 Spatial-Handbook 的分工

- Spatial-Handbook 的 `foundations/3dgs-family/` 收 3D **重建** 側
- 本倉的 `3d-aware-generation/` 收 3D **生成** 側
- 跨 ref 用 `bridge-to-spatial/3d-aware-video-gen.md`

## Dissection wishlist

- [x] **[World Labs / Marble](./world-labs.md)** ✅ (anchor written)
- [ ] 4D Gaussian / Gaussian-video 動態 3DGS
- [ ] DreamFusion → Magic3D → 後續 text-to-3D 演化
- [ ] Generative Gaussian Splatting (2503.13272) — v2 spec 列為 `3d-explicit` canonical anchor

## §8 共通 pitfall

- Multi-view consistency 仍脆
- 3D 表徵 → 物理規律的 coupling 不明確（3DGS 不是 mesh 不易加 contact）
- 推理 cost 高（多視角 + 時間維）
