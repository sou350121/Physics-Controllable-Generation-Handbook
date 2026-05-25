# Long-Horizon Rollout

> Drift / error accumulation / 跨 clip 銜接 —— 影片 WM 與 latent WM 的共同戰場。

## 三類解法

1. **架構解** —— Joint rollout（一次生整段）；sliding KV cache（Genie-2 / Decart）；hierarchical（兩層時間尺度，Cosmos-Reason+Predict / TECO）
2. **訓練解** —— Teacher forcing → scheduled sampling → DAgger style；rollout-aware loss
3. **推理解** —— Iterative refinement（PDE-Refiner 線）；guidance；replay buffer

## 關鍵 metrics

- FVD 隨 rollout length 增長曲線
- 守恆律違反率隨時間
- Identity 漂移（物體外觀 drift）
- 跨 clip 銜接的 perceptual gap

## Dissection wishlist (3 篇)

- [ ] Genie-2 / Decart sliding WM 設計
- [ ] Cosmos-Reason+Predict hierarchical 路線
- [ ] PDE-Refiner 的迭代精修在 video WM 的可遷移性

## §8 共通 pitfall

- Joint rollout 把長度寫死，跨 clip 銜接靠 image-init 不穩
- AR rollout 的 exposure bias 在 long horizon 放大
- Hierarchical 模型訓練收斂難
