# Use Case: Robotics Data Generation

> 用生成 video / latent / sim 補真實 demo 不足。VLA pre-training 的最大瓶頸 = 資料瓶頸。

## 三條 sub-route

1. **Pure video gen** —— Cosmos-Predict / Sora robot variants 生成大量手臂操作 video，無 action label → 用 inverse dynamics 補
2. **Action-conditioned WM** —— Genie-2 / V-JEPA-2 / DreamerV4，原生 action token，可直接用於 RL or imitation
3. **Sim-augmented gen** —— Genesis / Isaac 生 ground truth + 視覺 domain randomization

## 關鍵指標

- 生成資料 → policy 訓練 → real success rate（最終 ground truth）
- 比例：純合成 / 合成+少量真實 / 純真實 → Pareto

## Dissection wishlist (2-3 篇)

- [ ] Cosmos robotics 微調 pipeline 與 PI 對比
- [ ] V-JEPA-2 action-cond 對 VLA pre-train 的提升實測
- [ ] RoboCasa-style 合成 vs 真實 demo 的 Pareto

## 與 VLA-Handbook bridge

詳見 [`/bridge-to-vla/generative-data-for-vla.md`](../../bridge-to-vla/overview.md)
