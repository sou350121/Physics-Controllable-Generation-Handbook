# Use Case: Embodied Policy Rollout

> WM-as-policy（Dreamer 派）/ MPC-on-WM（model-based RL）。WM 直接用於決策。

## 兩派

1. **Actor-critic on WM** —— DreamerV3/V4 路線
2. **MPC on WM** —— PETS / TD-MPC / sampling-based plan

## 關鍵問題

- WM-rollout reliability vs decision quality
- 跨 embodiment 範化
- Sim2real 與 WM-real 的 gap

## Dissection wishlist (2 篇)

- [ ] DreamerV4 on real robots
- [ ] MPC-on-WM 的 sampling efficiency
