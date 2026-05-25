# Latent World Models

> 在 latent space 做 rollout，再選擇性 decode 像素。給 agent control 用最划算的一條路。

## 5-axis defaults

- `output=latent`（部分另出 `pixel-video` 為 visualization）
- `injection=implicit-from-data` or `constraint-loss`
- `control=action|image-prompt`
- `temporal=latent-rollout` or `hierarchical`
- `domain=robotics|driving|generalist`

## Anchor methods

| Method | 年份 | 重點 |
|---|---|---|
| DreamerV3 | 2023 | RSSM latent dynamics + actor-critic；agent control 範式 |
| DreamerV4 | 2025 | 大幅 scale + 多任務泛化 |
| V-JEPA | 2024 | latent 預測 + self-supervised pretrain |
| V-JEPA-2 | 2025 | 加 action conditioning，跨向 robotics |
| Genie-2 | 2025 | latent action token；interactive game WM |
| Decart | 2025 | 即時 latent WM，sliding KV cache |
| TECO | 2024 | hierarchical latent rollout |

## 關鍵 tension

1. **latent rollout 省 compute 但失去 visualization** — debug 失效模式靠 decode，但 decode 自己會引入額外誤差
2. **action token 怎麼設計**：連續 vs 離散 vs hybrid (Genie-2 用 latent action)
3. **跨 task / cross-embodiment 泛化** — V-JEPA-2 vs Dreamer 在這軸上正面對撞

## Dissection wishlist (5-7 篇)

- [ ] DreamerV3 → V4 演化、scale 與多任務
- [ ] V-JEPA 與 V-JEPA-2 (action) 對比
- [ ] Genie-2 latent action 設計
- [ ] Decart 即時 WM 的 sliding 設計
- [ ] TECO hierarchical
- [ ] latent vs pixel WM 在 agent control 的實測對比（指向 crossing/pixel-vs-latent-physics）

## §8 共通 pitfall

- latent dynamics 收斂到 trivial 解（mode collapse） — V-JEPA 的著名挑戰
- action conditioning 的範化失敗 — 訓練見過的 action 分布外崩
- Visualization 失真導致 debug 誤判
