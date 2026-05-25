# Diffusion Physics

> Diffusion score function 加物理梯度 / classifier guidance / 物理 prior。

## 5-axis defaults

- `output=pixel-video|mesh|particle`
- `injection=guidance-gradient`
- `control=text|trajectory|force|contact`
- `temporal=clip-parallel`（少數 latent-rollout）
- `domain=generalist|fluid|rigid`

## Anchor methods

| Method | 年份 | 重點 |
|---|---|---|
| PhysDiff | 2024 | classifier guidance with physics |
| ContactGen-diffusion | 2025 | 接觸條件 diffusion |
| ForceGen-diffusion | 2025 | 力條件 diffusion |
| MotionDiffuse / PhysMotion | 2024-25 | 人體運動物理約束 |

## Dissection wishlist (3 篇)

- [ ] PhysDiff classifier guidance 機制與失效
- [ ] Force/Contact conditioning 在 diffusion 中的設計（與 controllability-mechanisms 互鏈）
- [ ] Score-conditioned 與 aux-loss 的 Pareto 比較

## §8 共通 pitfall

- Guidance scale 過大 → 採樣崩塌；過小 → 物理失效
- Classifier 自身不精準時 guidance 反向誤導
