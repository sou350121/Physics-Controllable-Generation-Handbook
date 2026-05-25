# Evaluation: Physics

> 怎麼判斷「生成的影片/場景物理合理」 —— 這個 zone 是整個 handbook 的 ground truth。

## 三層評估

1. **Perceptual quality** —— FVD / IS / CLIP-score（傳統視覺）
2. **Physics plausibility** —— VBench-Physics / PhysBench / 守恆律違反率
3. **Downstream task** —— 用生成資料訓 VLA / policy，看真實任務 success

## Anchor benchmarks

| Benchmark | 主軸 |
|---|---|
| VBench / VBench-Physics | 物理屬性多維打分 |
| PhysBench | 物理理解專測 |
| PhyGenBench | physics-aware video gen |
| WorldModelEval | agent control 視角 |
| PDEBench | 神經 PDE solver |
| RoboEval / EvalGen | downstream policy 評估 |

## 守恆律違反 metrics

- 質量守恆（fluid）
- 動量守恆（rigid collision）
- 能量守恆（pendulum / oscillator）
- 接觸無穿透（rigid / soft）
- 因果一致性（物體不無中生有 / 消失）

## Dissection wishlist (3-4 篇)

- [ ] VBench-Physics 設計與 limitation
- [ ] PhysBench 評測項目
- [ ] downstream-task evaluation（生成資料 → VLA success 提升）
- [ ] 守恆律違反 metrics 的可重現 implementation

## §8 共通 pitfall

- Benchmark Goodhart —— 高分模型在新場景仍崩
- Human eval 噪聲大 + 偏 perceptual，物理感判斷不可靠
- Downstream task evaluation 才是 ground truth，但成本最高
