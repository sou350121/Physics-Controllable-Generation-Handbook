# Material & Dynamics

> 流體 / 剛體 / 軟體 / 顆粒 / 布料的生成。Domain coupling 最強的一條。

## 5 個子領域

| 子域 | Anchor |
|---|---|
| Fluid | NeuralSPH · NeuralMPM-fluid · GraphCast (weather as fluid) |
| Rigid | ContactNets · ContactGen · MeshDiffusion |
| Soft | NeuralCloth · NeuralFEM |
| Granular | NeuralMPM-granular · PBD-NN |
| Cloth | NeuralCloth · ClothNet |

## 與 neural-surrogates / differentiable-simulators 的分工

- `neural-surrogates/` 強調「替代 PDE solver」的視角（評估標準是 PDE residual）
- `differentiable-simulators/` 強調「sim itself」（評估標準是 sim 自身 metrics）
- 本 zone 強調「**生成**特定材質動力學」的視角（評估標準是視覺合理 + 物理合理）

## Dissection wishlist (4-5 篇)

- [ ] NeuralMPM 系列（fluid / granular）
- [ ] ContactGen / ContactNets 對比
- [ ] NeuralCloth / cloth diffusion
- [ ] Rigid contact discontinuity 處理
- [ ] Cross-material 統一架構（Genesis-NN）

## §8 共通 pitfall

- Cross-material generalization 幾乎不存在
- Contact discontinuity 仍是學習瓶頸
- 大形變 / 拓撲變化（裂紋 / 撕裂）失敗
