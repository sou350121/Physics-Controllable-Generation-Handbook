# Neural Surrogates

> 用 NN 替代 PDE solver。物理 inductive bias 最強的一條 —— 而且在自家領域已經 productionized（GraphCast、ECMWF AIFS）。

## 5-axis defaults

- `output=field|mesh|particle`
- `injection=hard-PDE` or `constraint-loss`
- `control=physical-param|3d-prompt`
- `temporal=autoregressive`（多數）
- `domain=fluid|weather|rigid|soft|granular|bio`

## Anchor methods

| Method | 領域 | 重點 |
|---|---|---|
| GraphCast | 氣象 | DeepMind 2023；超越 IFS；現入 ECMWF prod |
| AIFS | 氣象 | ECMWF 自家 transformer 線 |
| MeshGraphNet | rigid/soft | DeepMind 線 GNN PDE |
| FNO (Fourier Neural Operator) | PDE 通用 | Spectral parametrization |
| PDE-Refiner | PDE 通用 | iterative refinement 改 long-rollout drift |
| NeuralMPM / NeuralSPH | 粒子 / 流體 | 替代 MPM/SPH solver |
| Pangu-Weather | 氣象 | 華為線 |
| GenCast | 氣象 ensemble | DeepMind 2024 |

## 為什麼這條獨立於 video WM

- Surrogate 不生成「視覺」，生成「場」 —— evaluation 標準完全不同
- 守恆律 / 邊界條件 / 數值穩定性 是核心 —— 這些在 video WM 是 implicit failure
- 與 scientific-discovery use-case 強耦合

## Dissection wishlist (5 篇)

- [ ] GraphCast 架構 + 上 prod 路線
- [ ] FNO vs MeshGraphNet（spectral vs spatial）
- [ ] PDE-Refiner 解 long-horizon drift 的方法
- [ ] NeuralMPM 在 fluid/granular 的 sim2real
- [ ] Pangu / GenCast 與 GraphCast 對比

## §8 共通 pitfall

- Long-rollout drift —— 自迴歸 surrogate 累積誤差，PDE-Refiner 嘗試解
- Boundary condition OOD —— 訓練時沒見過的邊界拓撲
- Conservation violation —— 即使 hard-PDE 架構也只能保證特定守恆律
- 與真實量測的 gap —— surrogate 訓在 sim/reanalysis，不在真實 sensor
