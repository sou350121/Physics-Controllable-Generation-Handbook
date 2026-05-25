# Benchmarks

| 路線 | Benchmark |
|---|---|
| [video-physics/](video-physics/) | VBench-Physics, PhysBench, PhyGenBench |
| [world-model/](world-model/) | WorldModelEval, DreamerSimEval |
| [robot-data/](robot-data/) | Generated data → policy success rate 量測 |
| [scientific/](scientific/) | PDEBench, WeatherBench |
| [controllability/](controllability/) | Controllability-fidelity Pareto bench |

## 為什麼分這 5 類

對應 ontology Axis 1 (output) × Axis 5 (domain) 的主要交點。

## Dissection wishlist (3-4 篇)

- [ ] VBench-Physics 設計、項目、limitation
- [ ] PhysBench 與 VBench-Physics 的對比
- [ ] PDEBench 在 neural surrogate 評測的位置
- [ ] Robot policy 下游量測（用生成資料訓 policy → real success）
