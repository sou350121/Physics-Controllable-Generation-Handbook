# Sim-vs-Gen Data

> 「給 VLA pre-training / driving WM 該用生成資料還是 sim 資料？」

## 問題

兩派各有強烈經驗主義主張：
- **Sim 派**：MuJoCo / Genesis / Isaac 的物理 ground truth 嚴格、可量、可參數化；生成資料是「彩色雜訊」
- **Gen 派**：真實 video 預訓出來的 WM 自然涵蓋長尾、光照、紋理；sim 訓出來的 policy 不能跨 sim2real

## Pareto frontier

| 維度 | Sim | Gen |
|---|---|---|
| Physics ground truth | 強 | 弱（隱式） |
| 視覺多樣性 | 弱（需 domain randomization） | 強 |
| 可控性（force / contact / param） | 強 | 弱 |
| 規模 | 中（GPU env 數限制） | 大（pre-trained video） |
| sim2real gap | 大 | 小 |
| 邊際成本 | 中（GPU compute） | 高（大型生成模型） |

## Empirical evidence

- RoboCasa 系列證明合成資料能上 prod policy
- Cosmos / PI data engine 都是 Sim + Gen + Real **混用**
- DreamerV4 在純 sim 訓練後 real transfer 的限制
- V-JEPA-2 純真實 video 預訓 + 少量 action label 的成功

## Open question

- Optimal mix ratio 是多少？跨 task 變不變？
- 生成資料的 **diversity** vs sim 資料的 **physics fidelity**，哪個對 policy 更關鍵？
- Sim 資料 → 餵生成模型 → 再生成資料 的閉環是否會 distribution shift？

## Dissection wishlist

- [ ] PI data engine 細節
- [ ] Cosmos pipeline 中 sim / gen / real 比例
- [ ] RoboCasa vs 真實 demo 的 ablation
