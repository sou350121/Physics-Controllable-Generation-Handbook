# Data Engine

> Sim → Gen → Real 的資料閉環。生成模型的「燃料系統」。

## 三層管道

1. **Real → Gen** —— 真實 video / demo 訓 base WFM
2. **Sim → Gen** —— Isaac/Genesis pull video 補足 long-tail
3. **Gen → Real** —— 用生成資料訓 VLA / policy 在 real 部署，迭代

## Anchor pipelines

| 線 | 重點 |
|---|---|
| NVIDIA Cosmos data pipeline | Omniverse + Isaac + 真實資料整合 |
| RoboCasa | 機器人合成場景 |
| PI data engine | Physical Intelligence 內部資料管道 |
| Open X-Embodiment | 跨 embodiment 真實 demo 聚合 |

## Dissection wishlist (2-3 篇)

- [ ] Cosmos pipeline 細節（Omniverse / Isaac / web video curation）
- [ ] RoboCasa 合成場景設計
- [ ] Real-Gen-Real 閉環的成功與失敗案例

## §8 共通 pitfall

- 資料 quality > quantity，但 curation 成本爆
- Sim2real gap 在資料層就被放大
- Distribution shift across iteration
