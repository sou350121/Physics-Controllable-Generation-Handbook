# Use Case: Scientific Discovery

> Neural surrogate / generative physics 在科學模擬的應用。本倉與「純 AI」最遠的一角，但也是最 productionized 的。

## 四個子場

| 子場 | Anchor |
|---|---|
| Weather | GraphCast · Pangu · AIFS · GenCast |
| Climate | NeuralGCM · ACE2 |
| Molecular | AlphaFold · MD surrogate |
| Engineering CFD | FNO · MeshGraphNet 線 |

## 與 neural-surrogates / material-and-dynamics 的分工

- 那兩個 zone 是「方法視角」
- 本 use-case 是「**領域視角**」—— 同樣 GraphCast 既出現在 neural-surrogates 也出現在這

## Dissection wishlist (2-3 篇)

- [ ] GraphCast 上 prod 路線（ECMWF 整合）
- [ ] AlphaFold-physics 線（不是 structure prediction，是 dynamics）
- [ ] CFD surrogate 在工業設計的應用
