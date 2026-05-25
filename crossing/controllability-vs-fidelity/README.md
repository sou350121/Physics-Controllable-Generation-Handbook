# Controllability-vs-Fidelity

> 加越多 conditioning，生成質量越會被「拉壞」？兩端的 Pareto。

## 問題

- 加 text → 生成可控但細節走樣（常見現象）
- 加 trajectory → 物體走對軌跡但物理感弱
- 加 force / contact → 物理對但視覺崩
- 多 modal 同時加 → 互相干擾（multi-conditioning interference）

## Pareto frontier

| 方法類型 | Controllability | Fidelity |
|---|---|---|
| Text-only (Sora/Veo) | 中 | 高 |
| Text+image (SVD/Cosmos-img2vid) | 中-高 | 高 |
| Action-only (Genie-2) | 高 | 中 |
| Trajectory-cond (Cosmos-Drive) | 高 | 中-高 |
| Force/contact (ForceGen/ContactGen) | 高 | 中 |
| Multi (text+action+force) | 最高（理論） | 最低（實作） |

## Empirical evidence

- Classifier-free guidance scale 越大，controllability 越強，但採樣 fidelity 下降
- ControlNet 加多重 condition 互相壓制（Stable Diffusion 觀察）
- Cosmos-Predict 加 trajectory 後物體軌跡可控，但 fluid 細節品質下降

## Open question

- Multi-conditioning 的 fusion 機制（per-stage / cross-attn / token concat / hypernet）哪種損 fidelity 最少？
- 是否存在「fidelity-preserving controllability」的設計？

## Dissection wishlist

- [ ] Multi-conditioning fusion 方法對比
- [ ] Guidance scale 與 fidelity 的 Pareto 量化
