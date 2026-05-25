# Pixel vs Latent: Where Should Physics Live?

**Thesis**: 物理規律在 **pixel** 還是 **latent** 學，決定了 fidelity / cost / debuggability 的 Pareto。

## 三個維度比較

| 維度 | Pixel-WM | Latent-WM |
|---|---|---|
| Compute | 高（pixel decode 貴） | 低（latent 維度小） |
| Fidelity | 高（最終產出是視覺） | 受 decoder 限制 |
| Debug | 好（看影片即知失效） | 難（latent 失效不可視） |
| Agent control | 不適合（autoregressive 太慢） | 適合（rollout 即時） |
| Cross-domain | 弱（domain-specific decoder） | 強（latent 抽象度高） |

## 結論建議

- **影片內容 / 駕駛 sim**：pixel-WM 為主，因 visualization 即產品
- **Agent control / RL**：latent-WM 為主，速度與 reward signal 質量
- **混合策略**：訓 latent-WM，僅在評估 / debug 時 decode（DreamerV4 思路）

## TODO

- [ ] 加入實測對比：DreamerV4 vs Cosmos-Predict 在同 task 的 sample efficiency
- [ ] 加入 hybrid 案例：Genie-2 latent + decode for visualization
