# Bridge: Video Pre-training for Action

> 🚧 **Stub** — waiting on anchor dissections. See [overview.md](overview.md) for design rationale.

## 主題

用 video pre-train 出 latent embedding 再接 action head — V-JEPA / Cosmos / Sora-style backbone 給 VLA 當預訓 backbone 的取捨。

## 預計內容

- **三條 backbone 選項**：V-JEPA latent / Sora-style pixel feature / Cosmos hybrid
- **接 action head 的工程模式**：frozen vs fine-tune、cross-attn vs concat
- **收益測量**：用 video pre-train 的 VLA 在 robot success rate 提升多少
- **未解問題**：哪種 pre-train signal 對下游 action policy 最有用

## 引用 anchor

- [`/foundations/latent-world-models/v-jepa-2.md`](../foundations/latent-world-models/v-jepa-2.md)
- [`/foundations/video-world-models/sora.md`](../foundations/video-world-models/sora.md)
- VLA-Handbook（sister repo）對應 pre-training dissection
