# Video World Models

> 直接生成像素影片。物理 implicit-from-data。最 scale-pilled 的一條路。

## 5-axis defaults

- `output=pixel-video`
- `injection=implicit-from-data`（少數加 constraint-loss）
- `control=text|image-prompt`（少數加 trajectory）
- `temporal=joint-rollout` or `autoregressive`
- `domain=generalist`（少數 robotics / driving 專版）

## Anchor methods

| Method | 年份 | 重點 |
|---|---|---|
| Sora | 2024 Q1 | 公眾 GPT-moment；長 clip joint-rollout；DiT 架構 |
| Veo / Veo-2 | 2024-25 | Google 線；強 text→video，物理感漸進 |
| Cosmos-Predict | 2025 | NVIDIA pre-trained WFM；給 robotics/driving 微調 |
| GAIA-1 / GAIA-2 | 2024-25 | Wayve 駕駛專用 |
| Stable Video Diffusion | 2023 Q4 | 開源 image-to-video baseline |
| Kling / Hunyuan-Video / Wan | 2024-25 | 中國線 SOTA |
| Veo-3 | 2025-26 | 加 audio 與更強連貫 |

## 關鍵 tension

1. **物理感從 scale 還是 inductive bias 來？** Sora 派賭 scale，PhysGen 派加 constraint
2. **Joint rollout vs autoregressive** — Sora 走 joint 解 drift，但限長度；Genie 走 AR 撐長度，但 drift 重
3. **Text conditioning 的物理可控性差** — 加 trajectory / force 是目前的延伸方向

## Dissection wishlist (預計 8-10 篇)

- [ ] Sora 架構與物理 failure mode 二手實測（Liu et al, "How Far Is Sora Physics-Aware")
- [ ] Veo-2/3 與 Sora 對比
- [ ] Cosmos-Predict 訓練資料管道 + 加 robotics conditioning
- [ ] GAIA-2 駕駛專用設計
- [ ] Hunyuan-Video / Wan / Kling SOTA 比較
- [ ] Stable Video Diffusion 作為 baseline 的限制
- [ ] DiT vs U-Net 在 video 的取捨
- [ ] Joint vs AR rollout 實測對比

## §8 共通 pitfall（zone 級）

- Long-shot consistency 依然脆 — clip 切換、相機運動劇烈時 object identity 漏
- 物理：流體 splash、剛體碰撞、紙張褶皺 — 高頻細節的「物理破綻」最易暴露
- Text-only conditioning 無法精準控制力 / 接觸時序 — 要做 robotics 必須補 trajectory/action 接口
