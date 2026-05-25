# Foundation Physics Models

> 朝「物理 FM」的整合嘗試。等價於影片版的 GPT moment + 物理感。

## 候選 FMs

| 系統 | 主張 |
|---|---|
| NVIDIA Cosmos | World Foundation Model；多 domain 微調 |
| OpenAI Sora (long-term) | 通用 video FM；物理 implicit |
| Google Genie | interactive WM FM |
| Meta V-JEPA-2 | 跨 embodiment latent FM |
| WorldLabs 線 | 3D FM 視角 |
| AlphaFold 系列 | 蛋白質物理 FM 範式 |

## 關鍵問題

1. **Multi-domain transfer** —— 一個 FM 能不能同時做 robotics + driving + media + science？
2. **Physics-aware 與否** —— scale 賭注 vs explicit physics injection
3. **Open vs closed** —— Cosmos 部分開源，Sora/Veo 不開源；對研究社群影響
4. **與 LLM-FM 的 layered architecture** —— Cosmos-Reason (LLM) + Cosmos-Predict (video FM) 的範式

## Dissection wishlist (1-2 篇先做 anchor)

- [ ] Cosmos World Foundation Model 整體架構與 release strategy
- [ ] FM-for-physics 與 LLM-FM 的 layered combo
- [ ] (未來) 跨 domain transfer 實測

## §8 共通 pitfall

- 「FM」名詞被 hype 推高，實質單 domain 居多
- Closed FM 無法被獨立驗證物理感
- LLM + WM layered 之間的訊號傳遞瓶頸
