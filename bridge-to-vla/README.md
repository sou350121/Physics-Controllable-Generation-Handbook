# Bridge to VLA-Handbook

> 三冊一體中，本倉是 generation 端、VLA-Handbook 是 action 端。這個目錄收兩端的交界。

## 三篇邊界 essay

| 篇 | 主題 |
|---|---|
| [generative-data-for-vla.md](generative-data-for-vla.md) | 生成資料能否替代真實 demo 訓 VLA |
| [world-model-as-policy.md](world-model-as-policy.md) | DreamerV4 / V-JEPA-2 直接當 policy 的範式 |
| [video-pretraining-for-action.md](video-pretraining-for-action.md) | 用 video pre-train 出 latent embedding 再接 action head |

## 為什麼是 3 篇不是 5 篇

刻意保持稀疏 —— bridge 不該變成第二個 foundations。每篇收一個明確的「兩冊主題交界」。

新增 bridge essay 的條件：
1. 跨 generation × action 的明確設計問題
2. VLA-Handbook 與本倉都不能單獨完整覆蓋
3. 至少 2 篇 anchor 方法可以引用

## 跟 VLA-Handbook 同步

- VLA-Handbook 也有 `theory/03-engineering/` 與 `embodiments/` 等視角會引用 generation 線
- 建議互開 PR，bridge 文件兩邊各放一份 mirror（或 cross-link）
