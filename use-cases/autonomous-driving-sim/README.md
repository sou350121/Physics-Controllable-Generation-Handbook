# Use Case: Autonomous Driving Sim

> Closed-loop driving world model —— 沒有它就沒有 long-tail 場景的安全驗證。

## Anchor 系統

| 系統 | 重點 |
|---|---|
| Wayve GAIA-1 / GAIA-2 | 駕駛 video WM，trajectory conditioning |
| Tesla world model | 自家路線，從 occupancy 出發 |
| Cosmos-Drive | NVIDIA 微調 line |
| DriveWM / DriveDreamer | 學術 line |

## 三個工程問題

1. **Long-horizon** —— 30s+ rollout 不漂移
2. **多視角一致** —— 6 個 camera 同步
3. **Closed-loop** —— policy 動作回饋 WM，需要可微 / 可快速重採

## Dissection wishlist (2 篇)

- [ ] GAIA-2 設計與 closed-loop 整合
- [ ] Cosmos-Drive trajectory conditioning 工程
