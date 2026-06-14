# Use Case: Autonomous Driving Sim

> 閉環駕駛模擬 —— 沒有它就沒有 long-tail 場景的安全驗證。而駕駛剛好把本手冊命題演成一套清楚的**三方分工**：**重建給外觀（sensor-faithful）、生成給長尾多樣性、物理+反應式 agent 給動力學**。三者各管一塊，誰越界誰出事。

## 核心命題：外觀靠重建、長尾靠生成、動力學靠物理與反應

駕駛這個領域已經自己把「generation for appearance, physics for dynamics」分得很乾淨：

- **重建（被信任的那半）** —— 把真實 log 重建成可重渲染的場景，像素來自真實量測，**metric-scale、感測保真**（NeuRAD 連 rolling shutter / LiDAR ray-drop 都建）。死穴：**只能重渲染 log 看過的**，大幅改視角/移 actor 就外推進未觀測區。
- **生成（補多樣性）** —— 生成式駕駛 WM（GAIA-2 / Cosmos-Drive）造長尾場景，可控性已收斂到 **3D-box + HDMap + ego-action + text**。但**全是 open-loop 或 action-conditioned-rollout，VALIDATED 於資料增廣、DEMO-級於閉環驗證**。
- **物理 + 反應（動力學那半）** —— 車輛動力學 + **反應式多 agent**。**反應性本身是一條保真度軸**，不是預設給定的。

## AV 模擬契約（什麼必須真、什麼可以生成）

| 必須物理 / 感測上真 | 可以生成 / 採樣 |
|---|---|
| 車輛動力學（ego 對自己指令的反應） | 外觀：材質、光照、紋理 |
| 感測模型（含 rolling shutter、LiDAR ray-drop / beam divergence / intensity） | 天氣、時段、季節 |
| **metric 3D 幾何與尺度**（box、road、遮擋） | 長尾**場景佈局**（cut-in、VRU、碰撞） |
| **agent 反應性**（閉環、非 log-replay） | log 之外的資產/場景多樣性 |
| ego-action → next-observation 的耦合（閉環） | 域/地區遷移 |

最嚴謹的系統**把幾何/物理鎖在軌道上、只讓生成變外觀**：Cosmos-Drive 用 HDMap+LiDAR 錨定幾何（生成嚴格只是外觀），是本手冊命題最乾淨的 VALIDATED-增廣案。

## 開環陷阱（為什麼閉環是硬需求）

開環 metric 會騙人——這是 [閉環，否則白搭](./closed-loop-or-bust.md) 整篇的主題：① open-loop nuScenes 規劃可被「只吃 ego-status 的 MLP」刷穿（量的是模仿不是駕駛）；② 即使閉環，agent 若非反應式（IDM 太被動）仍會高估 planner。**NeuroNCAP** 把命題端到端兜起來：重建外觀 + 物理碰撞情境 + 閉環，證實「開環好 ≠ 閉環安全」。

## Anchor 系統

| 系統 | 重點 | 解構 / 來源 |
|---|---|---|
| GAIA-1 / GAIA-2 (Wayve) | 駕駛 video WM，3D-box + ego-action + 多視角；8.4B | [生成式駕駛 WM](./driving-world-models.md) · [GAIA-2 foundation](../../foundations/video-world-models/gaia-2.md) |
| Cosmos-Drive (NVIDIA) | HDMap+LiDAR ControlNet 錨定幾何，生成只是外觀 | [生成式駕駛 WM](./driving-world-models.md) · arXiv 2506.09042 |
| UniSim / NeuRAD / Street Gaussians | 真實 log 重建成可重模擬場景（cam+LiDAR、metric） | [神經重建模擬](./neural-reconstruction-sim.md) |
| NeuroNCAP / DriveArena | 閉環驗證（重建式 / 生成式） | [閉環，否則白搭](./closed-loop-or-bust.md) |

## 本區 Dissections

- [生成式駕駛世界模型](./driving-world-models.md) — GAIA-2 / Cosmos-Drive / Vista：控制契約、VALIDATED-增廣 vs DEMO-閉環
- [神經重建模擬](./neural-reconstruction-sim.md) — UniSim / NeuRAD / Street Gaussians：感測保真的 real2sim，及「只能重渲 log 看過的」限制
- [閉環，否則白搭](./closed-loop-or-bust.md) — 開環陷阱、反應式 agent、NeuroNCAP 端到端證據

## 與 sister handbook 的對應

駕駛 WM 是通用 video-WM 的 embodiment 鏡像（同樣 action-conditioning + long-horizon + compounding-error），但 AV 多了硬的感測/metric/反應性約束。對接 [Spatial-Handbook 駕駛 embodiment](https://github.com/sou350121/Spatial-Intelligence-Handbook/tree/main/embodiments/driving)。資料集錨點：nuScenes（開環陷阱展品）、Waymo Open、nuPlan（閉環 + 反應式 agent benchmark）。

## 未來前沿

- **生成式閉環驗證**：把生成 WM 當 AV stack 的*評測器*仍不可信（低 FID/FVD 可掩蓋安全關鍵幻覺）；DriveArena 是起點。
- **混合 production 路線**：重建（外觀）+ 顯式動力學/反應式 agent（物理）+ 生成（長尾）的三方縫合，是落地形態。
- **反應性沒有 ontology 槽**：5 軸目前無「反應性」軸，是個 taxonomy 缺口。
