# Use Case: Aerial Sim

> 物理可控生成給無人機自主：6-DoF 自由運動 + 螺旋槳尾流 + 風擾 —— 比 driving 多兩個自由度，比 manipulation 多了「掉下來就壞」的硬性約束。

## 為什麼 aerial 自成一格

不能直接套用 driving / robotics sub-route：

- **6-DoF free motion** —— 沒有 lane / 沒有桌面，policy 失敗就是墜機；rollout drift 容忍度比 driving 低一個量級
- **Sensor stack 不同** —— IMU + GNSS + optical flow + downward range，視覺只是其一；WM 需要同時生視覺與 IMU 一致的軌跡
- **Rotor / aerodynamics 是死角** —— propeller wake、ground effect、rotor-rotor interaction 在主流 video WM (Cosmos / Sora) 裡基本不存在；這是 aerial 與其他 use-case 最大的物理差異
- **Wind / turbulence** —— 外部擾動是 first-class 物理量，不是「噪聲」
- **HDR + 小目標** —— 天空到地面 14+ stops；其他無人機 / 電線 / 鳥是像素級小物件，photoreal 與否直接決定能否 train avoidance policy

## Anchor 系統

| 系統 | 重點 | 來源 |
|---|---|---|
| Aerial Gym Simulator | Isaac Gym backend，GPU 並行數千架 multirotor，PyTorch JIT geometric controller，ray-cast depth/segmentation | NTNU ARL，arXiv 2305.16510 (2023) / 2503.01471 (v2 2025) |
| Flightmare | Unity 渲染 + 解耦 physics，multi-modal sensor，RL API 並行數百架 | UZH RPG，CoRL 2020 |
| Swift (Champion-Level Drone Racing) | Sim-trained RL policy，real-world 擊敗人類冠軍；sim + 真實世界資料混訓 | UZH RPG，Nature 2023-08-31 |
| Dream to Fly | DreamerV3 + raw pixel → collective thrust/bodyrates；端到端 vision-based agile flight；sim + real | UZH RPG，arXiv 2501.14377 (2025) |
| PX4 SITL + Gazebo | 工業界 baseline，flight controller-in-loop，visual fidelity 低 | PX4 / Open Robotics |
| AirSim (legacy) | 早期 photoreal aerial sim，MSR 2022 起 deprecated | Microsoft (archived) |
| Cosmos Predict 2.5 (FPV camera) | NVIDIA WFM，官方 demo 含 FPV quadcopter over harbor，LoRA/DoRA fine-tune | NVIDIA，2025 |
| DJI / Skydio 內部 sim | Closed source，量產級驗證流程 | [TBD: 公開資料極少] |

## 三條 sub-route（對齊 robotics-data-gen 切法）

1. **Pure video gen** —— Cosmos / Sora aerial fine-tune，生 FPV / overhead drone footage 給感知模型 pre-train。痛點：rotor wake、IMU 一致性、ground effect 都不在訓練分布內，目前主要當「視覺 augmentation」用，不直接驅動 control。
2. **Action-conditioned aerial WM** —— Dream to Fly (DreamerV3 on quadrotor) 是目前最清楚的代表：raw pixel + collective thrust / bodyrates token，sim→real zero-shot agile flight。比 driving WM 更接近 closed-loop policy 而不是場景生成器。
3. **Sim-augmented** —— Aerial Gym / Flightmare + domain randomization；Swift 是這條路的 SOTA proof point（Nature 2023，世界冠軍級）。重點不是 photoreal，而是 dynamics fidelity + 大規模 parallelization。

> 三條路在 aerial 上的關係比 robotics 更明顯：(3) sim-augmented 是當前唯一證明過 real-world champion-level 落地的；(2) action-WM 是研究熱點；(1) pure video gen 仍在「能不能生對」的早期。

## 關鍵指標

- **Sim-to-real success rate** —— Swift 在 head-to-head 對人類冠軍取得 fastest lap (Nature 2023)；這是當前 aerial sim-to-real 最硬的 benchmark
- **GPU 並行度** —— Aerial Gym 可同時模擬數千架 multirotor（geometric controller 在 GPU 上跑）；Flightmare 數百架
- **Photorealism for vision policy** —— FPV racing gate detection、obstacle avoidance、小目標（無人機/電線）；photoreal 決定 perception backbone 能否 transfer
- **Dynamics fidelity** —— ground effect / rotor wake / IMU bias / wind gust 是否建模；多數 video WM 在這欄是 0 分
- **Multi-drone scaling** —— swarm scenario，rotor-rotor aerodynamic interaction 是 next frontier

## 與 sister handbook 的 bridge

這是本 handbook **第一個**直接餵到 Spatial-Handbook `embodiments/aerial/` 的 use-case（aerial 是 Spatial 最深的 embodiment）：

- **6-DoF dynamics** —— Spatial 的 `embodiments/aerial/dynamics_and_control_primer.md` 提供 quadrotor 動力學/控制基礎；本 use-case 生成的軌跡必須符合那邊的約束
- **VIO ground truth** —— Spatial 講 visual-inertial odometry 的消費端；aerial WM (本側) 是 VIO 訓練/驗證資料的生成端 —— 兩邊要對齊 IMU 噪聲模型 + camera-IMU extrinsic
- **VLA for drones** —— 若 Spatial 規劃 aerial-VLA，pre-training 資料瓶頸的解法走這裡 sub-route (1) / (2)

詳見 [`/bridge-to-spatial/aerial-embodiment.md`](../../bridge-to-spatial/overview.md)（待寫）。

## Dissection wishlist (3-4 篇)

- [ ] Aerial Gym Simulator —— GPU 並行 + geometric controller + ray-cast rendering 工程拆解（與 Isaac Lab aerial 對比）
- [ ] Swift (Champion-Level Drone Racing) —— sim-trained RL + real-world data 混訓的具體 pipeline，怎麼跨越 sim-to-real gap
- [ ] Flightmare vs PX4 SITL vs Isaac Sim aerial —— 三套 aerial sim 的 dynamics fidelity / rendering / RL throughput 對比
- [ ] Dream to Fly (DreamerV3 on quadrotor) —— action-conditioned aerial WM 第一個 real-world champion-grade proof，與 driving WM (GAIA-2) 的差異點
- [ ] [TBD] 合成 aerial video 給 drone-VLA pre-train —— 目前是否有公開 work，還是仍在內部（DJI / Skydio）
