<!-- ontology-5axis output=N/A injection=sim-in-loop control=action|trajectory|force|physical-param temporal=streaming domain=robotics|rigid -->

# Aerial Gym Simulator (NTNU-ARL / NVIDIA Isaac Gym)

## 1. One-paragraph TL;DR

Aerial Gym 不是「另一個通用 sim」 — 它是 **drone-only、跑在 Isaac Gym backbone 上的 thousand-env GPU 並行平台**，由 NTNU Autonomous Robots Lab（Kostas Alexis 組）在 2023 ICRA workshop 首發、2025-03 升級為 v2 並投 IEEE RA-L（arxiv 2503.01471）。它存在的理由是：MuJoCo MJX / Brax / Genesis 對 manipulation / locomotion 友善，但對 **multirotor airframe（under-/fully-/over-actuated）、geometric controller stack、ray-cast 深度/LiDAR 感測器**這條 drone-specific 鏈幾乎都要自己拼；Flightmare / AirSim 則是 CPU bottleneck，幾十個並行就到頂。Aerial Gym 把這條鏈 GPU 化到 **2^16 ≈ 65k envs、4.43M physics steps/s**，並把 **NVIDIA Warp 客製 ray-cast renderer** 接進來，讓 vision-based navigation policy 「一小時內訓完」並 sim2real 成功（quadrotor 位置誤差 0.09 m）。代價是：**不可微**（issue #58 由作者本人 close 確認）、**沒有 rotor wake / 沒有 wind model**（issue #47 still open）、且 Isaac Gym 已被 NVIDIA 標 deprecated，IsaacLab port 還在開發中。對本 handbook 的價值：它是 **aerial domain 的 sim-in-loop oracle + vision data factory** — 給 pixel-WM / VLA-for-drone 一條「跑得起來」的真實感測管線，而不是「拿來訓物理生成模型本身」。

## 2. Core mechanism

Aerial Gym 不發明新 physics solver，而是「**在 Isaac Gym rigid-body backbone 之上組一條 drone-專用 stack**」，所有東西都駐 GPU：

```
                ┌─────────────── Aerial Gym (NTNU-ARL) ───────────────┐
                │                                                      │
   URDF/cfg ──▶ │  ┌── State (per-env, GPU tensor) ──┐                 │
                │  │  p ∈ R³  v ∈ R³  q ∈ R⁴  ω ∈ R³ │                 │
                │  │  rotor RPM ω_m ∈ R^n_rotors      │                 │
                │  └─────────────────┬────────────────┘                │
                │                    │                                  │
                │  ┌── Motor dyn ────▼─── 1st-order, asymm τ_up/τ_dn ┐ │
                │  │  ω̇_m = (ω_cmd − ω_m)/τ                          │ │
                │  │  T_i = k_T · ω_m,i² (quadratic thrust)          │ │
                │  └─────────────────┬───────────────────────────────┘ │
                │                    │                                  │
                │  ┌── Allocation ───▼── arbitrary mixer (under/full/  │
                │  │   over-actuated)        ┌──────────────────────┐  │
                │  │   F_b, τ_b = Σ T_i r_i  │ linear+quadratic drag│  │
                │  └─────────────────┬───────┘ (no rotor wake)      │  │
                │                    │         └──────────────────────┘│
                │  ┌── Isaac Gym ────▼── rigid-body step (PhysX GPU) ┐ │
                │  └─────────────────┬─────────────────────────────────┘
                │                    │                                  │
                │  ┌── Sensors (Warp ray-cast, ~10× Isaac native) ────┐ │
                │  │  depth cam / RGB-D / LiDAR / hemisphere / ToF /  │ │
                │  │  IMU (Gaussian + bias random walk)                │ │
                │  └─────────────────┬─────────────────────────────────┘ │
                │                    │                                  │
                │  ┌── Controllers (parallel, on GPU) ────────────────┐ │
                │  │  position → velocity → attitude → rate → RPM     │ │
                │  └──────────────────────────────────────────────────┘ │
                │                                                      │
                │   ⬇ Gymnasium API → rl-games / sample-factory / CleanRL
                └──────────────────────────────────────────────────────┘
```

關鍵設計選擇：

- **Drone-first inductive bias**：state 不是「rigid body 一坨」而是 multirotor 完整鏈（rotor RPM + motor τ + allocation matrix），任意 n-rotor airframe 一個 cfg 切換
- **Geometric controller on GPU**：position/velocity/attitude/rate/RPM 五層 controller 全 batched，policy 可以接任一層做 action space（這是與 PX4 SITL 拼出來最大的差別）
- **Warp ray-cast renderer**：不用 Isaac Gym 原生 raster，而是 BVH-accelerated ray-cast，per-env mesh update + vertex annotation，論文 claim 比 Isaac Gym native ~10×
- **Motor model含對稱不對稱 time constant**：spool-up vs spool-down 不同 τ，這是 sim2real 最值錢的細節之一
- **不可微**：v2 paper 沒提，issue #58 作者直接 close — 走的是 model-free RL，不是 first-order policy gradient

## 3. 五軸定位 + 同軸對手

| 軸 | Aerial Gym | Flightmare (ETH RPG) | gym-pybullet-drones | RotorPy | OmniDrones (PKU/NVIDIA) | AirSim / PX4 SITL |
|---|---|---|---|---|---|---|
| Output | N/A (state + depth/LiDAR/RGB via Warp) | N/A (Unity render + Flightlib) | N/A (PyBullet render) | N/A (state-only) | N/A (Omniverse render) | N/A (Unreal render) |
| Injection | sim-in-loop（Isaac Gym PhysX） | sim-in-loop | sim-in-loop | sim-in-loop（純 Python ODE） | sim-in-loop（Omniverse PhysX） | sim-in-loop（CPU EKF/Gazebo） |
| Control | **action+trajectory+force+param**（5 層 controller）| action+trajectory | action+trajectory | param+trajectory（含 aero） | action+trajectory+param | action+trajectory |
| Temporal | streaming | streaming | streaming | streaming | streaming | streaming |
| Domain | **robotics+rigid（aerial-only）** | aerial+rigid | aerial+rigid | aerial+rigid（aerodynamics-first） | aerial+rigid | aerial+rigid+driving |
| Parallel envs (典型) | **65,536 / 4.43M steps·s⁻¹** | ~10s (Unity 限) | ~100 (CPU) | 1 (教學用) | thousands (Omniverse) | 1-2 (Unreal 限) |
| Differentiable | ❌ | ❌ | ❌ | ✅ (Python autograd-friendly) | ❌ | ❌ |
| Sim2real demo | ✅ quadrotor 0.09 m | ✅ [Champion-Swift](../../use-cases/aerial-sim/champion-level-drone-racing.md) (Kaufmann '23) | partial | educational only | partial | ✅ (PX4 widely) |

**同軸對手分群**：

- **「CPU 教學 / 學術 baseline」**：RotorPy（純 Python、可微 / 教學）、gym-pybullet-drones（OpenAI Gym 經典 wrapper）
- **「GPU 大規模 RL」**：**Aerial Gym（本篇）**、OmniDrones（Omniverse 線、商用整合更深）
- **「photoreal + 工程鏈完整」**：AirSim（Unreal 渲染、Microsoft 已 archive 2022）、Flightmare（Unity + ETH 自家 racing demo）、PX4 SITL（hardware-in-loop 黃金標準，但跑不快）
- **「通用 sim 跑 drone」**：Isaac Sim/Lab 通用版（無 drone-specific controller / aero）、Genesis-rotor（v0.x 有 quadrotor demo，但無 multi-airframe 與 LiDAR 鏈）

Aerial Gym 的獨佔位置 = **「GPU 並行 × 多 airframe × 多層 controller × ray-cast depth/LiDAR」四個都要的時候沒對手**；只要少一條（不需要 LiDAR / 不需要 over-actuated / 需要可微）對手都更便宜。

## 4. ⚡ shines / ❌ breaks

### ⚡ 真正領先的 regime

- **Vision-based navigation policy 一小時內訓完**：論文 claim depth-camera + cluttered forest 一小時訓到 deploy-ready；state-based 一分鐘內。這個級距在 Flightmare / AirSim 沒法達到（CPU 並行天花板）。
- **任意 n-rotor airframe 的 RL**：under-actuated（標準 quadrotor）→ fully-actuated（傾轉旋翼）→ over-actuated（hexarotor with tilt）一個 cfg 切換並 batched 訓練，這是學術 fixed-wing / tilt-wing 研究最缺的工具
- **真實 LiDAR / ToF 模型**：Ouster OS-0/1/2、Intel D455、Luxonis Oak-D、ST VL53L5CX 預設 cfg；Warp ray-cast 出 depth + segmentation + vertex-id，做 obstacle-avoidance VLA 訓練資料工廠很省事
- **sim2real 不需 fine-tune**：v2 paper 報 motor-control policy 直接部署、最高 2.83 m/s vision navigation；關鍵是 motor first-order dyn + asymmetric τ_up/τ_dn 把 spool 行為抓對了

### ❌ Known failure modes

- **不可微（issue [#58](https://github.com/ntnu-arl/aerial_gym_simulator/issues/58)）**：使用者直接問「Aerial Gym Simulator is differentiable?」 — 作者把 issue close 掉，結論是 **不可微**。要做 first-order policy gradient / model-based diff control（如 DiffMPC、Diff-OC）就出局。對比 Genesis MPM-solver / DiffTaichi / RotorPy 是劣勢。
- **沒有 rotor wake、沒有 wind（issue [#47](https://github.com/ntnu-arl/aerial_gym_simulator/issues/47) open since 2025-07）**：drag 是 linear + quadratic 二項展開，但 propeller-propeller interaction、ground effect、ceiling effect、blade flapping 統統 0；多旋翼緊密編隊或近地面任務 sim-to-real gap 立刻爆。Wind issue 至今無 maintainer 回覆。
- **Multi-drone scaling 弱**：issue [#57](https://github.com/ntnu-arl/aerial_gym_simulator/issues/57)（已關）作者建議「自己 fork、自己加」，沒有原生 swarm primitive；對比 OmniDrones 把 multi-agent 當 first-class 是劣勢
- **Isaac Gym 已 deprecated**：NVIDIA 自己掛 deprecated 標籤，Aerial Gym v2 仍依賴它；IsaacLab / Isaac Sim port 在 README 標「under development」（截至 2025-04 v2.0.0 release）。新環境裝起來會撞 Ubuntu 20.04-only / Python 3.7-3.8 / driver ≥ 470 的老牆。
- **環境初始化崩潰（issue [#52](https://github.com/ntnu-arl/aerial_gym_simulator/issues/52)）**：使用者報「all 16 envs crash at timestep 1」，render 只剩藍牆，maintainer 至今未回 — RL navigation example 第一次跑起來不一定能跑
- **VAE / 學習感測 module 缺**：issue [#50](https://github.com/ntnu-arl/aerial_gym_simulator/issues/50) 指 vision navigation reference 用的 VAE encoder 沒附訓練腳本與資料，要 reproduce paper 結果得自己重建這條
- **小物件 photorealism 弱**：Warp ray-cast 給的是 depth + segmentation，不是 PBR；窗框、樹葉細枝、室內小物件的視覺真實感比 Unreal-based AirSim / Isaac Sim 差一截 — 要訓「跨域 sim2real 的 vision policy」要另接 Cosmos / Isaac Sim 補幀

## 5. Reproduction notes

- **OS / Python**：Ubuntu 20.04 + Python 3.8 是 sweet spot（22.04 撞 Isaac Gym preview release，要打 patch）；NVIDIA driver ≥ 470.74
- **GPU 預算**：單卡 RTX 3090 / 4090 跑 4096-8192 envs 舒服；要 reproduce 4.43M steps/s benchmark 需要 H100 級
- **Install order**：
  1. 下 Isaac Gym Preview 4（NVIDIA developer 帳號）→ `pip install -e isaacgym/python`
  2. `git clone ntnu-arl/aerial_gym_simulator` → `pip install -e .`
  3. RL framework 三選一：`rl-games` / `sample-factory` / `cleanrl`（Gymnasium API 都通）
  4. 跑 `position_setpoint_task.py` 驗 state-based；要 vision 就跑 `navigation_task_depth.py`
- **典型踩坑**：
  - Isaac Gym pip install 後 import 失敗 → `LD_LIBRARY_PATH` 要加 `isaacgym/python/isaacgym/_bindings/linux-x86_64`
  - Warp version 不對 → v2 要 `warp-lang>=1.0`
  - 第一次跑 RL example 看到藍牆 / 立刻 reset → 對齊 [#52](https://github.com/ntnu-arl/aerial_gym_simulator/issues/52)；先跑 state-based 驗 install，再開 vision
  - 雙開大量 envs 時 PhysX GPU memory 漲很快，要先測 `--num_envs 256` 再爬
- **NTNU 釋出範圍**：simulator + 5 個 task（position setpoint, velocity tracking, depth navigation, LiDAR navigation, MPC tracking）+ controllers + sensor cfg；**VAE encoder / 真實 forest dataset / sim2real flight log 未釋出**

## 6. Cross-line synthesis

對本 handbook（generation 視角）的接點：

- **vs [Cosmos](../foundation-physics-models/cosmos-wfm.md)（aerial scenes）**：Cosmos-Predict 對地面駕駛 / 室內 manipulation 訓得多，aerial / 空拍視角資料稀；可把 Aerial Gym 當「aerial sim → Warp depth/RGB → Cosmos 條件 generation」的 ground-truth 工廠（同 [Genesis](./genesis.md) 餵 Cosmos 的模式 1：作為訓練資料源）
- **vs [Champion-Swift](../../use-cases/aerial-sim/champion-level-drone-racing.md)（Kaufmann et al. Nature 2023）**：Swift 用 Flightmare + on-policy RL 把 drone racing 推到人類冠軍水平；Aerial Gym v2 等於「同一條 recipe 但 GPU 並行 10-100× + 多 airframe」，是接續 Swift 路線最直接的工具
- **與 4 條路線怎麼接**：
  - **pixel-WM**：Aerial Gym depth + RGB → 訓 aerial video WM；缺 photoreal 要疊 Cosmos
  - **latent-WM**：跑 64k env rollout → 餵 [DreamerV4](../latent-world-models/dreamer-v4.md)-style latent 訓 aerial agent
  - **diff-sim**：❌ 直接出局（不可微）；要 first-order 就走 RotorPy / Brax-quadrotor
  - **neural surrogate**：可用 Aerial Gym GPU rollout 蒸 surrogate dynamics model（contact-light、適合 NN fit）
- **與 Spatial-Handbook 接**：Aerial Gym Warp ray-cast 出 depth + vertex-segmentation，正好餵 spatial-side 的 occupancy / NeRF / 3DGS 訓練；走 `bridge-to-spatial/` 目錄
- **與 VLA-Handbook 接**：aerial-VLA 訓練的事實工具 — controller 多層任挑做 action space，task cfg 切換不同 airframe

## 7. References

**Canonical**
- Kulkarni, Rehberg, Alexis. "Aerial Gym Simulator: A Framework for Highly Parallelized Simulation of Aerial Robots." arxiv 2503.01471 (2025-03), IEEE RA-L accepted. https://arxiv.org/abs/2503.01471
- Kulkarni et al. "Aerial Gym -- Isaac Gym Simulator for Aerial Robots." arxiv 2305.16510 (2023, ICRA workshop RS4UAVs).
- Repo: https://github.com/ntnu-arl/aerial_gym_simulator (v2.0.0, 2025-04-28; Apache-2.0)
- Docs: https://ntnu-arl.github.io/aerial_gym_simulator/

**Anchor 二手 / 鄰近**
- Song et al. "Flightmare: A Flexible Quadrotor Simulator." CoRL 2020. — ETH 線、Unity 渲染、Aerial Gym 直接對手
- Kaufmann et al. "Champion-level drone racing using deep reinforcement learning." Nature 2023 (Swift). — sim-to-real RL aerial 黃金 demo
- Folk, Tao, Cohen. "RotorPy: A Python-based Multirotor Simulator with Aerodynamics." arxiv 2306.04485 (ICRA-WS 2023). — Python-first、aerodynamics-first、可微
- Makoviychuk et al. "Isaac Gym: High Performance GPU-Based Physics Simulation for Robot Learning." NeurIPS Datasets 2021. — backbone
- Mittal et al. "Orbit / Isaac Lab: A Unified Simulation Framework for Interactive Robot Learning Environments." RA-L 2023. — Aerial Gym v3 預計遷移目標
- OmniDrones (Xu et al. 2024) — 同期 GPU drone sim，Omniverse 線對手

## 8. Pitfall log

| # | Issue | Severity | 摘錄 | Workaround |
|---|---|---|---|---|
| 8.1 | [#58](https://github.com/ntnu-arl/aerial_gym_simulator/issues/58) "Aerial Gym Simulator is differentiable?" (2025-10-24, closed) | **High**（路線選擇） | 作者 close 確認不支援可微 | 要 diff-sim 用 RotorPy / Brax-quadrotor；要 first-order policy gradient 出局 |
| 8.2 | [#47](https://github.com/ntnu-arl/aerial_gym_simulator/issues/47) "Wind Simulation" (2025-07-07, open) | High（sim2real） | 「seeks guidance on realistically simulating wind ... may be IsaacGym limitation」maintainer 至今未回 | 自己在 force injection layer 加 stochastic wind field；ground effect 同樣要自己加 |
| 8.3 | [#52](https://github.com/ntnu-arl/aerial_gym_simulator/issues/52) "Strange behavior and continuous environment resets" (open) | Medium | 所有 16 envs 在 timestep 1 即 crash，render 顯示藍牆，maintainer 未回 | 先跑 `position_setpoint_task` 驗 install，再啟用 vision task；檢查 asset path |
| 8.4 | [#48](https://github.com/ntnu-arl/aerial_gym_simulator/issues/48) "Support for Fisheye Camera and RayCaster Lidar" (open) | Medium | fisheye 與 RayCaster LiDAR 模型尚缺 | 自己擴 Warp ray-cast kernel；或先用 pinhole + 後處理 distort |
| 8.5 | [#50](https://github.com/ntnu-arl/aerial_gym_simulator/issues/50) "Lack of modules and datasets to train a VAE model" (open) | Medium | paper 用的 VAE encoder 沒附訓練腳本/資料 | 自己訓 VAE on Warp-rendered depth；或改 end-to-end CNN policy |
| 8.6 | [#57](https://github.com/ntnu-arl/aerial_gym_simulator/issues/57) "How to implement multi-drone missions" (closed) | Medium | swarm 不是 first-class；作者建議自行 fork | 要 swarm RL 改用 OmniDrones |
| 8.7 | Isaac Gym deprecated by NVIDIA | High（長期） | Aerial Gym v2 仍綁 Isaac Gym Preview 4（Ubuntu 20.04 + Python 3.7/3.8 + driver≥470） | README claim IsaacLab port 「under development」；新環境裝起來會撞老牆，先確認 Ubuntu / Python 版本 |
| 8.8 | No rotor wake / blade flapping / ground effect | High（physics fidelity） | Drag 僅 linear + quadratic 二項；近地面 / 編隊任務 sim2real gap 立刻爆 | 對 racing / single-drone open-air 任務影響有限；緊密編隊 / 室內 / 近地懸停請外加 surrogate aero model |
| 8.9 | Warp renderer 非 PBR | Medium | 只給 depth + seg + vertex-id，沒 photoreal RGB | 要 photoreal sim2real 把 depth 套到 Cosmos / Isaac Sim 補幀；或接 NeRF-baked 場景 |

> 註：上述 GitHub issue 編號 / 狀態 / 日期截至 web fetch 時點（2026-05）；後續可能 close / 新增。`[TBD: verify v3 IsaacLab port 是否已發佈]`
