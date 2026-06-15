<!-- ontology-5axis output=N/A injection=sim-in-loop control=action|trajectory|force|param temporal=streaming domain=robotics|rigid -->

# NVIDIA Isaac Sim

## 1. One-paragraph TL;DR

Isaac Sim 不是「另一個 GPU sim」也不是生成模型 — 它是 **建在 Omniverse / OpenUSD 之上的生產級 photoreal 機器人模擬器 + 合成資料工廠**，2026-06-04 釋出 **v6.0.0、Apache-2.0**（GitHub `isaac-sim/IsaacSim`）。它存在的理由跟本區的研究型 diff-sim（[MJX](./mujoco-mjx.md) / [Genesis](./genesis.md) / [Warp](./nvidia-warp.md)）正交：那些賣的是「千環並行吞吐 + 可微梯度」，Isaac Sim 賣的是 **RTX 光線追蹤 photoreal 渲染 ×（PhysX 5 / Newton 可選）成熟物理 × OpenUSD 場景授權 × Replicator 合成標籤管線**。對「物理可控生成」這本手冊，它的 USP 只有一條：它是 **NVIDIA 資料工廠的渲染層，也是 [Cosmos](../foundation-physics-models/cosmos-wfm.md) 的直接上遊** —— Replicator 內建的 **`CosmosWriter`** 把 Isaac Sim 渲出來的 RGB / depth / segmentation / Canny-edge 五路 AOV 直接輸出成影片，餵 **Cosmos-Transfer**（sim→real 貼皮），這就是官方文件白紙黑字的 data-factory pipeline。代價：**重 GPU / 必須 RTX 光追硬體**、Omniverse 與 OpenUSD 學習曲線陡、**PhysX 路徑不可微**（可微的未來押在 Newton，仍 roadmap / 實驗性）、版本churn 大（Isaac Gym → Isaac Lab、Kit 引擎版號跳動）。一句話：在這本手冊裡，Isaac Sim 是「**古典渲染那條外觀供應線的本體 + Cosmos 的料源**」，不是拿來訓物理生成模型本身。

## 2. Core mechanism

Isaac Sim 不發明 solver，而是把四層既有 NVIDIA 技術疊成一條「authored 3D 場景 → 確定性 RTX 渲染 + 物理 → 免費完美標籤 → Cosmos 上遊」的管線：

```
        ┌──────────────────────── NVIDIA Isaac Sim 6.0 ────────────────────────┐
        │                                                                       │
   USD ─┼─▶ OpenUSD 場景（Omniverse 平台）── 資產 / 燈光 / 材質 / 機器人 URDF→USD │
        │                    │                                                  │
        │   ┌── PhysX 5 ─────▼── 剛體 / articulation / GPU dynamics ──┐         │
        │   │  （或 Newton：Warp-based，6.0 起可選 backend，experimental）│       │
        │   └────────────────┬──────────────────────────────────────────┘       │
        │                    │                                                  │
        │   ┌── RTX 光線追蹤 render（path-traced AOV）──────────────────┐        │
        │   │  RGB · depth · instance/semantic seg · normals · edge ··· │        │
        │   │  6.0 新增：NuRec 3DGS 經 Fabric Scene Delegate 進場景      │        │
        │   └────────────────┬──────────────────────────────────────────┘        │
        │                    │                                                  │
        │   ┌── Replicator（合成資料層, Python API）────────────────────┐         │
        │   │  domain randomization（材質/燈光/物理 propery 隨機化）     │         │
        │   │  Writers：BasicWriter / KITTI / COCO / ★ CosmosWriter      │         │
        │   └────────────────┬──────────────────────────────────────────┘         │
        │                    │  CosmosWriter：RGB / depth / seg / shaded-seg / Canny
        │                    │  → MP4（每路一檔）+ PNG 序列，1280×720（範例）       │
        └────────────────────┼───────────────────────────────────────────────────┘
                             ▼
              Cosmos-Transfer（Multi-ControlNet：vis/edge/depth/seg 各路 0.0–1.0 權重）
                             ▼  sim→real 貼皮：低保真渲染 → photoreal
                        photoreal 合成訓練資料
```

關鍵設計選擇：

- **OpenUSD 為單一真相**：所有資產先轉 USD 才能進場景；場景組合 / 引用 / layer 都是 USD composition，這是 Omniverse 與一般 game-engine 最大的差別（也是學習曲線的來源）。
- **確定性 RTX 渲染 = 免費完美標籤**：因為場景是 authored 的，depth / segmentation / optical flow / 2D-3D bbox 全是 ground-truth，零標註成本 —— 這正是「古典渲染外觀供應線」的本錢（見 [3d-aware § 外觀邊供應線](../3d-aware-generation/overview.md)）。
- **`CosmosWriter` = 上遊接口**：官方 Replicator tutorial 直接寫「emits canny / depth / segmentation AOVs as videos ready as inputs for Transfer1」；五路模態各對應 Cosmos-Transfer 的一條 ControlNet 分支。這是 Isaac Sim 對本手冊的唯一 load-bearing 事實。
- **多 physics backend（6.0）**：PhysX 5（GPU 加速剛體，預設）與 **Newton**（Warp-based、differentiable 路線）可選；**PhysX 不可微**（見 §4），可微仍靠 Newton 整合（arXiv `2511.04831` Isaac Lab paper 點出此分工，Newton GA 排在 2026 GTC）。
- **三層 NVIDIA stack**：Omniverse（USD 平台 / 中樞）→ **Isaac Sim（sim + 渲染）** → **Isaac Lab** RL 框架（取代 Isaac Gym + Orbit）。Isaac Sim 是中間那層 sim+render，不是 RL 框架本身。

## 3. 五軸定位 + 同軸對手

| 軸 | Isaac Sim | UE（AirSim / CARLA） | Gazebo | MJX / [Genesis](./genesis.md)（研究 diff-sim） |
|---|---|---|---|---|
| Output | **N/A**（state + photoreal RGB/depth/seg AOV） | N/A（Unreal 渲染） | N/A（Gz/OGRE 渲染，較不 photoreal） | N/A（state + 簡易/批渲染） |
| Injection | **sim-in-loop**（PhysX 5 / Newton） | sim-in-loop（Chaos physics） | sim-in-loop（DART/ODE/Bullet 可選） | sim-in-loop（+ 可微 hard-constraint） |
| Control | action / trajectory / force / param | action / trajectory | action / trajectory | action / trajectory / force / contact / param |
| Temporal | streaming | streaming | streaming | streaming |
| Domain | **robotics / rigid（+ photoreal 渲染）** | robotics / driving / aerial | robotics / rigid | robotics / rigid（+ soft，研究向） |
| 渲染保真度 | **RTX 光追 photoreal（最高一檔）** | 高（game-engine PBR） | 中（較假） | 低 / 功能性（非賣點） |
| 可微 | ❌（PhysX）；Newton roadmap | ❌ | ❌ | ✅（MJX/Genesis-MPM 部分可微） |
| 合成資料管線 | **✅ Replicator + CosmosWriter（USP）** | 部分（自寫） | 弱 | 弱（自寫 renderer 接管線） |

**同軸對手分群**：

- **「game-engine photoreal sim」**：UE-based 的 **AirSim**（Microsoft 已 archive 2022）/ **CARLA**（自駕、UE + Chaos physics）。渲染保真度與 Isaac Sim 同檔，但**沒有 OpenUSD 資產生態 + 沒有 Replicator→Cosmos 這條官方料源管線**，且物理是 Chaos 不是 PhysX 5。
- **「ROS 原生、輕量」**：**Gazebo（Gz）**，DART/ODE/Bullet 物理可選，社群最廣、最省 GPU，但**明顯不 photoreal**，做 vision sim2real 要另接增強。
- **「研究型 diff / GPU 並行 sim」**：[MJX](./mujoco-mjx.md) / [Genesis](./genesis.md) / [Warp](./nvidia-warp.md) / [Aerial Gym](./aerial-gym.md) —— **不同類別**：它們賣可微梯度 + 千環吞吐，渲染只是功能性；Isaac Sim 賣 photoreal + 資料工廠，不賣可微。要 first-order policy gradient 或 contact 梯度，Isaac Sim 直接出局。

Isaac Sim 的獨佔位置 = **「photoreal RTX 渲染 × OpenUSD 場景生態 × Replicator 合成標籤 × 官方直連 Cosmos」四個都要的時候沒對手**；少任一條（不需 photoreal / 不需 Cosmos 料源 / 需要可微）對手都更便宜。

## 4. ⚡ shines / ❌ breaks

### ⚡ 真正領先的 regime

- **合成資料工廠 + Cosmos 上遊**：Replicator 出 domain-randomized photoreal 資料 + 完美標籤；`CosmosWriter` 五路 AOV（RGB / depth / seg / shaded-seg / Canny-edge）直出 MP4 餵 Cosmos-Transfer。這條 **sim 出粗糙 → Transfer 貼皮** 是 NVIDIA / Wayve 採用的 production pattern（見 [Cosmos § 自駕 closed-loop](../foundation-physics-models/cosmos-wfm.md)）。
- **免費完美標籤**：authored USD 場景 → 確定性 RTX 渲染 → depth / seg / flow / bbox 全 ground-truth，零標註成本，這是真實資料拿不到的（occlusion / amodal mask 都免費）。
- **OpenUSD 工業生態 + 6.0 NuRec 3DGS**：6.0 經 Fabric Scene Delegate 把 **NuRec 3D Gaussian Splatting** 真實重建場景塞進 USD 與 mesh 共存（多 GPU、燈光互動、MaterialX）—— 等於把「重建外觀供應線」（3DGS）併進「古典渲染外觀供應線」同一場景。
- **完整機器人資產 + sim-in-loop 物理**：URDF→USD、PhysX 5 GPU dynamics、teleop demo 採集、ROS 橋接 —— digital-twin / robotics-data-gen 落地齊全（見 [use-cases robotics-data-gen](../../use-cases/robotics-data-gen/overview.md)）。

### ❌ Known failure modes

- **不可微（PhysX 路徑）**：要 first-order policy gradient / 可微 system-id / 差分控制器，Isaac Sim PhysX 出局；可微只能等 Newton 整合（experimental / roadmap，arXiv `2511.04831`）。對比 MJX / Genesis-MPM / DiffTaichi 是結構性劣勢。
- **重 GPU / 必須 RTX 光追**：官方系統需求最低 **RTX 3070 / 8GB VRAM / 32GB RAM**，舒適 RTX 4080 / 16GB，理想 RTX 6000 Ada / 48GB；**無 RT Core 的卡（A100 / H100）官方不支援**（`UNVERIFIED：6.0 是否放寬此限`）。對純算力訓練機（A100/H100 farm）是硬牆。
- **Omniverse + OpenUSD 複雜度**：所有資產要先轉 USD；場景是 USD composition（layer / reference / variant），不是「拖個模型進場」。初次上手成本遠高於 Gazebo / PyBullet。
- **版本 churn 大**：Isaac Gym（deprecated）→ Isaac Lab；Kit 引擎版號（6.0 = Kit 110）跳動；6.0 一度是 Early Developer Release（需從 source build）才轉 GA。新驅動相容性踩雷（見 §8）。
- **sim2real domain gap**：RTX 再 photoreal 仍「看起來假」（材質 / 光照 / 感測器噪聲分布差），這正是 Cosmos-Transfer / 3DGS 重建 / Carla2Real 存在的理由 —— Isaac Sim 自己 own「可控但假」，真實感要往下游貼皮（見 [3d-aware overview 三條供應線](../3d-aware-generation/overview.md)）。
- **多 viewport / 新硬體崩潰**：RTX 5080（Blackwell）加第二 viewport 觸發 `ERROR_DEVICE_LOST`（issue #264）；RTX Lidar + GPU dynamics 同開會 crash（官方 known-issues）。

## 5. Reproduction notes

- **GPU 預算**：最低 RTX 3070（8GB）；做 Replicator photoreal SDG 建議 ≥ RTX 4080（16GB）；多相機 / 大場景 / NuRec 3DGS 走 RTX 6000 Ada（48GB）。驅動 Linux ≥ 535.129.03（官方建議；新驅動如 595.79 反而踩雷見 §8）。
- **安裝**：6.0 起 GitHub `isaac-sim/IsaacSim` 開源（Apache-2.0），可 pip / container / 從 source build；早期 6.0 Early Developer Release 須 build from source，GA（2026-06-04）後有正式發行。
- **CosmosWriter 最小管線**：
  1. 搭 / 引用 USD 場景（倉庫導航 demo 是官方範例）
  2. `rep.create.render_product(camera_path, (1280, 720))` 建 render product
  3. 掛 `CosmosWriter` → 輸出 RGB / depth / seg / shaded-seg / Canny 五路（MP4 + PNG 序列）
  4. 五路餵 Cosmos-Transfer，各 ControlNet 分支 `0.0–1.0` 權重調 adherence vs 創造力
- **典型踩坑**：
  - 新 NVIDIA 驅動（如 595.79）Isaac Sim 偵測不到 CUDA → 降回 580（issue #537）
  - RTX Lidar + GPU dynamics 同開 crash → 關 GPU dynamics、broad-phase 設 MBP（官方 known-issues）
  - aarch64（Jetson / GB200）pip 裝 6.0.0 失敗（IsaacLab issue #5053）
  - 第一次起 Omniverse Kit 載 shader cache 很慢，別當當機
- **NVIDIA 釋出範圍**：simulator + Replicator + CosmosWriter tutorial + 機器人資產庫（USD）+ ROS 橋；Cosmos 權重在 Cosmos repo 另取（見 [cosmos-wfm](../foundation-physics-models/cosmos-wfm.md)）。

## 6. Cross-line synthesis

對本 handbook（generation 視角）的接點：

- **vs [Cosmos](../foundation-physics-models/cosmos-wfm.md)（直接上遊）**：Isaac Sim 是 Cosmos 的**料源**，不是對手。NVIDIA 線標準 pattern：Isaac Sim Replicator 出 (state, RGB, depth, seg, edge) → `CosmosWriter` 打包 → **Cosmos-Transfer** 貼 photoreal 皮。axis 2 從 `data-only` 跳到 `sim-in-loop` 的 cleanest production 接法（見 [Cosmos §10 cross-line](../foundation-physics-models/cosmos-wfm.md)）。
- **vs 外觀邊三條供應線**：Isaac Sim = ① **古典渲染**那條的本體（authored 場景 + 確定性 RTX render + 內建物理）；它的 sim2real domain gap 正是 ② **重建**（[3DGS](../3d-aware-generation/generative-gaussian-splatting.md)）與 ③ **生成**（Cosmos）存在的理由。6.0 的 NuRec 3DGS 把 ② 併進同一 USD 場景（見 [3d-aware § 外觀邊供應線](../3d-aware-generation/overview.md)）。
- **vs 研究 diff-sim**：[MJX](./mujoco-mjx.md) / [Genesis](./genesis.md) / [Aerial Gym](./aerial-gym.md) 是**不同類別**（可微 + 千環吞吐 vs photoreal 資料工廠）。要 contact 梯度 / first-order PG 走前者；要 photoreal 合成資料 + Cosmos 料源走 Isaac Sim。本區 overview 的「生產 sim 平台」段把這條分工講清。
- **與 4 條路線怎麼接**：
  - **pixel-WM**：Isaac Sim → CosmosWriter → Cosmos-Transfer 是教科書接法（本身就是 NVIDIA pixel-WM 資料管線）
  - **latent-WM**：USD 場景 rollout 出 (obs, action) 餵 [DreamerV4](../latent-world-models/dreamer-v4.md)-style latent 訓練（`UNVERIFIED：是否有官方 Isaac→latent-WM recipe`）
  - **diff-sim**：❌ PhysX 不可微出局；可微等 Newton
  - **neural surrogate**：PhysX rollout 可蒸 surrogate dynamics，但非 Isaac Sim 賣點
- **與 use-cases 接**：robotics-data-gen / digital-twin / autonomous-driving-sim 的合成資料引擎（見 [robotics-data-gen overview](../../use-cases/robotics-data-gen/overview.md)、[autonomous-driving-sim overview](../../use-cases/autonomous-driving-sim/overview.md)）。

## 7. References

**Canonical / 官方**
- Isaac Sim Documentation 6.0.0 — "What Is Isaac Sim?". https://docs.isaacsim.omniverse.nvidia.com/6.0.0/index.html
- Isaac Sim 6.0 GA 公告（2026-06-04）— GitHub Discussion `isaac-sim/IsaacSim#655`；Apache-2.0（`isaac-sim/IsaacSim/blob/main/LICENSE`）
- **Cosmos Synthetic Data Generation（CosmosWriter tutorial）** — Replicator → Cosmos-Transfer 官方管線，五路 AOV / MP4+PNG / 1280×720 / ControlNet 權重。https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_cosmos.html
- Isaac Sim 系統需求（最低 RTX 3070/8GB，理想 RTX 6000 Ada/48GB；無 RT Core 卡不支援）— docs.isaacsim … /installation/requirements
- Isaac Sim Known Issues — docs.isaacsim.omniverse.nvidia.com/latest/overview/known_issues.html

**Anchor 二手 / 鄰近**
- Mittal et al. "Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning." arXiv [2511.04831](https://arxiv.org/abs/2511.04831)（2025）— Isaac Sim 上的 RL 框架；PhysX vs Newton 分工、PhysX 不可微出處
- Makoviychuk et al. "Isaac Gym." NeurIPS 2021 D&B, [2108.10470](https://arxiv.org/abs/2108.10470)— deprecated 前身（GPU 千環，不談梯度）
- Newton（Linux Foundation 2025-09，Warp-based、differentiable，GTC 2026 GA 排程）— Isaac Sim 可微未來
- "NVIDIA's Isaac Sim 6.0 Ships With NuRec Gaussian Splatting." radiancefields.com（2026）— 6.0 NuRec 3DGS / Kit 110 第三方報導
- "NVIDIA Isaac Sim, Omniverse, and Cosmos – The Robotics & AI Simulation Ecosystem Explained." ridgerun.ai — 三層 stack 釐清

## 8. §8 Pitfall log

| # | Issue / 來源 | Severity | 摘錄 | Workaround |
|---|---|---|---|---|
| 8.1 | PhysX 不可微（arXiv `2511.04831`） | **High**（路線選擇） | Isaac Sim 預設 PhysX 路徑無解析梯度；可微靠 Newton 整合（experimental / roadmap） | 要 first-order PG / contact 梯度走 [MJX](./mujoco-mjx.md)/[Genesis](./genesis.md)；可微等 Newton GA |
| 8.2 | 無 RT Core 卡不支援（官方需求） | **High**（硬體） | A100 / H100 等無 RT Core 卡官方不支援；最低 RTX 3070/8GB | 渲染機用 RTX 4080+/Ada；純算力 farm 不能跑 Isaac Sim 渲染（`UNVERIFIED：6.0 是否放寬`） |
| 8.3 | issue [#537](https://github.com/isaac-sim/IsaacSim/issues/537) 新驅動偵測不到 CUDA | High（裝機） | 升 NVIDIA 595.79 後 Isaac Sim 偵測不到 CUDA、起不來 | 降回驅動 580；釘官方建議驅動版本 |
| 8.4 | issue [#264](https://github.com/isaac-sim/IsaacSim/issues/264) RTX 5080 第二 viewport 崩 | Medium（新硬體） | Blackwell（RTX 5080）加第二 viewport → `ERROR_DEVICE_LOST` | 暫用單 viewport；等 Blackwell 路徑修好 |
| 8.5 | RTX Lidar + GPU dynamics crash（官方 known-issues） | Medium | RTX Lidar 與 GPU dynamics 同開觸發 crash | 關 GPU dynamics、broad-phase 設 MBP |
| 8.6 | IsaacLab issue [#5053](https://github.com/isaac-sim/IsaacLab/issues/5053) aarch64 裝不上 | Medium（邊緣裝置） | aarch64（Jetson / GB200）pip 裝 6.0.0 失敗 | 用官方 container / 等 aarch64 wheel；x86_64 不受影響 |
| 8.7 | OpenUSD / Omniverse 學習曲線 | Medium（上手） | 資產須先轉 USD；場景 = USD composition，非「拖模型進場」 | 先跑官方 warehouse / Replicator tutorial 建心智模型；別硬套 game-engine 直覺 |
| 8.8 | sim2real domain gap（RTX 仍「假」） | High（fidelity） | photoreal 渲染仍與真實感測分布有 gap（材質/光照/噪聲） | 下游接 Cosmos-Transfer 貼皮 / [3DGS](../3d-aware-generation/generative-gaussian-splatting.md) 重建 / domain randomization 拉寬分布 |
| 8.9 | 版本 churn（Gym→Lab、Kit 版號、Early→GA） | Low–Medium（長期） | Isaac Gym deprecated；6.0 一度 Early Developer Release 須 source build | 釘 GA 版本（6.0.0 GA 2026-06-04）；別跟 nightly / early dev |

---

*寫作日期：2026-06-15。Isaac Sim / Newton / CosmosWriter 都在快速演進 —— Newton 可微整合進度、6.0 對 H100 渲染的支援、CosmosWriter 對 Transfer2.5 的對接細節，建議 6 個月內回查官方 changelog。標 `UNVERIFIED` 處未在官方文件二次確認。*
