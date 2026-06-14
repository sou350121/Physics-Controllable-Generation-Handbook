# Bridge: Aerial Embodiment — 生成端造資料 × 感知端消費資料

> **本倉 (Physics-Gen) aerial-sim = upstream data engine** · **Spatial-Handbook `embodiments/aerial` = downstream consumer**（VIO / dynamics / avoidance）
> **同一架無人機、相反的資料流向**：generation 端**生**訓練/驗證 footage，perception 端**吃**它去跑 VIO / 控制 / 避障。

**Status:** v1 — opinionated draft. 對應 use-case [`../use-cases/aerial-sim/overview.md`](../use-cases/aerial-sim/overview.md) §「與 sister handbook 的 bridge」留的 hook（原標「待寫」）。跨倉 benchmark / 數字一律標 `UNVERIFIED`，無捏造。

**TL;DR:** 兩冊收的是**同一架無人機、相反的資料端**。Physics-Gen 的 aerial-sim（Cosmos FPV fine-tune、Dream-to-Fly action-WM、Aerial Gym/Flightmare）是**資料生產者**：吐出 FPV / overhead footage + IMU-一致軌跡 + 多感測 stack。Spatial 的 `embodiments/aerial`（VIO、dynamics primer、event-camera、real-flight gotchas）是**資料消費者**：拿這些去訓 VINS-Fusion / OpenVINS / DROID-SLAM，去驗避障 policy。能不能接得起來，**不取決於畫面多 photoreal，而取決於兩端有沒有 agree on 同一套 IMU 噪聲模型 + camera-IMU extrinsic**——這是整篇唯一真正會爆的 seam，也是 §3 的核心。畫面對、IMU 假，generated VIO 資料就是**廢的**（trains VIO to estimate a sensor that doesn't exist）。

---

## 1. TL;DR（一句版）

> **Physics-Gen 端管 appearance，Spatial 端管 dynamics；兩端在 IMU contract 上握手，否則生成的 VIO 資料不可用。**

- **生成端不該重講 perception 端的東西**：VIO 演算法內部、quadrotor 動力學/控制、min-snap 軌跡優化、避障 policy 設計、真飛 ops 坑——這些是 Spatial `embodiments/aerial` 的領地，生成端只**消費它的約束**，不重新解釋。
- **生成端該交付的是「資料」**：IMU-一致的軌跡 footage、vision+IMU+depth 多感測 stack、3D map / occupancy、event stream、swarm 資料。
- **物理由 dynamics 擁有，不由 video model 擁有**：見 Physics-IQ 的教訓（realism ≠ physics）——影片模型生得出「看起來像在飛」的畫面，但**不保證軌跡服從四旋翼動力學**；那是 Spatial dynamics primer 的硬約束（§3）。

---

## 2. 兩端契約（interface table）

每一條 handoff：生成端**提供**什麼、感知端**消費**什麼、seam 在哪。

| 契約欄位 | Physics-Gen 端（aerial-sim）提供 | Spatial 端（`embodiments/aerial`）消費 | seam 在哪 |
|---|---|---|---|
| **視覺 footage** | FPV / overhead 影格（Cosmos fine-tune、Dream-to-Fly rendered frames） | perception backbone pre-train、gate / 小目標偵測 | ✅ 對齊：rendered frame 拿來 pre-train 視覺已被多次驗證 |
| **IMU 軌跡** | 與影格時間對齊的 accel / gyro 序列 | VIO（VINS-Fusion / OpenVINS）的 inertial 通道 | 🔴 **最大 seam**：兩端 IMU 噪聲模型一旦不一致，generated VIO 資料直接報廢（§3） |
| **camera-IMU extrinsic** | 生成時隱含的 sensor 幾何（多半未明確宣告） | VIO 標定（`T_cam_imu`）、tight-coupling 前提 | 🔴 extrinsic 不對齊 → VIO 學到錯的剛體變換，real 機上發散 |
| **軌跡動力學可行性** | 影片 prior 外推的相機運動 | dynamics primer：4-control / 6-DoF under-actuated、~200 Hz state、<10 ms latency | ⚠ video model **無法 enforce** 動力學；生成軌跡可能物理不可飛（§3） |
| **depth / 3D map / occupancy** | 多感測 stack 的深度通道、occupancy | 避障 policy、SLAM map 對照 | ⚠ 生成深度多為 static、無 metric scale 保證；避障需真實尺度 |
| **event stream** | （目前**無**生成端覆蓋） | Spatial 有 `event_camera_for_aerial` 解構（lighting-OOD 修法） | 🔴 整段空白——乾淨機會（§5b） |
| **swarm / multi-drone** | （兩端**皆空**） | Spatial `swarm/` 為空 | 🔴 rotor-rotor interaction 兩端都未解（§5a） |
| **raw-camera 部署** | Dream-to-Fly 只證過 rendered-frame HIL | 真飛需 real-camera 影像 | ⚠/`UNVERIFIED` 是否有人證過 rendered→real-camera 閉環（§5c） |

---

## 3. 核心契約：IMU 噪聲模型 + camera-IMU extrinsic（為何不對齊就廢）

這是整篇的 intellectual core。其他欄位錯了頂多「資料品質差」；**這一欄錯了，generated VIO 資料是負資產**——它會主動把 VIO 教歪。

**VIO 在估什麼。** Visual-inertial odometry 不是「看影片估位置」，而是把**相機觀測**與**IMU 預積分**做 tight coupling：IMU 在影格之間積分出運動先驗，相機 feature 再修正它。整個融合的數學前提是一組**已知且固定的 IMU 噪聲參數**（accel / gyro 的 noise density、random-walk bias instability）與**已知的 camera-IMU extrinsic**（`T_cam_imu`，相機到 IMU 的剛體變換 + 時間 offset）。VINS-Fusion / OpenVINS 都把這些當**標定常數**餵進 estimator。

**為什麼生成端一錯就廢——具體機制：**

1. **IMU 噪聲模型不一致 → estimator 的不確定度全錯。** 若生成端用了一個「乾淨」或隨手設的 IMU 模型，但消費端 VIO 假設的是真機那組（含 prop-vibration 引入的 bias drift），那麼：generated 資料裡 IMU 看起來太準 → VIO 學會**過度信任 IMU**、欠信視覺 → 一上真機（IMU 髒）就發散。反過來，生成端噪聲過大也會把 VIO 教成欠信 IMU。**VIO 的權重全靠噪聲模型決定，餵錯模型 = 餵錯權重先驗。**
2. **camera-IMU extrinsic 不一致 → 學到錯的剛體變換。** 影格與 IMU 軌跡若不是用**同一組 `T_cam_imu`** 生出來的，視覺位移與慣性位移在資料裡就「對不上」一個固定變換。VIO 會把這個矛盾吸收成一個**錯誤的 extrinsic 估計**或殘留誤差；真機上 extrinsic 是另一個值 → 系統性漂移。
3. **時間同步（time offset）是隱形殺手。** 影格時間戳與 IMU 時間戳若沒對齊到生成端宣告的 offset，等於人為注入一個 cam-IMU 時延 → tight-coupling 的雅可比全錯。

**Spatial 的 failure atlas 把這些寫成了真實故障模式**：`vio/github_failure_atlas` 記了 prop-vibration 引起的 IMU bias drift、static-init divergence 等真機坑。**生成端必須對著這份 atlas 設計**——目標不是生「乾淨的 IMU」，而是生**和真機同分布的髒 IMU**（含螺旋槳振動譜、起飛瞬態），否則 VIO 在 sim 裡學到的魯棒性根本沒涵蓋真機會遇到的退化。

**一句話契約：** generated VIO footage 可訓的**充要條件**是——影格、IMU 序列、extrinsic、time offset **全部由同一組、且與消費端 VIO 標定一致的 sensor 參數生成**。少一個對齊，這批資料就不是「品質差一點」，而是**會主動損害 VIO**。`UNVERIFIED`：目前是否有公開 aerial WM 明確宣告其 IMU 噪聲參數 + extrinsic 供下游核對——據本側資料未見標準做法。

---

## 4. Handoff map（什麼東西、往哪個方向過）

```
Physics-Gen aerial-sim  (PRODUCER)        Spatial embodiments/aerial  (CONSUMER)
──────────────────────────────────        ──────────────────────────────────────
文字 / 單圖 / camera traj / action                真實/合成感測 → 演算法 → 控制
   │ Cosmos FPV · Dream-to-Fly · Aerial Gym          │ VIO · dynamics · avoidance
   ▼                                                  ▼
 ┌─ FPV / overhead footage ─────────────► perception pre-train          (✅)
 ├─ IMU-一致軌跡 ──────────[同一 IMU 模型]──► VIO 訓練/驗證               (🔴 契約見 §3)
 ├─ camera-IMU extrinsic ─────────────────► VIO 標定                    (🔴)
 ├─ multi-sensor stack (vision+IMU+depth) ─► 多感測融合                  (⚠)
 ├─ 3D map / occupancy ───────────────────► 避障 / SLAM 對照            (⚠ scale)
 ├─ event stream ─────────────────────────► event-camera VIO           (🔴 生成端空白)
 └─ swarm 資料 ───────────────────────────► swarm                      (🔴 兩端皆空)

反向（CONSUMER → PRODUCER，生成端應「消費」其約束、不重講）：
 ◄─ VIO 演算法內部 · quadrotor 動力學/控制 · min-snap 軌跡優化 ·
    避障 policy 設計 · 真飛 ops 坑  ── 這些是 Spatial 的領地，生成端讀它來約束自己。
```

方向規則（與 [`overview.md`](overview.md) 寫作 rule 一致）：**生成端是 producer，只交付資料 + 宣告 sensor 契約；演算法側的一切由 Spatial 擁有，本倉不重新解釋。**

---

## 5. 開放 seam（未解）

### 5a · 🔴 Swarm —— 兩端都空

Spatial `embodiments/aerial/swarm/` 目前是空的；生成端這邊 Aerial Gym 自己也把 **rotor-rotor aerodynamic interaction** 標為未解。所以「生成 swarm 訓練資料」這條路**兩端同時缺**：沒有消費端 spec 可對齊，也沒有生產端模型能生對多機尾流耦合。這不是工程坑，是 open frontier——**誰先定義 swarm 的 sensor + 互擾契約，誰就能讓另一端動起來**。

### 5b · 🔴 Event-stream synthesis —— 乾淨機會

Spatial 有 `event-camera/event_camera_for_aerial_dissection.md`：event camera 是被點名的 **lighting-OOD 修法**，是 Swift 的 #1 failure（極端光照下 frame camera 失效）。**但生成端完全沒有合成 event 資料的覆蓋。** 這是本 bridge 最乾淨的機會缺口：消費端已經把「為什麼需要 event」「event 怎麼救 lighting-OOD」講透了，等的就是一個能生**時間一致 event stream**（而非把 RGB 影片事後 diff 成假 event）的生成端方法。

### 5c · ⚠ Raw-camera visual gap —— 閉環未證

Dream-to-Fly 證的是 **rendered-frame** 的 HIL（Hardware-in-the-loop）閉環，**不是 real-camera 部署**。從「生成/渲染影格驅動 policy」到「真實相機影格驅動同一 policy」之間的 visual gap 有沒有被閉合——`UNVERIFIED`，本側無公開 work 可佐證。這條一旦閉合，生成端就從「視覺 augmentation」升級為「可直接驅動真飛 control 的資料源」。

---

## Boundary / References

**本倉（生成端，PRODUCER）：**
- aerial-sim 總覽 → [`../use-cases/aerial-sim/overview.md`](../use-cases/aerial-sim/overview.md)
- Dream to Fly 解構 → [`../use-cases/aerial-sim/dream-to-fly.md`](../use-cases/aerial-sim/dream-to-fly.md)
- 合成 aerial 訓練資料 → [`../use-cases/aerial-sim/generative-aerial-data.md`](../use-cases/aerial-sim/generative-aerial-data.md)
- 同類 bridge（同表徵/相反方向的寫法範本）→ [`3d-aware-video-gen.md`](3d-aware-video-gen.md) · [`overview.md`](overview.md)

**Spatial-Handbook（感知端，CONSUMER）：**
- aerial 總覽 → https://github.com/sou350121/Spatial-Intelligence-Handbook/tree/main/embodiments/aerial
- VIO（消費 generated footage 的那一端）→ https://github.com/sou350121/Spatial-Intelligence-Handbook/tree/main/embodiments/aerial/vio
- dynamics & control primer（§3 動力學硬約束來源）→ https://github.com/sou350121/Spatial-Intelligence-Handbook/blob/main/embodiments/aerial/dynamics_and_control_primer.md
- event-camera 解構（§5b 缺口的消費端 spec）→ https://github.com/sou350121/Spatial-Intelligence-Handbook/blob/main/embodiments/aerial/event-camera/event_camera_for_aerial_dissection.md
- real-flight production gotchas（真飛 ops 坑，生成端應消費）→ https://github.com/sou350121/Spatial-Intelligence-Handbook/blob/main/embodiments/aerial/real_flight_production_gotchas.md

**外部 anchor（事實標 `UNVERIFIED` 者為本側未直接核對）：**
- Swift (Champion-Level Drone Racing) — UZH RPG, **Nature 2023-08-31**（event-camera 為其 lighting-OOD 修法）
- Dream to Fly — UZH RPG, **arXiv 2501.14377** (2025)，rendered-frame HIL 閉環 `UNVERIFIED real-camera 是否閉合`
- Aerial Gym Simulator — NTNU ARL, **arXiv 2305.16510 / 2503.01471**（rotor-rotor interaction 標未解）
- Physics-IQ（realism ≠ physics 的教訓來源）— `UNVERIFIED canonical link`，詳見本倉 foundations 對應 dissection
- VINS-Fusion / OpenVINS / DROID-SLAM（消費端 VIO/SLAM）— 解構在 Spatial-Handbook，非本倉
