<!-- ontology-5axis output=pixel-video|3d-explicit injection=data-only control=image-init|trajectory temporal=clip-parallel domain=robotics -->

# Generative Aerial Data — 外觀靠生成、動力學靠物理 解構

> 本篇是 generation 端對 aerial 的 **核心論點落地**：**生成負責 APPEARANCE（外觀、novel-view、可標註偵測資料、photoreal domain randomization），物理負責 DYNAMICS（推力、阻力、重力、螺旋槳尾流、可靠的 metric scale）**。為什麼進 aerial-sim anchor 名單？因為 aerial 把這個 contract 逼到最尖：6-DoF 自由運動 + 「掉下來就壞」的硬約束，意味著任何讓 generation 去「猜動力學」的捷徑都會在真實飛行裡墜機。而每一個 *已驗證* 的 aerial 結果（SOUS VIDE/FiGS、FlightDiffusion、UAV-Sim）都把 dynamics 外包給一個明確的物理模型，generation 只碰外觀 —— 這不是巧合，是這條路目前唯一 work 的架構。
> This dissection applies the handbook's central thesis to aerial: **generation owns appearance, physics owns dynamics.** It earns an anchor slot because aerial is where the contract is most load-bearing — a policy that trusts hallucinated dynamics does not get a soft failure, it crashes. The split is the lesson, not an implementation detail.
> Dynamics substrate（物理 sim 本身的拆解）由 [aerial-sim-stack.md](./aerial-sim-stack.md) 擁有；本篇 **不重複** teardown 物理 sim，只把它當作 generation 配對的底座引用。

## 1. TL;DR — the split + the strongest proof

**論點裁決：強支持（strongly supported）。** 在所有 *已驗證* 的 aerial 系統裡，DYNAMICS 一律來自物理模型，generation 只生 APPEARANCE。最乾淨的證明是 **SOUS VIDE / FiGS**（arXiv [2412.16346](https://arxiv.org/abs/2412.16346)）：飛行動力學 = 10 維半運動學四旋翼模型（ACADOS integrator）；視覺觀測 = 3D Gaussian Splatting（最高 130 fps 渲染）。兩者 **嚴格不互碰** —— 3DGS 從不接觸 dynamics，物理從不接觸 appearance。視覺運動策略 SV-Net 在 100k–300k 合成 image/state-action pair 上訓練，**zero-shot** 部署，105 次真實飛行；ablation 5/5（0 碰撞）、novel task 51/60（85%），且能抗 30% 質量變化、40 m/s 陣風、60% 亮度變化。

The strongest single proof: **a 3DGS renderer and a quadrotor physics model wired side-by-side, never crossing, yields a zero-shot real-world policy.** That is the thesis, validated.

```
   ┌──────────────────────────── THE SPLIT PIPELINE ───────────────────────────┐
   │                                                                            │
   │   APPEARANCE  (generated / neural-rendered)   DYNAMICS  (simulated)        │
   │   ─────────────────────────────────────────   ──────────────────────────  │
   │                                                                            │
   │   3DGS / NeRF / video-diffusion                physics model               │
   │     ├─ photoreal RGB frame                       ├─ thrust / drag / gravity│
   │     ├─ novel-view (15m→50m altitude)             ├─ prop-wake / ground eff.│
   │     ├─ labeled detection bbox                    ├─ wind gust / turbulence │
   │     └─ domain randomization (light/texture)      └─ rigid-body 6-DoF state │
   │              │                                            │                │
   │              │  image_t                       state_t / action_t           │
   │              ▼                                            ▼                 │
   │        ┌─────────────────────────────────────────────────────┐            │
   │        │  PAIR  (image_t , state_t , action_t)                │            │
   │        │  → visuomotor policy training / perception training  │            │
   │        └─────────────────────────────────────────────────────┘            │
   │              ▲                                            ▲                 │
   │   generation NEVER writes here ─────────────┘   physics NEVER writes here  │
   │   (no dynamics)                                 (no pixels)                 │
   └────────────────────────────────────────────────────────────────────────────┘
   契約：generation 提供 image_t（外觀）；物理提供 state_t/action_t（動力學）；
   兩條流在 pairing 處匯合，但各自的「筆」不越界。
```

## 2. The contract — what generation / neural-rendering CAN vs CANNOT provide

這是進場前要簽的契約。把 generation 當外觀引擎用，它很強；把它當動力學引擎用，它會騙你。

| 能力 | **CAN provide（generation/neural-rendering）** | **CANNOT provide（需物理模型/感測）** |
|---|---|---|
| Appearance / photorealism | ✅ photoreal RGB frame、紋理、光照變化（domain randomization） | — |
| Novel-view synthesis | ✅ 從低空訓練影像合成高空視角（UAV-Sim：15m→50m） | — |
| Labeled detection data | ✅ 自帶 bbox 的合成偵測資料（NeRF/3DGS 已知幾何 → 可投影標註） | — |
| Lighting / weather OOD | ✅ 亮度/天候/季節 randomization 擴 perception 分布 | — |
| Thrust / drag / gravity | — | ❌ 必須來自 rigid-body 物理模型 |
| Prop-wake / ground effect | — | ❌ 主流 video WM（Cosmos/Sora）分布外，幾乎為 0 |
| Wind gust / turbulence | — | ❌ first-class 物理擾動量，非「噪聲」 |
| Reliable metric scale | — | ❌ monocular SfM/3DGS 只到任意尺度（見 §5）；絕對尺度需 LiDAR/metric-depth |
| 6-DoF action consistency | △ 可條件於軌跡生影片（外觀層） | ❌ 不保證符合可飛動力學（FlightDiffusion 是 *事後* 檢查，非 *強制*） |

> 契約一句話：**generation 賣的是 photons，不是 forces。** 把 forces 的需求轉嫁給它，等於把墜機風險寫進 training set。
> Contract in one line: **generation sells you photons, not forces.** Outsource the forces to it and you write crash risk into your training set.

## 3. Validated proof points — 贏的系統都架構化地切開

以下三個是 *已驗證* 並且 **贏** 的系統。共同點：它們都明確把 appearance 與 dynamics 切開，而切開正是它們贏的原因。

### 3.1 SOUS VIDE / FiGS — [VALIDATED, 最乾淨的證明]
arXiv [2412.16346](https://arxiv.org/abs/2412.16346)
- **架構切分**：flight dynamics = 10-D 半運動學四旋翼模型（ACADOS integrator）；visual obs = 3D Gaussian Splatting，渲染最高 **130 fps**。
- **資料**：100k–300k 合成 image/state-action pair；visuomotor policy **SV-Net**。
- **部署**：**zero-shot**，105 次真實飛行。
- **數字**：ablation **5/5（0 碰撞）**；novel task **51/60（85%）**；robust to **30% 質量變化、40 m/s 陣風、60% 亮度變化**。
- **關鍵句**：**3DGS 從不接觸 dynamics，物理從不接觸 appearance** —— 這就是本 handbook 論點的字面證據。

### 3.2 FlightDiffusion — [VALIDATED]
arXiv [2509.14082](https://arxiv.org/html/2509.14082)
- **架構切分**：Diffusion（**Wan 2.2 I2V**）生 FPV 影片；一個 **獨立的 VO 模組（ORB-SLAM3）** 從影片回復 pose → command。
- **關鍵**：diffusion **不強制 dynamics**（動力學一致性是 *事後* 檢查，非生成時的硬約束）。
- **數字**：真實飛行 vs VICON，平均位置誤差 **0.25 m（RMSE 0.28）**、朝向誤差 **0.19 rad**；sim-to-real 成功率 **62.8%（sim） vs 61.7%（real）**，統計上等價（ANOVA **F=0.394, p=0.541**）。
- **讀法**：生成模型把外觀做到夠真，**但動力學是另一條管線（VO）回推出來的** —— 又一次把 dynamics 外包。

### 3.3 UAV-Sim — [VALIDATED, perception]
ICRA 2024，arXiv [2310.16255](https://arxiv.org/abs/2310.16255)
- **角色**：NeRF 合成 **帶 bbox 標註** 的 novel-view UAV 影像；從 **15m 訓練影像** 渲染 **50m 高空** 視角。
- **數字**：YOLOv8n 在 hybrid（real+synthetic） vs real-only：**+55.85% mAP50、+47.25% mAP50:95**（@50m）。
- **結論**：**hybrid 打敗 real-only 也打敗 synthetic-only** —— generation 的價值在「補真實資料拿不到的視角/標註」，不是取代真實資料，更不是生動力學。

> 三者共識：**generation 提供 appearance（含可標註偵測資料、novel-view、photoreal DR）；dynamics 永遠來自物理模型或事後幾何回推。** 沒有反例。

## 4. Why generation can't own dynamics — Physics-IQ 的鐵錘

如果還想讓 generation 兼管動力學，這一節是反證。**Physics-IQ**（arXiv [2501.09038](https://arxiv.org/abs/2501.09038)，DeepMind）直接量到：

- **視覺真實度與物理理解 *不相關*：Pearson r = −0.46（不顯著）。**
- Physics-IQ 分數（100% = 真實上限）：VideoPoet **24.1%**、Runway Gen-3 **18.4%**、Stable Video Diffusion **13.5%**、Sora **8.7%** —— 然而 **Sora 的視覺真實度最佳（55.6%）**。
- 最差類別：**solid mechanics（固體力學）** —— 正是無人機關心的剛體/碰撞行為。

> 一句話：**看起來最真的影片，物理常常最爛。** 視覺真實度是 appearance 軸的指標，跟動力學軸正交甚至微負相關。任何「畫面夠真所以動力學也對」的推論，被這個 r=−0.46 直接否決。
> The prettiest video is often the least physical. Realism is an appearance metric; it is orthogonal (slightly anti-correlated) to dynamics. This is why generation must not own the dynamics half of the contract.

旁證（reconstruction-only，不是 perception training 用途，列此以免被誤引）：**DroneSplat**（CVPR 2025，arXiv 2503.16964）做 in-the-wild 無人機 3DGS（SAM2 遮動態 distractor + DUSt3R 補 sparse-view），PSNR **24.53 vs 3DGS 22.43**（low-dynamic）。它的目的是 **NVS（novel-view synthesis），不是訓 perception** —— 屬於 appearance 軸的重建工具，不提供 dynamics。

## 5. 五軸定位 + the metric-scale trap

**五軸座標**（見 [ontology.md](../../cheat-sheet/ontology.md)）：

| Axis | 值 | 說明 |
|---|---|---|
| Output | `pixel-video` \| `3d-explicit` | video-diffusion 出像素影片；NeRF/3DGS 出顯式 3D（含可投影標註） |
| Injection | `data-only` | 物理 *不* 注入 generation；外觀層純資料驅動，dynamics 由外掛物理模型負責 |
| Control | `image-init` \| `trajectory` | I2V 從首幀起 / 條件於軌跡生影片（仍只控外觀，不保證可飛） |
| Temporal | `clip-parallel` | 一次生固定窗口 clip（外觀片段）；長序列銜接非本軸強項 |
| Domain | `robotics` | aerial 在本 ontology 屬 robotics 子類 |

> 注意 `injection=data-only`：這正是論點的形式化 —— **generation 不承載物理注入**，所以它只能在 appearance 軸發力；dynamics 必須由 pipeline 的另一半（物理 sim）提供。

**The metric-scale trap（必讀陷阱）。** Monocular SfM/COLMAP 只能重建到 **任意尺度（arbitrary scale）**；3DGS 繼承這個無尺度性質；絕對尺度需要 **LiDAR / metric-depth** 才能釘住。對 aerial 這是致命的：**gate / obstacle 的「距離」是飛行安全量。** 一個尺度自由的 3DGS 場景可以畫面完美，但 gate 在 3 m 還是 5 m 完全沒被約束 —— policy 學到的避障距離可能整體縮放錯。

> 規則：**任何用 monocular-only 重建場景訓練 aerial perception/policy 的 pipeline，distance 量都不可信，除非引入 metric anchor（LiDAR / metric-depth / 已知 baseline stereo）。** 這是「外觀對 ≠ 距離對」的 aerial 版本，與 §4 的「真實 ≠ 物理」同源。

## 6. Cross-line synthesis — 把這條路接回 handbook

把 generation 的兩條主線各歸其位，都落在 **appearance 側**：

- **Pixel-WM 線（Cosmos-as-appearance）**：[cosmos-wfm.md](../../foundations/foundation-physics-models/cosmos-wfm.md)。NVIDIA Isaac + Cosmos-Transfer 工作流把 **PHYSICS + DR 跑在 Isaac**，再由 **Cosmos 只變 APPEARANCE**，明言「preserving robot motions, object positions, scene structure」（arXiv 2503.14492）—— **在平台層就把 split 寫死了**。Cosmos 在這裡是外觀增強器，不是動力學引擎。
  - **DEMO 級**：Cosmos Predict2 有 FPV quadcopter 範例（drone over harbor），frame 用 **OWL-ViT** 自動標註餵 detector，**但沒有 vs real 的準確度報告**，作者自己警告「expect misses」。歸 **DEMO**，不可當 validated。
- **3DGS 線（Real2Sim2Real）**：[generative-gaussian-splatting.md](../../foundations/3d-aware-generation/generative-gaussian-splatting.md)。真實飛行採集 → 3DGS 重建外觀 → 配物理模型 → 訓 policy → 飛回真實（SOUS VIDE/FiGS 正是這條閉環的 aerial 範式）。3DGS 提供 photons，物理提供 forces。
- **與 dynamics substrate 的關係**：本篇刻意 **不** teardown 物理 sim —— 那是 [aerial-sim-stack.md](./aerial-sim-stack.md) 的職責。本篇生成的外觀必須坐在那邊的 6-DoF 動力學上才成立。Swift 的端到端 sim-to-real 證據見 [champion-level-drone-racing.md](./champion-level-drone-racing.md)；總圖見 [overview.md](./overview.md)。
- **6-DOF 為何更難**：**ANWM**（Aerial World Model，arXiv 2512.21887）[DEMO] 條件於軌跡生 aerial 影片，但 **僅 Unreal-sim 評測、無真實飛行**，論文自承「does not explicitly model rigid-body dynamics or realistic aerodynamics」。**aerial 的 6-DOF action space 比地面機器人 3-DOF 更難** —— 這正是為什麼 aerial WM 至今沒有 validated 的「generation 自帶動力學」案例。

**Honest closed-source note [INTERNAL/UNKNOWN]：** Skydio 承認「synthetic + real」混訓但無細節；DJI Terra 5.0 出貨 3DGS mapping（是 *輸出產品*，非 *訓練 pipeline*）；Autel 無公開 sim/generative 揭露。**這些具體做法一律視為 UNVERIFIED，不引為論點證據。**

> 收束：generation 的每一條主線（pixel-WM、3DGS）放進 aerial，落點都在 **appearance**。動力學的位置始終空著，由物理填。這就是 trilogy generation 端的結論。

## 7. References（分組）

**VALIDATED — split 的正面證據**
- SOUS VIDE / FiGS — arXiv [2412.16346](https://arxiv.org/abs/2412.16346)（10-D quadrotor + ACADOS / 3DGS 130 fps / SV-Net / 105 flights / 85% novel）
- FlightDiffusion — arXiv [2509.14082](https://arxiv.org/html/2509.14082)（Wan 2.2 I2V + ORB-SLAM3 / 0.25 m / sim 62.8% vs real 61.7% / F=0.394 p=0.541）
- UAV-Sim — ICRA 2024，arXiv [2310.16255](https://arxiv.org/abs/2310.16255)（NeRF labeled novel-view / +55.85% mAP50）

**VALIDATED — 反證與重建工具**
- Physics-IQ — arXiv [2501.09038](https://arxiv.org/abs/2501.09038)（DeepMind；r=−0.46；VideoPoet 24.1% / Sora 8.7% 但 realism 55.6%）
- DroneSplat — CVPR 2025，arXiv [2503.16964](https://arxiv.org/abs/2503.16964)（in-the-wild 3DGS / PSNR 24.53 / NVS-only，非 perception training）

**VALIDATED — 合成偵測資料集（小目標 hybrid recipe）**
- SynDroneVision — arXiv [2411.05633](https://arxiv.org/abs/2411.05633)
- DrIFT — arXiv [2412.04789](https://arxiv.org/abs/2412.04789)（synthetic→real domain-shift 研究）

**DEMO（無 vs-real 準確度）**
- Cosmos Predict2 FPV — drone-over-harbor + OWL-ViT auto-label；作者警告「expect misses」
- NVIDIA Isaac + Cosmos-Transfer 工作流 — arXiv [2503.14492](https://arxiv.org/abs/2503.14492)（PHYSICS+DR 在 Isaac，Cosmos 只變 appearance）
- ANWM Aerial World Model — arXiv [2512.21887](https://arxiv.org/abs/2512.21887)（Unreal-sim only / 自承不建模 rigid-body dynamics）

**INTERNAL / UNKNOWN（視為 UNVERIFIED）**
- Skydio「synthetic + real」混訓（無細節）· DJI Terra 5.0 3DGS mapping（輸出非訓練）· Autel（無揭露）

**Cross-handbook**
- generated→VIO 資料契約：[../../bridge-to-spatial/aerial-embodiment.md](../../bridge-to-spatial/aerial-embodiment.md)
- Spatial aerial VIO（生成影像的消費端）：https://github.com/sou350121/Spatial-Intelligence-Handbook/tree/main/embodiments/aerial/vio
- Spatial event-camera（lighting-OOD 緩解）：https://github.com/sou350121/Spatial-Intelligence-Handbook/blob/main/embodiments/aerial/event-camera/event_camera_for_aerial_dissection.md

## §8 Pitfall log

1. **[CRITICAL · src: Physics-IQ 2501.09038] 用「畫面夠真」推斷「動力學夠真」。** r=−0.46（不顯著），Sora realism 55.6% 但 Physics-IQ 僅 8.7%。**Workaround**：appearance 與 dynamics 分軸驗收；動力學一律由物理模型提供，generation 的 realism 不得當動力學保證。

2. **[CRITICAL · src: monocular SfM/COLMAP + 3DGS] metric-scale trap：monocular 重建只到任意尺度，gate/obstacle 距離不可信。** **Workaround**：引入 metric anchor（LiDAR / metric-depth / 已知 baseline stereo）釘絕對尺度；否則所有 distance-dependent 行為（避障、過門）需標記為 scale-unverified。

3. **[HIGH · src: FlightDiffusion 2509.14082] 誤以為「條件於軌跡生影片」= 影片符合可飛動力學。** 該系統的動力學一致性是 *事後* 由 VO（ORB-SLAM3）回推檢查，**生成時不強制**。**Workaround**：軌跡條件視為外觀控制；可飛性必須由獨立物理/VO 模組驗證，不可假設生成器已保證。

4. **[HIGH · src: Cosmos Predict2 FPV demo] 把 DEMO 當 VALIDATED 引用。** FPV 範例用 OWL-ViT 自動標註但 **無 vs-real 準確度**，作者自承「expect misses」。**Workaround**：DEMO 級資料只做 pre-train / augmentation，下游必須 hybrid 真實資料並回報 real 上的指標（對齊 UAV-Sim 的 hybrid recipe）。

5. **[HIGH · src: ANWM 2512.21887] 以為已有「generation 自帶 aerial 動力學」的 validated 系統。** ANWM 僅 Unreal-sim 評測、無真實飛行，且自承不建模 rigid-body dynamics / aerodynamics；aerial 6-DOF 比地面 3-DOF 更難。**Workaround**：dynamics 仍交物理 sim（[aerial-sim-stack.md](./aerial-sim-stack.md)）；aerial WM 動力學自洽性視為未解。

6. **[MEDIUM · src: UAV-Sim 2310.16255] 用 synthetic-only 訓 detector。** hybrid 打敗 real-only 也打敗 synthetic-only；純合成有 domain gap（參見 DrIFT 2412.04789 的 synthetic→real shift）。**Workaround**：永遠 synthetic + small-real 混訓；小目標（無人機/電線/鳥，<0.01% 像素）尤其依賴少量真實樣本錨定。

7. **[MEDIUM · src: DroneSplat 2503.16964 引用脈絡] 把 NVS/reconstruction 工具誤當 perception-training 資料源。** DroneSplat 目的是 novel-view 合成，不是訓 perception。**Workaround**：明確區分「重建外觀」與「生成可標註訓練資料」；前者需再接標註/物理才進 training。

8. **[MEDIUM · src: Skydio/DJI/Autel 公開資料] 把 closed-source 廠商做法當論點證據。** Skydio「synthetic+real」無細節、DJI Terra 3DGS 是輸出非訓練、Autel 無揭露。**Workaround**：一律標 [INTERNAL/UNKNOWN] / UNVERIFIED，不納入論點推導。

> **UNVERIFIED 標註彙整**：Cosmos Predict2 FPV（DEMO，無 vs-real）· ANWM（DEMO，sim-only）· Skydio/DJI Terra/Autel（INTERNAL/UNKNOWN）· 任何「生成器自帶可飛動力學」的宣稱（無 validated 案例）。本篇所有數字均附 arXiv URL；超出上述來源的具體宣稱一律視為 UNVERIFIED，不予引用。
