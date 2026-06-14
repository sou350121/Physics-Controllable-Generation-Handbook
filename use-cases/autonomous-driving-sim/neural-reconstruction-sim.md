<!-- ontology-5axis output=3d-explicit|pixel-video injection=data-only control=camera|trajectory temporal=streaming domain=driving -->

# 神經重建模擬 —— 真實 log 重建成可重模擬場景 解構

> 「我們把一條真實採集的 log 變成一個**閉環、可重新模擬**的數位場景：移得動車、增刪得了 actor、換得了視角 —— 但它只知道感測器**那天真的看過**的東西。」
> 為什麼進名單：自駕模擬的契約有兩半。生成式 world model（[driving-world-models.md](./driving-world-models.md)）負責長尾與「沒發生過的事」，**神經重建**負責另一半 —— **metric-scale、感測保真的外觀**，像素直接來自真實量測，是整條 closed-loop 評估裡**唯一被信任**的那半。沒有它，policy 在純生成像素上跑出來的數字沒人敢簽字；有了它，重建的硬天花板（只能重渲 log 看過的）又恰好定義了生成式 WM 要補的那道縫。

---

## 1. TL;DR（真實量測 → 可重渲染場景：被信任的那半；但只能重渲染 log 看過的）

**一句話**：UniSim / NeuRAD 這類系統把**單條真實駕駛 log（多相機 + LiDAR）反演成一個顯式 3D 神經場**，可在**任意新視角 / 新軌跡**下重渲染出 metric-scale 的相機與 LiDAR 觀測，並支援**移動 SDV、增刪 / 重置 actor、反應式避讓**等閉環操作。**它強在感測保真**（NeuRAD 連 rolling shutter、ray-drop 都建模），這是它「被信任」的根本 —— 像素不是想像出來的，是真實量測的重投影。**它的硬限制同樣根本**：一切只在**原 log 觀測過的範圍內**成立；大幅改視角或挪走 actor，就會把渲染推進「沒被任何感測器看過」的外推區（UniSim 用一個 **CNN 補未見區**，就是這個外推本質的 tell）。**結論：重建的多樣性被 log 邊界鎖死 —— 這正是生成式 WM 要接手的縫。**

---

## 2. 核心機制（neural feature field + 靜 / 動分解 + 未見區補全；NeRF → 3DGS 的速度躍升）

三個共通設計，外加一條決定「能不能即時閉環」的演化線：

- **(a) Neural feature field（神經特徵場）**：把場景表徵成一個可微的 3D 場 —— 不是只存 RGB，而是存**特徵向量**，再經神經 renderer 解出相機像素與 LiDAR 觀測。**NeuRAD 用單一統一的 Neural Feature Field 同時餵相機與 LiDAR 兩個感測頭**，所以兩種模態天然空間一致。
- **(b) 靜 / 動分解（static / dynamic decomposition）**：靜態背景用一套表徵（NeuRAD / UniSim 的背景場、Street Gaussians 的靜態點雲），每個動態 actor 用**獨立的 per-actor 表徵**（UniSim 的 per-actor network、DrivingGaussian 的 composite dynamic Gaussian graph、Street Gaussians 的 per-object 動態高斯）。**這層分解就是「可編輯」的來源** —— 能單獨平移、刪除、重置某個 actor，正因為它在表徵層就是可尋址的獨立物件。**S³Gaussian 更進一步：自監督完成靜 / 動分解，無需任何 3D 標註。**
- **(c) 未見區補全（unseen-region completion）**：單條 log 不可能把每個面都拍到。改視角後露出的「沒被看過的區域」必須被填 —— **UniSim 明確用一個 CNN（含可學的動態物先驗）去 hallucinate 這些補洞**。**這是整套方法外推本質的最誠實 tell**：一旦渲染依賴補全，像素就不再是「被量測的」，可信度從這裡開始衰減。
- **(d) NeRF → 3DGS 的速度躍升**：早期路線（MARS / Block-NeRF）是 NeRF，**渲染慢、訓練久**。3D Gaussian Splatting 把表徵換成顯式高斯點雲，**Street Gaussians 報 135 FPS @1066×1600、約 30 分鐘訓練、比 NeRF 快約 100×**。**這 2 個數量級的速度，正是「即時閉環渲染」從不可能變可能的那一步** —— closed-loop 需要 policy 動作回饋後快速重渲染（呼應 [closed-loop-or-bust.md](./closed-loop-or-bust.md)）。

```
真實 log（多相機 RGB + LiDAR 點雲 + 位姿）
      │  反演 / 擬合
      ▼
┌─────────────────────────────────────────────┐
│  靜態背景場         ＋   per-actor 動態表徵    │  ← (b) 靜/動分解＝可編輯來源
│  (neural feature field / 3DGS)                │
└─────────────────────────────────────────────┘
      │  新相機 pose / 新 ego+actor 軌跡（control=camera|trajectory）
      ▼
[ 神經 renderer ] ──→ 相機像素（pixel-video, metric-scale）
      │            └─→ LiDAR 觀測（含 ray-drop / intensity, NeuRAD）
      ▼
露出未觀測區？ ──→ [(c) CNN 補未見區] ←─ 可信度從這裡開始掉
```

---

## 3. 對比表（重建 / 感測 / 可微軌跡 / metric / 限制）

| 系統 | 表徵 | 感測模態 | 可編輯軌跡 / actor | Metric-scale | 速度 / 規模 | 核心限制 |
|---|---|---|---|---|---|---|
| **UniSim** (Waabi, CVPR'23 Highlight, [2308.01898](https://arxiv.org/abs/2308.01898)) | neural feature grids（靜背景）+ per-actor nets + **CNN 補未見區** | cam **+** LiDAR（閉環多感測） | ✓ **2m 車道平移、升降 SDV、增刪 / 重置 actor、反應式正面避讓**、稀疏資料外推 km 級高速 | ✓ 真實感測 → metric | per-scene；單 log 覆蓋限外推品質 | **single domain log** — 「small domain gap」但**外推受單 log 覆蓋限制**；公開頁無 PSNR/LPIPS（UNVERIFIED） |
| **NeuRAD** (Zenseact, CVPR'24, [2311.15260](https://arxiv.org/abs/2311.15260); **開源** [neurad-studio](https://github.com/georghess/neurad-studio)) | 單一統一 **Neural Feature Field**（cam + LiDAR 共用） | cam **+** LiDAR；**明確建模 rolling shutter（逐像素 / 逐點時戳）、beam divergence、ray-drop + intensity(MLP)、per-camera embedding** | ✓ NVS + actor 編輯（多資料集開箱） | ✓ | 5 個 AD 資料集 SOTA | 仍 per-log 重建；**精確 PSNR/SSIM 頁面未給（UNVERIFIED）** |
| **MARS** ([2307.15058](https://arxiv.org/abs/2307.15058), 開源) | **NeRF**、instance-aware（靜 / 動分離網） | **cam-only** | ✓ instance 級 | ✓ | per-scene、NeRF 慢 | 無 LiDAR；NeRF 渲染慢 |
| **Street Gaussians** ([2401.01339](https://arxiv.org/abs/2401.01339), ECCV'24) | 點雲 + 3DGS、**4D spherical harmonics** 表動態外觀、可優化追蹤位姿 | cam（LiDAR 先驗） | ✓ 可優化追蹤位姿 | ✓ | **135 FPS @1066×1600、~30 分訓練、比 NeRF ~100× 快、KITTI/Waymo SOTA** | 動態外觀靠 4D-SH 近似；仍 per-log |
| **DrivingGaussian** ([2312.07920](https://arxiv.org/abs/2312.07920), CVPR'24) | incremental 靜態 3DGS + **composite dynamic Gaussian graph**（每物件） | 環景多相機（LiDAR 先驗） | ✓ per-object graph | ✓ | 環景一致 | 大型動態場景 graph 維護成本；per-log |
| **S³Gaussian** ([2405.20323](https://arxiv.org/abs/2405.20323)) | 自監督靜 / 動分解、**4D 場** | cam | ✓（分解後可編輯） | ✓ | — | **無需 3D 標註**；仍受 log 覆蓋限制 |
| **Block-NeRF** (Waymo/Berkeley, CVPR'22, [2202.05263](https://arxiv.org/abs/2202.05263)) | NeRF、外觀 embedding / 曝光控制 / **分塊融合** | cam | **✗ 靜態，無動態 actor / 無閉環** | ✓ | **城市級**（SF Alamo Sq, 2.8M 圖, 3 個月） | **只有靜態外觀**；無 actor、無閉環 |

> **讀表三條軸線**：(1) **感測完備度** —— UniSim / NeuRAD 收 cam + LiDAR 閉環雙感測，MARS / Street GS / Block-NeRF 偏 cam（LiDAR 退為先驗）；(2) **速度 / 規模換軸** —— NeRF 線（MARS / Block-NeRF）慢但 Block-NeRF 上得了城市級，3DGS 線（Street GS / DrivingGaussian）快到 135 FPS 但通常 scene-scale；(3) **動態能力** —— 除 Block-NeRF 外都靠 per-actor / per-object 分解拿到 actor 編輯，**而這正是閉環的前提**。

---

## 4. ⚡ 強 / ❌ 崩

### ⚡ 強：metric-scale 感測保真

- **像素來自真實量測 → 被信任的那半**：重建的 RGB / LiDAR 是真實觀測的重投影，不是生成想像。這是它能進 closed-loop 評估、policy 數字敢簽字的根本理由。
- **NeuRAD 連物理感測效應都建**：**rolling shutter（逐像素 / 逐點時戳）、beam divergence、ray-drop + intensity、per-camera embedding** —— 不是「看起來像」，是把感測器的物理成像鏈一起重建。對需要感測級保真的 AV 評估，這是決定性的。
- **metric-scale，不是相對尺度**：源於真實 LiDAR / 標定，幾何是公尺單位的；軌跡、距離、碰撞判定都可信。
- **3DGS 線買到 2 個數量級速度**：Street Gaussians **135 FPS / ~30 分訓練 / ~100× 快於 NeRF** —— 即時閉環渲染因此可行。
- **可編輯閉環**：UniSim 的 **2m 車道平移、升降 SDV、增刪 / 重置 actor、反應式正面避讓** 證明「重建 ≠ 只能回放」，在 log 邊界內可做反事實。

### ❌ 崩：多樣性受 log 邊界限制

- **核心限制 —— 只能重渲染原 log 看過的**：表徵是某條 log 的觀測擬合，**大幅改視角 / 移走 actor 就會外推進未觀測區**。UniSim 的「未見區 CNN」就是這個外推本質的 tell —— 一旦補全介入，像素不再是被量測的，保真度從此衰減。
- **多樣性被 log 邊界鎖死**：重建**不會生出 log 裡沒出現過的新 agent 行為、新天氣、新長尾事件**。它能重組（換視角、挪 actor），不能無中生有。
- **single-log 覆蓋是天花板**：UniSim 自承 small domain gap，但外推品質**受單條 log 覆蓋限制**；沒拍到的角度、沒出現的物體，沒有真值可依。
- **per-log / per-scene 成本**：每條 log 要重建（NeuRAD / Street GS 等多為 per-log）；不是一個模型涵蓋全分佈。
- **公開 metric 多為 UNVERIFIED**：UniSim 公開頁無 PSNR/LPIPS；NeuRAD 稱 5 資料集 SOTA 但頁面未給精確 PSNR/SSIM。**本篇不臆造數字。**

---

## 5. 五軸定位

```
output     = 3d-explicit | pixel-video   ← 顯式 3D（neural feature field / 3DGS / NeRF）＋ 渲染出像素影片與 LiDAR
injection  = data-only                   ← 物理 / 外觀全靠真實 log 擬合；無 PDE、無守恆 loss、無 sim-in-loop
control    = camera | trajectory         ← 新相機 pose（NVS）＋ ego/actor 軌跡編輯（移車 / 增刪 actor）
temporal   = streaming                   ← 即時連續重渲染，支援 policy 回饋的閉環滾動（3DGS 線使其可行）
domain     = driving                     ← AD 道路場景；Check 9c 白名單外，明確宣告 driving
```

- **Check 9b（Output × Injection）**：`3d-explicit × data-only` = ✓，`pixel-video × data-only` = ✓（cheat-sheet/ontology.md 矩陣兩格皆合法）—— **無需 §8 例外解釋**。
- **Check 9c（generalist 白名單）**：不在 Sora / Veo / Cosmos 白名單；明確標 `driving`，與 [overview.md](./overview.md) 預設一致。
- **與生成式 WM 的軸差**：本篇 `injection=data-only` 且 metric-scale —— 跟 [driving-world-models.md](./driving-world-models.md)（GAIA-2 / Cosmos-Drive 同樣 `data-only` 但 `output=pixel-video`、無 metric 3D）的關鍵差在**有顯式 3D + LiDAR 感測重建**。重建是「真實量測的可微回放」，生成是「分佈的想像採樣」。
- **與通用 3DGS 的關係（NO-DUP）**：本篇**不重拆通用 3DGS 表徵**；那是 foundation 的 [generative-gaussian-splatting.md](../../foundations/3d-aware-generation/generative-gaussian-splatting.md) 的事（生成端，從文字 / 單圖外推、必須 hallucinate）。本篇是**重建端**（從真實多感測 log 反演），資訊來源相反。

---

## 6. 跨路線綜合（與 driving-world-models 互補：重建給外觀、生成給長尾；連 closed-loop-or-bust）

| 路線 | 它給什麼 | 它缺什麼 | 怎麼接 |
|---|---|---|---|
| **[driving-world-models.md](./driving-world-models.md)**（GAIA-2 / Cosmos-Drive 生成式 WM） | **長尾、未發生事件、新天氣 / 新 agent**（分佈外採樣） | metric-scale 感測保真弱、無真值 LiDAR | **重建給外觀，生成給長尾** —— 用重建場景當 metric backbone，生成式 WM 注入 log 裡沒有的長尾變化。**這就是 AV 契約的兩半合一。** |
| **[closed-loop-or-bust.md](./closed-loop-or-bust.md)**（閉環評估的硬要求） | 閉環評估的方法論與門檻 | 需要可快速重渲染、可微的場景 | 3DGS 線的 **135 FPS** 讓 policy 動作回饋後即時重渲成為可能 —— 重建是 closed-loop 的**算力可行性那一塊** |
| **本篇（神經重建）** | **metric-scale、感測保真、被信任的外觀**（log 邊界內可編輯閉環） | **多樣性受 log 邊界鎖死** | 提供可信 backbone；把「沒看過的」交給生成式 WM |

- **與 foundation 生成 3DGS（[generative-gaussian-splatting.md](../../foundations/3d-aware-generation/generative-gaussian-splatting.md)）**：同表徵（3DGS）兩個相反方向 —— 生成端從文字 / 單圖**外推 hallucinate**，重建端從真實 log **反演量測**。pipeline 上可互補：重建場景當 anchor，生成補 corner-case。
- **跨 handbook（Spatial 的 3DGS 建圖 / on-board mapping）**：Spatial-Handbook 的機載建圖（https://github.com/sou350121/Spatial-Intelligence-Handbook/tree/main/embodiments/aerial/on-board-mapping）走**感知 / 定位**視角的 3DGS 重建；本篇走**模擬 / 重渲染**視角。同一族 3DGS 重建技術，下游目的不同（建圖定位 vs 場景再模擬）。
- **本軸結論回扣 ontology**：`injection=data-only` 在這裡的天花板不是物理一致性的缺失（重建本就不宣稱模擬物理），而是**多樣性 = log 邊界**。要突破，要嘛接生成式 WM 補分佈，要嘛接可微 sim 補 dynamics —— 重建自己補不了。

---

## 7. 參考

**Canonical**：
- **UniSim**: Yang, Z. et al. (2023). "UniSim: A Neural Closed-Loop Sensor Simulator." **CVPR 2023 (Highlight)**, Waabi / Univ. of Toronto. arXiv **2308.01898**. https://arxiv.org/abs/2308.01898
- **NeuRAD**: Tonderski, A. et al. (2024). "NeuRAD: Neural Rendering for Autonomous Driving." **CVPR 2024**, Zenseact. arXiv **2311.15260**. https://arxiv.org/abs/2311.15260 · **開源**: neurad-studio https://github.com/georghess/neurad-studio
- **MARS**: Wu, Z. et al. (2023). "MARS: An Instance-aware, Modular and Realistic Simulator for Autonomous Driving." arXiv **2307.15058**（開源）. https://arxiv.org/abs/2307.15058
- **Street Gaussians**: Yan, Y. et al. (2024). "Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting." **ECCV 2024**. arXiv **2401.01339**. https://arxiv.org/abs/2401.01339
- **DrivingGaussian**: Zhou, X. et al. (2024). "DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes." **CVPR 2024**. arXiv **2312.07920**. https://arxiv.org/abs/2312.07920
- **S³Gaussian**: Huang, N. et al. (2024). "S3Gaussian: Self-Supervised Street Gaussians for Autonomous Driving." arXiv **2405.20323**. https://arxiv.org/abs/2405.20323
- **Block-NeRF**: Tancik, M. et al. (2022). "Block-NeRF: Scalable Large Scene Neural View Synthesis." **CVPR 2022**, Waymo / UC Berkeley. arXiv **2202.05263**. https://arxiv.org/abs/2202.05263

**同倉交叉**：[driving-world-models.md](./driving-world-models.md) · [closed-loop-or-bust.md](./closed-loop-or-bust.md) · [overview.md](./overview.md) · [../../foundations/3d-aware-generation/generative-gaussian-splatting.md](../../foundations/3d-aware-generation/generative-gaussian-splatting.md) · [../../cheat-sheet/ontology.md](../../cheat-sheet/ontology.md)
**跨 handbook**：Spatial-Handbook 機載 3DGS 建圖 https://github.com/sou350121/Spatial-Intelligence-Handbook/tree/main/embodiments/aerial/on-board-mapping

---

## §8 踩坑日誌

### 8.1 跨軸合規性（Check 9b / 9c）
- **9b**：`3d-explicit × data-only` ✓、`pixel-video × data-only` ✓（兩格皆合法）→ 無需例外解釋。
- **9c**：非 generalist 白名單；明確標 `driving`，與 [overview.md](./overview.md) 一致。
- **Descriptive note（Control × Domain）**：`trajectory` 屬 `driving` 典型，符合 ontology §181 描述；`camera` 為 2025 主流 NVS 控制。

### 8.2 已知限制（paper / 公開頁確認）

| # | 來源 | 描述 | Severity | Workaround |
|---|---|---|---|---|
| 8.2.1 | UniSim paper / project page | **single-log 覆蓋是外推天花板**；「small domain gap」但外推品質受單條 log 限制 | High | 多 log 融合 / 接生成式 WM 補分佈 |
| 8.2.2 | UniSim 機制 | **未見區靠 CNN 補全** —— 改視角露出未觀測區即 hallucinate，可信度下降 | High（保真） | 限制視角偏移幅度；標記補全區為低信賴 |
| 8.2.3 | 全線共有 | **多樣性 = log 邊界**：不生 log 沒出現的 agent / 天氣 / 長尾 | High（多樣性） | 交給 [driving-world-models.md](./driving-world-models.md) |
| 8.2.4 | NeuRAD / Street GS | 多為 **per-log / per-scene** 重建，非單模型覆蓋全分佈 | Medium | 工程化批量重建管線 |

### 8.3 量測缺口（UNVERIFIED — 本篇不臆造）

| # | 項目 | 狀態 |
|---|---|---|
| 8.3.1 | UniSim PSNR / LPIPS | **UNVERIFIED** —— 公開頁未給數字 |
| 8.3.2 | NeuRAD 精確 PSNR / SSIM | **UNVERIFIED** —— 稱 5 個 AD 資料集 SOTA，頁面未給精確值 |
| 8.3.3 | Street Gaussians 135 FPS / ~30min / ~100× | paper 宣稱值，已標來源；其餘系統未逐一複現對照 |

### 8.4 結構性批判
- **`injection=data-only` 在重建端的天花板 ≠ 物理一致性缺失**：重建本就不宣稱模擬物理，它宣稱**真實量測的可微回放**。真正天花板是**多樣性被 log 邊界鎖死**。
- **「未見區 CNN」是最誠實的 tell**：任何重建系統一旦要補未觀測區，就已踏出「被量測」的安全區。評估時應把補全區與量測區分級對待，否則保真度被高估。
- **NeRF→3DGS 的速度躍升買的是「可行性」不是「多樣性」**：135 FPS 讓即時閉環可行，但不擴大場景分佈 —— 速度與多樣性是兩個正交問題，別混淆。

### 8.5 待釐清項目（[TBD]）
- [TBD] UniSim / NeuRAD 官方 PSNR/LPIPS/SSIM 精確數字（待 paper 表格核對）
- [TBD] 各系統閉環評估下 policy 數字的可比 benchmark（與 [closed-loop-or-bust.md](./closed-loop-or-bust.md) 對齊）
- [TBD] per-log 重建的工程成本（wall-clock / GPU）跨系統對照
- [TBD] 重建 backbone + 生成式 WM 注入的混合管線是否有公開實作

---

> **Pulsar maintenance**：本篇是 AV 契約「重建外觀」那半，與 [driving-world-models.md](./driving-world-models.md)（生成長尾）互補、與 [closed-loop-or-bust.md](./closed-loop-or-bust.md)（閉環門檻）相連。核心論點 = **重建給 metric-scale 感測保真的外觀（NeuRAD 連物理感測效應都建），但只能重渲 log 看過的 → 多樣性被 log 邊界鎖死，這正是生成式 WM 要補的縫**。daily monitoring keyword：「UniSim Waabi sensor simulation」「NeuRAD neurad-studio」「Street Gaussians driving」「DrivingGaussian」「S3Gaussian」「Block-NeRF」「closed-loop neural reconstruction autonomous driving」。下次相關 release 後重 audit §8.3 的 UNVERIFIED 數字。
