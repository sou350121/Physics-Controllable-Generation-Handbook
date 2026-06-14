<!-- ontology-5axis output=N/A injection=sim-in-loop-infer control=param temporal=streaming domain=robotics -->

# 孿生保真度契約 —— 視覺孿生 vs 可預測孿生 解構

> 本篇不解構某一個生成模型，而是解構一條**契約條款**：當有人說「我有一個 digital twin」，他到底是有一個**長得像**的東西（視覺孿生），還是一個**能替你預測下一刻會發生什麼**的東西（可預測孿生）？這兩者差的那一層，跨 robotics / 工業 / 手術**完全一致**。主要證據：
> - **A Survey on Physics-aware / Interactive Digital Twins from Visual Reconstruction**，arXiv [2504.13159](https://arxiv.org/pdf/2504.13159) —— 把「photoreal 重建 ≠ 可用孿生」講到逐字的那篇 survey；taxonomy 從 appearance-only → physics-integrated。
> - **Surgical Digital Twins survey**，arXiv [2512.00019](https://arxiv.org/html/2512.00019v1) —— 把契約講得**最清楚**的領域：preop 靜態解剖孿生 vs intraop 即時同步孿生。
> - **NVIDIA Omniverse / OpenUSD + Sensor RTX**（開發者 blog / 行銷頁）—— 工業孿生的旗艦；但「閉環」是 simulation-only 的 reference blueprint（見 §3，標 `BLUEPRINT` / `MARKETING`）。
> - **McKinsey 2024 製造業 digital-twin 調查** —— **86% 視 DT 為 mission-critical，僅 14% 已接到 live robot fleet**。這個 86/14 落差就是「live-sync 契約幾乎沒人關上」的硬數據。
> - **MaterialGS / 3DGS-for-sensing**，arXiv [2511.20348](https://arxiv.org/abs/2511.20348)；**PhysGaussian / SplatSim / RoboGSim** —— 把物理 bolt 上 Gaussian Splatting 的前緣。
>
> **為什麼進名單**：本手冊反覆講「外觀靠生成、動力學靠物理」。digital twin 是這條命題**商業壓力最大**的落地點——因為太多人把一個 NeRF/3DGS photoreal 重建直接叫「孿生」，然後拿去做決策。本篇把那條紅線劃死：**photoreal 重建是視覺孿生；只有加上物理（動力學）+ 即時狀態同步，才升級成可預測孿生。** 而且這條紅線不是學術潔癖——它正是 §3 那個 86/14 落差的根因，也是 §4 手術界**已經寫進臨床流程**的東西。

## 1. TL;DR —— photoreal 重建是視覺孿生

**一句話契約**：你能 render 出它，不代表你能 predict 它。

- **視覺孿生（visual twin）**：NeRF / 3DGS / photogrammetry 重建。它交付的是**外觀**——從任意視角看起來像真的。survey 2504.13159 的原話最狠：**「a visual reconstruction can look photorealistic yet lack the physical grounding necessary for interactive simulation」**。它是 **appearance-only**。
- **可預測孿生（predictive twin）**：在視覺孿生之上，**加兩樣東西**——
  1. **物理（動力學）**：mass / density / friction、articulation（joints / DOF）、**獨立的 collision geometry**（不是渲染用的那層 mesh）、material、以及 **dynamic state**。少任何一樣，你都只能看不能算。
  2. **即時狀態同步（live state sync）**：孿生裡的「當下」必須隨真實系統的「當下」一起變。沒有這條，你預測的是**昨天那個物體**。

> **核心論點：跨 robotics / 工業 / 手術，缺的那一層永遠是「動力學 + 同步」，從來不是外觀。生成管外觀與場景；物理 + 同步管行為與「當下」。** 外觀這一半，2025 的生成/重建技術已經**做得太好了**——好到掩蓋了另一半根本沒做。本篇就是要把被外觀掩蓋的那半挖出來。

## 2. 三層保真 + 契約表

survey 2504.13159 的 taxonomy（appearance-only → hybrid → physics-integrated → generative-physics）本質上在爬一個梯子。把它重切成**三層保真**，每層問一個不同的問題：

| 層 | 問題 | 內容 | 來源 |
|---|---|---|---|
| **L1 幾何（geometry）** | 它**長**什麼樣？ | 外觀 + 形狀。NeRF / 3DGS / photogrammetry / CT / MRI。**視覺孿生只到這層。** | 2504.13159（appearance-only）；手術=CT/MRI |
| **L2 動力學（dynamics）** | 它**會怎麼動**？ | mass/density/friction、articulation(joints/DOF)、**獨立 collision geometry**、material、組織生物力學(FEA/deformable) | 2504.13159（physics-integrated）；手術 survey FEA/bioheat |
| **L3 即時狀態同步（live sync）** | 它**現在**是什麼狀態？ | dynamic state 隨真實系統實時更新；intraop 隨組織形變更新 | DT-fidelity 文獻（sync 是定義性的）；手術 intraop |

**契約表**——哪些**必須忠實**（faithful，不能近似），哪些**可生成 / 近似**（generated / approximated）：

| 維度 | 必須忠實（faithful） | 可生成 / 近似（generated / approx） |
|---|---|---|
| 外觀 / 紋理 | — | ✅ 生成/重建管這塊；photoreal 是可近似的 |
| 度量尺度（metric scale） | ✅ **絕對尺度必須對**（碰撞/受力/劑量全靠它） | ❌ 不能猜——monocular 重建這裡會塌（見 §8.2） |
| 質量 / 密度 / 摩擦 | ✅ 動力學行為的根 | 🔶 可從先驗/材質庫**估**，但要標不確定度 |
| articulation / DOF | ✅ joints 接錯，整個運動學就錯 | ❌ 不能省略 |
| collision geometry | ✅ 必須**獨立於**渲染 mesh | 🔶 可簡化（凸包/原語），但要保守 |
| material（給 sensing） | 🔶 看用途：給 sensor sim 要對 | ✅ MaterialGS 這類可從 camera-only 推（§6） |
| **dynamic state（live）** | ✅ **這就是「可預測」的定義性條件** | ❌ 不能離線——離線就退回 L1 視覺孿生 |
| 場景背景 / 干擾物 | — | ✅ 可程序化生成/隨機化 |

> **讀法**：左欄是**契約的硬約束**——缺一條，這個孿生對「預測」就不成立。右欄是**生成的用武之地**。一個健康的孿生工程，是把左欄一項項關死，把右欄交給生成去 scale。把右欄做得再漂亮（photoreal），都補不了左欄任何一個洞。

## 3. 工業孿生：外觀/結構已產品化、live-sync 是沒人關上的縫

工業是孿生**最像產品**的地方——但也是契約被**行銷話術**糊得最厲害的地方。

**事實層（外觀/結構，已產品化）**：NVIDIA **Omniverse** 用 **OpenUSD Stage** 把 CAD + sensor 數據聚合成一個可組合的場景，**Sensor RTX** 負責 ray-traced 的 sensor 渲染。這套把 **L1 幾何 + sensor 外觀**做到了工業級——這部分是真的、可買的。

**但「閉環」的故事 vs 事實**（這是本節要劃的紅線）：

- ★ NVIDIA 開發者 blog 講的「閉環」是 **simulation-only**——合成 sensor → robot brain → actuation，**整個迴圈跑在 sim 裡**。它**不是**真實 sensor 回饋進孿生（不是 L3 live-sync）。blog 自己說明這是 **reference blueprint**，且**沒有客戶部署 / 沒有數字** → 標 **`BLUEPRINT`**。
- ★ 行銷 blog 列了一串客戶（Foxconn / KION / Schaeffler …）用 Omniverse 做工廠孿生，但**沒有任何量化成效**（沒有 sync 延遲、沒有預測準確率、沒有 fleet 規模）→ 標 **`MARKETING`**。

**所以**：工業孿生今天賣的，**主要是 L1（幾何/外觀）+ sim-only 的 L2（在 sim 裡跑物理）**。真正的 **L3 live-sync（真實 robot fleet 的狀態實時灌回孿生）**，公開證據裡**基本是空的**。

> ★ **McKinsey 2024 的硬數據把這個縫量出來了**：**86% 製造商視 digital twin 為 mission-critical，但只有 14% 已經接到 live robot fleet。** 換句話說——**契約裡 L1+L2 大家都在做，唯獨 L3（live-sync）這一條，幾乎沒人關上。** 這個 86/14 落差不是工程進度問題，它**就是本篇契約的實證**：缺的那層永遠是同步，不是外觀。

> ⚠️ 常被引用的「BMW 30% greenfield 工廠先建數字孿生」這個數字，在兩個 NVIDIA 官方頁面上**都查不到出處** → 標 `UNVERIFIED`，不採用為論據。

## 4. 手術孿生：契約講得最清楚

如果工業是契約被糊得最厲害的地方，**手術就是契約被講得最清楚的地方**——因為在手術裡，把 L3 同步做錯是會出人命的，所以這個領域**被迫**把「靜態」和「即時」分得一清二楚。

survey 2512.00019 把手術孿生切成**三層**，正好對齊本篇的 L1/L2/L3：

- **L1 解剖 / 幾何**：CT / MRI 重建出病患特異的解剖結構。
- **L2 組織物理 / 生物力學**：FEA、deformable registration、bioheat（熱傳）方程——讓組織**會形變、會傳熱**。
- **L3 生理即時態**：術中隨真實組織狀態同步。

★ **而契約最鋒利的一刀，是 survey 對 preop vs intraop 的切分**：

| | **術前規劃孿生（preop）** | **術中孿生（intraop）** |
|---|---|---|
| 本質 | **靜態解剖**（手術前的快照） | **即時同步**（隨組織形變更新） |
| 需要 live-sync？ | **不需要**——它就是個高保真靜態模型 | **必須**——這就是它存在的理由 |
| 對應本篇 | L1（+ 離線 L2 規劃） | L1 + L2 + **L3** |
| 瓶頸 | 重建精度 | ★ **real-time deformable registration 是限制臨床部署的瓶頸** |

> **這就是本篇最該被記住的一句**：preop 孿生**沒有 live-sync 需求**（它本來就只是術前的靜態解剖，當「視覺/規劃孿生」用完全合格）；intraop 孿生**的全部價值就在 live-sync**（組織一形變，模型必須跟上）。**同一個器官、同一套 CT，差別只在那條 L3 同步軸開沒開**——這把「視覺孿生 vs 可預測孿生」的分界，講得比任何工業 marketing 都乾淨。

**VALIDATED（手術領域已落地的硬證據）**：

- **HeartFlow FFR-CT** —— 從病患 CT 算出**病患特異的冠脈血流**（fractional flow reserve），**FDA-cleared、商用**。這是「可預測孿生」真正臨床落地的標桿：它不只長得像血管，它**算得出血流**。
- **MRI 引導微波消融孿生** —— 把孿生耦合 **bioheat 方程**，預測熱劑量分布。L2 物理（熱傳）+ L3（術中 MRI 同步）兩條都真。

survey 的建議也和本篇契約表一致：**hybrid**——對 critical physics 做**選擇性的高保真生物物理**（L2/L3 全開），其餘走**快速 workflow sim**（近似即可）。這正是 §2 契約表「左欄關死、右欄交給近似」的臨床版。

## 5. 五軸定位（同步 = temporal 軸是定義性的）

本篇頂部標 `output=N/A`（不交付生成物，解構的是契約/驗收標準）。重點落在 **Temporal = `streaming`**——因為**「即時狀態同步」本身就是一條 temporal 條款**。

| 軸 | 值 | 為什麼 |
|---|---|---|
| Output | `N/A` | 解構**契約/驗收標準**，不交付生成物。對齊 ontology「純評測/契約 → output=N/A」（[ontology](../../cheat-sheet/ontology.md) Axis 1 N/A 條款）。 |
| Injection | `sim-in-loop-infer` | 可預測孿生在**推理時**把 sim 接進迴圈：真實狀態灌進來、sim 往前推一步、回灌預測。HeartFlow 的 patient-CFD、intraop 的 deformable registration 都是 infer-time sim-in-loop。視覺孿生**沒有**這條 injection → 才只能看不能算。 |
| Control | `param` | 孿生由**顯式物理參數**驅動（mass / density / friction / stiffness / 組織生物力學係數）——契約表 L2 那一整欄就是 `param`。 |
| **Temporal** | **`streaming`** | ★ **這條是定義性的。** 「live-sync」= 連續時間、隨真實系統滾動更新，無固定 clip 窗口。**preop（靜態）≈ `single-frame`/快照；intraop（同步）= `streaming`**——五軸裡，**就是 Temporal 這一格，把視覺孿生和可預測孿生分開**。 |
| Domain | `robotics` | 以工業/具身孿生為主錨（手術=`bio` 為輔）；非 `generalist`（白名單只給 Sora/Veo/Cosmos-Predict，見 9c）。 |

> **一句話把 ontology 與本篇命題對齊**：**視覺孿生 = L1 幾何 + Temporal 退化成快照；可預測孿生 = L2(`param` 物理) + Temporal=`streaming`(live-sync) + Injection=`sim-in-loop-infer`。** DT-fidelity 文獻說得直接——**fidelity = 精度 + synchronization 機制，而 synchronization 是定義性的、不是可選的**。這在五軸裡，就是「Temporal 軸不能是快照」。

> **Injection × Temporal 相容性註記**（ontology 9b descriptive note）：`sim-in-loop-infer` 只對 iterative paradigm 有意義（連續 `streaming` 滾動）。可預測孿生的 live-sync 全屬連續 streaming 更新，相容；preop 靜態快照**不構成** iteration，因此**進不了** `sim-in-loop-infer`——這從 ontology 層面解釋了為什麼靜態重建只是視覺孿生。

## 6. 跨路線綜合（連 real2sim-twins；與 AV closed-loop 同源）

本篇是 digital-twin zone 的**契約層**——它定義「什麼才算可預測孿生」，其他子線提供把外觀做出來的手段：

| 子線 | 提供什麼 | 與本篇的關係 |
|---|---|---|
| [real2sim-twins.md](./real2sim-twins.md)（從真實重建可用 sim 資產） | 把真實物體重建成**帶物理/articulation 的 sim-ready 資產** | 這正是把 L1 視覺孿生**升級到 L2** 的工序：補 mass/friction/joints/collision。real2sim 做得好不好，就看它補了多少左欄硬約束。 |
| [generative-gaussian-splatting.md](../../foundations/3d-aware-generation/generative-gaussian-splatting.md)（3DGS 生成/重建） | **L1 外觀**——photoreal 的可微表徵 | 3DGS 本身**只到 L1**。**MaterialGS**（2511.20348）從 **camera-only 3DGS** 推 material masks 給 sensor sim——注意這是加 **material-for-sensing，不是動力學**（補的是「外觀給感測」，不是「行為」）。**PhysGaussian / SplatSim / RoboGSim** 才是把物理（L2）bolt 上 GS 的前緣。**別把「給 GS 加 material」誤當「給 GS 加動力學」**（§8.4）。 |
| **本篇（twin-fidelity-contract）** | **L2+L3 的驗收標準** | 給上面兩條設**及格線**：外觀真不真是它們的事；**有沒有物理 + 有沒有 live-sync**，是本篇的事。 |

★ **最關鍵的跨倉命題**：本篇和自駕的 [closed-loop-or-bust.md](../autonomous-driving-sim/closed-loop-or-bust.md) **是同一條命題的兩個臉**——

- 自駕問：「你的 sim 動力學那一半驗收了嗎？」答案是 **closed-loop + reactivity**（動作要真的改變世界）。
- 孿生問：「你的 twin 可預測那一半驗收了嗎？」答案是 **physics + live-sync**（狀態要真的同步「當下」）。

兩者**同源**：都是在說「**外觀（生成/重建）只是一半，另一半是會不會隨真實互動/狀態演化**」。closed-loop 的 `sim-in-loop-infer` 和可預測孿生的 `sim-in-loop-infer` 是**字面同一條 injection**——自駕把它叫「閉環」，孿生把它叫「live-sync」。**缺的那層，自駕是反應性，孿生是同步；本質都是 Temporal/Injection 那條被人省略的軸。**

> **meta-lesson（與 closed-loop-or-bust 共享）**：**保真度是分層的、要逐項投資的，不是一句「我的孿生很 photoreal」就帶過。** 跨 robotics / 工業 / 手術 / 自駕，被省略的永遠是動力學 + 同步那一層——而那一層，恰恰是「可預測」與「只是好看」的全部差別。

## 7. 參考

主要
- *A Survey on Physics-aware / Interactive Digital Twins from Visual Reconstruction.* arXiv [2504.13159](https://arxiv.org/pdf/2504.13159).（appearance-only → hybrid → physics-integrated → generative-physics taxonomy；「photorealistic yet lack physical grounding」原句；metric-scale 模糊 / 動態變形 / 從視覺欠約束估物理參數 三挑戰）
- *Surgical Digital Twins.* arXiv [2512.00019](https://arxiv.org/html/2512.00019v1).（三層：解剖/幾何 + 組織物理(FEA/bioheat) + 生理即時態；**preop 靜態 vs intraop 即時同步**；real-time deformable registration 為臨床瓶頸；VALIDATED: HeartFlow FFR-CT / MRI 微波消融 bioheat；建議 hybrid 選擇性高保真）
- *MaterialGS / 3DGS-for-sensing.* arXiv [2511.20348](https://arxiv.org/abs/2511.20348).（camera-only 3DGS → material masks → sensor sim；material-for-sensing 非動力學）

工業（分級標註）
- NVIDIA **Omniverse / OpenUSD + Sensor RTX** —— OpenUSD Stage 聚合 CAD+sensor、Sensor RTX 渲染。`BLUEPRINT`：開發者 blog 「閉環」為 simulation-only reference blueprint、無客戶部署/數字。`MARKETING`：行銷頁列 Foxconn/KION/Schaeffler 等客戶但無量化成效。
- **McKinsey 2024** 製造業 digital-twin 調查 —— **86% mission-critical / 14% 已接 live robot fleet**（live-sync 契約落差的硬數據）。

物理 bolt-on GS 前緣
- **PhysGaussian** · **SplatSim** · **RoboGSim** —— 把動力學物理加到 Gaussian Splatting 的代表線（L2 前緣）。

DT-fidelity 方法論
- DT-fidelity 文獻：**fidelity = 精度 + synchronization 機制**（同步是定義性的、非可選）；**multi-fidelity 標配**（低保真自動孿生診斷、高保真 surrogate 用於 critical physics / tight Sim2Real）。

同倉交叉
- [real2sim-twins.md](./real2sim-twins.md)（L1→L2 升級工序） · [overview.md](./overview.md) · [../autonomous-driving-sim/closed-loop-or-bust.md](../autonomous-driving-sim/closed-loop-or-bust.md)（同源命題：閉環≈live-sync） · [../../foundations/3d-aware-generation/generative-gaussian-splatting.md](../../foundations/3d-aware-generation/generative-gaussian-splatting.md)（L1 外觀錨） · [../../cheat-sheet/ontology.md](../../cheat-sheet/ontology.md)

## §8 踩坑日誌

| # | 坑 | 嚴重度 | 來源 | 繞法 |
|---|---|---|---|---|
| 8.1 | **把 photoreal 重建（NeRF/3DGS）直接叫「digital twin」並拿去做決策** —— 它是視覺孿生，沒有物理也沒有同步 | 🔴 High | survey 2504.13159：「a visual reconstruction can look photorealistic yet lack the physical grounding necessary for interactive simulation」 | 任何「孿生」聲明先問：有 L2 物理嗎？有 L3 live-sync 嗎？兩條缺一就只標「視覺孿生」，不准下預測結論 |
| 8.2 | **monocular 重建拿去算碰撞/受力/劑量** —— **metric-scale 模糊**，絕對尺度沒解出來 | 🔴 High | survey 2504.13159：metric-scale 模糊是核心挑戰（monocular 無法解絕對尺度） | 度量尺度列為**必須忠實**（契約表）；用已知尺寸/深度/多視角/sensor 標定錨定絕對尺度，否則所有物理量都不可信 |
| 8.3 | **以為「閉環/孿生已產品化」就等於 live-sync 已落地** —— Omniverse 的閉環是 simulation-only blueprint，不是真實 sensor 回饋 | 🔴 High | NVIDIA 開發者 blog（閉環=sim-only reference blueprint，無部署/數字 → `BLUEPRINT`）；行銷頁客戶清單無成效 → `MARKETING`；McKinsey 86/14 | 嚴格區分 **sim-only 閉環（L2 在 sim 裡跑）** vs **真實 live-sync（L3 真狀態回灌）**；引用工業案例前先確認是哪一種 |
| 8.4 | **把「給 3DGS 加 material」誤當「給 3DGS 加動力學」** —— MaterialGS 補的是 material-for-sensing（外觀給感測），不是行為 | 🟠 Medium | MaterialGS 2511.20348：camera-only → material masks → sensor sim | material（給 sensing）vs 動力學（mass/friction/articulation）分開記；要動力學請看 PhysGaussian/SplatSim/RoboGSim 那條線 |
| 8.5 | **preop 靜態孿生硬塞 live-sync 需求 / 或反過來 intraop 用靜態模型** —— 兩種孿生的契約不同 | 🟠 Medium | 手術 survey 2512.00019：preop=靜態解剖(無 sync 需求) vs intraop=即時同步(隨組織形變) | 先定孿生**屬於哪一類**再定契約：preop 當高保真靜態/規劃孿生即可；intraop 必須 L3，且 real-time deformable registration 是已知瓶頸 |
| 8.6 | **從視覺欠約束直接估物理參數當 ground-truth** —— 物理參數從外觀反推是 under-constrained | 🟠 Medium | survey 2504.13159：「從視覺欠約束估物理參數」列為核心挑戰 | mass/density/friction 從先驗/材質庫估時**標不確定度**；critical physics 用 hybrid 選擇性高保真（手術 survey 建議），別讓估出來的參數冒充量測值 |
| 8.7 | **引用 BMW「30% greenfield 先建孿生」當論據** —— 兩個 NVIDIA 官方頁皆查無出處 | 🟡 Low | 本輪未在 NVIDIA 官方頁找到出處 → `UNVERIFIED` | 不採用為論據；要用先回官方原始頁逐字核對，否則保留 `UNVERIFIED` 標 |
| 8.8 | **「即時同步」想當第六軸塞進 ontology** —— 其實它就是 Temporal=`streaming` + Injection=`sim-in-loop-infer` | 🟡 Low (open) | 本篇 §5：sync 是定義性的，但已被現有兩軸覆蓋（streaming 滾動 + infer-time sim-in-loop） | 不拆新軸；用 Temporal(streaming) × Injection(sim-in-loop-infer) 表達「live-sync」；與 closed-loop-or-bust §8 的「反應性無專屬格子」並列為 ontology 邊界註記 |

[TBD: verify 8.7 — BMW 30% greenfield 數字，回 NVIDIA 官方頁逐字核對；查無則永久保留 UNVERIFIED]
[TBD: verify 8.3 — Omniverse 行銷頁是否在本輪之後補出任何量化 live-sync 成效；若有則可從 MARKETING 升級]
