<!-- ontology-5axis output=N/A injection=sim-in-loop-train control=action|trajectory|camera temporal=streaming domain=robotics|driving -->

# CARLA-Air —— 空地一體的城市級模擬 解構

> Tianle Zeng、Yanci Wen、Hong Zhang。arXiv [2603.28032](https://arxiv.org/abs/2603.28032)（v1 2026-03-30 / v2 2026-04-22；cs.RO·cs.AI·cs.CV·cs.HC）。程式碼 [github.com/louiszengCN/CarlaAir](https://github.com/louiszengCN/CarlaAir)（2026-03-19 建立，約 991★，維護中）。專案站 [carla-air.com](https://www.carla-air.com/)。
>
> **為什麼值得收進 aerial-sim 名單**：前面七套 aerial sim（[對比見此](./aerial-sim-stack.md)）拼的是「動力學精度 × 並行吞吐 × 畫面真實感」這個三角；CARLA-Air 不在這個三角裡比，它補的是**所有純空中模擬器都缺的一塊**——把無人機放進一個**有規則車流、有社會化行人、城市級 photoreal** 的世界，而且**空中與地面共用同一個物理 tick、同一個渲染器**。對「低空經濟 / 城市空域 / 空地協同 / 跨視角感知 / 具身導航（VLN-VLA）」這些題目，它是目前最對口的公開模擬基座。它在 ontology 上唯一同時佔住 `domain=robotics`（空中）與 `domain=driving`（地面）兩格——這就是它的真正座標。

## 一句話總結

**CARLA-Air 不是新模擬器、也不是 CARLA 的 fork——它是一層「組合膠水」：把 Microsoft AirSim 的多旋翼飛控塞進 CARLA 的 Unreal Engine 世界，只改了上游 3 個檔案、約 35 行（`CARLAAirGameMode` 一邊繼承 CARLA、一邊組合 AirSim），讓無人機與地面車輛在同一進程、同一物理 tick、同一渲染器裡被嚴格時空一致地共模擬。** 它的賣點**不是**空氣動力學、**也不是**吞吐，而是**城市真實感 + 空地統一**。

但有一顆必須講清楚的星號：**它的飛行動力學就是 AirSim 原封不動的剛體 6-DoF**——論文只說「aerodynamically consistent multirotor dynamics」，**沒有**記載任何 rotor/blade-element 模型、馬達動態、阻力係數、地面效應或風擾建模。所以拿它練「貼地高速、強風、螺旋槳尾流」這類靠空氣動力學的科目，會跟單獨用 AirSim 一樣不夠（見 [generative-aerial-data](./generative-aerial-data.md) 與 [Swift 殘差法](./champion-level-drone-racing.md) 的對照）。它的吞吐也只有 **~20 FPS 單環境**，沒有 GPU 並行多環境、不可微、也沒有任何 sim-to-real 真機實證。把這顆星號看懂，才知道該拿它做什麼、不該拿它做什麼。

## 怎麼運作（架構）

核心是「**兩套物理引擎、一個 Unreal 進程**」：CARLA 用 PhysX 管地面車輛與行人，AirSim 用 FastPhysics 管無人機，兩者掛在同一個 Unreal Engine 4.26 渲染迴圈裡。關鍵工程點是**時間與座標的嚴格對齊**——無人機物理跑在獨立 CPU 執行緒約 1000 Hz，渲染 tick 約 20 Hz（等於每出一幀畫面、無人機物理已經走了約 50 個 substep），而 CARLA 的左手座標系與 AirSim 的 NED 座標系被對齊到**誤差 0.0000 m**。

```
   ┌──────────── 單一 Unreal Engine 4.26 進程（渲染 tick ~20 Hz）────────────┐
   │                                                                          │
   │   CARLA 0.9.16 (PhysX)                AirSim 1.8.1 (FastPhysics)         │
   │   ├ 地面車輛 / 行人 / 交通規則         ├ 多旋翼剛體 6-DoF 飛控            │
   │   └ 隨渲染 tick 同步                   └ 無人機物理 ~1000 Hz（獨立執行緒）│
   │            └────── 座標對齊：CARLA 左手系 ↔ AirSim NED（誤差 0 m）──────┘ │
   │                              │                                            │
   │                     共用渲染器 + 共用時間軸                               │
   │                              ▼                                            │
   │   最多 18 種同步感測：RGB / depth / 語義+實例分割 / LiDAR / radar /        │
   │                       表面法向 / IMU / GNSS / 氣壓計 …（空地同場景）       │
   │                              │                                            │
   │   對外：CARLA Python API（89/89 測試過）+ AirSim Python API + ROS2（63 topic）│
   └──────────────────────────────────────────────────────────────────────────┘
        只改上游 3 檔、~35 行：CARLAAirGameMode「繼承 CARLA、組合 AirSim」
```

三個設計選擇決定了它的性格：

- **「組合」而非「重寫」**：它不另造物理、不 fork CARLA，而是把兩個成熟堆疊接起來。好處是**兩邊的原生 Python API 與 ROS2 都零改動可用**（CARLA 89/89 API 測試通過、ROS2 跑 63 個 topic），既有 CARLA 自駕程式碼幾乎能直接搬上來、再加一架無人機；壞處是**它繼承了兩邊的所有侷限**，尤其 AirSim 的剛體飛行模型與「上游已封存（archived）」的版本鎖死問題。
- **單一 tick 共模擬，而非橋接（bridge）**：很多空地系統用 ROS bridge 把兩個獨立模擬器串起來，代價是同步開銷與時序漂移。CARLA-Air 把無人機直接放進 CARLA 的 Unreal 場景，**省掉跨進程同步**，換來嚴格的空地時空一致——這對「無人機看著地面車流做決策」這類跨視角任務是硬需求。
- **城市真實感免費繼承**：因為渲染器就是 CARLA 的，無人機視角直接拿到 CARLA 的 photoreal 城市（測過 Town01–05、Town10HD 等 13 張城市圖、14 種天氣預設）。這是它相對純空中模擬器最大的單點優勢。

## 五軸定位與同類對手

| 系統 | output | injection | control | temporal | domain | 一句話差異 |
|---|---|---|---|---|---|---|
| **CARLA-Air** | N/A | sim-in-loop-train | action·trajectory·camera | streaming | **robotics + driving** | 唯一空地同 tick + 城市 photoreal |
| AirSim（單獨） | N/A | sim-in-loop-train | action·trajectory·camera | streaming | robotics | 有無人機 photoreal，但無豐富地面車流生態 |
| Flightmare | N/A | sim-in-loop-train | action·trajectory | streaming | robotics | Unity photoreal aerial，無地面交通 |
| Isaac-Pegasus | N/A | sim-in-loop-train | action·trajectory·param | streaming | robotics | Omniverse RTX + PX4，photoreal 但無城市駕駛生態 |
| 基底 CARLA | N/A | sim-in-loop-train | action·trajectory | streaming | driving | 純地面，沒有空中 |

> **Cross-axis 必要說明**（呼應 ontology Check 9b/9c）：CARLA-Air 是模擬器，故 `output=N/A`；它在 `domain` 軸上同時標 `robotics|driving`，是因為它真的把兩個 domain 放進同一場景共模擬——這不是含糊其辭，而是它存在的理由。它沒有用到 `generalist`（非白名單），因此不觸發 Check 9c。

它真正的對手其實不是別的 aerial sim，而是「**任何想同時要城市車流生態 + 空中視角**」的需求方案：要嘛接受 ROS bridge 串兩個模擬器的同步開銷，要嘛用 CARLA-Air 的單 tick 共模擬。

## 強在哪 / 崩在哪

**⚡ 強（真正領先的場景）**

- **空地統一、單一物理 tick**：無 bridge 同步開銷，空中與地面嚴格時空一致——做空地協同、跨視角（cross-view）感知、車-機協作這類任務的首選公開基座。
- **城市級 photoreal 免費繼承**：13 城市圖 × 14 天氣，無人機俯視/斜視城市場景的畫面真實感遠勝純空中模擬器。
- **18 種同步多模態感測**：一個場景同時出 RGB/depth/分割/LiDAR/radar/IMU/GNSS… 對「建多模態資料集、餵感知或具身模型」極友善（搭 [generative-aerial-data](./generative-aerial-data.md) 看資料用途）。
- **零改動復用既有生態**：CARLA + AirSim 雙 Python API + ROS2 全保留，舊程式碼搬遷成本低。
- **讓 archived 的 AirSim 飛控續命**：AirSim 上游 2022 已封存，CARLA-Air 把它的飛控接到一個仍在維護的環境裡——對還在用 AirSim 資產的團隊是一條延壽路。

**❌ 崩（已知失效邊界）**

- **飛行動力學只到 AirSim 剛體級**：無 rotor/blade-element、無馬達動態、無地面效應、無風擾建模。貼地高速、強風、尾流交互這些**靠空氣動力學的科目它撐不住**——這跟單獨用 AirSim 是同一個天花板。
- **~20 FPS、單環境、無 GPU 並行多環境**：論文實測「3 車 + 2 行人 + 1 無人機 + 8 感測器」約 19.8±1.1 FPS（純地面 baseline 28.4 FPS，整合開銷約 30%）。比起 Aerial Gym/Isaac 的數千並行環境，**RL 吞吐低好幾個數量級**；GPU 並行多環境是作者列的 future work，現在沒有。
- **不可微**：別指望 first-order policy gradient / diff-MPC 這條路。
- **無 sim-to-real 實證**：驗證全在模擬內（如 precision landing 誤差 <0.5 m），**沒有任何真機遷移結果**。
- **工程毛邊**：換地圖要**重啟整個進程**；多無人機 >2 雖能跑但未正式驗證；高密度場景效能未充分刻畫（作者標為 active engineering target）。
- **版本鎖死在老堆疊**：UE 4.26 / CARLA 0.9.16 / AirSim 1.8.1，且 AirSim 上游已 archived——長期維護有風險。

## 復現

- 原始碼 + 預編譯二進位都有：Ubuntu 20.04/22.04（HuggingFace / 百度網盤）與 Windows 11 x86_64；版本 v0.1.7（2026-03）。
- 跑得起來的證據：CARLA API 89/89 測試通過、ROS2 63 topic 驗證、RL 流程 357 次 spawn/destroy 零崩潰。
- **License 需自行確認**：README 寫程式碼 MIT + CARLA 資產 CC-BY，但 GitHub 自動偵測回傳 `NOASSERTION`（沒解析出單一 SPDX）。商用前請以實際 LICENSE 檔為準（`UNVERIFIED`）。
- 論文署名單位、以及 README 自稱的「peer-reviewed」皆**未在公開頁面查證**（arXiv 本身是技術報告）——標 `UNVERIFIED`。

## 跨路線綜合

放回本手冊「外觀靠生成、動力學靠物理」的框架：**CARLA-Air 是『物理 + 真實外觀』那層 substrate 的一個特例——特別之處在於它的『場景』不只是地形，而是一個有車流、有行人、有交通規則的活城市。** 因此：

- 對 **pixel-WM / 生成路線**：它是極好的**條件 ground truth 工廠**——空地同場景出 18 模態，可拿 CARLA-Air 的幀＋幾何當條件去餵 [Cosmos](../../foundations/foundation-physics-models/cosmos-wfm.md) 這類 video-WM 做 photoreal 增廣，補足純空中模擬器拿不到的「城市動態背景」。
- 對 **動力學**：它**不解決**空氣動力學問題（那仍是 AirSim 天花板）——要高保真飛行動力學，仍得回到 [RotorPy / 殘差法](./aerial-sim-stack.md) 那條路。它的價值在 domain × 外觀，不在 aero。
- 對 **空地具身 / VLN-VLA**：規則車流 + 社會化行人當無人機的動態背景，是「無人機在城市裡看著人車做決策」這類任務難得的公開資料源。
- 跨冊：它生成的城市空地資料，最終要餵給 Spatial-Handbook 的感知端消費——對齊問題（尤其 IMU 噪聲模型與 camera-IMU extrinsic）見 [Bridge: Aerial Embodiment](../../bridge-to-spatial/aerial-embodiment.md)；它和純空中七套的取捨見 [Aerial Sim Stack 對比](./aerial-sim-stack.md)。

## 參考

- **CARLA-Air**（本篇主體）—— arXiv [2603.28032](https://arxiv.org/abs/2603.28032) · [github.com/louiszengCN/CarlaAir](https://github.com/louiszengCN/CarlaAir) · [carla-air.com](https://www.carla-air.com/) · [HuggingFace papers](https://huggingface.co/papers/2603.28032)
- **基底**：CARLA 0.9.16（自駕模擬）· AirSim 1.8.1（Microsoft，2022 已 archived）· Unreal Engine 4.26
- **同類取捨**：[Aerial Sim Stack 對比](./aerial-sim-stack.md)（純空中七套）· [Generative Aerial Data](./generative-aerial-data.md)（資料用途）

## §8 踩坑日誌

| # | 坑 | 嚴重度 | 來源 | 繞法 |
|---|---|---|---|---|
| 8.1 | **飛行動力學僅 AirSim 剛體級**，無 rotor aero / 地面效應 / 風擾 | 🔴 High | 論文僅稱「aerodynamically consistent」，無 aero 細節 | 高保真飛行動力學另接 RotorPy / NeuroBEM 殘差；CARLA-Air 用於場景與感知 |
| 8.2 | **~20 FPS 單環境、無 GPU 並行多環境** | 🔴 High（RL 規模） | 論文實測 19.8±1.1 FPS；GPU 並行列為 future work | 大規模 RL 走 Aerial Gym/Isaac；CARLA-Air 用於真實感資料與評測 |
| 8.3 | **無 sim-to-real 真機實證**（驗證全在模擬內） | 🟠 Medium | 論文僅報 in-sim precision landing <0.5 m | 真機遷移需自行補殘差辨識（參 Swift），別假設零樣本 |
| 8.4 | **換地圖要重啟整個進程**；多無人機 >2 未正式驗證 | 🟠 Medium | repo README / issues | 場景批次設計避免頻繁切圖；多機需自行壓測 |
| 8.5 | **版本鎖 UE4.26 / CARLA0.9.16 / AirSim1.8.1，且 AirSim 上游已 archived** | 🟠 Medium（長期維護） | GitHub repo 依賴 | 接受版本凍結；長期方案考慮 Cosys-AirSim / Project AirSim 路線 |
| 8.6 | **不可微** | 🟡 Low | 論文無可微分宣稱 | 需梯度走 crazyflow（JAX）等可微基座 |
| 8.7 | **License 自動偵測 NOASSERTION**（README 稱 MIT+CC-BY） | 🟡 Low | GitHub API vs README 不一致 | 商用前以實際 LICENSE 檔為準 `UNVERIFIED` |
| 8.8 | 署名單位與「peer-reviewed」**未公開查證** | 🟡 Low | arXiv 技術報告 | 引用時標 `UNVERIFIED`，勿當已發表期刊 |
