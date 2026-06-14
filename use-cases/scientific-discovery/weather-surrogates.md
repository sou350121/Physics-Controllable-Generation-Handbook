<!-- ontology-5axis output=field injection=architecture-bias-soft|data-only control=param temporal=autoregressive domain=weather -->

# 天氣神經代理 —— 本手冊唯一上線生產的成功 解構（Weather & Climate Neural Surrogates — the Handbook's Only Operationally-Deployed Success）

> 這頁不重拆任何引擎內部（GNN message-passing、diffusion sampler、FNO spectral conv 已在 foundation 各自有專文）。**這頁回答一個 use-case 層的問題**：在本手冊收錄的所有 surrogate 路線中，**哪一條真的進了 24/7 生產環境、與物理求解器並肩跑、被一個國家級／跨國氣象機構簽字背書**？答案只有一個領域——**全球中期天氣預報**。它進名單，不是因為精度數字漂亮，而是因為它是**唯一一個把 surrogate「契約」走到成熟形態的真實案例**：deterministic 會 blur → diffusion ensemble 修正 → 業務上**與物理模型並行驗證、非取代**。這正是其他所有 surrogate 子場（molecular dynamics、engineering CFD、climate projection）還在 benchmark 或 research-preview 階段時，唯一可以指著說「它在生產」的存在性證明。

---

## 1. TL;DR

- **天氣是本手冊唯一 operationally deployed 的 surrogate 成功**。判準很嚴：不是「跑贏 benchmark」，不是「雲端 demo 可調用」，而是 **operational**——進入氣象機構的 24/7 主跑管線、有 on-call、有 SLA、有與既有物理系統的 hand-off 協議。目前只有 **ECMWF AIFS** 真正跨過這條線（AIFS Single 2025-02-25、AIFS ENS 2025-07-01 operational）。
- **整條路線的核心轉折是 deterministic → probabilistic**。第一代確定性模型（**GraphCast / Pangu-Weather / FourCastNet**）證明了「**architecture inductive bias + 大量 reanalysis 數據 > 嚴格 PDE 求解**」——它們在 mean forecast skill 上擊敗了業務級數值天氣預報（NWP），但都繼承同一個結構病：**用 MSE 類 loss 訓練 → 系統性 blur（模糊化）**，在極端事件、tail risk、颱風強度上失真。
- **修正來自 diffusion ensemble**。**GenCast**（DeepMind, Nature 2024）把確定性 GNN regression 換成 conditional diffusion，輸出 50+ 隨機成員，**這就是 deterministic→probabilistic 的轉折點**——它恢復了確定性模型被 blur 掉的尖銳結構與機率尾部，並在極端熱冷、強風、熱帶氣旋路徑上反超物理 ensemble。
- **成熟的 surrogate 契約 = 並行、非取代**。ECMWF 反覆明說 AIFS 與物理 IFS 是 **complementary（互補）、不是 replacement**。AI 跑得快、能耗低（~1000× 降）、氣旋路徑更準（~20% 改善），但物理模型提供物理一致性錨點與已驗證的可信度基線。**兩者並行交叉驗證**，才是 surrogate 從「論文」走到「業務」的真正成熟形態。
- 一句話總結整頁的論點：**天氣證明了 surrogate 可以上線——但只有在它「承認自己會 blur、用 ensemble 補上 tail、並接受與物理模型並行驗證」的前提下**。這三件事缺一不可，也正是其他 surrogate 子場還沒走完的路。

---

## 2. 確定性世代（GraphCast / Pangu / FourCastNet）+ 為什麼會 blur

第一代神經天氣代理共享同一個 use-case 突破：**首次在真實業務指標上，廣泛擊敗各國氣象機構的數值天氣預報主力系統**。三條路線用三種不同 backbone 達成（引擎細節見各自 foundation 專文，這裡只取 use-case 角度）：

- **GraphCast**（DeepMind, *Science* 2023, [arXiv 2212.12794](https://arxiv.org/abs/2212.12794)）：icosahedral multi-mesh GNN、autoregressive、**訓於 ECMWF ERA5 reanalysis**；**0.25°（~28km）解析度、10 天 lead time、6h autoregressive step**、數百個變數。關鍵實證：在 1,380 個驗證目標中，**對 ECMWF HRES（業界最佳業務確定性系統）勝出 90.3%**；推論 **<1 分鐘** vs 超算數小時。它是把「surrogate 廣泛擊敗 NWP」這個 fact 補上的第一個 data point。
- **Pangu-Weather**（Huawei, *Nature* 2023）：3D Earth-Specific Transformer；1 小時到 7 天 lead time 上**勝 NWP、約 ~10,000× 快**；是**首個被 ECMWF 公開託管（hosted）的純資料驅動模型**——「served（可被外部調用）」這一步的里程碑，但 served ≠ operational（見 §4 區分）。
- **FourCastNet**（NVIDIA, [arXiv 2202.11214](https://arxiv.org/abs/2202.11214)）：**基於 FNO（具體為 AFNO, Adaptive Fourier Neural Operator）**——這條線**把本頁與 neural-operator 直接連起來**，是 spectral-method 譜系在天氣上的代表。週預報 **<2 秒、約 ~45,000× 快**；它在**快時間尺度變數**（地面風、降水、水氣）上強，但**中程 skill 低於 GraphCast / Pangu**——speed 與 skill 的取捨在這條路線上最明顯。

**★ 為什麼會 blur（這是第一代的結構性病灶，也是整頁論證的樞紐）**：

確定性模型用 **MSE 類 loss** 學一個 point estimate。但天氣的下一刻是**機率分布**，不是單一值。當風暴的真實位置不確定時，MSE 最小化的最優解**不是賭一個尖銳位置，而是輸出一個平滑、攤平的場**——這就是 blur 的數學根源：

- **double penalty（雙重懲罰）**：若模型賭一個尖銳的風暴位置但賭錯了，它被罰兩次——一次是「該有風暴的地方沒有」，一次是「不該有風暴的地方有」。
- **penalty ~ ℓ²**：MSE 的懲罰隨偏差**平方**成長，所以模型的理性策略是**規避尖銳賭注、向平滑靠攏**，最終預測收斂到**接近 ensemble-mean** 的模糊場。
- **後果**：極端事件 intensity（颶風中心氣壓、極端降水峰值）被系統性 over-smooth、低估；tail risk 失真。這不是 bug，是 MSE-on-deterministic-output 的**必然產物**。

修法有二：**(a) 改 ensemble / diffusion**——直接建模分布而非點估計，從根上繞開 double penalty（這是 §3 的 GenCast）；**(b) 改良 spectral loss**——在頻域加項懲罰過度平滑，恢復小尺度尖銳結構。第一代用 (b) 緩解，第二代用 (a) 真正修正。

---

## 3. 機率世代（GenCast diffusion ensemble 修 blur）

**GenCast**（DeepMind, *Nature* 2024, [arXiv 2312.15796](https://arxiv.org/abs/2312.15796)）是 **deterministic → probabilistic 的轉折點**，也是本手冊「diffusion 修 blur」最乾淨的真實案例。它把 GraphCast 的確定性 GNN regression 換成 **conditional diffusion model**，target 從「下一個大氣狀態的點估計」改成「下一個大氣狀態的**條件分布**」：

- **diffusion ensemble**：用不同 noise seed 反覆 sample，產出 **50+ 個隨機的 15 天成員**；ensemble spread 來自 diffusion 的隨機性，而非物理擾動結構。**這正是修 blur 的關鍵**——建模整個分布，double penalty 不再逼模型平滑化，尖銳的風暴結構與機率尾部被恢復。
- **規格**：**0.25° 解析度、12h step（vs GraphCast 6h，減半 rollout 深度→減半 drift）、80+ 變數**。
- **速度**：**單張 Cloud TPU v5、8 分鐘出一條 15 天軌跡**（per member，成員間並行）；對比物理 ensemble 在數千 CPU core 跑數小時。
- **勝物理基準**：在 1,320 個驗證目標中，**對 ECMWF ENS（51 成員業務 ensemble）勝出約 ~97%**；**lead time > 36h 區段升到 99.8%**。
- **它強在哪（恰好是第一代 blur 最嚴重的地方）**：**極端熱／冷、強風、熱帶氣旋路徑**（論文以颱風 **Hagibis** 路徑預測 demo）、**風力發電出力預測**。並且**開源**。

把 §2 與 §3 並讀，論證閉環就清楚了：**GraphCast 解決 mean forecast skill；GenCast 解決 uncertainty quantification 與 tail。** 前者證明 surrogate 能贏 NWP 的平均技巧，後者補上「能贏 NWP 的機率與極端」——而**後者才是 surrogate 能進業務 ensemble 的最後一公里**，因為氣象業務的核心產品就是機率與極端預警，不是單一確定值。

> **延伸補充（research，擴大版圖）**：**Aurora**（Microsoft, *Nature* 2025）是一個 **1.3B 參數的 foundation model**，訓於 >100 萬小時地球物理資料，可微調到**空氣品質、海浪、熱帶氣旋**等下游任務——把神經代理從「純天氣」**擴到大氣化學與多圈層**。它代表 surrogate 的「foundation-model 化」方向，但屬 research/微調示範，尚非 §4 意義下的 operational。

---

## 4. 上線生產（ECMWF AIFS：與物理 IFS 互補不取代）

這一節是整頁的**重心**，因為它是「operationally deployed」這個 claim 的**唯一硬證據**。先把三個常被混為一談的部署層級拆乾淨——**這個區分是本頁的方法論貢獻**：

| 部署層級 | 定義 | 本頁案例 |
|---|---|---|
| **research** | 論文 + 開源 weights，能復現 | GraphCast 原始發表 · GenCast · Aurora |
| **served** | 被機構公開**託管**、外部可調用，但非主跑管線 | Pangu-Weather（首個 ECMWF 公開託管的純資料驅動模型） |
| **deployed (operational)** | 進入 **24/7 主跑業務管線**、有 SLA、與既有系統有 hand-off 協議 | **ECMWF AIFS（唯一）** |

**ECMWF AIFS（Artificial Intelligence Forecasting System）—— 最強的「上線生產」證據**（[ECMWF 官方公告](https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwfs-ai-forecasts-become-operational)）：

- **AIFS Single（確定性）於 2025-02-25 進入 operational**，**24/7 與物理 IFS 並行（parallel）運行**。
- **AIFS ENS（50 成員機率 ensemble）於 2025-07-01 進入 operational**——把 §3 的 probabilistic 路線正式帶進業務 ensemble 產品線。
- **量化收益**：**熱帶氣旋路徑約 ~20% 改善**；**能耗約 ~1000× 降低**（相對於跑同等物理預報的算力）。
- **最關鍵的態度——契約的成熟形態**：**ECMWF 明確聲明 AIFS 與 IFS 是 complementary（互補）、非取代（not a replacement）**。

為什麼「並行、非取代」才是 surrogate 契約的**成熟形態**，而非過渡權宜？

1. **物理模型是可信度錨點**。IFS 基於守恆律與已驗證數十年的物理，提供 surrogate 沒有的**物理一致性保證**與**長期可信度基線**。surrogate 的 architecture-bias-soft（見 §6）**不嚴格守恆**，長 rollout 會累積非物理 drift——並行的物理跑就是它的「對照組」與安全網。
2. **並行 = 持續交叉驗證**。把 AI 與物理放在同一業務管線同時跑，是**對 surrogate 最誠實的線上監測**：一旦 AI 在某個 regime 偏離物理基線，立即可見。這把「surrogate 會在 OOD 氣候 regime 失準」這個已知風險，從「希望它不出事」變成「出事看得見」。
3. **這正是 surrogate 進真實高風險系統的通用模板**。不是「AI 取代物理」，而是「**AI 補物理的速度/成本/某些技巧短板，物理補 AI 的守恆/可信度短板，兩者並行驗證**」。天氣是第一個把這個模板走通並寫進業務流程的領域——這就是它配當本手冊**唯一 operationally deployed 成功**的全部理由。

---

## 5. 對比表（系統 × 解析度 / 技法 / 速度 / 勝物理基準 / 部署狀態）

| 系統 | 機構 / 年 | 解析度 · lead time · step | 核心技法 | 速度（vs 物理） | 勝物理基準 | 部署狀態 |
|---|---|---|---|---|---|---|
| **GraphCast** | DeepMind 2023 | 0.25° · 10 天 · 6h AR | icosahedral multi-mesh **GNN**（architecture-bias-soft）· MSE → **blur** | **<1 min**（vs 超算數小時）| **勝 HRES 於 1,380 目標的 90.3%** | research（啟發 AIFS） |
| **Pangu-Weather** | Huawei 2023 | （次小時–7 天）| 3D Earth-Specific **Transformer** · MSE → blur | **~10,000×** | 1h–7day **勝 NWP** | **served**（首個 ECMWF 公開託管純資料驅動模型） |
| **FourCastNet** | NVIDIA 2022 | （週預報）| **FNO / AFNO**（connects to neural-operator）· spectral | **<2 s · ~45,000×** | 強於**快時間尺度**變數；中程 skill **低於** GraphCast/Pangu | research |
| **GenCast** ★ | DeepMind 2024 | 0.25° · 15 天 · 12h AR | **diffusion ensemble**（50+ 成員）→ **修 blur** | **8 min / 15 天**（單 TPU v5）| **勝 ENS 於 1,320 目標的 ~97%（>36h→99.8%）** | research（開源） |
| **Aurora** | Microsoft 2025 | （多任務）| **1.3B foundation model**（訓 >1M hr）→ 微調空氣品質/海浪/氣旋 | UNVERIFIED | UNVERIFIED（任務而異）| research（foundation + 微調示範） |
| **ECMWF AIFS** ★★ | ECMWF 2025 | （業務全球）| graph-encoder + transformer（AIFS Single 確定性 / AIFS ENS 50 成員）| **能耗 ~1000× 降** | 氣旋路徑 **~20% 改善** | **deployed（唯一 operational：Single 2025-02-25 · ENS 2025-07-01）** |

> 讀表要點：**速度欄**各系統測法與基準不同，僅作數量級對照，非同台 benchmark。**勝物理基準欄**的 90.3% / ~97% / 99.8% 來自各論文自報的驗證目標統計（GraphCast vs HRES；GenCast vs ENS）。**部署狀態欄**是本頁最該看的一欄——只有 AIFS 是 **deployed**，其餘是 research 或 served。★ = deterministic→probabilistic 轉折；★★ = 唯一上線生產。Aurora 的速度/勝基準標 **UNVERIFIED**：grounding 未提供對應的量化 sourced fact。

---

## 6. 五軸定位 + 跨路線綜合

**本頁五軸 header**：`output=field | injection=architecture-bias-soft·data-only | control=param | temporal=autoregressive | domain=weather`。逐軸說明本頁（use-case 綜合視角）為何這樣標，以及它與 foundation 各引擎標註的關係：

- **output=`field`**：所有天氣代理輸出的都是**連續大氣場**（風、溫、壓、濕的格點/mesh 狀態），不是像素影片、不是 3D 顯式表徵。與 [GraphCast](../../foundations/neural-surrogates/graphcast.md) / [GenCast](../../foundations/neural-surrogates/gencast.md) / [Pangu](../../foundations/neural-surrogates/pangu-weather.md) / [FNO](../../foundations/neural-surrogates/fno.md) 一致。
- **injection=`architecture-bias-soft` ＋ `data-only`**（本頁取 use-case 綜合標法，故雙標）：這條路線的物理**主要靠兩件事進模型**——(a) backbone 的球面/mesh 對稱性等**架構偏置**（soft inductive bias，**不保證守恆**，這也是 §4「需與物理並行」的根因）；(b) 訓於 ERA5 等海量真實 reanalysis 的**純資料驅動**隱式學習。注意：foundation 專文會把 **GenCast 額外標 `guidance-gradient`**（diffusion 的 score-based sampling）、把訓練的 weighted-MSE 標 `aux-loss`——本 use-case 頁不重拆這些引擎內機制，只取「整條路線靠 soft bias + data」的綜合面，與 [ontology cheat-sheet](../../cheat-sheet/ontology.md) Axis 2 的 `architecture-bias-soft`（anchor 含 GraphCast）對齊。
- **control=`param`**：天氣代理的「控制」其實是**初始條件（initial-condition）**——餵當前/前一步大氣狀態作為 conditioning，沿時間 rollout。在 ontology 中歸 `param`（與 GraphCast/Pangu/GenCast header 一致），不是 text/force/layout 那類顯式語意控制。
- **temporal=`autoregressive`**：一步一步往前生成，下一步 condition 上一步（GraphCast 6h、GenCast 12h step）。典型代價是 **drift 累積**——GenCast 把 step 從 6h 拉到 12h 正是為了減半 rollout 深度、壓制 drift。
- **domain=`weather`**：全球大氣 forecasting。依 cheat-sheet Axis 5，`weather` 與 `fluid` 故意重疊（同 Navier–Stokes 系公式、不同 benchmark/社區習慣）——FourCastNet 的 FNO 在本倉 foundation 標 `domain=fluid`，但其天氣應用屬本頁 `weather`，這個重疊是設計使然，不強行合併。

**跨路線綜合（這頁的價值在「連」，不在「拆」）**：

- **連 deterministic 三引擎**：GraphCast（GNN）/ Pangu（Transformer）/ FourCastNet（FNO）是**同一 use-case 的三種 backbone bet**，共享 `field × architecture-bias-soft × autoregressive × weather` 的五軸骨架，也共享 **MSE→blur** 的同一病灶。差異只在空間 backbone 與速度/skill 取捨（見 §5 表）。
- **連 probabilistic 轉折**：GenCast 在同一五軸骨架上,**只動 injection 一軸**（加 diffusion 的機率採樣），就把 blur 修掉、把 tail 補上——這是本手冊「injection 軸升級 = 質變」最清楚的真實示範。
- **連部署契約**：AIFS 把上述全部收斂進**業務管線 + 與物理並行驗證**，完成 surrogate 契約。
- **連 limits 與 discovery**：本頁講的是「成功」與「契約」；**未解的另一半**——extreme intensity 殘留低估、OOD 氣候 regime（warmer-than-training）外推無保證、長 rollout（>10d）非物理 drift——留給姊妹頁 [surrogate-limits-and-discovery.md](./surrogate-limits-and-discovery.md) 展開。本頁的 §4 並行契約，正是業務上**現階段管理這些 limits 的手段**。
- **連 foundation 引擎內部**：要看 GNN multi-mesh / diffusion sampler / FNO spectral conv 怎麼運作，去 [GraphCast](../../foundations/neural-surrogates/graphcast.md) · [GenCast](../../foundations/neural-surrogates/gencast.md) · [Pangu-Weather](../../foundations/neural-surrogates/pangu-weather.md) · [FNO](../../foundations/neural-surrogates/fno.md)；本頁刻意不重拆。
- **連 use-case 母頁**：本子場在 [overview.md](./overview.md) 的「四個子場」中對應 Weather + Climate 兩格。

---

## 7. 參考

- GraphCast — Lam et al., "Learning skillful medium-range global weather forecasting," *Science* 2023 · [arXiv 2212.12794](https://arxiv.org/abs/2212.12794)
- GenCast — Price et al., "Probabilistic weather forecasting with machine learning," *Nature* 2024 · [arXiv 2312.15796](https://arxiv.org/abs/2312.15796)
- Pangu-Weather — Bi et al., "Accurate medium-range global weather forecasting with 3D neural networks," *Nature* 2023（首個 ECMWF 公開託管的純資料驅動模型）
- FourCastNet — Pathak et al., *基於 FNO/AFNO* · [arXiv 2202.11214](https://arxiv.org/abs/2202.11214)
- Aurora — Microsoft, "A foundation model for the Earth system," *Nature* 2025（1.3B,訓 >1M 小時地球物理資料）· **UNVERIFIED**（無 URL in grounding）
- ECMWF AIFS operational — ECMWF, "ECMWF's AI forecasts become operational," 2025 · [ecmwf.int 官方公告](https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwfs-ai-forecasts-become-operational)
- 本倉 cross-links：[surrogate-limits-and-discovery.md](./surrogate-limits-and-discovery.md) · [overview.md](./overview.md) · [../../foundations/neural-surrogates/graphcast.md](../../foundations/neural-surrogates/graphcast.md) · [../../foundations/neural-surrogates/gencast.md](../../foundations/neural-surrogates/gencast.md) · [../../foundations/neural-surrogates/pangu-weather.md](../../foundations/neural-surrogates/pangu-weather.md) · [../../foundations/neural-surrogates/fno.md](../../foundations/neural-surrogates/fno.md) · [../../cheat-sheet/ontology.md](../../cheat-sheet/ontology.md)

---

## §8 踩坑日誌

- **坑 1：把 served 當成 deployed。** Pangu 被 ECMWF「公開託管、可調用」很容易被誤報成「已上線生產」。**served ≠ operational**——operational 的硬判準是「進 24/7 主跑業務管線 + SLA + 與既有系統 hand-off」。本頁只把 **AIFS** 算 deployed，§4 的三層表就是為了堵這個常見錯報。寫 surrogate 成功案例時，先問「它在 research / served / deployed 哪一格」。
- **坑 2：把 blur 當「模型不夠大/數據不夠多」的工程問題。** blur 是 **MSE-on-deterministic-output 的數學必然**（double penalty + ℓ² penalty → 最優解平滑化），不是 scale 不夠。再大的確定性模型、再多的數據，只要 loss 仍是 point-estimate MSE，仍會 over-smooth 極端。**修法是換目標（diffusion/ensemble 建模分布）或改 spectral loss，不是繼續堆參數。**
- **坑 3：以為 AI 取代了物理模型。** 媒體常把「AI 天氣預報擊敗超算」讀成「AI 取代 NWP」。ECMWF 立場相反——AIFS 與 IFS **complementary、並行驗證**。把這個讀錯，會誤判整個 surrogate 落地路徑：成熟形態不是替換，是**速度/成本短板 ↔ 守恆/可信度短板的互補並行**。
- **坑 4（cheat-sheet Axis 2 cross-axis note）：`architecture-bias-soft` ≠ 守恆保證。** 這條路線的 soft inductive bias（球面/mesh 對稱）只是「物理 flavor」，**不保證守恆律/長期一致性**——這直接推導出 §4「必須與物理並行」的工程結論。若有人把天氣代理當 `hard-constraint` 來信任長 rollout，會踩 OOD regime 與非物理 drift 的雷。本頁 header 故意只標 soft + data-only，不標 hard-constraint。
- **坑 5（cheat-sheet Axis 5 fluid/weather 重疊）：別因 FourCastNet 在 foundation 標 `domain=fluid` 就把它排除在天氣外。** `weather` 與 `fluid` 在 ontology 中故意重疊（同公式、不同社區/benchmark）。FourCastNet 引擎側標 `fluid`，其天氣應用屬本頁 `weather`——這是分流習慣，不是矛盾。
- **坑 6：Aurora 的量化收益不要從記憶補。** grounding 只給 Aurora 的「1.3B / 訓 >1M 小時 / 微調到空氣品質·海浪·氣旋」這幾個 sourced fact，**沒有給速度與勝基準數字也沒給 URL**——本頁對 Aurora 的速度/勝基準一律標 **UNVERIFIED**，§5 表與 §7 已照辦。寫科學代理頁時，凡 grounding 未附 URL 的具體數字，一律標 UNVERIFIED，不臆造。
