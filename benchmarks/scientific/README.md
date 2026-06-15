# Scientific Benchmarks

> neural-surrogate 的科學評測：PDE 求解保真度、天氣預報 skill score、長程 rollout 穩定性，以及最常被忽略的守恆誤差（NN 是否守住質量/能量/動量）。

## 這類在測什麼

這一格的對象是「用神經網路替代傳統 PDE solver / 數值天氣模型」的 surrogate。評測要回答三個層次的問題：

1. **逐點精度**：surrogate 的輸出和高解析度數值參考解差多少？標準是 RMSE / nRMSE，天氣領域再加 ACC（anomaly correlation coefficient）。
2. **長程是否飄移**：surrogate 通常 autoregressive 地 rollout 多步，誤差會累積，可能爆掉或退化成模糊均值。測 rollout stability、frequency drift。
3. **是否守恆**：這是 surrogate 相對純數值解最致命的弱點。FNO / GraphCast 這類模型**不內建**質量/動量/能量守恆，純靠資料擬合，所以全域守恆量會隨 rollout 持續漂移。PDEBench 把這點顯式做成指標（cRMSE），是少數正面測量守恆違背的 benchmark。

換句話說：一個 surrogate 可以 RMSE 很漂亮，卻在物理上不自洽（不守恆、power spectrum 被抹平）。誠實的評測必須把這兩條線分開看。

## Benchmark 全景

| Benchmark | 測什麼 | 主要 metric | 已知缺陷 / 可被 game | 出處 |
|---|---|---|---|---|
| PDEBench | 多類時間相依 PDE（流體/反應擴散/Burgers/CFD 等）surrogate 精度與物理一致性 | RMSE, nRMSE, MaxErr, **cRMSE（守恆違背）**, bRMSE（邊界）, fRMSE（低/中/高頻 Fourier 帶） | 多為合成模擬資料、固定解析度；高分不保證真實湍流場景遷移 | arXiv 2210.07182 |
| WeatherBench（v1） | 全域中期天氣預報 skill（data-driven 對標數值模型） | RMSE, ACC（少數變量/解析度） | 任務窄、解析度粗；早期 baseline 易被後來模型刷爆 | arXiv 2002.00469 |
| WeatherBench 2 | 下一代 data-driven 全域天氣模型統一評測（對標 ECMWF IFS） | RMSE, ACC, SEEPS（降水）, Bias；**CRPS + spread-skill（ensemble）**；power spectra | deterministic ML 模型靠最小化 MSE → 預測模糊均值，**RMSE-skill 漂亮但 power spectrum 被抹平**；spectra 對得上只是「真實」的必要非充分條件 | arXiv 2308.15560 |
| The Well | 大規模多 regime 物理模擬資料集（16 datasets，含湍流/輻射冷卻/相對論性 MHD，約 15TB） | 跨資料集統一 RMSE 類指標（HDF5 + 標準化座標/正規化） | 主要是資料集而非排行榜協議；守恆/長程指標未被強制當主指標 | arXiv 2412.00568（NeurIPS 2024 D&B） |
| Matbench Discovery | 材料穩定性預測：機器學習力場/模型篩 candidate 晶體的命中率 | 穩定度判定（相對 convex hull）、precision/recall、discovery rate | 排行榜分數可被「發現量」敘事虛胖（見下方 GNoME caveat） | matbench-discovery.materialsproject.org |

### GraphCast / GenCast 的評分方式（WeatherBench 2 體系內）

- **GraphCast**（DeepMind, Science 2023, arXiv 2212.12794 UNVERIFIED — 此處引用其後續評測脈絡）是 deterministic 模型，按 RMSE / ACC 對標 ECMWF IFS。它正是 WeatherBench 2 指出的「deterministic ML 靠 MSE 優化 → 小尺度變異性顯著下降、spectrum 被抹平」的典型：**skill score 贏，但物理一致性（power spectrum）輸**。
- **GenCast**（DeepMind, arXiv 2312.15796）是 diffusion-based ensemble 模型，用 **CRPS** 評分；論文宣稱在 1320 個（變量×lead time×層）組合中 97.4% 顯著優於 ECMWF ENS（p<0.05），lead time >36h 時達 99.8%。ensemble + CRPS 正是為了繞開 deterministic 模型的模糊偏置——機率預報不需要 hedge 成均值，因此能保住小尺度變異性。

## 怎麼誠實讀分數

- **skill score（RMSE/ACC/CRPS）≠ 物理一致性。** 一個 deterministic 模型可以靠輸出模糊均值把 RMSE 壓低，但 power spectrum 被抹平、小尺度結構消失。WeatherBench 2 原話：spectrum 對得上「是 realism 的必要非充分條件」。
- **deterministic spectral blur 是系統性假象。** 最小化 MSE 的模型會「賭注下在分佈均值上」（hedge their bets），lead time 越長越模糊。看到漂亮的 RMSE-skill 曲線，先去查 power spectra 與 ensemble spread。
- **NN surrogate 不內建守恆 → 會 drift。** FNO / GraphCast 這類純資料驅動 surrogate 沒有硬約束守住質量/動量/能量，全域守恆量隨 rollout 累積誤差。看 PDEBench 要把 **cRMSE** 跟 RMSE 分開讀；只報 RMSE 而不報守恆誤差的論文，等於把最關鍵的物理弱點藏起來（後續有 PCNO / 守恆校正 FNO 等方法試圖補，arXiv 2505.24579 UNVERIFIED 細節）。
- **發現量虛胖（GNoME 式）。** 材料領域 DeepMind GNoME 宣稱發現 220 萬新晶體，但 UCSB 等獨立分析一個子集後指出，依「credible / useful / novel」三項檢驗「尚未找到任何顯著新穎的化合物」，多為已知材料的瑣碎變體。排行榜命中率高 ≠ 真正有用的新發現。讀「discovery rate」時要區分「通過穩定性判定」與「真有科學價值」。

## 現況與缺口

- **最大缺口：守恆/物理一致性很少被當主指標。** 多數 surrogate 論文以 RMSE 為頭條，cRMSE / power spectrum / spread-skill 退居附錄甚至不報。PDEBench 與 WeatherBench 2 提供了正確的指標集合，但社群慣性仍偏向單一逐點誤差。
- **長程穩定性測得不足。** 真實部署需要長 horizon autoregressive rollout，但很多 benchmark 只報短 horizon。frequency drift、blow-up、退化成均值這些失效模式缺乏統一協議。
- **真實資料 vs 合成模擬。** PDEBench / The Well 多為數值模擬產生的乾淨資料，對真實觀測（含噪聲、稀疏採樣）的遷移性評測仍薄弱。

## 連回

- 上一層總覽：[../overview.md](../overview.md)
- 物理評測基礎：[../../foundations/evaluation-physics/overview.md](../../foundations/evaluation-physics/overview.md)
- neural-surrogate 模型卡：[GraphCast](../../foundations/neural-surrogates/graphcast.md) · [GenCast](../../foundations/neural-surrogates/gencast.md) · [FNO](../../foundations/neural-surrogates/fno.md) · [Pangu-Weather](../../foundations/neural-surrogates/pangu-weather.md) · [MeshGraphNet](../../foundations/neural-surrogates/meshgraphnet.md)
- 守恆違背地圖：[../../crossing/conservation-violation-atlas/overview.md](../../crossing/conservation-violation-atlas/overview.md)
- 科學發現應用：[../../use-cases/scientific-discovery/overview.md](../../use-cases/scientific-discovery/overview.md)

## 參考

- PDEBench: An Extensive Benchmark for Scientific Machine Learning — arXiv 2210.07182（cRMSE/bRMSE/fRMSE 物理一致性指標）
- WeatherBench: A Benchmark Dataset for Data-Driven Weather Forecasting — arXiv 2002.00469
- WeatherBench 2: A Benchmark for the Next Generation of Data-Driven Global Weather Models — arXiv 2308.15560（RMSE/ACC/CRPS/spread-skill/power spectra；deterministic blurring caveat）
- GenCast: Diffusion-based Ensemble Forecasting for Medium-Range Weather — arXiv 2312.15796（CRPS 對標 ECMWF ENS）
- GraphCast: Learning Skillful Medium-Range Global Weather Forecasting — Science 2023（arXiv 2212.12794，UNVERIFIED 編號待核）
- The Well: A Large-Scale Collection of Diverse Physics Simulation Datasets — arXiv 2412.00568（NeurIPS 2024 Datasets & Benchmarks）
- Matbench Discovery（materialsproject）+ GNoME「發現量虛胖」獨立批評（UCSB credible/useful/novel 三檢驗，UNVERIFIED 出處編號）
- 守恆校正 / physics-consistent neural operator（PCNO、conservation-preserved FNO 等）— 例 arXiv 2505.24579（UNVERIFIED 細節）
