<!-- ontology-5axis output=field injection=architecture-bias-soft|aux-loss control=param temporal=autoregressive domain=weather -->

# Pangu-Weather

## 1. One-paragraph TL;DR

Pangu-Weather 是華為雲（Huawei Cloud, Shenzhen）2023 年發表在 *Nature* 的中期天氣預報神經代理（Bi, K., Xie, L., Zhang, H., Chen, X., Gu, X., Tian, Q., "Accurate medium-range global weather forecasting with 3D neural networks", *Nature* **619**, 533–538, 5 July 2023; DOI 10.1038/s41586-023-06185-3; arxiv 2211.02556）。它比 [GraphCast](./graphcast.md) 早約 4 個月發表，是**第一個在 *Nature*/*Science* 級期刊上聲稱於確定性預報精度上系統性超越 ECMWF IFS HRES 的 ML 天氣模型**。對本 handbook 的意義：與 GraphCast 構成「同任務、不同 inductive bias」的 A/B 對照——Pangu 押 **3D Earth-Specific Transformer (3DEST) + 多時間步級聯**，GraphCast 押 **icosahedral GNN + 6h autoregressive**——兩者最後在 IFS 對齊上殊途同歸，是 surrogate handbook 裡探討 transformer vs GNN 歸納偏置最乾淨的對位實驗。值得標 `architecture-bias-soft`（3DEST 的 Earth-specific positional bias 是空間/球面 prior，但**不保證**質量/動量/能量守恆）；不是 `data-only` —— 把 3DEST 視為純資料驅動會低估其架構偏置的訊號強度。

## 2. Core mechanism

```
ERA5 state at t (0.25°, 13 pressure levels, 5 upper + 4 surface vars)
        │
        │  Patch embedding (3D patches)
        ▼
   3D Earth-Specific Transformer (3DEST)
   ┌───────────────────────────────────────┐
   │ - Swin-style shifted-window attention │
   │ - Earth-specific positional bias       │
   │   (latitude / longitude / pressure)    │
   │ - Encoder-Decoder (vision-transformer  │
   │   backbone, ~256M params)              │
   └───────────────────────────────────────┘
        │
        ▼
ERA5 state at t + Δt  (Δt ∈ {1h, 3h, 6h, 24h})

   Hierarchical Temporal Aggregation (HTA, 推理時組合):
      24h × n  +  6h × m  +  3h × k  +  1h × j   = lead time T
      (greedy 拆解，最少 step → 最少累積誤差)
```

關鍵設計（與 GraphCast 對位看最清楚）：

- **3D 而非 2D**：把 13 個 pressure levels 當第三個 spatial 軸（不只是 channel），用 3D patch embedding 進 ViT。動機是大氣動力學在垂直方向有強耦合（geopotential / temperature / wind），2D-per-level 會丟掉這個結構。
- **Earth-specific positional bias**：在 Swin 的 relative position bias table 上加經緯/壓力的 Earth-aware encoding，讓網絡能區分赤道 vs 極區、地表 vs 平流層——這是 `architecture-bias-soft` 的精華（v2 ontology Axis 2）。
- **Hierarchical Temporal Aggregation (HTA)**：訓**四個獨立模型**（Δt = 1h, 3h, 6h, 24h），推理時按 lead time 貪婪拆解（例 7d = 168h → 7×24h，不是 168×1h 或 28×6h）。直接後果：累積誤差幾何下降。GraphCast 只用單一 6h step 做 autoregressive rollout，是 Pangu 的對立設計。
- **參數量**：每個 Δt 模型 ~256M（合計 ~1B 跨四模型）；遠多於 GraphCast 的 ~36.7M——架構偏置更重，參數也吃得更多。
- **訓練資料**：ERA5 reanalysis 1979–2017（39 年），0.25° lat/lon × 13 pressure levels；驗證 2018+。
- **推理速度**：單張 V100，7-day forecast ~1.4s（paper claim ~10⁴× faster than IFS；參考 Huawei Cloud blog 與 paper §)。

Injection 軸值 `architecture-bias-soft|aux-loss`：3DEST 的 Earth-aware position 屬 soft architectural prior（不嚴格守恆）；訓練 loss 是 latitude-weighted MSE，可視為 weighted aux-loss。**不是** `hard-constraint`——架構不保證任何 PDE residual 為零。

## 3. 五軸定位 + 同軸對手

| Axis | Pangu-Weather | [GraphCast](./graphcast.md) | [FNO](./fno.md) | FourCastNet | GenCast |
|---|---|---|---|---|---|
| **Output** | `field` (3D atmos state) | `field` | `field` | `field` | `field` (ensemble) |
| **Injection** | architecture-bias-soft\|aux-loss (3DEST, Earth-aware bias) | architecture-bias-soft\|aux-loss (icosahedral GNN symmetry) | hard-constraint (Fourier ops on PDE solution operator) | architecture-bias-soft (AFNO = FNO 變種 + ViT token mixing) | guidance-gradient (diffusion ensemble) |
| **Control** | param (initial state) | param | param | param | param |
| **Temporal** | autoregressive (hierarchical 1/3/6/24h cascade) | autoregressive (6h) | autoregressive | autoregressive (6h) | autoregressive + diffusion |
| **Domain** | weather | weather | weather/fluid (general PDE) | weather | weather |

同軸對手摘要：

- **[GraphCast](./graphcast.md)** (Lam et al., *Science* 14 Nov 2023, DeepMind)：晚 4 個月發表、廣度更大（1380+ verification targets vs Pangu 主推 z500/t850/msl/u10/v10 與 88 個 2018 颱風）。架構選擇正相反：mesh topology bias vs Earth-aware transformer bias。Operational：ECMWF 在 ai-models 框架同時 host 兩者，已知 Pangu 在 **2–4 day lead time 的 TC POD/FAR 領先**（Ho et al. 2024 比較研究，*Weather* RMetS）。
- **FourCastNet** (Pathak, J. et al., arxiv 2202.11214, NVIDIA, Feb 2022)：更早一年的 ML 天氣模型，用 **Adaptive Fourier Neural Operator (AFNO)**——把 FNO 的 spectral mixing 塞進 ViT token-mixer。0.25° 解析度、7-day forecast <2 s。**精度顯著落後** Pangu/GraphCast：paper 報 5-day z500 RMSE，FourCastNet ≈ 462.5 vs Pangu ≈ 296.7 vs operational IFS ≈ 333.7（Bi et al. 2023 Fig. 2 ）。
- **[FNO](./fno.md)**：通用 PDE solution operator；Pangu/FourCastNet 都是它的特化變種。FNO 本身在球面網格上不天然 fit FFT，後續球面 SFNO 解掉這個 mismatch。
- **AIFS** (ECMWF, 2024–2025, arxiv 2509.18994)：ECMWF 自家 transformer-based model，**2025-02-25 進入 operational（1.0.0）**，2025-08-27 升級 1.1.0，AIFS-ENS **2025-07-01 operational**。比 IFS 在中期預報誤差低 5–15%，TC 軌跡領先但**強度低估**——與 Pangu 同病。
- **GenCast** (Price et al., *Nature* Dec 2024, DeepMind)：GraphCast 的 diffusion-based ensemble 後繼；Pangu 沒有對應 ensemble 升級版（Huawei 之後轉做 Pangu-α LLM，氣象 follow-up 不顯著）。

## 4. Where it shines / where it breaks

### ⚡ Shines

- **熱帶氣旋軌跡 (early-stage)**：論文 §Fig.4 比較 2018 年 88 個 named TCs，mean direct position error 全程低於 ECMWF-HRES；最知名案例為 **Typhoon Yutu**：Pangu 比 HRES **早 48 小時** 給出正確的菲律賓登陸路徑。
- **5-day z500 RMSE**：Pangu ≈ 296.7 vs operational IFS ≈ 333.7 vs FourCastNet ≈ 462.5——同期所有 ML 模型最低。
- **Lead time 1–7 天確定性精度**：對 z500/t850/u10/v10/msl 等 headline variables 全面領先 IFS（論文 claim）。
- **推理速度**：單張 V100，7-day forecast ~1.4 s（vs IFS 上千 CPU cores 數小時）。Operational cost 比 IFS 低約 4 個 orders of magnitude。
- **2–4 day lead TC POD/FAR**：2024 多模型橫評（Ho et al., *Weather* RMetS, [TBD: verify exact DOI]）報 Pangu 在 2–4 天 POD 最高、FAR 最低，略勝 GraphCast/FourCastNet。

### ❌ Breaks

- **熱帶氣旋強度仍低估**：軌跡好但中心氣壓/最大風速系統性偏弱——與 GraphCast、AIFS 同病，是 deterministic MSE-trained ML weather models 的共通結構性缺陷（Bi et al. 2023 §論文外的 ECMWF 與 NOAA 後續評估）。
- **無 ensemble**：deterministic-only；對 tail risk / 颶風路徑機率分布無法量化。DeepMind 用 GenCast 補上，Pangu 沒有對應後繼。
- **長 rollout drift**：autoregressive 即便有 HTA 拆解，>10 天仍 over-smooth；spectral energy 在小尺度衰減（與 GraphCast §8.4 同病）。
- **降水**：ERA5 precipitation reanalysis 本身質量差，Pangu 在 precip 變量上不如 z/t/u/v 等動力變量（共通病）。
- **垂直層數有限**：13 個 pressure levels（vs GraphCast 37 個）——平流層動力學表達力弱。
- **GitHub repo 限制**：BY-NC-SA 4.0，**商業使用明確禁止**；ONNX 格式無訓練 code（只放 inference）；4 個 24h/6h/3h/1h ONNX 各 ~1.1GB；ECMWF ai-models-panguweather 有第三方訓練/推理整合。
- **GPU inference 痛點**：repo issue #67 與 yanxingjianken.github.io 報告 ONNX-GPU runtime 配置依 CUDA 11.6 + cuDNN 8.2.4 嚴格鎖死，許多用戶被迫退回 CPU（每步 ~4 分鐘 vs GPU 秒級）。
- **Lite version 限制**：repo issue #70, #75 反映 lite checkpoint（1% compute 訓練）至今未完全釋出。

## 5. Reproduction notes

- **Repo**：[`github.com/198808xc/Pangu-Weather`](https://github.com/198808xc/Pangu-Weather)（BY-NC-SA 4.0，**非商用**）。
- **Mirror / operational wrapper**：[`github.com/ecmwf-lab/ai-models-panguweather`](https://github.com/ecmwf-lab/ai-models-panguweather)（與 GraphCast / FourCastNet 共用 `ai-models` CLI）。
- **Checkpoints**：4 個 ONNX—— `pangu_weather_{1,3,6,24}.onnx`，各 ~1.1 GB（Google Drive / 百度網盤）。**只 inference，無訓練 code**。
- **Dependencies**：`numpy` / `netCDF4` / `pygrib` + ONNX Runtime；GPU 需 CUDA 11.6 + cuDNN 8.2.4（Linux）或 8.5.0.96（Windows）。
- **Data**：ERA5 from Copernicus CDS（年度 TB 級）；google-cloud arco-ERA5 mirror 可用。輸入需 0.25° 全球 13 pressure levels × 5 upper + 4 surface vars，**順序與單位錯一個就 NaN**。
- **GPU 預算**：
  - Inference：單張 V100 16GB，7-day forecast ~1.4 s（論文）。實務上需 ~16–24 GB 啟動 ONNX session。
  - **Training**：論文未公開完整訓練細節；Huawei 自報每模型 192 V100 × 約 16 天（HTA 共 4 個模型 → ~3 月集群時間）。對外部 lab 不可行。
- **典型踩坑**：
  - ONNX-GPU CUDA/cuDNN 版本嚴格鎖死（repo issue #67）；錯版本→ silent fallback CPU，速度差 100×。
  - HTA 拆解順序：論文與 repo 推薦 greedy（先 24h，再 6h，再 3h，再 1h）；隨意順序會引入額外誤差。
  - Lite version 至今未完整釋出（issue #70, #75）；想小規模重訓需自寫 pipeline。
  - 與 GraphCast 比較時注意：Pangu 用 13 pressure levels，GraphCast 用 37 個，verification 需重採樣對齊。

## 6. Cross-line synthesis

- **vs [GraphCast](./graphcast.md)（transformer vs GNN bet）**：兩者同任務、同數據、同目標——架構偏置選擇相反。GraphCast 用 mesh topology 編碼空間結構，Pangu 用 Earth-aware position bias + 3D patch；最終精度相近（Pangu 在 z500 RMSE 略勝、GraphCast 在 verification 廣度勝）。**Ontology 啟示**：兩者都標 `architecture-bias-soft` 而非 `data-only`——架構選擇是 surrogate handbook Axis 2 最有信號的軸，比 loss 設計更決定 generalize 行為。
- **vs [FNO](./fno.md)（spectral 路線）**：Pangu 完全不走 spectral；FourCastNet（AFNO = FNO + ViT）走部分 spectral，精度顯著落後 Pangu/GraphCast。經驗結論：**球面氣象上，spectral inductive bias 不如 Earth-aware/mesh inductive bias**——這是 surrogate 設計者拿來反推 PDE benchmark 結果（FNO 在 Navier-Stokes 上贏）的重要對照。
- **HTA 與 hierarchical Temporal**：Pangu HTA 是「同一空間、多時間 step 訓練 + 推理拼接」，與 ontology v2 Axis 4 `hierarchical` 在精神接近但**未滿足正式定義**（後者要兩層 planner/renderer）。本 dissection 標 `temporal=autoregressive`，HTA 在 §2 內文解釋。
- **與 [GenCast](./graphcast.md#genCast) 的關係**：Pangu 沒有對應 diffusion ensemble；若要為 Pangu 補 tail-risk 量化，方法上可直接用 ensemble perturbation 包 Pangu（已有第三方嘗試，見 `kelvinfkr/Perturbation_AI_weather`）。
- **Surrogate × video WM**：與 GraphCast 同——`field` output 空間與 `pixel-video` 不可直接比較；但 weather field 渲染後可作 high-fidelity world-model 下游應用。
- **Cross-handbook bridge**：Pangu / GraphCast / AIFS 三件套是 surrogate 進 operational 的存在證明，對應 VLA-Handbook「foundation model 進 production」議題的氣象版本。

## 7. References

**Canonical**：

1. Bi, K., Xie, L., Zhang, H., Chen, X., Gu, X., Tian, Q. "Accurate medium-range global weather forecasting with 3D neural networks." *Nature* **619**, 533–538 (5 July 2023). DOI: 10.1038/s41586-023-06185-3. PubMed PMID 37407823. All authors affiliated with **Huawei Cloud, Shenzhen, China**.
2. Bi, K., Xie, L., Zhang, H., Chen, X., Gu, X., Tian, Q. "Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast." arXiv:2211.02556 (Nov 2022 v1 / 修訂版作為 Nature paper 的 preprint)。

**Secondary / operational evaluation**：

3. Lam, R. et al. "Learning skillful medium-range global weather forecasting." *Science* **382**, 1416–1421 (14 Nov 2023). DOI 10.1126/science.adi2336. [GraphCast 對照]
4. Pathak, J. et al. "FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators." arXiv:2202.11214 (Feb 2022).
5. Lang, S. et al. "An update to ECMWF's machine-learned weather forecast model AIFS." arXiv:2509.18994 (Sep 2025). [AIFS operational 1.0/1.1 status]
6. Ho, S. et al. "Comparative evaluation of data-driven weather forecast models performance for medium- to extended-range weather forecasting and tropical cyclone genesis in 2024." *Weather* (RMetS), 2024 [TBD: verify exact volume/pages].
7. ECMWF news release: "ECMWF's ensemble AI forecasts become operational" (2025-07-01); ECMWF ai-models open-source framework: `github.com/ecmwf-lab/ai-models`.
8. Huawei Cloud official announcement: huawei.com/en/news/2023/7/pangu-ai-model-nature-publish (5 July 2023).

## 8. §8 Pitfall log

| § | Source | Issue / Observation | Severity | Workaround |
|---|---|---|---|---|
| §8.1 | `198808xc/Pangu-Weather` issue #67 (Aug 2024) + Yan Xingjian teaching note (Apr 2025) | ONNX-GPU runtime 配置嚴格鎖 CUDA 11.6 + cuDNN 8.2.4；不符 → silent fallback CPU，每步 ~4 min vs GPU 秒級 | High | 用 Docker 鎖版本；或改 ECMWF `ai-models-panguweather` wrapper（自帶相依管理） |
| §8.2 | `198808xc/Pangu-Weather` issue #66 (Aug 2024) | 推理時 masking 行為與訓練不一致；用戶質疑 | Med | 參考 ecmwf-lab fork 的 inference loop；社區尚無共識 |
| §8.3 | `198808xc/Pangu-Weather` issue #69 (Feb 2025) | EarthSpecificBias table 維度與輸入 shape 對不上；自訂 patch 大小會炸 | Med | 嚴格用論文預設 patch (2, 4, 4)；不要改 |
| §8.4 | `198808xc/Pangu-Weather` issue #70 + #75 (2025) | Lite version checkpoint 至今未完整釋出；issue #75 (Nov 2025) 仍開 | Med | 用 full 4 個 ONNX；或第三方 `zhaoshan2/pangu-pytorch` 24h-only PyTorch 重實作（精度未驗證） |
| §8.5 | `198808xc/Pangu-Weather` issue #74 (Aug 2025) | Surface 變量輸出量級異常（用戶報告） | Med | 嚴格按論文單位/順序餵 input；驗證 ERA5 GRIB→NetCDF 轉檔無誤 |
| §8.6 | Bi et al. 2023 §Fig. 4 + ECMWF AIFS 2025 報告 | 熱帶氣旋**強度**（中心氣壓、峰值風速）系統性低估，雖**軌跡**領先 IFS | High | 與物理-based intensity model post-hoc 組合；或等 ensemble 版本（社區 `Perturbation_AI_weather` 嘗試中） |
| §8.7 | Bi et al. 2023 §論文 + repo LICENSE | BY-NC-SA 4.0 + ERA5 ECMWF 雙重 license 限制；**商業使用明確禁止** | Info | 商業部署需自行用 ERA5 重訓（無 training code 公開）；或用 AIFS（更寬鬆 license）/ GraphCast Apache-2.0 |
| §8.8 | Ho et al. 2024 *Weather* + ECMWF 2024–25 evaluation | 與 GraphCast 比較在不同 lead time / 變量互有勝負；不是 strict dominance | Info | Operational 同時跑兩者作 multi-model ensemble（ECMWF 實務）；不要假設 Pangu 全面勝出 |
| §8.9 | 共通病（Pangu / GraphCast / AIFS） | 長 rollout (>10 天) over-smoothing；小尺度 spectral energy 衰減 | High | 停在 7–10 天；或用 PDE-Refiner / GenCast-style diffusion correction |

---

**TBD checklist for next pass**：

- 確認 Ho et al. 2024 *Weather* (RMetS) 的 DOI 與 exact 比較表。
- 確認 Huawei 自報 Pangu 訓練 compute（192 V100 × 16 天 × 4 models）的原始引用——目前是 paper §Methods + 二手 blog。
- Pangu-α LLM 與 Pangu-Weather 共用 backbone 嗎？兩者 codebase 不同 repo，paper 用「Pangu」當品牌但架構是 Swin-3D 變種而非 Pangu-α transformer——需要再 disambig。
- 確認 Pangu 訓練資料是 1979–2017（39 年, paper claim）還是 43 年（Huawei press release claim）——口徑不一。
- 驗證 ECMWF ai-models-panguweather repo 的 operational 用途（official forecast pipeline 是否真的引用 Pangu，還是僅 evaluation）。
