<!-- ontology-5axis output=field injection=architecture-bias-soft|aux-loss|guidance-gradient control=param temporal=autoregressive domain=weather -->

# GenCast

## 1. One-paragraph TL;DR

GenCast 是 DeepMind 2024 年 12 月發表在 *Nature* 的**機率性中期天氣預報神經代理**（Price, I., Sanchez-Gonzalez, A., Alet, F. et al., "Probabilistic weather forecasting with machine learning", *Nature* **637**, 84–90, published online 4 December 2024; DOI 10.1038/s41586-024-08252-9; arxiv 2312.15796）。它是 [GraphCast](./graphcast.md) 的後繼，把確定性 GNN regression 換成 **conditional diffusion model**，輸出 **50+ 成員的 ensemble**，並在 ECMWF ENS（51 成員 operational ensemble）的 1320 個 verification combinations 上**勝出 97.2%**（lead time > 36h 區段更達 99.8%）。對本 handbook 的意義：它是 surrogate handbook 裡**第一個 Axis 2 injection 從 `architecture-bias-soft` 升級到含 `guidance-gradient`** 的 anchor——把 diffusion 的 score-based sampling 嫁接到 GraphCast 的 mesh GNN 上，補上確定性 surrogate 在 **tail risk / 極端事件 / 颶風強度機率** 上的結構性弱點。換句話說：GraphCast 解決了 **mean forecast skill**，GenCast 解決了 **uncertainty quantification**——這在 surrogate 進 operational 的最後一公里是必須過的關。

## 2. Core mechanism

GenCast 是一個**球面 graph 上的 conditional diffusion model**，target 是「下一個 12h 大氣狀態的條件分布」而非 point estimate：

```
ERA5 state at t-12h, t (0.25°, 13 levels, 6 surf + 5 atmos vars)
        │
        │  Encoder (grid → icosahedral mesh, refined 6×)
        ▼
   Conditional input embedding c = f_enc(x_{t-12h}, x_t)
        │
        │   ┌─────────────────────────────────────┐
        │   │  Diffusion sampler (DDPM / DDIM):    │
        │   │  z_K ~ N(0,I)                        │
        │   │  for k = K..1:                       │
        │   │    z_{k-1} = denoise(z_k, k, c)      │
        │   │    └─ Processor: sparse-transformer  │
        │   │       on icosahedral mesh (不同於    │
        │   │       GraphCast 的 dense MP-GNN)     │
        │   └─────────────────────────────────────┘
        ▼
ΔX (residual) → x_{t+12h} = x_t + ΔX
        │
        │  Repeat autoregressively, 30 steps × 12h = 15-day forecast
        │  Run 50+ times with different noise seeds → ensemble
        ▼
50-member trajectory ensemble (probability distribution over 15-day futures)
```

關鍵設計（與 GraphCast 對位看最清楚）：

- **12h step（vs GraphCast 6h）**：減半 rollout 深度→減半 drift；diffusion sampling 本身比 deterministic forward pass 貴，step 拉大攤平成本。
- **Sparse transformer processor on icosahedral mesh**：sparse attention 跟著 mesh edges 走（不是 dense ViT，也不是 GraphCast 的 dense message-passing GNN）。`google-deepmind/graphcast` README 明文標示「Processor uses a sparse transformer with a different graph connectivity pattern from GraphCast」。
- **Conditional diffusion**：典型 score-based / DDPM 訓練——加噪、學去噪、推理時從 N(0,I) 反向 sample。每個 forecast 跑 ~20–30 denoising steps × 30 autoregressive steps = ~600–900 NN evals per ensemble member。
- **Ensemble via noise sampling**：跟 IFS-ENS 的 SPPT / SKEB 物理擾動完全不同——GenCast 的 ensemble spread 來自 diffusion 隨機性（noise seed + sampling stochasticity），不是顯式 perturbation 結構。
- **解析度 / 變量**：0.25° lat/lon、**13 pressure levels**（與 GraphCast 的 37 不同；與 Pangu 對齊），6 surface + 5 atmospheric vars。
- **Compute**：單張 Google Cloud TPU v5，**15-day forecast 8 分鐘**（per member，所有 member 並行）。對比 IFS-ENS 在數千 CPU cores 數小時——cost reduction ~10⁴ orders。

Injection 軸的雙軌標註：(i) Encoder/Decoder/Processor 在球面 mesh 上的對稱性 → `architecture-bias-soft`（與 GraphCast 同源）；(ii) Diffusion 的 score function 推理時做隨機梯度修正 → `guidance-gradient`；(iii) Training 用 weighted MSE on noisy targets → `aux-loss`。**三標並列**才能在 ontology v2 表達「同時繼承 GraphCast 架構偏置 + 引入 diffusion-based score sampling」這個複合機制——這正是 v2 cross-axis 9b 矩陣鼓勵的 `output=field × injection=guidance-gradient`「✓」格的範例。

## 3. 五軸定位 + 同軸對手

| Axis | GenCast | [GraphCast](./graphcast.md) | [Pangu-Weather](./pangu-weather.md) | AIFS-ENS | ECMWF IFS-ENS |
|---|---|---|---|---|---|
| **Output** | `field` (ensemble of 50+) | `field` (deterministic) | `field` (deterministic) | `field` (ensemble) | `field` (ensemble 51) |
| **Injection** | architecture-bias-soft\|aux-loss\|**guidance-gradient** | architecture-bias-soft\|aux-loss | architecture-bias-soft\|aux-loss | architecture-bias-soft\|guidance-gradient | hard-constraint (PDE solver) |
| **Control** | param (initial state, 2 frames) | param | param | param | param + SPPT/SKEB perturbations |
| **Temporal** | autoregressive (12h × 30) | autoregressive (6h) | autoregressive (hierarchical) | autoregressive | streaming (PDE integration) |
| **Domain** | weather | weather | weather | weather | weather |

同軸對手摘要：

- **[GraphCast](./graphcast.md)**（DeepMind, *Science* Nov 2023）：本 method 的直接前身。GraphCast = mean forecast、GenCast = ensemble；架構共用 encoder/decoder，processor 從 dense MP-GNN 換成 sparse transformer。**評估互補**：在 mean RMSE 上 GenCast 跟 GraphCast 接近（不是設計目標），但在 CRPS / ensemble spread / extreme-event recall 上 GenCast 完勝——是 deterministic→probabilistic 升級的教科書案例。
- **[Pangu-Weather](./pangu-weather.md)**（Huawei, *Nature* Jul 2023）：同期 deterministic 對手，無對應 diffusion ensemble 版本（Huawei 之後重心轉到 Pangu-α LLM）；社區嘗試 `kelvinfkr/Perturbation_AI_weather` 用 perturbation 包 Pangu 但不在同個量級。
- **AIFS-ENS**（ECMWF, operational 2025-07-01, arxiv 2509.18994）：ECMWF 自家 transformer-based ensemble，2025 年 7 月進 operational。架構非 diffusion，是 perturbation-driven transformer ensemble。GenCast vs AIFS-ENS 在 2025 多家評估中互有勝負，沒有 strict dominance（ECMWF 把兩者都當 supplementary product）。
- **IFS-ENS**（ECMWF, 51-member operational ensemble）：物理 baseline。GenCast 在 1320 個 verification combinations 上勝出 97.2%（Price et al. Fig. 1）；> 36h lead time 區段勝率 99.8%——是 ML weather model 對 IFS-ENS 第一次系統性壓制。
- **與 GenCast 同類 diffusion-weather**：NeuralGCM (Google, *Nature* Jul 2024) 走 hybrid neural-physical 路線、Stormer (Microsoft, 2024) 是 transformer regression、Aurora (Microsoft, *Nature* May 2025) 是 foundation model ——只有 GenCast 把 diffusion 機制本身放在中心。

## 4. Where it shines / where it breaks

### ⚡ Shines

- **97.2% verification win rate vs IFS-ENS**：Price et al. *Nature* 2024 §Fig. 1：在 1320 個 (variable, level, lead time, region) 組合上 GenCast > IFS-ENS in CRPS 的比例。> 36h lead time 區段達 99.8%——這在 IFS 60 年 operational 歷史上前所未見。
- **極端事件 tail risk**：對 z500 / t850 / msl 的 90/99 percentile events，GenCast ensemble 比 IFS-ENS recall 更高、Brier score 更低（Price et al. §Fig. 2）。直接結果：颶風 / 熱浪 / 寒潮的 probability of detection 提升。
- **熱帶氣旋路徑機率錐**：GenCast 50-member ensemble 對 TC track 的 uncertainty cone 比 IFS-ENS 窄且 better-calibrated——對保險業 / 應急管理直接可用。**這是 deterministic GraphCast 完全做不到的**（GraphCast 只給單一最可能軌跡）。
- **Compute cost**：15-day 50-member forecast，單張 TPU v5 ~8 min × 50 = ~6–7 小時 sequential，但**所有 member 完全並行**——50 個 TPU 並行 ~8 min total。IFS-ENS 在數千 CPU cores 上 ~數小時。Cost-per-forecast 降約 1000×。
- **Diffusion 的天然不確定性表達**：score-based sampling 是 generative model 中**唯一在數學上 well-posed 的 conditional distribution sampler**——比 dropout-ensemble、deep-ensemble 等替代方案在理論上更乾淨。

### ❌ Breaks

- **Mean forecast skill 不勝 GraphCast**：deterministic mean 在大多數變量上 GenCast 跟 GraphCast 持平或略遜——這是設計取捨，不是 bug。**若任務需要 single best forecast，仍用 GraphCast**。
- **50-member 計算成本**：雖比 IFS-ENS 便宜 1000×，但仍是 GraphCast 的 ~50× 推理成本。對單張 GPU 用戶來說從「~1 min」變「~hours」，部署友善度顯著下降。
- **Diffusion sampling latency**：每 step ~20–30 denoising iterations × 30 AR steps × 50 members = ~30k–45k NN evals per forecast cycle。對 nowcasting (<3h) 用例幾乎無法用，主用 medium-range (3–15 day)。
- **GPU OOM 痛點**：`google-deepmind/graphcast` issue **#135**（finetune on GDAS 0.25° 在 8×H100 上仍 OOM）；issue **#106**（first-run inference 慢，cache 沒生效）；issue **#108**（ensemble parallel 邏輯不明）——open issues 反映實務部署細節遠未完善。
- **Sparse transformer 訓練不穩**：相比 GraphCast 的 dense MP-GNN，sparse-attn processor 在不同硬體上數值精度差異更敏感；issue **#113** 報告 AMD CPU 上 demo 行為與 NVIDIA GPU 不一致。
- **Conditional 是「上 2 幀 → 下 1 幀」**：需要連續 2 個 timestep 作 conditioning（t-12h, t），對 cold-start 與 reanalysis 中斷段不友善。
- **強度 underestimation 部分緩解但未根除**：GenCast 的 ensemble spread 讓 tail-risk **可被量化**，但 single-member intensity bias 仍存在——ensemble mean 上 TC 中心氣壓仍偏弱，只是 spread 涵蓋了真值（Price et al. §Fig. 5）。
- **OOD climate regime**：與 GraphCast 同病——訓在 ERA5 1979–2018，對 climate-change regime shift 無強保證；**GenCast 是 weather model，不是 climate model**。

## 5. Reproduction notes

- **Repo**：[`github.com/google-deepmind/graphcast`](https://github.com/google-deepmind/graphcast)（Apache-2.0；GraphCast + GenCast 共用 repo）。
- **Checkpoints**（4 個 GenCast 變種）：
  - **GenCast 0.25deg <2019**：full 0.25° / 13 levels / 6× icosahedral mesh，trained ERA5 1979–2018，可 evaluate 2019+。
  - **GenCast 0.25deg Operational <2022**：fine-tuned on HRES-fc0 2016–2021，與 operational analyses 對齊（不只 reanalysis）。
  - **GenCast 1.0deg <2019**：低解析度，memory 友善。
  - **GenCast 1.0deg Mini <2019**：4× refined mesh，Colab demo 用，**精度不代表正式 model**（repo README 明文 disclaimer）。
- **Entry point**：`gencast_mini_demo.ipynb`（Colab）——load data → sample → compute loss → gradients；數小時內可跑通。
- **Data**：ERA5 from Copernicus CDS / Google Cloud `gs://gcp-public-data-arco-era5`；operational variant 需要 HRES-fc0 (ECMWF MARS access)。
- **GPU 預算**：
  - Inference (mini demo, 1°)：單張 A100 40GB 可跑，~1–2 min per ensemble member × N members。
  - Inference (full 0.25°)：單張 TPU v5 ~8 min per member；GPU 上 H100 也可跑但 OOM 風險高（issue #135）。
  - **Training**：論文未公開完整 compute；社區估 ~32 TPU v5 × 數週量級。**重訓對外部 lab 不可行**。
- **典型踩坑**：
  - JAX + Haiku 版本鎖死（同 GraphCast §8.1）；新版 JAX 不相容，pin requirements.txt。
  - Ensemble GPU parallelization 邏輯不明（issue #108）——num_ensemble_members=4 on 1 GPU vs =1 on 4 GPUs 是否等價？官方未明確說。
  - AMD CPU 跑 mini demo 行為異常（issue #113）；建議 NVIDIA GPU 或 TPU。
  - Operational variant 的 HRES-fc0 input 與 ERA5 schema 不同，切換 checkpoint 須換 normalization stats（issue #110）。
  - First-run inference 慢（issue #106）——JAX JIT compile 開銷未被 cache，30+ 分鐘 warm-up。

## 6. Cross-line synthesis

- **vs [GraphCast](./graphcast.md)（deterministic → probabilistic 升級）**：架構繼承（encoder/decoder/mesh）、processor 換 sparse transformer、output 從 point estimate 換成 conditional diffusion distribution。**Ontology v2 啟示**：同一 surrogate 家族內，Axis 2 injection 從單一 `architecture-bias-soft` 演化到複合 `architecture-bias-soft|guidance-gradient`——這是 Axis 2 v2 設計的關鍵範例（v1 沒法區分「靜態架構偏置」與「動態 score 引導」）。
- **vs [Pangu-Weather](./pangu-weather.md)（transformer regression）**：Pangu 走 dense ViT + HTA，沒走 generative path。GenCast 證明 **diffusion ensemble > perturbation ensemble** 在球面氣象上的可行性——Pangu 若補 ensemble，社區實驗（`Perturbation_AI_weather`）做加性擾動，效果未達 GenCast 級。
- **Surrogate × generative-diffusion 雙線交叉**：GenCast 是本 handbook 罕見的「surrogate 線」與「diffusion 線」**真正合流**的條目。對應 ontology v2 cross-axis 9b 矩陣的 `field × guidance-gradient`「✓」格——v2 鼓勵這類複合機制標註。下游意義：把 weather field 渲染為 video 給 high-fidelity world model 下游使用時，**ensemble 提供天然的 video generation 多樣性**——是 surrogate→video WM bridge 的物質基礎。
- **Diffusion-on-mesh 範式對其他 surrogate 的啟發**：FNO / NeuralMPM / MeshGraphNet 都可以原則上換上 diffusion head 變 ensemble 版本。社區已有 DiffusionPDE、Score-based PDE solver 等嘗試（arxiv 2308.01138 等）——但都未達 GenCast 在 operational 評估上的成熟度。
- **與 VLA / diff-sim 的關係**：GenCast 在氣象 forecasting 不做 control，與 VLA 的 action 一端無直接耦合。**但 diffusion-based ensemble 的 uncertainty quantification 概念**可直接遷移到 diff-sim 領域——「對 contact 不確定性做 ensemble rollout」是 robotics 社區 2025+ 的新方向。
- **Cross-handbook bridge**：GenCast 對應 VLA-Handbook「foundation model + uncertainty quantification」議題的氣象 instance；對 Spatial-Handbook 則無直接對應（spatial 是感知，不是 forecasting）。

## 7. References

**Canonical**：

1. Price, I., Sanchez-Gonzalez, A., Alet, F., Andersson, T. R., El-Kadi, A., Masters, D., Ewalds, T., Stott, J., Mohamed, S., Battaglia, P., Lam, R., Willson, M. "Probabilistic weather forecasting with machine learning." *Nature* **637**, 84–90 (2025; published online 4 December 2024). DOI: 10.1038/s41586-024-08252-9.
2. Price, I. et al. "GenCast: Diffusion-based ensemble forecasting for medium-range weather." arXiv:2312.15796 (Dec 2023 v1 preprint, 修訂版作為 Nature paper)。

**Secondary / operational evaluation**：

3. Lam, R. et al. "Learning skillful medium-range global weather forecasting." *Science* **382**, 1416–1421 (14 Nov 2023). DOI 10.1126/science.adi2336. [GraphCast 直接前身]
4. Lang, S. et al. "An update to ECMWF's machine-learned weather forecast model AIFS." arXiv:2509.18994 (Sep 2025). [AIFS-ENS 對手]
5. DeepMind Blog. "GenCast predicts weather and the risks of extreme conditions with state-of-the-art accuracy." 4 Dec 2024. <https://deepmind.google/blog/gencast-predicts-weather-and-the-risks-of-extreme-conditions-with-sota-accuracy/>
6. ECMWF Blog / news: "AI ensemble forecasts (AIFS-ENS) become operational" (2025-07-01).
7. CSIRO news release: "AI weather models can now beat the best traditional forecasts" (Dec 2024).
8. ECMWF ai-models-gencast wrapper: `github.com/ecmwf-lab/ai-models-gencast` (community integration into ai-models CLI alongside GraphCast / Pangu / FourCastNet)。

## 8. §8 Pitfall log

| § | Source | Issue / Observation | Severity | Workaround |
|---|---|---|---|---|
| §8.1 | `google-deepmind/graphcast` issue **#106** | First-run GenCast inference 在 GPU 上 ~30–35 min；二次跑仍同樣慢（JIT cache 未生效）。打破「JAX warm-up 後快」的預期 | High | 預先 warm up 並 pickle compiled XLA graph；或固定 batch shape 避免 recompile |
| §8.2 | `google-deepmind/graphcast` issue **#135** | Fine-tune GenCast on GDAS 0.25° 在 **8×H100** + host RAM offload 仍 OOM；社區無確定 fix | High | 改用 1° checkpoint；或降 ensemble batch 為 1；或等 DeepMind 釋出 official fine-tune recipe |
| §8.3 | `google-deepmind/graphcast` issue **#108** | Ensemble GPU parallelization 邏輯不明：num_ensemble_members=4 on 1 GPU 是否等價於 =1 on 4 GPUs？官方未明說 | Med | 視為**不等價**處理；ensemble 數值 reproducibility 需固定 seed + 固定 GPU 拓樸 |
| §8.4 | `google-deepmind/graphcast` issue **#113** | `gencast_mini_demo.ipynb` 在 AMD CPU / 無 GPU 環境上行為異常；fallback CPU path 未充分測試 | Med | NVIDIA GPU 或 TPU 為主；CPU only 視為 demo-only |
| §8.5 | `google-deepmind/graphcast` issue **#110** | Operational checkpoint（HRES-fc0 fine-tuned）與 ERA5 input schema 不同；切換 checkpoint 沒換 normalization stats 會 silent garbage output | High | 對 checkpoint 嚴格綁 normalization stats；inference 前 sanity check 一個已知 lead time 的 RMSE |
| §8.6 | `google-deepmind/graphcast` issue **#127** + #149 | GPU 上 GenCast 跑通與 retrain 流程文件不足；社區需自行拼湊 | Med | 參考 `ecmwf-lab/ai-models-gencast` wrapper；或 issue thread 內社區腳本 |
| §8.7 | Price et al. 2024 §Fig. 5 + 共通病 | TC **強度**（中心氣壓 / 峰值風速）即使 ensemble 也偏弱；ensemble spread 涵蓋真值但 single-member 仍 underestimate（與 GraphCast / Pangu / AIFS 同病） | High | 用 ensemble 90/99 percentile 而非 mean；或 post-hoc bias correction |
| §8.8 | 共通病 | OOD climate regime（warmer-than-training）外推**未驗證**；GenCast 是 weather model 非 climate model | High | 不要做 multi-year 預測；嚴格在 medium-range (15 day) 內使用 |
| §8.9 | Repo README + cross-axis 9b 註記 | `output=field × injection=guidance-gradient` 雖在 v2 矩陣為「✓」，但 diffusion-on-physical-field 是新範式，spectral energy 在 small scale 的行為與 deterministic GraphCast 不同；長 rollout (>15 天) 行為**未充分評估** | Med | 嚴格停在 15 天；對 small-scale spectral 做 PSD diagnostic 後再延伸 |
| §8.10 | DeepMind blog + repo | Mini checkpoint「performance is reasonable, [but] is not representative of the GenCast models」——demo 容易誤導 | Info | 評估 / benchmark 必用 full 0.25° checkpoint；Mini 只做 sanity check |

---

**TBD checklist for next pass**：

- 確認 *Nature* 2024 paper 的 Issue number 與最終 pagination（637 卷 84–90 頁是 online-first 標示，print 版可能微調）。
- 驗證 GenCast vs AIFS-ENS 在 2025 多家評估中的勝負細節（ECMWF 內部報告、JAMES papers）。
- 補 NeuralGCM (*Nature* Jul 2024) 與 Aurora (*Nature* May 2025) 的 cross-reference——它們是 GenCast 同期 hybrid / foundation-model 對手，值得在 §3 表格擴充一輪。
- 補 ecmwf-lab/ai-models-gencast 的 operational 整合狀態（是否進 official forecast pipeline）。
