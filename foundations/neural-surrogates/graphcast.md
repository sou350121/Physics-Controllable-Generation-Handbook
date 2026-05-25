<!-- ontology-5axis output=field injection=hard-PDE|constraint-loss control=physical-param temporal=autoregressive domain=weather -->

# GraphCast (+ GenCast)

## 1. One-paragraph TL;DR

GraphCast 是 DeepMind 2023 年在 *Science* 發表的中期天氣預報神經代理模型（Lam et al., "Learning skillful medium-range global weather forecasting", Science, 14 Nov 2023）。它是**第一個在 1380+ 個驗證項目中、5–10 天 lead time 上系統性擊敗 ECMWF IFS HRES（業界金標準數值預報）的 ML 模型**，且推理只需單張 TPU v4 約 60 秒 vs IFS 在 supercomputer 集群上跑數小時——速度提升約 1000×。對本 handbook 的意義：它是「neural surrogate 已經 productionize、進 operational pipeline」這條路線最硬的存在證明，把 surrogate 從 PDE benchmark 拉進真實業務系統。GenCast（Price et al., Nature, Dec 2024）是其 diffusion-based ensemble 後繼，補上 GraphCast 在極端事件 / 不確定性量化上的弱點。

## 2. Core mechanism

GraphCast 的核心是一個 **Graph Neural Network on icosahedral multi-mesh**：

```
ERA5 state at t (0.25°, 37 levels, ~6 vars)
        │
        │  Encoder (grid → mesh)
        ▼
   Icosahedral mesh (refined 6×, multi-resolution edges)
        │
        │  Processor: 16 GNN layers, message passing
        │  across all mesh resolutions simultaneously
        ▼
        │  Decoder (mesh → grid)
        ▼
ERA5 state at t + 6h (residual prediction)
```

關鍵設計：

- **Multi-mesh**：把 icosahedron 從 mesh_0（12 nodes）連續細分到 mesh_6（40,962 nodes），所有層的邊**同時存在於同一張圖**裡——一次 message passing 同時做局部細節與全球遠距 teleconnection。這是相對於 MeshGraphNet 純單一解析度 mesh 的關鍵升級。
- **6-hour autoregressive step**：預測 t→t+6h 的 residual，rollout 至 10 天=40 步。
- **Training loss**：訓練時直接 unroll 12 steps（3 天）做 fine-tune，與真實 ERA5 reanalysis 算 weighted MSE（pressure-level weighted、area weighted）。
- **Resolution**：0.25° lat/lon ≈ 28 km，6 surface vars + 5 atmospheric vars × 37 pressure levels ≈ 235M values per state。
- **Parameters**：~36.7M（小到不可思議——遠少於同期 LLM）。

Injection 軸：GNN 的 permutation/translation symmetry 對球面結構是 **soft inductive bias**（不是嚴格守恆律），所以 `injection=hard-PDE|constraint-loss` 並列——架構偏 hard 一端，訓練 loss 不顯式加 PDE 殘差但靠 ERA5 reanalysis 隱含。

## 3. 五軸定位 + 同軸對手

| Axis | GraphCast | Pangu-Weather | AIFS | FourCastNet | GenCast |
|---|---|---|---|---|---|
| **Output** | `field` (3D atmos state) | `field` | `field` | `field` | `field` (ensemble) |
| **Injection** | hard-PDE\|constraint-loss (GNN symmetry) | implicit-from-data (3D Earth-Specific Transformer) | implicit-from-data (transformer) | hard-PDE (Fourier ops, FNO 變種) | score-conditioned (diffusion) |
| **Control** | physical-param (initial state) | physical-param | physical-param | physical-param | physical-param |
| **Temporal** | autoregressive (6h) | hierarchical (1/3/6/24h cascade) | autoregressive (6h) | autoregressive (6h) | autoregressive + diffusion sampling |
| **Domain** | weather | weather | weather | weather | weather |

同軸對手摘要：

- **Pangu-Weather** (Bi et al., Nature, Jul 2023, 華為)：早 GraphCast 幾個月發表，首個聲稱 beat IFS 的 ML 模型，用 **3D Earth-Specific Transformer + hierarchical temporal aggregation**（1h/3h/6h/24h 四個模型 cascade 拼長 lead time）。Pangu 在某些變量上略勝 GraphCast，但 GraphCast 在 verified-against-IFS 的廣度上更全面。
- **AIFS** (ECMWF, 2024)：ECMWF 自家的 GNN+transformer hybrid，operational evaluation 自 2024 進入，[TBD: verify AIFS 是否已 fully operational 還是 still parallel run as of 2026-05]。
- **FourCastNet** (Pathak et al., NVIDIA, 2022)：Adaptive Fourier Neural Operator，更早的 ML 天氣模型，精度不如 GraphCast 但速度更快，已被 GraphCast/Pangu 在大多數指標超越。
- **GenCast** (Price et al., Nature, Dec 2024, DeepMind)：GraphCast 後繼，**diffusion model on the same mesh**，輸出 50-member ensemble，補 GraphCast 確定性預報在 tail risk / 颶風路徑機率分布上的弱點。

## 4. Where it shines / where it breaks

### ⚡ Shines

- **3–10 day medium-range deterministic skill**：對 z500、t850、msl 等 headline variables，5–10 天 RMSE / ACC 系統性優於 IFS HRES（Lam et al. 報告 ~90% 的 verification targets）。
- **Speed**：單 TPU v4 / 單 A100，~60s for 10-day forecast vs IFS 在 ~1000 CPU cores 上 ~1 小時。Operational cost 降 3+ orders of magnitude。
- **Cyclone track**：Lam et al. 聲稱對熱帶氣旋**軌跡**預測優於 IFS（注意是 track，不是 intensity，見下）。

### ❌ Breaks

- **Extreme event intensity underestimation**：autoregressive MSE training 系統性地把預報 over-smooth——尤其颶風中心氣壓、極端降水峰值。ECMWF 與多位作者在 talks 中明確指出這是當前 ML weather models 的共通病。GenCast 部分緩解（ensemble spread 提供 tail-risk 量化），但 deterministic GraphCast 本身仍有此問題。
- **Tropical cyclone intensity**：軌跡好、強度差。`deepmind/graphcast` GitHub issue 與 ECMWF/NOAA 評估報告皆有對應 caveat [TBD: verify specific GitHub issue numbers]。
- **Out-of-distribution climate regimes**：模型 train 在 ERA5 1979–2017，2018+ 評估；對 climate-change regime shift（warmer-than-training distribution）的外推能力**未有強保證**。
- **Conservation**：架構不嚴格保證 mass / energy 守恆——稱「hard-PDE」是因為 spherical symmetry，但 PDE residual 不為零。長 rollout（>10 天）會出現非物理 drift。
- **Precipitation**：ERA5 precipitation 本身 reanalysis 質量較差，導致 GraphCast 在降水變量上不如 z/t/u/v 等動力學變量。

## 5. Reproduction notes

- **Repo**：`github.com/deepmind/graphcast`（Apache-2.0）。Jax-based。
- **Checkpoints**：DeepMind 釋出三檔 checkpoint：full 0.25° model、small 1° model、ENS-style small model（用於 demo）[TBD: verify exact checkpoint list as of 2026-05]。
- **Data**：ERA5 reanalysis from Copernicus CDS（需註冊，年度約 TB 級下載）。Google Cloud 上有 mirror（`gs://gcp-public-data-arco-era5`）。
- **GPU budget**：
  - Inference 0.25° model：單張 A100 80GB 可跑 10 天 rollout，~1 分鐘。
  - **Training from scratch**：原文用 32 個 TPU v4，~3 週 [TBD: verify exact compute]——對一般 lab 不可行。
- **典型踩坑**：
  - JAX + Haiku 版本鎖死；新版 JAX 不相容，需用 repo 指定 commit 的依賴。
  - ERA5 變量命名與 pressure level 對齊；少一個變量 inference 就 NaN。
  - Mesh 預生成的 pickle 必須對應 resolution。
  - 想 fine-tune 時 unrolling depth × batch × mesh size 會爆 memory；多數復現工作只做 inference，不重訓。

## 6. Cross-line synthesis

- **vs FNO（spectral）**：FNO 在 spectral domain 做 global mixing，理論上對 PDE 解集有 universal approximation，但球面網格不天然 fit FFT；GraphCast 用 icosahedral mesh + GNN 解掉這個 mismatch。經驗上 GraphCast 在球面氣象上明確優於 FourCastNet 系（FNO 變種）。
- **vs Pangu transformer**：Transformer 路線（Pangu、AIFS）vs GNN 路線（GraphCast）——兩者收斂到相近精度，差異在 inductive bias 來源：Pangu 靠 3D Earth-Specific position encoding，GraphCast 靠 mesh topology。Operational 來看 ECMWF 同時跑兩種，作為 multi-model ensemble。
- **Ensemble via GenCast**：GraphCast 確定性 → GenCast diffusion-based ensemble 是 DeepMind 自家的 deterministic→probabilistic 升級路徑。對應 ontology axis 2：`hard-PDE` → `score-conditioned`。
- **Surrogate × VLA / video WM**：GraphCast 完全在 `field` output 空間，與 pixel-video / latent-WM 路線在 evaluation 標準上**不可直接比較**——這正是 overview.md 強調「surrogate 獨立於 video WM」的原因。但若把 weather field 渲染成 visualization video，可作為 high-fidelity world model 的下游（例如氣象節目自動化、災害可視化）。
- **與 diff-sim 的關係**：GraphCast 不可微地用於 control（雖然 JAX 全程可微，但 control horizon 對氣象無意義）。差分模擬器路線在 robotics / contact 上才有意義，氣象 surrogate 一般只做 forecasting，不做 inverse design。

## 7. References

**Canonical**：

1. Lam, R., Sanchez-Gonzalez, A., Willson, M., et al. "Learning skillful medium-range global weather forecasting." *Science* 382, 1416–1421 (14 November 2023). DOI: 10.1126/science.adi2336.
2. Price, I., Sanchez-Gonzalez, A., Alet, F., et al. "Probabilistic weather forecasting with machine learning." *Nature* 637, 84–90 (December 2024). [GenCast paper, [TBD: verify exact issue / page numbers]]
3. Bi, K., Xie, L., Zhang, H., et al. "Accurate medium-range global weather forecasting with 3D neural networks." *Nature* 619, 533–538 (July 2023). [Pangu-Weather]

**Secondary / operational evaluation**：

4. ECMWF AIFS technical memos & blog posts on AI model parallel runs [TBD: verify specific 2024–2025 ECMWF technical memorandum numbers].
5. Pathak, J., et al. "FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators." arXiv:2202.11214 (2022).
6. WeatherBench 2: Rasp et al., "WeatherBench 2: A benchmark for the next generation of data-driven global weather models." J. Adv. Model. Earth Syst. (2024) [TBD: verify exact citation].

## 8. §8 Pitfall log

| § | Source | Issue / Observation | Severity | Workaround |
|---|---|---|---|---|
| §8.1 | `deepmind/graphcast` GitHub issues | JAX / Haiku version drift breaks reproducibility on fresh installs [TBD: verify specific issue numbers] | Med | Pin to repo's `requirements.txt`; use Docker image if provided |
| §8.2 | Lam et al. 2023 §S4 + community talks | Cyclone **intensity** (central pressure, peak wind) under-predicted vs ECMWF HRES, even when track is better | High | Use GenCast ensemble; combine with physics-based intensity model post-hoc |
| §8.3 | ECMWF blog / AIFS papers | ML models inherit ERA5 reanalysis biases (esp. precipitation, polar regions) | Med | Train on operational analyses, not just reanalysis; not yet standard practice |
| §8.4 | Multiple ML-weather post-mortems | Long rollout (>10 days) shows non-physical smoothing & spectral energy decay | High | PDE-Refiner-style iterative refinement; or stop at 10 days (current operational practice) |
| §8.5 | Climate community critique | OOD generalization to warmer climate (vs 1979–2017 training distribution) is **untested for operational climate prediction** | High | GraphCast is a **weather** model, not a climate model—do not extrapolate |
| §8.6 | `deepmind/graphcast` README | Full 0.25° checkpoint requires ~40 GB GPU memory for 10-day rollout | Low | Use small 1° checkpoint for prototyping |
| §8.7 | ECMWF AIFS / integration status | As of [TBD: verify 2026-05-25 status] AIFS / GraphCast-style models run **in parallel** to IFS, not as primary deterministic forecast | Info | Treat ML forecasts as supplementary; IFS still the headline operational product as of [TBD: verify] |
| §8.8 | GenCast paper Sec. limitations | Diffusion sampling cost: 50-member ensemble ≈ several minutes per cycle, still 100× faster than IFS-ENS but not free | Low | Acceptable trade for tail-risk coverage |

---

**TBD checklist for next pass**:

- Confirm exact *Science* DOI / volume / pages for Lam et al. (currently 382:1416 cited; verify against publisher).
- Confirm GenCast *Nature* publication metadata (year/volume/pages).
- Confirm ECMWF AIFS operational status as of 2026-05.
- Pull specific GitHub issue numbers from `deepmind/graphcast` matching §8.1 / §8.6 claims.
- Verify FourCastNet / Pangu intensity-underestimation references (currently second-hand).
