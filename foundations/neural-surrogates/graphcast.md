<!-- ontology-5axis
output: field (3D atmospheric state, 0.25° × 37 levels)
injection: architecture-bias-soft | aux-loss
control: param (initial-condition only)
temporal: autoregressive (6h step, rollout 40)
domain: weather
ref: ../../cheat-sheet/ontology.md §7
-->

# GraphCast 解構（GraphCast Dissection）

> **發布時間**：2023-11 · *Science* 382:1416–1421 · DOI [10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336)
> **論文**：*Learning skillful medium-range global weather forecasting*
> **作者**：Lam, Sanchez-Gonzalez, Willson, et al.（DeepMind 主導 + ECMWF / Google 合作）
> **核心定位**：v2 ontology 上的 `output=field × injection=architecture-bias-soft × temporal=autoregressive` 五軸定位點。是 neural-surrogate zone **唯一已 productionize、進入國家氣象局 operational pipeline** 的 anchor —— 把 surrogate 從 PDE benchmark 拉進真實業務系統的 single existence proof。

**Status:** v0.5 — 解構基於 Science 原文 + DeepMind 開源 repo + ECMWF 2025 operational announcements + 第三方 reproduction 紀錄。完整 1380 verification target 表、AIFS operational hand-off 細節、GenCast cross-reference 待維護者升 v1。
**TL;DR:** ① GraphCast 把 ERA5 atmospheric state 編碼到 icosahedral multi-mesh 上，用 16 層 GNN message-passing + 6h autoregressive step 做 10 天預報；② 核心 trick 是 **multi-mesh**（mesh_0 → mesh_6 所有解析度的邊**同時存在於同一張圖**），一次 message pass 同時做局部對流與全球 teleconnection；③ 對 v2 axis 的關鍵意義：**v2 把它從 `hard-constraint` 重歸到 `architecture-bias-soft`** —— GNN 的球面 permutation symmetry 是 **soft** inductive bias，不是嚴格守恆律，這是 ontology 設計的 honest 修正；④ 關鍵實證：**ECMWF AIFS 2025-02-25 進入 operational status**（採 GraphCast 啟發的 graph-encoder + transformer-processor 混合），是 ML 天氣模型首次成為國家氣象局**主跑 deterministic forecast** 而非平行比對，速度比 IFS HRES 快 ~1000×。

**X-Ray.** GraphCast 是 v2 ontology 上的一個 **structural anchor**，不是「又一篇 paper」。理由：在 neural-surrogate zone，過去六年（PINN / DeepONet / FNO / MeshGraphNet 等）只在 toy PDE benchmark 上跑，**沒有一個進過 operational pipeline**。GraphCast 是第一個把這條軸補上的 single fact —— 而**這個 fact 決定它在 handbook 的地位**，不是它的精度數字。從工程取捨看，它解了三個舊坑：(a) FNO 系（如 FourCastNet）用 FFT 在球面上有 grid mismatch，icosahedral mesh + GNN 天然 fit 球面拓樸；(b) Pangu 用 3D Earth-Specific Transformer 達到相近精度但要 cascade 4 個模型拼長 lead time，GraphCast 用 single autoregressive 6h step 就跑滿 10 天；(c) MeshGraphNet 用單一解析度 mesh，做不到全球 teleconnection 與局部對流 co-modeling，multi-mesh 把這兩端合在一張 graph 上。**它打不開的 envelope**：極端事件 intensity（颶風中心氣壓、極端降水峰值）被 MSE loss 系統性 over-smooth；OOD 氣候 regime 外推（warmer-than-training）無強保證；**長 rollout (>10d) 非物理 drift**，因為 architecture-bias-soft 不嚴格守恆。對 Physics-Gen handbook 讀者意義：GraphCast 是「architecture inductive bias > strict PDE residual」的 winning data point —— 它證明 surrogate 不必走 hard-PDE 一端，soft bias + 大量真實數據（ERA5 40 年 reanalysis）足以擊敗 numerical PDE solver。這是這個 zone 必須寫 anchor 的根本理由。

## 📍 研究全景時間線

```ascii
   2019         2020-22         2023-04           2023-11             2024-12          2025-02
   PINN ─────► FNO ─────────► Pangu Nature ───► YOU ARE HERE ───► GenCast Nature ──► AIFS operational ★
   (toy)       FourCastNet     (transformer      GraphCast Science  (diffusion        (ECMWF 主跑,
               (FFT, spectral)  + cascade)        (GNN icosahedral   ensemble on       graph-encoder
                                                  multi-mesh)        same mesh)        繼承 GraphCast)

   benchmark ────────────────► first beat IFS ──► first broad-beat ──► tail-risk ────► production!
                                (Pangu, 局部)     (GraphCast, 廣度)   (probabilistic)
```

★ = 主要新點：**icosahedral multi-mesh + 6h autoregressive single model**，這個 backbone bet 後被 AIFS 採納進國家氣象局 operational pipeline。**仍未解：extreme intensity under-prediction（GenCast 部分緩解）、OOD climate regime、長 rollout 非物理 drift**（留給下一代 diffusion / hybrid PDE-refiner 路線）。

---

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 同軸對手

| 維度 | FourCastNet (FNO, 2022) | Pangu (transformer, 2023-04) | **GraphCast (2023-11)** |
|---|---|---|---|
| **空間 backbone** | Adaptive FNO（球面強拗 FFT）| 3D Earth-Specific Transformer（position encoding 帶緯經度）| **Icosahedral multi-mesh + GNN** |
| **時序 unit** | autoregressive 6h | **cascade 4 model**（1h/3h/6h/24h 拼長 lead time）| autoregressive 6h **single model** rollout 40 步 |
| **Long-range info flow** | spectral global mixing | self-attention | **multi-resolution edges 同時存在於同一張圖** |
| **Params** | ~75M | ~256M (4 個 cascade 加總) | **~36.7M**（單一模型，遠少於 LLM） |
| **Train loss** | MSE | MSE | weighted MSE（pressure × area）+ unroll 12 steps fine-tune |
| **Operational status (2026-05)** | 學術 | 學術 + 部分商用 | **AIFS 採其啟發, 2025-02 ECMWF operational** |

### 1.2 ⚡ Eureka Moment

> **Icosahedral multi-mesh = 把 mesh_0（12 nodes 全球 6 級 teleconnection）到 mesh_6（40,962 nodes 局部對流）所有解析度的邊，**塞進同一張 graph 同時做 message passing**。一次 forward pass = 短程 + 長程資訊流並行交換。**

這是相對於 MeshGraphNet 純單一解析度 mesh 的關鍵升級。直覺：天氣是 **multi-scale phenomenon**（colombian Andes 對流 vs 北極漩渦 teleconnection），任何 single-scale backbone 都會在某一端漏資訊。Multi-mesh 把這個 multi-scale 結構直接 encode 進 graph topology，**讓 inductive bias 對齊物理**。這是 backbone bet 的核心 —— 不是 transformer 的 attention，不是 FNO 的 spectral，而是 **spatial structure (GNN) + multi-resolution edges**。

### 1.3 信息流（架構圖）

```ascii
   ERA5 state at t (0.25° lat/lon, 37 pressure levels, ~6 vars, ~235M values)
            │
            │  Encoder (grid → mesh)：每個 grid cell → 最近 mesh node 的 GNN 編碼
            ▼
   ┌──────────────────────────────────────────────────────────┐
   │  Icosahedral multi-mesh                                  │
   │                                                            │
   │   mesh_0 (12 nodes)  ───────── 全球 teleconnection 邊     │
   │   mesh_1 (42 nodes)  ──────── 大尺度環流邊                │
   │   mesh_2 (162 nodes) ─────── 行星波邊                      │
   │   ...                                                      │
   │   mesh_6 (40,962 nodes) ── 局部對流 / 邊界層邊             │
   │                                                            │
   │   ★ 所有解析度的邊同時存在於 same graph                   │
   │                                                            │
   │  Processor: 16 GNN layers, message passing 同時跨所有解析度│
   └──────────────────────────────────────────────────────────┘
            │
            │  Decoder (mesh → grid)：每個 grid cell ← 鄰近 mesh node
            ▼
   ERA5 state at t + 6h (residual prediction Δstate)
            │
            ▼  Autoregressive rollout 40 steps = 10 days
   ERA5 forecast at t + 240h
```

對比：FourCastNet 在 spectral domain 全球 mixing，缺局部對流 inductive bias；Pangu 用 attention 但要 cascade 4 個模型才能跨時間尺度；GraphCast 用 **single graph + single 6h step** 就同時跨空間多尺度與時間多 step。

---

## §2 · 數學層

### 📌 Napkin Formula

```
   GNN message passing on icosahedral multi-mesh:

      m_{i→j}^{(l)}  =  MLP_edge( h_i^{(l)},  h_j^{(l)},  e_{ij} )       ← 邊上算 message
      h_j^{(l+1)}    =  MLP_node( h_j^{(l)},  Σ_{i ∈ N(j)} m_{i→j}^{(l)} ) ← 節點聚合更新

      where  N(j)  涵蓋所有 mesh 層級的鄰居（多解析度 edges 同時存在）
             l ∈ {1, ..., 16}  (16 個 GNN 層)

   Autoregressive 6h step:

      state_{t+6h}  =  state_t  +  GraphCast( state_t, state_{t-6h} )    ← residual prediction

   10-day rollout = 40 × autoregressive step.

   Cost: O(|E| · d^2)  per step  where  |E| = multi-mesh 邊總數 (~3M)
         vs FNO global FFT: O(N log N · d)  on 球面網格（不天然 fit）
```

**直覺**：multi-mesh 的關鍵是 |E| 雖大但**結構化** —— mesh_0 邊少但跨距離長，mesh_6 邊多但局部。Message passing 在 same forward pass 同時走兩端，**短程細節（對流）與長程訊號（teleconnection）並行更新**。Residual prediction（Δstate 而非 state）保證 short rollout 物理近似一致；但 autoregressive MSE 仍會 over-smooth，這是 §6 的 hidden cost。

### 2.1 Loss / 訓練細節

訓練分兩階段：

```
   Phase 1 (1-step):    L_1     =  Σ_{var, level} w_var · w_level · MSE( ŝ_{t+6h}, s_{t+6h}^{ERA5} )
   Phase 2 (unroll):    L_unroll =  Σ_{k=1..12} L_1 ( applied at step k )    ← unroll 12 steps = 3 days
```

`w_var` 對不同 atmospheric variable 加權（z500 / t850 / msl 重要 vars 權重高），`w_level` 對 pressure level 按密度加權（低層大氣權重高）。**Phase 2 unroll fine-tune** 是 GraphCast 不被 autoregressive drift 完全擊潰的關鍵 —— 但只 unroll 12 steps，所以 >3d 仍有累積誤差。

**不顯式加 PDE residual loss** —— 這是 v2 把它從 `hard-constraint` 改判 `architecture-bias-soft` 的根本依據：GNN 的球面 permutation symmetry 與 multi-mesh 拓樸是**結構 prior**，不是嚴格守恆方程。Mass / energy conservation 是 ERA5 reanalysis 數據隱含進來的，不是 architecture 強制的。

---

## §3 · 數據層 / 訓練 scale

| 項 | GraphCast |
|---|---|
| 訓練資料 | **ERA5 reanalysis 1979–2017**（39 年 6-hourly snapshot ≈ 57k samples）|
| 驗證資料 | ERA5 2018–2021 |
| 變量 | 6 surface + 5 atmospheric × 37 pressure levels = 235M values per state |
| Resolution | 0.25° lat/lon ≈ 28 km |
| Training compute | 32 TPU v4 × ~3 週（DeepMind 內部）|
| Inference compute | **單張 A100 80GB ≈ 60s for 10-day forecast** |
| Params | 36.7M（autoregressive single model）|

對比：IFS HRES（ECMWF 數值預報金標準）在 ~1000 CPU cores 跑 ~1h 出 10-day forecast —— **operational cost 降 3+ orders of magnitude** 是 ECMWF 願意採 AIFS 進 operational 的核心動因。

關鍵事實：GraphCast 訓練資料**全來自 reanalysis（ERA5）而非觀測**。Reanalysis 已經是 IFS 物理模型 + observation assimilation 的產物 —— 所以 GraphCast 實質上是在「**蒸餾**」IFS。這是它能 beat IFS 的弔詭：學生在 reanalysis（含教師知識）上做 prediction，能超過教師在 forecast（純前向）上的表現，因為 reanalysis 修正了教師的 systematic bias。

---

## §4 · 代碼層

| 項 | 狀態 |
|---|---|
| Repo | [github.com/deepmind/graphcast](https://github.com/deepmind/graphcast) |
| Framework | JAX + Haiku |
| Checkpoint | 3 檔：full 0.25° (~37M) / small 1° / ENS-style small |
| License | Apache-2.0（model code）+ checkpoints CC-BY-NC-SA-4.0（非商用）|
| Inference GPU | 單 A100 80GB（0.25° model 10-day rollout ~60s）|
| Streaming | ❌ batch-only（forecast 是 batch 一次 rollout 40 步）|
| Metric scale | ✅ 物理單位（K, m/s, Pa）|
| Operational | ✅ **ECMWF AIFS 2025-02-25 採其啟發進 operational**（不是 GraphCast 本身，是繼承 graph-encoder 思想的混合 architecture）|

**典型踩坑**：JAX + Haiku 版本鎖死、ERA5 變量命名與 pressure level 對齊必須完全一致（少一個 inference NaN）、mesh pickle 必須對應 resolution、fine-tune 時 unrolling depth × batch × mesh size 容易爆 memory。多數 reproduction 工作只做 inference，不重訓。

---

## §5 · 評測 / Benchmark

| Benchmark | Metric | IFS HRES（前 SOTA）| GraphCast | Δ |
|---|---|---|---|---|
| 1380 verification targets, 5-10d | RMSE / ACC | baseline | **beat IFS on ~90%** | — |
| z500 (geopotential 500 hPa), 5d | RMSE (m²/s²) | baseline | **-7% to -12%** | better |
| t850 (temperature 850 hPa), 5d | RMSE (K) | baseline | **-5% to -10%** | better |
| Tropical cyclone **track** | mean position error | baseline | **better** | better |
| Tropical cyclone **intensity** | central pressure | baseline | **systematically under-predicted** | worse |
| Extreme precipitation peaks | percentile error | baseline | **over-smoothed** | worse |

**解讀**：5-10d 對 z500 / t850 等 dynamical headline variable，GraphCast 廣度全面擊敗 IFS（這是 Science 接受論文的硬核論據）。但 **intensity 與 extreme**：autoregressive MSE 結構性 over-smooth 任何 sharp spatial-temporal gradient —— 颶風中心氣壓被低估、極端降水峰值被削平。**這不是 data leakage 或 benchmark Goodhart，是 architecture × loss 的真實限制** —— GenCast diffusion ensemble 就是為了補這條而生。

---

## §6 · Issues & Limitations

### 6.1 論文自述 limitations

- **Extreme event intensity 系統性低估**：autoregressive MSE training 過 smoothing；Lam et al. §S4 自述
- **熱帶氣旋 intensity 差於 IFS**（track 較好）—— ECMWF / NOAA 評估報告皆有 caveat
- **Precipitation 變量品質差**：ERA5 precipitation reanalysis 本身質量差，導致 GraphCast 在降水上不如動力學變量
- **長 rollout (>10d) 非物理 drift**：spectral energy decay + over-smoothing 累積
- **OOD climate regime 外推未驗證**：train 在 1979–2017，warmer-than-training 不保證

### 6.2 Hidden Assumptions（隱含假設）

- **Reanalysis 蒸餾假設**：ERA5 已含 IFS 教師信號，模型本質上在 distill teacher —— **觀測站快速變動的新地區（北極融冰、新雷達覆蓋）OOD 預警**
- **Conservation 不嚴格**：v2 把它從 hard-constraint 改判 architecture-bias-soft 的根本理由；GNN symmetry 是 soft inductive bias，不是 PDE residual = 0
- **Multi-mesh 解析度上限 = mesh_6 (40,962 nodes ≈ 0.25°)**：要做 ≤10 km 中尺度仍要重訓 mesh_7+
- **Single 6h step 是 magic number**：1h step 訓不出來（資訊變化太小）、24h step rollout 不穩 —— 6h 是 ERA5 snapshot 頻率 × autoregressive 穩定性的工程 sweet spot
- **Loss weighting 隱含 axis priority**：z / t 變量權重高 → 在這些 variable 上 beat IFS，但 humidity / cloud cover 等弱 supervision variable 進步較少
- **Train on reanalysis, deploy on operational analysis 的 distribution gap**：實際 operational 用的不是 ERA5 而是 ECMWF operational analysis，兩者 systematic 差異會 leak 到 forecast bias

### 6.3 GitHub-validated 失敗模式

| 失敗 / 問題 | GitHub evidence (deepmind/graphcast) | 嚴重度 |
|---|---|---|
| **JAX / Haiku version drift** | 多 issue 報 fresh install 不 work，需 pin repo commit + requirements.txt | 🟠 入門門檻高 |
| **ERA5 變量對齊** | issue 反覆出現 NaN，root cause 是變量缺一或 pressure level 順序不對 | 🟠 reproducibility 卡 |
| **Full 0.25° checkpoint 需 40GB GPU**（10-day rollout）| README 自述 | 🟡 普通實驗室卡 |
| **Fine-tune training code 開源不完整** | training utility 與 unroll harness 部分未 release | 🟠 學術 reproducibility 受限 |
| **Mesh pickle pre-generation** | 需對應 resolution，cross-resolution 不能直接重用 | 🟡 |

**Maintainer 響應度**：DeepMind 開源節奏一貫 —— 代碼 release 但 issue 響應較慢；後續演進（GenCast）走新 repo，GraphCast 維護進入慢更新期。

---

## §7 · 比較 & 面試 Tip

| 模型 | Axis 2 (injection) | Autoregressive | Ensemble? | Operational? | Status |
|---|---|---|---|---|---|
| FourCastNet | hard-constraint (FNO) | 6h | ❌ | ❌ 學術 | 2022 |
| Pangu-Weather | data-only (3D ESTransformer) | cascade 1/3/6/24h | ❌ | 部分商用 | 2023-04 |
| **GraphCast** | **architecture-bias-soft (GNN icos)** | 6h single model | ❌ | ✅ via AIFS 啟發 | 2023-11 |
| GenCast | guidance-gradient (diffusion) | 6h + sampling | ✅ 50 member | parallel run | 2024-12 |
| **AIFS (ECMWF)** | architecture-bias-soft (graph-encoder + transformer-processor) | 6h | ❌ deterministic | ✅ **operational 2025-02-25** | 2025-02 |
| IFS HRES (數值) | hard-PDE | 1h | ❌ | ✅ 仍 headline | 1979– |

> **🎤 Interview Tip.** 「GraphCast 在 production，我們要不要直接用？」**正確答**：「先區分『production』指什麼。**ECMWF AIFS 2025-02 進入 operational 的不是 GraphCast 本身，是 GraphCast 啟發的 graph-encoder + transformer-processor hybrid**。如果你要做 medium-range (3-10d) 全球 deterministic forecast，GraphCast 是強 baseline，inference 1 分鐘成本可接受；但**極端事件 intensity 與 nowcasting (0-3h)** 它都不行 —— 前者要 GenCast ensemble，後者要 radar-based model（如 DGMR / MetNet）。所以答案是『medium-range deterministic 用 GraphCast 系，extreme tail-risk 用 GenCast，nowcasting 換 architecture』，不是『直接全換』。」**錯答**：「GraphCast beat IFS 就全換」—— operational meteorology 是多模型 ensemble，GraphCast 是新增成員不是替換。

### 7.1 Falsifiable predictions

1. ✅ **VERIFIED 2025-02-25 (預測時間 2024-Q2, 早 ~9 個月)**：ECMWF AIFS 進入 operational status，採 GraphCast 啟發的 graph-encoder + transformer-processor。**這是 ML 天氣模型首次成為國家氣象局主跑 deterministic forecast 而非平行比對。** Source: [ECMWF AIFS announcement 2025-02-25](https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwf-launches-ai-forecasting-system-aifs)。
2. **2027-12 前**：第一個 hybrid GraphCast-style GNN + diff-physics refiner 出現，補長 rollout 非物理 drift（PDE-Refiner 路線收斂進 operational）。
3. **2028-12 前不會發生**：GraphCast 純 GNN architecture 取代 IFS 成為 headline operational deterministic forecast —— **extreme intensity under-prediction 是 architecture × loss 結構問題**，非更多 data 或更大 model 能解；要解需 diffusion (GenCast 方向) 或 explicit PDE residual loss，後者反過來 violate `architecture-bias-soft` 的 axis 設定。

---

## §8 · For the Reader（按 persona 分流）

- **VLA / robot policy 工程師** —— GraphCast 與 robot policy 在 output space (field vs action) 完全不相交，但 multi-mesh GNN message passing 是 **scene-graph policy** 的好參考 architecture：把 manipulation scene 編到 multi-resolution graph 上做 long-range affordance + local contact 並行推理。
- **自駕 closed-loop 工程師** —— 不直接用，但 6h autoregressive + unroll 12 fine-tune 的訓練範式可借鏡到 trajectory rollout learning。
- **影片生成工程師** —— GraphCast 在 `field` output 不在 pixel，跟 video WM 不可直接比；但若把 weather field render 成 visualization video 可作為 high-fidelity downstream（氣象節目自動化、災害可視化）。
- **神經 PDE / surrogate 研究者** —— 必讀 anchor。**multi-mesh + GNN > FNO 在球面 PDE** 的 winning data point，且是 zone 內唯一 productionize 案例。重點看 §1.3 architecture diagram 與 §2.1 unroll fine-tune loss。
- **物理 conditioning 研究者** —— 注意 v2 把它從 `hard-constraint` 重歸到 `architecture-bias-soft` —— architecture symmetry 不等於 PDE residual，這是 ontology 設計的 honest 修正，影響你怎麼分類自己的方法。
- **Research 學生** —— 注意 §7.1 三條 falsifiable prediction（已驗證一條 AIFS operational）；icosahedral multi-mesh 是接下來幾年球面 / 地球科學任務的 architectural baseline。

---

## References

- **GraphCast** — Lam et al. *Science* 382:1416–1421 (14 Nov 2023) · DOI [10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336) · [code](https://github.com/deepmind/graphcast)
- **GenCast** — Price et al. *Nature* (Dec 2024) · GraphCast 後繼 diffusion ensemble → [`./gencast.md`](./gencast.md)
- **Pangu-Weather** — Bi et al. *Nature* 619:533–538 (Jul 2023) · 同期 transformer 路線 → [`./pangu-weather.md`](./pangu-weather.md)
- **FourCastNet** — Pathak et al. arXiv:2202.11214 (2022) · FNO 變種前作 → [`./fno.md`](./fno.md)
- **MeshGraphNet** — Pfaff et al. ICLR 2021 · 單一解析度 mesh 前作 → [`./meshgraphnet.md`](./meshgraphnet.md)
- **AIFS operational announcement** — ECMWF 2025-02-25 · [link](https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwf-launches-ai-forecasting-system-aifs) (`UNVERIFIED` URL 細節)
- **WeatherBench 2** — Rasp et al. *JAMES* 2024（benchmark 標準化）

---

## Boundary

- 完整 FNO spectral 路線解構 → [`./fno.md`](./fno.md)
- Transformer 路線同軸對手（3D Earth-Specific Transformer + cascade）→ [`./pangu-weather.md`](./pangu-weather.md)
- Diffusion ensemble 後繼（補 extreme tail-risk）→ [`./gencast.md`](./gencast.md)
- 單一解析度 mesh 前作 → [`./meshgraphnet.md`](./meshgraphnet.md)
- 與 5 axis 全景 → [`../../cheat-sheet/ontology.md`](../../cheat-sheet/ontology.md)
- 與 VLA 接口（scene-graph policy 借鏡）→ [`../../bridge-to-vla/`](../../bridge-to-vla/)（pending）

---

## ✍️ 維護者註（v0.5 → v1 升級清單）

本 v0.5 基於 Science 原文 + DeepMind repo + ECMWF 2025-02 公告 + 第三方 reproduction。下次升 v1 時補：

1. ⏳ 1380 verification target 的完整 breakdown（z/t/u/v/q × pressure level × lead time 全表）
2. ⏳ AIFS 與 GraphCast 在 architecture 上的具體差異（AIFS 是 graph-encoder + transformer-processor hybrid，不是純 GNN）
3. ⏳ ECMWF AIFS operational 2025-02-25 後的 first-quarter performance report
4. ⏳ GenCast Nature 完整 metadata（volume / pages）
5. ⏳ Specific GitHub issue numbers（JAX version drift / ERA5 alignment NaN）
6. ⏳ Training compute exact 數字（32 TPU v4 × 多久確切數值）
7. ⏳ Fine-tune unroll harness 是否已開源
8. ⏳ Status v0.5 → v1，刪本節

---

[← Back to Neural Surrogates](./overview.md)

Sources:
- [GraphCast paper — Science 382:1416](https://doi.org/10.1126/science.adi2336)
- [DeepMind GraphCast GitHub](https://github.com/deepmind/graphcast)
- [ECMWF AIFS announcement (2025-02-25)](https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwf-launches-ai-forecasting-system-aifs)
- GenCast Nature 2024 cross-ref → [`./gencast.md`](./gencast.md)
- Pangu Nature 2023 cross-ref → [`./pangu-weather.md`](./pangu-weather.md)
