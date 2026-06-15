<!-- ontology-5axis output=3d-explicit injection=data-only control=text|image-init|camera temporal=clip-parallel domain=rigid -->

# Generative Gaussian Splatting (GGS) — Schwarz, Müller, Kontschieder 2025

## 1. One-paragraph TL;DR

**Generative Gaussian Splatting: Generating 3D Scenes with Video Diffusion Priors**（arxiv 2503.13272，Meta Reality Labs Zurich，Katja Schwarz / Norman Müller / Peter Kontschieder，2025-03-17，ICCV 2025）解決具體 gap：**video diffusion 生 N-view 影像好看，但 view 間 3D 一致性會漂**（同面牆兩 view 顏色不同；turn back 物件位置偏）。GGS 賭注：不要 post-hoc 拿生成 video fit 3DGS（Stable-DreamFusion / InstantMesh 路線），而是把 **3DGS 表徵當 video diffusion 中間 bottleneck** — LDM U-Net 出 per-view feature → epipolar transformer → Gaussian splat 參數 → 顯式 splat → 渲染回 multi-view feature → decode 像素。**3D 一致性靠表徵硬性保證**。v2 ontology line 32 把它列為 `output=3d-explicit` canonical anchor 的根本原因：跟 [World Labs Marble](./world-labs.md)（closed product，無 paper）相對，GGS 是這條路線唯一有 paper / benchmark / reproducibility 的學界 anchor，FID 比同類無 3D 表徵 baseline 改善 ~20%（RealEstate10K + ScanNet++）。

## 2. Core mechanism

```
camera poses (target N views, Plücker coords)
      │
      ▼                              ┌── single reference image (optional, image-init)
[ Latent Video Diffusion U-Net ]  ◀──┤
      │  per-view latent feature maps │── text prompt (optional, via base VDM)
      ▼                              └── depth supervision (training only, when available)
[ Epipolar Transformer ]
      │   cross-view feature aggregation
      ▼
[ Gaussian Splat decoder ]
      │   per-pixel → (μ, Σ, α, SH) tuples
      ▼
Feature-3DGS field (3D primitives carry feature vectors, not just RGB)
      │
      ├──→ rasterize at any view → feature map → 2D decoder → RGB image
      └──→ direct upsample → 3D radiance field (NeRF-style query)

Training loss = LDM denoising loss + multi-view rendering loss (+ optional depth loss)
Inference     = denoise latent → epipolar transform → splat → render N consistent views
```

關鍵點：
- **3D 表徵在 denoising 之後，render 之前** — diffusion 在 2D latent space 工作（沿用 pretrained VDM 權重），輸出 commit 到顯式 3D，N views 自動一致
- **Feature-3DGS（非 RGB-3DGS）**：每 Gaussian 帶 feature vector，2D decoder 變 RGB；splat 數可遠少於 vanilla 3DGS 的百萬量級
- **Pose-conditional**：camera pose 用 Plücker 嵌入塞進 U-Net；text / image-init 走 VDM 既有通道
- **訓練資料**：RealEstate10K + ScanNet++（室內 RGB-D 掃描），對應 v2 `domain=rigid`

## 3. 五軸定位 + 同軸對手

```
output     = 3d-explicit             ← v2 spec canonical anchor; Feature-3DGS field
injection  = data-only               ← 跟 Sora 同類；無 PDE / 無 contact / 無守恆；3D 靠表徵不靠物理
control    = text | image-init | camera   ← camera 是必填（pose-conditional 是核心）
temporal   = clip-parallel           ← 一次性產 N views（typical 8-24）；非 autoregressive
domain     = rigid                   ← Check 9c 白名單外；訓練資料 RealEstate10K + ScanNet++ 全是 static rigid 室內外場景
```

**v2 Check 9b**：`3d-explicit × data-only` = ✓。**Check 9c**：不在 generalist 白名單；訓練 domain 是 static rigid，標 rigid。

**同軸對手（`output=3d-explicit`）**：

| 對手 | 出處 | 與 GGS 差異 |
|---|---|---|
| **[World Labs Marble](./world-labs.md)** | 商用 product（無 paper） | scene-level + 多模態入口 + product polish；但 closed weights / closed architecture。**GGS 是 Marble 的學界對應**：同 output 軸、同 injection 軸，差在透明度 |
| **GaussianAnything** (arxiv 2411.08033, ICLR 2025) | NIRVANALAN 等 | **object-level**（point cloud latent diffusion）vs GGS 的 **scene-level**；走「VAE + cascaded LDM」route，不靠 pretrained VDM；無 multi-view pose-conditional 設定 |
| **DreamFusion → Magic3D → MVDream 線** | Google / NVIDIA | **SDS 蒸餾 + per-scene optimization**（每 prompt 跑數小時）；GGS 是 **feed-forward**（amortized）。前者 object-centric，後者 scene-centric |
| **Cosmos 3D variants** (NVIDIA) | 視 release | Cosmos 主線仍 pixel-video 為主，3D 是 derived；GGS 是 3D-first |
| **LatentSplat** (對比 baseline) | 學界 | GAN decoder 在大 baseline 崩 → GGS diffusion route 的勝場 |
| **ViewCrafter** (對比 baseline) | 學界 | trajectory 對，但跨 frame content drift；GGS 用顯式 3D 直接消滅此 failure |

**跨軸對手**（不同 output 軸但搶同應用）：

| 對手 | output | 差異 |
|---|---|---|
| [Sora](../video-world-models/sora.md) / [Veo](../video-world-models/veo.md) | pixel-video | view 看完場景消失；GGS 的 .splat 場景可重複 render 任意 view |
| [Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md) | pixel-video | NVIDIA 的 robotics bet，含 reward / sim-in-loop（injection 軸不同）；GGS 純 data-only |

**跨 handbook（Spatial `foundations/3dgs-family/`）**：Spatial 線收 **重建**（Kerbl 2023 原版 / 4D-GS / SuGaR / Mip-Splatting），從 N 張真實照片 fit GS。GGS 同產 3DGS，但**資訊來源相反**：從文字 / 單圖外推，必須 hallucinate 遮擋面。對 robotics scene synthesis 來說兩者互補。

## 4. Where it shines / where it breaks

### ⚡ shines

- **Multi-view consistency 結構性保證**：N views 來自同 splat field，view 間無 drift — ViewCrafter / 純 video-WM 結構上做不到
- **Large viewpoint extrapolation**：比 LatentSplat 在大基線強（GAN decoder 在大基線崩）
- **Feed-forward inference**：vs DreamFusion-SDS 每 prompt 跑 hours，GGS 一次 forward 出 scene
- **Pretrained VDM 復用**：站在 VDM prior 上，不從零訓
- **FID +~20%**：對比同類無 3D 表徵 baseline（RealEstate10K / ScanNet++）

### ❌ breaks

- **Domain narrow**：訓練在 RealEstate10K + ScanNet++ — 室內 / 房地產 / 掃描；OOD（戶外 / nature / stylized）未驗證
- **Static scene only**：跟 Marble 同病 — 無 dynamic / physics。`injection=data-only` 天花板（§8.4）
- **Pose 必填**：text-only / image-only 無 pose 時需 hallucinate trajectory
- **Splat count / scene scale 限制**：Feature-3DGS 雖省 splat，大場景仍可能 OOM；demo 以 room-scale 為主
- **無 open weights / 無官方 GitHub**（截至 2026-05）：project page 只有 paper + demo videos，需自行 reimplement
- **依賴 pretrained VDM**：Meta 內用 VDM 未公開，社區只能用 SVD / open-Sora 替代

## 5. Reproduction notes

**Code 狀態**：截至 2026-05-26 無官方 GitHub；project page 只有 paper + demo。需自行 reimplement。

**自架近似 pipeline**：SVD / open-Sora 當 VDM backbone → Plücker pose embedding 進 U-Net cross-attn → epipolar transformer（參考 EpiDiff / MultiDiff）→ Gaussian splat decoder（per-pixel μ/Σ/α/SH，參考 pixelSplat / MVSplat）→ 渲染用 gsplat 或原版 CUDA rasterizer。資料：RealEstate10K（公開）+ ScanNet++（學術 license）。Loss：LDM denoising + multi-view photometric + optional depth。

**GPU 預算**：訓練估 H100 × 8，1-2 週（[TBD]）；推理單 H100，per scene ~tens of seconds（feed-forward）— vs SDS 線每 prompt 1-2 小時是質變。

**典型踩坑**（pixelSplat / MVSplat 近親經驗）：Plücker embedding 數值需 normalize；splat count 隨場景爆需 prune（vanilla 3DGS densify 邏輯不能直接用，因為 splat 是 diffusion 生的非 optimize）；multi-view photometric loss 對 exposure / WB 敏感（RealEstate10K 是 YouTube 抓的）。

## 6. Cross-line synthesis

| 與其他路線怎麼接 | 接法 |
|---|---|
| **pixel-WM (Sora / Veo / Cosmos-Predict)** | GGS 出 static 3DGS scene → 用 Cosmos / Sora 補 dynamic agents。GGS 提供開放 splat field 給 video gen 當 reference（closed Marble 不可能做）|
| **latent-WM (V-JEPA-2 / Dreamer / Genie)** | GGS 場景當「世界 prior」；Genie 3 的 streaming-cache pixel route 與 GGS clip-parallel 3D route 哲學分歧 |
| **diff-sim (Genesis / MJX)** | splat field **不能直接給 contact dynamics**（3DGS 非 mesh，無顯式 surface）。要 mesh-extract（SuGaR / 2DGS-to-mesh）再餵 Genesis。**`injection=data-only` 硬瓶頸在此** |
| **surrogate (GraphCast / FNO)** | 無交集 |

**跨 handbook**：Spatial-Handbook `foundations/3dgs-family/` **重建線** + GGS **生成線** = 同表徵兩方向。pipeline：Spatial 從機器人攝影機 fit anchor scene → GGS 生 corner-case scene → 一起餵 Isaac Sim 做 VLA training scene multiplication。跟 VLA-Handbook：GGS 場景 → mesh extraction → sim → VLA policy 是合理 chain。

## 🚁 § aerial：GGS 是「生成」，aerial 用的是「重建」

[aerial-sim](../../use-cases/aerial-sim/overview.md) 從本檔連進來（`generative-aerial-data.md` 的「3DGS 線 Real2Sim2Real」），但要先講清一刀：**本檔主角 GGS 是生成側**（文字/單圖 → 外推 3D，需 hallucinate 遮擋面）；**aerial 實際用的是重建側**（真實飛行多視角拍攝 → 忠實 fit 3DGS）。兩者同產 3DGS 表徵、**資訊方向相反**。本倉 foundations 沒有獨立的「重建 anchor」（重建 method 解構在 [[spatial-handbook]] 的 `3dgs-family/`），所以把 aerial 重建範式補在這裡——它才是 aerial 外觀邊的現實主力。

**Real2Sim2Real 兩個 aerial 旗艦（scout 一手核驗）：**

- **FalconGym**（`2503.02198`，UIUC，IROS 2025）：用 **NeRF**（v1）重建真實競速空間 → 渲染無限合成過門影像 → 訓視覺策略（神經姿態估計 + Kalman + self-attention 控制器）→ zero-shot 上機。**95.8% 真機過門、~10 cm 平均誤差、38 cm 半徑門、30 次飛行 / 120 門**。**FalconGym 2.0**（`2510.02248`）把 NeRF→**GSplat** 加速 + Edit API：sim 100% 未見賽道（含定翼+四旋翼）、硬體 **98.6%（69/70 門）**。
- **SOUS VIDE**（`2412.16346`，Stanford MSL，RA-L 2025）：**FiGS** 模擬器 = **簡化動力學模型 + 3DGS 重建場景耦合**，渲染 **~130 fps** 訓 SV-Net（RGB + 光流 + IMU → thrust/body-rates @20 Hz，100k–300k MPC 對 + RMA）。**105 次硬體實驗，對 30% 質量 / 40 m/s 陣風 / 60% 亮度 / 物件+人 擾動穩健**。

兩者都是本手冊反覆講的 **photons（3DGS 給外觀）＋ forces（簡化動力學給力）** 拆分——SOUS VIDE 明說、FalconGym 同構。

**大場景 / 城市級 aerial 重建**（aerial-sim 外觀的真正引擎）：CityGaussian（`2404.01133`）· VastGaussian（`2402.17427`）· DroneSplat（`2503.16964`，in-the-wild 稀疏視角去動態干擾）· Horizon-GS（`2412.01745`，aerial-to-ground 統一）· AGS（`2409.00381`，空拍表面重建）。

**⚠ metric-scale 陷阱（aerial 必踩）**：SfM/3DGS 重建**尺度模糊**（差一個未知比例），但 drone 控制要**真實公尺**（推力/重力/速度）。aerial 範式的輕量解是 **ArUco fiducial**：SOUS VIDE 在錄影開頭放 ArUco tag 對齊 GSplat 到已知全域座標、mocap 只當 diagnostics；FalconGym 2.0 用 ArUco + COLMAP→world（Kabsch-Umeyama），**免昂貴 mocap**。

**和生成（Cosmos）的分工——互補不是競爭**：重建 owns「**拍得到的場景**」外觀（metric-anchorable、zero-shot 95.8–98.6%）；生成 owns「**拍不到的新情境**」（天氣/新物件/罕見事件）。aerial 的 generation 端仍 under-served（見 [Cosmos § aerial](../foundation-physics-models/cosmos-wfm.md)）；aerial-gen 反例是 FlightDiffusion（單幀→FPV 影片，非 sim-render 翻真）。動力學邊回 [Aerial Gym](../differentiable-simulators/aerial-gym.md) / RotorPy。

## 7. References

**Canonical**：
- Schwarz, K., Müller, N., Kontschieder, P. (2025). "Generative Gaussian Splatting: Generating 3D Scenes with Video Diffusion Priors." **arxiv 2503.13272** (2025-03-17). **ICCV 2025**. Meta Reality Labs Zurich. https://katjaschwarz.github.io/ggs/

**Foundational lineage**：
- Kerbl, Kopanas, Leimkühler, Drettakis (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." **SIGGRAPH 2023**, ACM TOG 42(4). — 所有 GS 衍生方法 root anchor
- Mildenhall et al. (2020). "NeRF." ECCV 2020. — 隱式 3D ancestor
- Rombach et al. (2022). "Latent Diffusion." CVPR 2022. — LDM base
- Blattmann et al. (2023). "Stable Video Diffusion." arxiv 2311.15127. — 開源 VDM backbone

**同領域對手**：
- Lan et al. (2024). "GaussianAnything." arxiv 2411.08033, **ICLR 2025** — object-level
- Charatan et al. (2024). "pixelSplat." CVPR 2024 — epipolar transformer 近親
- Chen et al. (2024). "MVSplat." ECCV 2024 — feed-forward GS baseline

**動態 / 4D Gaussian（GGS 不解決，v2 wishlist 下一篇）**：
- Wu et al. (2024). "4D Gaussian Splatting." CVPR 2024
- Yang et al. (2024). "Deformable 3D Gaussians." CVPR 2024

**aerial 重建 / Real2Sim2Real（§ aerial 引用）**：
- Miao, Shen, Mitra. "FalconGym." IROS 2025 · [2503.02198](https://arxiv.org/abs/2503.02198)；FalconGym 2.0 · [2510.02248](https://arxiv.org/abs/2510.02248)
- Low et al. "SOUS VIDE / FiGS." RA-L 2025 · [2412.16346](https://arxiv.org/abs/2412.16346)
- 大場景 aerial：CityGaussian [2404.01133](https://arxiv.org/abs/2404.01133) · VastGaussian [2402.17427](https://arxiv.org/abs/2402.17427) · DroneSplat [2503.16964](https://arxiv.org/abs/2503.16964) · Horizon-GS [2412.01745](https://arxiv.org/abs/2412.01745) · AGS [2409.00381](https://arxiv.org/abs/2409.00381)
- GS ＋ 物理（forces 另接）：PhysGaussian [2311.12198](https://arxiv.org/abs/2311.12198) · RoboGSim [2411.11839](https://arxiv.org/abs/2411.11839) · SplatSim [2409.10161](https://arxiv.org/abs/2409.10161) · GSWorld [2510.20813](https://arxiv.org/abs/2510.20813)

**二手**：Hugging Face papers (https://huggingface.co/papers/2503.13272) · alphaXiv (https://www.alphaxiv.org/overview/2503.13272)

## 8. §8 Pitfall log

### 8.1 跨軸合規性（Check 9b / 9c）

- **9b**：`3d-explicit × data-only` = ✓，無需解釋
- **9c**：不在白名單；訓練 domain 是 RealEstate10K + ScanNet++ 室內 static scene，標 `rigid` 與 [overview.md](./overview.md) 預設一致

### 8.2 Known limitations（paper / project page confirmed）

| # | 來源 | 描述 | Severity | Workaround |
|---|---|---|---|---|
| 8.2.1 | paper Sec on baselines | LatentSplat 在大基線崩 → GGS 的勝場，但反過來 **GGS 在過小基線可能不如 LatentSplat 的 GAN decoder 銳利** | Low | 小基線用其他工具 |
| 8.2.2 | paper Limitations | 訓練 domain 限於 RealEstate10K + ScanNet++（室內 + 房地產），OOD 戶外 / 風格化效果未驗證 | High | fine-tune on target domain |
| 8.2.3 | project page | 無 open weights / 無官方 code release（截至 2026-05-26）| High（reproducibility） | 自行 reimplement，用 pixelSplat / MVSplat 當骨架 |

### 8.3 Community-observed / structural pitfalls

| # | 來源 | 描述 | Severity | Workaround |
|---|---|---|---|---|
| 8.3.1 | inferred VDM / 3DGS 共有 issue | Pretrained VDM motion prior 在 static scene「想動」— frame 間可能 micro-drift；splat field freeze 是 mitigation 需驗證 | Medium | zero motion latent；ablation |
| 8.3.2 | inferred pixelSplat 近親 repo | Plücker pose embedding 數值穩定性 — pose degenerate（極端 roll）時 splat 預測退化 | Medium | normalize + clamp |
| 8.3.3 | inferred Feature-3DGS lineage | Feature splat → 2D decoder 若學到 view-specific bias，consistency 仍漏 | Medium | 強制 view-agnostic decoder |
| 8.3.4 | structural（同 [World Labs](./world-labs.md) §8.3.4） | 「重建樣子強，理解會發生什麼弱」— `injection=data-only` 共病 | High（simulation） | 接 Genesis / Isaac Sim 補 dynamics |
| 8.3.5 | RealEstate10K bias | YouTube 房地產曝光 / WB 跳動 → 模型可能學到「跨 view 顏色會跳」當合理 prior | Low-Med | color jitter normalize |

### 8.4 Structural critique

- **`injection=data-only` 的天花板**：GGS 「3D consistency 結構保證」本質是**幾何一致性**（同牆兩 view 是同 splat），非**物理一致性**（球從桌上掉的 trajectory）。解決 video-WM multi-view drift，沒碰物理。Marble 同病；要碰物理得加 `aux-loss` / `sim-in-loop`
- **跟 Marble 的學界 / 商業二分**：GGS = 學界 anchor（paper / benchmark / 可審計）；Marble = 商業 anchor（product / UX / closed）。同 ontology cell 收兩 anchor 是 v2 設計選擇
- **VDM dependency 長期 audit 風險**：VDM 進步（Sora 2 / Veo 3）拉抬 GGS 上限，但讓「顯式 3D 表徵」邊際 marginal value 遞減。需 2-3 年後重 audit

### 8.5 待釐清項目（[TBD]）

- [TBD] ICCV 2025 接收狀態（oral / poster）正式確認
- [TBD] FID ~20% 改善精確數字（需 paper Table 對照）
- [TBD] 訓練 GPU budget / wall-clock
- [TBD] Meta 內用 VDM backbone（自家 internal / SVD fork？）
- [TBD] Splat count per scene 典型值
- [TBD] 跟 Marble 同 benchmark 對比可能性
- [TBD] Code release 是否在 ICCV 2025 期間公布

---

> **Pulsar maintenance**：本篇是 `output=3d-explicit` 學界 anchor，與 [World Labs Marble](./world-labs.md) 互補（學界 vs 商業）。建議 reports/ daily monitoring keyword：「Generative Gaussian Splatting」「Katja Schwarz GGS」「Feature-3DGS diffusion」「Meta Reality Labs Zurich 3D generation」。下次 release 後重 audit §8.5。
