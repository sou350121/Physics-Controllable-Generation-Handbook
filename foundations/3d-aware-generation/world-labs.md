<!-- ontology-5axis output=3d-explicit injection=data-only control=text|image-init|camera temporal=clip-parallel domain=rigid -->

# World Labs Marble (gen-3D scenes from image/text)

## 1. One-paragraph TL;DR

World Labs（Fei-Fei Li 領銜，共同創辦人 Justin Johnson / Christoph Lassner / Ben Mildenhall）2024 年初成立、9 月出 stealth、2025 年 11 月發 Marble 第一版商用、2026-02-18 拿到 10 億美元 B 輪（AMD / Autodesk / Emerson / Fidelity / NVIDIA / Sea 領投，估值傳 ~10B）。Fei-Fei 的核心命題是「**spatial intelligence is the linchpin** — 不是 video，是 explorable 3D」。從本 handbook 的 generation 視角，這條路線跟 [Sora](../video-world-models/sora.md) / [Veo](../video-world-models/veo.md) / [Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md) 是顯式對手：**Sora-class 是 pixel-video（看一次的 60s clip，turn back 場景就漂走）**；Marble 是 **3d-explicit（Gaussian Splat / mesh，turn back 場景還在）**。本篇拆 Marble — 它在 output 軸跟 video-WM 完全錯開，但 injection 軸跟 Sora 是同類（`data-only`，沒有 PDE / contact / 守恆），這也是它最大限制：**幾何持久但動力學缺席**。注意：Marble 是 **closed product**，無 arxiv paper、無 open weights、無公開架構細節，本篇 §2 必然大量 TBD。

## 2. Core mechanism

公開資訊極少（無 paper, 無 model card；以下純從 release blog / TechCrunch / The Batch / NVIDIA Isaac Sim integration blog / VIVE Mars case study 推測）：

```
text prompt / single image / panorama / video / 3D layout (boxes+planes)
   │
   ▼
[??? multimodal encoder]                ← [TBD: 架構未公開；推測 latent diffusion-style]
   │
   ▼
[??? 3D scene generator]                ← [TBD: 推測 latent → 3DGS decoder 或 multi-view diffusion + GS fitting]
   │   training data: [TBD: 未公開；推測含多視角影片 + 合成 3DGS 場景]
   ▼
3D Gaussian Splat field (millions of gaussians: pos, scale, color, opacity, SH coefs)
   │
   ├─→ .ply / .spz export (raw GS)
   ├─→ collider mesh (低 poly, 物理碰撞用)
   └─→ high-quality triangle mesh (.glb / .gltf, 編輯用)
        │
        ▼
   [Chisel Editor]                       ← text-guided geometry / material / lighting edit
        │
        ▼
   downstream: Unreal / Isaac Sim / Blender / Three.js (via Spark renderer)
```

可確認的工程事實（非推測）：

- **輸出表徵**：3DGS 為主 — 與 [overview.md](./overview.md) `output=3d-explicit` 一致；同時提供 mesh 出口給傳統 DCC pipeline
- **多模態輸入**：text / image / panorama / video / box-plane layout（World API blog 點名五種）
- **場景尺度**："room-sized" 為主；更大場景用「compose multiple generations」拼接（自家 bigger-better-worlds blog 承認此限制）
- **靜態場景**：所有公開 demo 都是 static scene，**沒有 dynamic / physics simulation 介入**
- **Chisel editor**：text-prompt 改幾何（"move this wall"）+ AI 補材質/光照
- **Spark renderer**：自家開源，把 GS 塞進 Three.js 走瀏覽器渲染

[TBD: verify Marble 是否內部用了 NeRF / Mildenhall 的 NeRF lineage — 公開資料只說「Gaussian Splat」export，但內部 representation 是 NeRF→GS bake、還是直接 GS-native diffusion，未揭露]

[TBD: verify 是否 multi-view diffusion 先生 N 張一致 view 再 fit GS（類似 Stable-DreamFusion / InstantMesh 路線），還是 direct latent→GS]

[TBD: verify 訓練資料規模 / 是否包含 Mildenhall 在 Google 期間的合成 NeRF 資料 / 是否有 RealEstate10K-class video corpus]

## 3. 五軸定位 + 同軸對手

```
output     = 3d-explicit              ← 跟 Sora pixel-video 正交；跟 GaussianAnything 同類
injection  = data-only                ← 跟 Sora 同類；沒 PDE / 沒 sim / 沒守恆
control    = text | image-init | camera   ← 多模態 conditioning，含 panorama/video/layout (歸 image-init/camera 族)
temporal   = clip-parallel            ← 一次生整個 scene，但「時間」不是它的維度（static scene）
domain     = rigid                    ← Check 9c：generalist 白名單僅 Sora/Veo/Cosmos-Predict/Cosmos-WFM；Marble 是室內靜態場景生成，幾何是 rigid，故標 rigid
```

**為什麼 `domain=rigid` 而不是 `generalist`**：Check 9c 明文限制 — 只有 foundation video model 可標 generalist。Marble 雖然在「視覺風格」上 generalist（卡通 / 寫實都能生），但它的物理 domain 落在 static rigid scene；流體 / 軟體 / 顆粒都不是它的目標。若未來 Marble 加入 dynamic 場景（4D Gaussian / deformable），應重新評估 domain tag（可能落 `rigid|soft` 或某些 demo 拆 sub-tag）。

**為什麼 `temporal=clip-parallel` 而不是 `single-frame`**：3D scene 本身雖無時間軸，但生成過程是一次性產出整個場景（非 autoregressive frame-by-frame），跟 SVD 一次性產 24 幀的拓樸同類，故套 clip-parallel 而非 single-frame。若 audit script 對 3D static scene 質疑此選擇，可移到 `N/A` — 留 §8 解釋。

同軸對手（output=3d-explicit）：

| 對手 | 出處 | 與 Marble 差異 |
|---|---|---|
| **Generative Gaussian Splatting** (arxiv 2503.13272, Mar 2025) | 學界 | 開源；單 object / 小場景；無 product polish；是 anchor for `output=3d-explicit` |
| **GaussianAnything** (arxiv 2411.08033) | 學界 | object-level 3DGS gen；跟 Marble 的 scene-level 不同尺度 |
| **DreamFusion → Magic3D → MVDream** 線 | Google / NVIDIA 學界 | SDS 蒸餾 / NeRF representation；object-centric；Marble 是 scene-centric + GS-native |
| **Cosmos-3D variants** | NVIDIA | 仍以 video 為主出口，3D 是 derived；Marble 是 3D-first |
| **DeepMind Genie 3** (2025) | DeepMind | real-time frame-by-frame，**不出 explicit asset**（這是關鍵差異，TechCrunch 點名）；output 應歸 pixel-video / streaming-cache，不在 3d-explicit 軸上 |

**跨軸對手**（不同 output 軸但搶同一塊應用市場）：

| 對手 | output | 差異 |
|---|---|---|
| [Sora](../video-world-models/sora.md) / [Veo](../video-world-models/veo.md) | pixel-video | 看一段，看完場景消失；turn-back inconsistency 是 video-WM 死結 |
| [Cosmos-Predict / WFM](../foundation-physics-models/cosmos-wfm.md) | pixel-video（+ 3D variants） | NVIDIA 的 robotics-domain bet；含 sim-in-loop reward；Marble 完全靠 data |

**跨 handbook 對比**（Spatial-Handbook 視角）：

| 路線 | 視角 | output 同樣是 3DGS，差在哪 |
|---|---|---|
| Spatial-Handbook `foundations/3dgs-family/` | **重建**（perception 端，從多視角真實照片 fit GS） | input 是已存在的場景；幾何受真實照片約束 |
| 本篇 Marble | **生成**（generation 端，從 text/單張圖外推） | input 是想像 / 單一視角；幾何要自己編造 |

兩者 output 表徵同（都是 3DGS .ply），但**資訊來源完全相反** — Marble 在做的事是 Spatial 線「沒拍到的視角」的幻覺補完，所以它對 multi-view consistency 的要求其實比重建還高，因為沒有 ground truth 可錨定。

## 4. Where it shines / where it breaks

### ⚡ shines

- **Multi-view consistency**：3DGS-based 出口讓「turn back, 場景還在」— 這是 [Sora](../video-world-models/sora.md) / [Veo](../video-world-models/veo.md) 結構上做不到的事（pixel video 無 underlying 3D）
- **Explorable + exportable**：產出 .ply / .glb 可直接 drop 進 Unreal / Isaac Sim / Blender — 不是「看一段就沒了」的 video（NVIDIA blog 點名 Isaac Sim integration 加速 robotics sim 場景建構）
- **Persistent assets**：一次生成可重複用 — 跟 Genie 3 的 real-time frame-by-frame 形成對比（後者不出 exportable asset）
- **Multimodal entry**：text / single image / panorama / video / box-plane layout 五種入口都 work（World API blog 確認）
- **Style 廣度**：cartoon → photoreal 都有 demo；對 stylized rendering 友善（VIVE Mars virtual production case study）

### ❌ breaks

- **Dynamics / physics 完全缺席**：所有公開 demo 是 static scene；無 rigid body / fluid / soft body 動力學。社區評論（36kr/AIBase 中譯）一句精準：「**重建『樣子』強，理解『會發生什麼』弱**」
- **Illustration / 風格化 input 退化**：bdtechtalks 拿 fantasy tavern illustration 餵入，回吐「grainy and buggy」結果 — 訓練資料偏向 photoreal 3D render
- **Detail 隨距離衰減**："the more you move away from the original image, the less detailed the objects become"（bdtechtalks 直接觀察）— 對應 §8 已知 hallucination
- **Outdoor / 大場景 limited**：自家 blog 承認「room-sized」是甜蜜點；更大要拼接，拼接縫處的一致性 [TBD: verify 拼接機制 — 是 latent-level overlap 還是 post-hoc registration]
- **不適合 character / animal**：自家明文 — 「designed to create 3D environments rather than focusing on isolated or central objects, such as people or animals」
- **Closed model + paywall**：PLY / GLB export 需付費 plan；無 weights / 無 paper / 無 API rate-limit 透明度（World API 是 hosted）
- **No style memory**：MemU blog 點名 — 每次 generation 從頭開始，沒有 aesthetic preference 累積（屬產品 UX 而非模型結構，但對 production workflow 是痛點）

## 5. Reproduction notes

**無法 reproduce**：closed product。

可做的事：

- 申請 World API（hosted；定價未完全公開 [TBD: verify pricing tiers]）
- 用 Marble web UI 試 generation；下載 .ply 進 Blender / Three.js（需付費 plan）
- 開源替代：Generative Gaussian Splatting (2503.13272) / GaussianAnything (2411.08033) — 學界路線，object-centric 為主，**不是 1:1 平替**
- 自架 baseline 思路：multi-view diffusion (SV3D / Zero123++) → 多 view 出來後跑 3DGS fitting（gsplat / nerfstudio）→ 接近 Marble 的 pipeline shape，但場景連貫性會差一截

GPU 預算（自架近似版）：multi-view diffusion 一張圖 ~10s on H100，3DGS fitting per scene ~5-30 min on A6000。Marble hosted 推測在 cloud 端做，end-user 看到的是 ~minutes per scene（自家 blog 描述 "within minutes"）。

## 6. Cross-line synthesis

| 與其他 4 條路線怎麼接 | 接法 |
|---|---|
| **pixel-WM (Sora / Veo / Cosmos-Predict)** | 輸出空間完全錯開 — pixel-video vs 3d-explicit。實際 production 可組合：Marble 生 base 3D scene → 在裡面用 Cosmos / Sora 補 dynamic agents / actors（NVIDIA Isaac Sim integration 是這條路的雛形）。但 Marble 自身**不出時序**，動態要 outsource。|
| **latent-WM (V-JEPA-2 / Dreamer / Genie)** | latent-WM 走 action token + latent rollout；Marble 出 3DGS 給它當「世界 prior」。Genie 線的反例：Genie 3 跟 Marble 是直接競爭（前者 streaming-cache pixel，後者 clip-parallel 3D），路線哲學分歧。|
| **diff-sim (Genesis / MJX)** | Marble 出的 collider mesh 可直接餵 Genesis / Isaac Sim 做 contact dynamics — 這是 NVIDIA 投資 World Labs 的工程動機。Marble 做不到的 dynamics 由 sim 端補上；World Labs 自己沒走 sim-in-loop。|
| **surrogate (GraphCast / FNO)** | 幾乎無交集（不同 domain）。|

**與 Spatial-Handbook 跨倉 synthesis**：Spatial-Handbook 的 `foundations/3dgs-family/` 是重建線（INRIA 3DGS 原版 → 4D-GS → SuGaR → Mip-Splatting），跟 Marble 是「同表徵不同方向」。對 robotics 場景生成的 implication：先用 Spatial 線從真實照片重建幾個錨定 3DGS scene → 用 Marble 把它們之間的 transition / 未拍到的 corner case 生成補完 → 一起餵進 [Isaac Sim](../differentiable-simulators/) 跑 policy training。這條 pipeline 已被 NVIDIA 在 blog 中暗示。

## 7. References

**Official (high confidence)**：

- World Labs 官網 / About — https://www.worldlabs.ai/about（mission, team, founding members）
- "Generating Bigger and Better Worlds" blog — https://www.worldlabs.ai/blog/bigger-better-worlds（公開的能力陳述 + 已認的限制）
- "Announcing the World API" blog — https://www.worldlabs.ai/blog/announcing-the-world-api（五種 input 模態確認）
- Marble docs（Unreal export）— https://docs.worldlabs.ai/marble/export/gaussian-splat/unreal
- Fei-Fei Li substack "From Words to Worlds" — https://drfeifei.substack.com/p/from-words-to-worlds-spatial-intelligence（spatial intelligence framing）
- Fei-Fei Li TED Talk on spatial intelligence（2024，[TBD: verify exact date and TED video URL]）

**Secondary tech analyses (medium confidence; 含推測)**：

- TechCrunch 2025-11-12 "Fei-Fei Li's World Labs speeds up the world model race with Marble" — 商業 launch context
- The Batch (deeplearning.ai) — "World Labs Makes Its Marble Generative World Model Public, Adds Chisel Editing Tool"（Chisel editor 細節）
- bdtechtalks substack "What to know about World Labs Marble" — **唯一一篇詳細嘗試拆架構的二手分析**，含失敗案例（fantasy tavern illustration）
- Time Magazine "Inside Fei-Fei Li's Plan to Build AI-Powered Virtual Worlds"（business framing；本次 fetch 403）
- NVIDIA developer blog "Simulate Robotic Environments Faster with NVIDIA Isaac Sim and World Labs Marble"（robotics integration angle）
- a16z "What's In a World? Investing in World Labs"（VC narrative；非技術）
- MemU blog "World Labs Marble Creates Explorable 3D Worlds — But Forgets Your Style"（UX critique，style memory pitfall）

**Funding / business (low technical content)**：

- AI Insider / PYMNTS / CGTN — $1B Series B 2026-02-18 confirmation
- Andreessen Horowitz announcement "What's In a World?"

**Not yet found / 可能不存在**：

- World Labs 至 2026-05 為止**無 arxiv paper**發表（[TBD: re-verify after each release cycle]）
- 無 model card / 無 open weights / 無 reproducibility statement
- 共同創辦人 Ben Mildenhall 的 NeRF 線研究跟 Marble 內部架構連結是**社區推測**，非官方確認

## 8. §8 Pitfall log

### 8.1 跨軸合規性說明（Check 9b / 9c）

- **Check 9b（Output × Injection）**：本篇 `output=3d-explicit` × `injection=data-only`，矩陣中 ✓，無需特殊解釋。
- **Check 9c（generalist 白名單）**：Marble 不在白名單，本篇標 `domain=rigid`。理由見 §3 — 視覺風格廣度 ≠ 物理 domain 廣度；Marble 的物理場景是 static rigid。若日後 Marble 推 dynamic / deformable，需重新 tag。

### 8.2 Known limitations（officially confirmed）

| # | 來源 | 描述 | Severity | Workaround |
|---|---|---|---|---|
| 8.2.1 | bigger-better-worlds blog（官方） | 不適合 character / animal / 中心 object，只做 environment | High（限定 use case） | 對 character 改用其他工具（DreamGaussian / TripoSR）|
| 8.2.2 | bigger-better-worlds blog（官方） | "room-sized" 是甜蜜點；更大場景要 compose | Medium | 多次生成手動拼接 |
| 8.2.3 | Marble docs / pricing page | PLY / GLB export 需付費 plan | Low（商業限制） | API 走 hosted endpoint |

### 8.3 Community-observed pitfalls（二手實測）

| # | 來源 | 描述 | Severity | Workaround |
|---|---|---|---|---|
| 8.3.1 | bdtechtalks substack | Fantasy tavern illustration 入，輸出「grainy and buggy」；style domain bias 偏 photoreal | High（對 stylized art workflow） | 用 photoreal reference 或先在 prompt 強制 style |
| 8.3.2 | bdtechtalks substack | 「the more you move away from the original image, the less detailed the objects become」— 距離 hallucination | Medium | 在原視角附近探索；遠處不要當 ground truth |
| 8.3.3 | 36kr / AIBase 中譯評論 | 視覺一致但「a few steps after，會出現視覺扭曲 / hallucination」 | Medium | scope navigation；接受 close-to-origin 為有效範圍 |
| 8.3.4 | 36kr 評論（精準一句） | 「reconstruct '樣子' 強，understand '會發生什麼' 弱」— **無物理動力學** | High（對 simulation use case） | 接 Genesis / Isaac Sim 補 dynamics；或限定 visualization-only use case |
| 8.3.5 | MemU blog | No style memory — 每次 generation 從頭，無 user preference 累積 | Low（UX；非結構） | 自己維護 prompt template / style anchor 圖 |

### 8.4 Structural critique（generation 端視角）

- **`injection=data-only` 的天花板問題**：Marble 跟 Sora 在 injection 軸是同類 — 物理規律完全靠資料 implicit 學會。它在 output 軸換到 3d-explicit 解決了 multi-view consistency，但**沒解決物理 grounding 問題**。3DGS 是個 rendering primitive，不是 mesh，不易加 contact constraint，所以即使外接 diff-sim，也要先 mesh-extract（Marble 提供 collider mesh 算是迎合這條 pipeline 的工程妥協）。詳見 [crossing/controllability-vs-fidelity/](../../crossing/controllability-vs-fidelity/)。
- **跟 Spatial-Handbook 重建線的「資訊量不對稱」**：重建線從 N 張真實照片擬 GS，資訊量是 N×H×W×3；Marble 從 1 張圖外推整個 room，資訊量是 1×H×W×3 + LLM-style prior。多視角一致性看似達成，本質是「強 prior 補資料缺口」— 跟 §8.3.2 detail degradation 是同一個現象兩面。
- **無 paper 的長期 audit 風險**：本 handbook 對 closed product 的 dissection 必然有效期短。每次 World Labs release 應重 audit 本篇，標記 [TBD] 是否解。

### 8.5 待釐清項目（[TBD] 集中）

- [TBD] Marble 內部表徵是 NeRF / 3DGS / 其他 latent — Mildenhall 背景讓 NeRF lineage 是合理猜測，但官方未確認
- [TBD] 是否走 multi-view diffusion → GS fitting，還是 direct latent → GS decoder
- [TBD] 訓練資料規模 / corpus 構成 / 是否含合成 GS 場景
- [TBD] Scene compositing（拼接更大場景）的機制
- [TBD] World API 定價層級
- [TBD] Fei-Fei Li 2024 TED talk 確切日期與 URL
- [TBD] 是否有任何 World Labs 員工以個人名義發過跟 Marble 直接相關的 arxiv preprint
- [TBD] Sora-class video / Genie 3 與 Marble 在實際 user evaluation 上的盲測結果（目前 worldsimulator.ai 等三方 leaderboard 在演化中）

---

> **Maintenance note for Pulsar**：本篇是 closed-product dissection 的範本之一；每次 World Labs / Marble 主要 release（model version、技術 blog post、arxiv preprint）後應觸發 re-audit，更新 §2 / §8.5 的 [TBD] 狀態。建議放入 reports/ daily 監控的 keyword filter（"World Labs", "Marble", "Fei-Fei Li spatial intelligence", "Justin Johnson 3D generation", "Ben Mildenhall NeRF generation"）。
