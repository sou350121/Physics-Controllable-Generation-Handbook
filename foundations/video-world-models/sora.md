<!-- ontology-5axis output=pixel-video injection=data-only control=text|image-init temporal=clip-parallel domain=generalist -->

# OpenAI Sora 解構（Sora 1.0 / Sora-2 Dissection）

> **發布時間**：2024-02-15（Sora 1.0）· 2025-09-30（Sora-2）· **App sunset 2026-04-26 · API 退役 2026-09-24**
> **技術報告**：[*Video generation models as world simulators*](https://openai.com/index/video-generation-models-as-world-simulators/)（OpenAI, 2024-02）· [*Sora 2 is here*](https://openai.com/index/sora-2/) · [Sora 2 System Card](https://openai.com/index/sora-2-system-card/)
> **作者**：OpenAI Sora team（無 paper，無 author list；技術報告為 product post 格式）
> **核心定位**：v2 ontology 上落 `pixel-video / data-only / text|image-init / clip-parallel / generalist` —— **THE implicit-from-data scale-pilled reference point**。任何主張「加 aux-loss / sim-in-loop / hard-constraint」的方法都必須在此 baseline 上交差。

**Status:** v0.5 — closed model，無 paper，無 model size／訓練 FLOPs／資料來源公開。解構基於 OpenAI tech report + Sora-2 system card + PhyWorld (2411.02385) 結構性結論 + 社區拆解 (skywork / CineD / A2E) + Open-Sora 二手 reproduction。完整參數待 OpenAI 公開（但 sunset 在即，**多半永遠不會發**）。
**TL;DR:** ① DiT + spacetime patches 取代 U-Net + fixed-frame，使變長／變 aspect ratio／joint-rollout 60s 1080p 成為可能 ② 物理規律完全靠資料 implicit 學，**零 aux-loss / 零 PDE / 零 contact constraint** ③ 對本 handbook 的價值不在它「解了 physics」(沒有)，在於它**把 data-only 路線推到公開可見極限** —— 給其他 axis 提供了 falsifiable baseline ④ PhyWorld 結構性結論：in-distribution 完美外推、OOD 全面失敗、generalization feature priority `color > size > velocity > shape`（case-based not principled）。

**X-Ray.** Sora 在 v2 ontology 上是 `axis-2 injection = data-only` 那條路線的 **reference point** —— 不是因為它最好（Veo-3 / Kling 已在多項 case 上反超），而是因為它**最公開、最被研究、最徹底地把 scale-pilled 賭注押滿**。它解了三個工程坑：(a) joint-rollout 在 60s 量級內幹掉 AR 模型的 drift 累積；(b) spacetime patches 讓 variable resolution / aspect ratio 同 corpus 訓練成為可能（不再 letterbox / crop loss）；(c) re-captioning pipeline 讓 text-following 在多物件多動詞下穩定。但它打不開的 envelope 同樣清楚：剛體破裂 / brittle fracture 永遠像橡膠彈、物體穿透桌面、流體 / 煙 / 布料這類需要 volumetric simulation 的 case 即使 Sora-2 referee model 都救不回來。**它的價值對 handbook 讀者**：每次你提案一條 `data-only` 以外的路徑（diff-sim 接 video WM、sim-in-loop guidance、physics aux loss），你的 reviewer 第一句一定是「比 Sora-class 純資料路線好多少？」 —— 你必須有答案。Sunset 加重了這層意義：**Sora 是被 falsify 走的，不是被取代走的**（後面 §7.1 第 3 條）。

## 📍 研究全景時間線

```ascii
   2022          2023                2024-02              2025-09            2026-04-26       2027?
   VDM ────► Make-A-Video ────► YOU ARE HERE ────► Sora-2 ─────────► Sora app sunset ──► implicit
   Imagen     SVD               Sora 1.0            +audio              API 2026-09-24       路線
   Video      (per-frame VAE,   (spacetime          +referee model      退役                 全面被
              U-Net AR)         patches, DiT,                                                aux-loss
                                joint-rollout,                                               / sim-in-loop
                                clip-parallel) ★                                             取代?
   └─ AR drift ─────────────► joint-rollout ─────► referee-as-RLHF ──► commercial collapse ──►
      fixed-frame              variable everything   reward-filtered     scale-pilled 不夠
```

★ = 主要新點：spacetime patches + DiT + joint-rollout 三件套。**仍未解：剛體破裂 / 流體 / OOD 物理**（這些留給 axis-2 以外的路線）。

---

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 同期 video-gen

| 維度 | 早期 VDM / SVD / Imagen Video | Sora 1.0 |
|---|---|---|
| **Backbone** | U-Net + temporal layers | **Diffusion Transformer (DiT)** |
| **Tokenization** | Fixed frame count, fixed resolution | **Spacetime patches**（任意 H, W, T, aspect）|
| **VAE** | Per-frame spatial only | **3D VAE**（空間+時間同時壓縮）|
| **Rollout** | AR over frames（drift 累積）| **Clip-parallel**（整段一次 denoise）|
| **Text cond** | 用戶原始 prompt | **GPT re-captioned**（推理時自動 expand）|
| **Variable aspect** | Letterbox / crop | **原生支援** —— 訓練時就混入豎屏短影音 + 橫屏電影 |

Sora-2 (2025-09-30) 在 Sora 1 backbone 上加三件事（社區拆解，無 paper 細節）：

1. **Referee model**：抓 floating objects / penetration / 不自然運動，作為 reward signal 反饋至 re-training（看起來像 RLHF-for-physics 或 reward-filtered SFT；OpenAI 沒給細節）
2. **同管線 audio**：dialogue + SFX + 環境聲音與 video token 共生成
3. **Visual encoder 不變** —— system card 描述「inherits Sora 1 framework」

### 1.2 ⚡ Eureka Moment

> **核心 trick 一句話** —— **「Video generation 的 transformer moment 是 spacetime patches，不是 DiT 本身。」** DiT 在 image 端 (DiT paper, Peebles 2022) 早就證明可行，但 video 端真正解鎖 scale 的是把 variable-length / variable-resolution / variable-aspect-ratio 統一成 patch sequence —— 一條 corpus 從豎屏 TikTok 到橫屏電影都能直接餵。

直覺：U-Net 對 input shape 敏感（要 reshape / pad / crop），對應到 corpus 只能用 fixed-shape subset，這直接砍掉了 80% 真實 video 訓練資料的多樣性。Spacetime patches 讓資料端「能餵什麼就餵什麼」—— 這是把 scaling law 從 image 帶到 video 的關鍵 enabler。

### 1.3 信息流（架構圖）

```ascii
            早期 VDM / SVD                            Sora 1.0
   ─────────────────────────────              ─────────────────────────────

   Fixed 16-frame 512×512 clip                Raw video (任意 H, W, T, aspect)
            │                                          │
            ▼                                          ▼
     Per-frame VAE                              [3D VAE encoder]
     (spatial only)                             (空間+時間共壓縮)
            │                                          │
            ▼                                          ▼
   Latent frames (16 × h × w)                  Latent volume (h, w, t)
            │                                          │
            ▼                                  patchify into
   U-Net + temporal attn                      spacetime patches
   (AR over frames OR                                 │
    fixed-length joint)                               ▼
            │                                  Variable-length patch tokens
            │                                          │
            ▼                                          ▼
   Denoise → frame seq                        Diffusion Transformer (DiT)
                                              ★ joint clip-parallel denoise
   ★ drift 累積 (AR mode)                              │
   ★ 不能變 aspect                              GPT re-captioned text cond
                                                       │
                                                       ▼
                                              [3D VAE decoder] → pixel video

                                              ★ 60s 1080p, 變 aspect, 一次 rollout
```

---

## §2 · 數學層

### 📌 Napkin Formula

```
   Spacetime patch tokenization:

       video ∈ ℝ^(T × H × W × 3)
           │ VAE encode  →  latent ∈ ℝ^(t × h × w × c)     ← t = T/r_t, h = H/r_s
           │ patchify    →  N = (t/p_t) · (h/p_s) · (w/p_s) tokens
           ▼
       DiT( tokens, text_emb, timestep_emb )

   Cost (per denoising step):   O(N² · d)   ← attention over all patches
            vs 3D conv U-Net:    O(N · k³ · d)   ← local kernel k

       Sora: trade locality bias for全域時空關係建模 + variable shape
```

**直覺**：3D conv 便宜 (linear in N) 但 receptive field 受 layer depth 限制 —— 長 clip 的 60s 全域一致性靠 conv 不來。DiT 用 quadratic-in-N 的 attention 換來「任何兩個 patch（不論距離 T 還是 H/W）都能直接交換信息」—— 這對 60s 主角 identity / 相機運動一致性是必要的。代價：compute 量級遠高於 latent-WM 路線（V-JEPA / Dreamer 在 latent 空間 rollout，pixel-fidelity 不要求；Sora 每個 pixel 都要算）。

### 2.x 訓練細節（公開只到框架）

OpenAI 完全沒披露：model size、訓練 FLOPs、資料來源（YouTube? Shutterstock? 內部 partners?）、batch size、optimizer schedule。能確定的只有：

- **Joint training on images (T=1) + videos** —— image 當作 t=1 的 video，共用同個 DiT；這是 spacetime patch 設計順帶的福利
- **No paper-grade ablation** —— 連 "DiT 比 U-Net 好多少" 都沒給可比數字

`UNVERIFIED` 一切 quantitative training detail。Sunset 後多半永遠不會公開。

---

## §3 · 數據層 / 訓練 scale

| 維度 | Sora 公開 | 社區估算 / 二手轉述 |
|---|---|---|
| 訓練 video 數量 | ❌ 未公開 | 推估 100M~1B 量級 clips |
| 訓練 FLOPs | ❌ 未公開 | 推估 GPT-4 量級 |
| Re-caption pipeline | ✅ 確認用 GPT 風格 re-captioner | 細節未披露 |
| 變長 / 變 aspect 混訓 | ✅ 確認 | 比例未披露 |
| Sora-2 referee model | ✅ 確認存在 | 訓練資料未披露 |

**關鍵事實**：Sora 是 video-gen 第一個把 "scale + data + transformer" 三件套押滿的公開案例，但**它把訓練配方鎖在 OpenAI 內部**。Open-Sora / Open-Sora-Plan / HunyuanVideo / Wan 等開源路線都是在做「逆推 Sora 配方」。

---

## §4 · 代碼層

| 項 | 狀態 |
|---|---|
| Repo | ❌ 無 |
| Checkpoint | ❌ 無 |
| Paper | ❌ 無（只有 product blog + system card）|
| License | Closed |
| API | 2026-09-24 退役 |
| App | 2026-04-26 關閉 |
| Inference GPU | 未公開；二手轉述 ~$15M/day inference cost `UNVERIFIED single-source` |
| Streaming | ❌ clip-parallel only |
| Metric scale | N/A（不適用）|

**開源代理**（社區重建 Sora-class 模型的主要路線）：

- **[Open-Sora (hpcaitech)](https://github.com/hpcaitech/Open-Sora)** —— v2.0 (2025-03) 號稱 ~$200k 訓出對標 Hunyuan-Video / Runway Gen-3；v1.2 加 3D-VAE + rectified flow；v1.3 升 VAE + Transformer。代碼 / 權重 / data pipeline 全開。Paper [arxiv 2503.09642](https://arxiv.org/abs/2503.09642)
- **Open-Sora-Plan (PKU-YuanGroup)** —— 平行路線，v1.5 用 8B + 40M video 達 Hunyuan-class
- **CogVideoX (智譜)** —— 另一條開源 DiT-video，1.5 版重點改 motion 連續性
- **HunyuanVideo (Tencent)** —— 13B DiT-video，2024-12 開源；社區常拿來與 Sora 對打物理 case
- **Wan (阿里)** —— 中國線開源權重路線

**典型重建踩坑**（社區共識）：3D-VAE 訓不穩、變長 / 變 aspect 要 dynamic batching、re-caption pipeline 是必補、200k USD 是 v2.0 最低估算且 data + 工程人月不計。

---

## §5 · 評測 / Benchmark

| Benchmark | Metric | Sora 1.0 / Sora-2 | 其他 SOTA | 解讀 |
|---|---|---|---|---|
| **OpenAI 自評（report 內示例）** | 質性 | 60s 1080p、長一致性、多物件、變 aspect | 同期 SVD / Phenaki 30-50s, 720p | 質性領先確實，但無量化 |
| **PhyWorld (2411.02385)** | OOD physical law | **全面失敗** | 同期 model 同樣失敗 | 結構性，scale 不解 |
| **VBench-2.0 Physics (2026-03)** | Physics faithfulness | < 60% `TBD` 確切分數 | 大多數 model 同 regime | Sora 沒有顯著領先 |
| **PhyGenBench** | Physical commonsense | `UNVERIFIED` | `UNVERIFIED` | 待補 |

`TODO` 從 VBench-2.0 leaderboard 拉 Sora-480p vs Sora-2 確切分數。

**Sora-2 vs Sora-1 改善區**（社區實測）：
- ✅ 基礎人體運動、球類碰撞、玻璃破裂明顯好
- ❌ 流體 / 煙 / 布料、需要 volumetric simulation 的場景仍崩
- ⚠️ 改善不均衡 —— referee model 對「能被 reward model 抓到的 case」有效，對「reward model 自己也學不會的物理」（湍流、相變）無效 —— 經典 RLHF 局限

**警告**：Sora 的 in-distribution 視覺品質 + handbook 物理可靠度是兩個 metric。前者領先，後者結構性受限（§6）。

---

## §6 · Issues & Limitations

### 6.1 OpenAI report 自述 limitations（官方貼示例）

OpenAI 在 2024-02 原 report "Limitations" 段親自貼了示例：

- **玻璃杯掉地不會碎** —— 像橡膠彈
- **咬一口餅乾後**，餅乾上沒有缺口
- **物體穿越彼此 / 穿桌**（"objects passing through each other"）
- **椅子被人提起時形變不正確**

### 6.2 Hidden Assumptions

從架構推斷的隱含假設：

- **物理規律 = 資料分布的二階統計** —— 模型只能學「訓練集中常見的物理表象」；訓練集罕見的（罕見材料破裂模式、極端速度 collision）必然崩
- **DiT attention 能 implicit 學保守律** —— 但 PhyWorld 證實沒有；模型學的是 case 匹配
- **Re-captioning 不引入語意 bias** —— 實際上 GPT re-caption 會強化常見 prompt-result mapping，加劇 OOD 失敗
- **60s 是 attention compute 的 sweet spot** —— 不是「物理一致性可以撐 60s」的 evidence；換成需要 60s 演化的物理 (e.g. 慢速流體) 仍會崩
- **Sora-2 referee model = 物理 oracle** —— 但 referee 本身也是學的，對 referee 看不懂的物理 (湍流、相變) 無效

### 6.3 PhyWorld 結構性結論（atlas 聯動）

[Liu/Kang et al., "How Far is Video Generation from World Model: A Physical Law Perspective", arxiv 2411.02385 (ICML 2025)](https://arxiv.org/abs/2411.02385) 給了 data-only 路線最重要的 falsifiable 結論：

| Regime | 結論 |
|---|---|
| **In-distribution** | 完美外推 |
| **Combinatorial**（已見過的物理組合）| Scaling 有 measurable 改善 |
| **OOD（換新初始條件）** | **全面失敗** |
| **Generalization feature priority** | `color > size > velocity > shape` —— 模型擇近鄰 case 的順序揭示它根本沒在學 dynamics |

**含義**：scale + data 路線在 OOD 物理上有 **structural ceiling**，不是「再加 10× 資料就解」的問題。

### 6.4 失效案例對照表

| # | 來源 | 失效 | Severity | Workaround |
|---|---|---|---|---|
| 8.1 | OpenAI report | 玻璃杯落地不碎、像橡膠彈 | 🔴 high | 無 —— data-only 在剛體破裂上沒有 inductive bias |
| 8.2 | OpenAI report | 咬餅乾無缺口、liquid splash 失真 | 🔴 high | 同上 |
| 8.3 | OpenAI report | 物體互相穿透 / 穿桌 | 🔴 high | Sora-2 referee 部分緩解；fluid/cloth 仍崩 |
| 8.4 | PhyWorld (2411.02385) | OOD 全面失敗；case-based not principled | 🔴 structural | scaling 不解；需引入 inductive bias (aux-loss / sim-in-loop) |
| 8.5 | PhyWorld | 泛化 feature priority `color > size > velocity > shape` | 🔴 structural | 訓練資料要 deliberately decorrelate；或加 physics-aware augmentation |
| 8.6 | 社區實測 Sora-2 | steam / smoke / 布料仍崩 | 🟠 medium | 等 volumetric-aware backbone；現階段切 sim-in-loop |
| 8.7 | 商業層 | App 2026-04-26 關閉、API 2026-09-24 退役；峰值後下載量降 ~66%、報告日燒 $15M inference `TBD single-source` | 🔴 product | 工程依賴需切 Veo / Kling / Hunyuan / Open-Sora |
| 8.8 | 透明度 | OpenAI 從未公開 model size / 資料來源 / 訓練 FLOPs | 🟠 research | 學術 reproduction 全靠 Open-Sora / Open-Sora-Plan / Hunyuan 代理 |

**Maintainer 響應度**：N/A（closed model，無 issue tracker；OpenAI 透過 system card + product blog 單向溝通）。

---

## §7 · 比較 & 面試 Tip

| 同軸對手 | Output | Injection | Streaming | Open? | Status |
|---|---|---|---|---|---|
| **Sora 1.0 / Sora-2** | pixel-video | data-only (+referee) | ❌ clip-parallel | ❌ closed | sunset 2026-04-26 |
| [Google Veo / Veo-3](./veo.md) | pixel-video | data-only | ❌ | ❌ closed | shipping |
| Kling (快手) | pixel-video | data-only | ❌ | ❌ closed | shipping |
| HunyuanVideo (Tencent) | pixel-video | data-only | ❌ | ✅ 13B 開源 | shipped 2024-12 |
| Wan (阿里) | pixel-video | data-only | ❌ | ✅ 開源 | shipping |
| [Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md) (NVIDIA) | pixel-video | data-only + robotics 微調 | ❌ | ✅ 權重開放 | shipping |
| Open-Sora 2.0 | pixel-video | data-only | ❌ | ✅ 全開 | shipping |
| [V-JEPA / DreamerV4](../latent-world-models/v-jepa-2.md) | **latent** | data-only | ✅ | ✅ | shipping |
| [Genesis](../differentiable-simulators/genesis.md) | physics state | sim-in-loop | ✅ | ✅ | shipping |

> **🎤 Interview Tip.** 「Sora-2 出來了，我們的 video WM 是不是該迁過去？」
> **正答**：「Sora-2 在 audio + 基本剛體 case 上強過 Sora-1，但有兩個結構性限制讓我們不該迁。**一、商業風險**：app 2026-04-26 關閉、API 2026-09-24 退役 —— 把 production 依賴鎖到一個 OpenAI 自己都要 sunset 的服務上，是工程 anti-pattern。**二、物理結構天花板**：PhyWorld (2411.02385) 證實 data-only 路線在 OOD 物理上有 structural ceiling —— Sora-2 referee 模型把 in-distribution case 修得更好，但對 referee 自己學不會的物理 (湍流、相變、新材料破裂) 無效。我們該做的是把 video WM 當 **content / framing prior**，物理 closed-loop 走 diff-sim 或 latent-WM (V-JEPA / Dreamer) —— 那邊 streaming 又 inductive bias 更強。」
> **錯答**：「Sora-2 更強，所以迁過去」 —— 沒考慮 sunset 風險 + 沒區分 in-distribution 視覺品質 vs OOD 物理可靠度。

### 7.1 Falsifiable predictions

1. **2027-06 前**：第一篇 「Sora-class DiT + explicit physics aux-loss」 結合的 paper 會出，且在 PhyWorld OOD benchmark 上比純 data-only 提升 ≥ 30% absolute。如果到 2027-06 沒出，說明 aux-loss 對 video-gen scale 不 compatible（不太可能）。
2. **2026-09-24（Sora API 退役當日）前**：Veo / Kling / Hunyuan / Open-Sora 至少一家會推出 Sora API-compatible 替代品（drop-in replacement）—— 商業 demand 強到 inevitable。
3. **2028-12 前不會發生**：純 data-only video-gen（無 aux-loss、無 sim-in-loop、無顯式 3D 表徵）解開 brittle fracture / 流體 / 布料 OOD 案例 —— 這需要 inductive bias 而不是更多資料。Sora 路線的 sunset 不是商業失敗，是 **scale-pilled 路線在 physics 上被 falsify 走**。

---

## §8 · For the Reader（按 persona 分流）

- **影片生成工程師** —— Sora 是你的 axis-2 reference 但**不是你該依賴的 production backend**。讀 Open-Sora 2.0 paper (2503.09642) 學配方；production 切 Hunyuan / Wan / Veo。
- **VLA / robot policy 工程師** —— Sora 不適合 closed-loop agent（clip-parallel 不能 streaming，pixel rollout compute 太貴）。Video WM 接 VLA 走 V-JEPA / DreamerV4 路線；要 sim2real 走 [Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md)（Sora-class DiT 但 NVIDIA sim ecosystem 接口齊全）。
- **自駕 closed-loop 工程師** —— Sora-class 不適合 closed-loop。但可做 **offline scene reconstruction / data augmentation** —— [Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md) 是把 Sora-class 接 AD pipeline 的範例。
- **物理 conditioning 研究者** —— Sora 是你的 baseline，**你的 paper reviewer 第一個問題就是「比 Sora-class 純資料路線好多少」**。準備好 PhyWorld OOD benchmark 數字 + 至少一個 brittle fracture / fluid case 的對照。
- **神經 surrogate / diff-sim 研究者** —— Sora 是 axis-2，你在 axis-2 以外的 wedge。引用它作為「為什麼 implicit 不夠」的 motivation 就好，不用 reproduce。
- **Research 學生** —— 三件事：(a) 讀 PhyWorld (2411.02385) —— 它把 scale-pilled 路線的 structural ceiling 寫死了；(b) 讀 Open-Sora 2.0 paper —— 學現代 video-gen 工程；(c) 不要相信 OpenAI report 沒給數字的 claim —— closed model 無法 falsify。

---

## References

- **Sora 1.0** — OpenAI, 2024-02-15 · [Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/)
- **Sora 2** — OpenAI, 2025-09-30 · [Sora 2 is here](https://openai.com/index/sora-2/) · [Sora 2 System Card](https://openai.com/index/sora-2-system-card/)
- **PhyWorld** — Kang/Yue/Lu et al., *ICML 2025* · [arXiv:2411.02385](https://arxiv.org/abs/2411.02385) · [phyworld.github.io](https://phyworld.github.io/)
- **Sora Review (Liu et al.)** — [arXiv:2402.17177](https://arxiv.org/abs/2402.17177)
- **Sora as a World Model? Survey** — [arXiv:2403.05131](https://arxiv.org/abs/2403.05131)
- **Open-Sora 2.0** — hpcaitech · [arXiv:2503.09642](https://arxiv.org/abs/2503.09642) · [github](https://github.com/hpcaitech/Open-Sora)
- **PhyGenBench / VBench-2.0** (intrinsic faithfulness, 2026-03) · [arXiv:2503.21755](https://arxiv.org/html/2503.21755v1)
- **DiT (architecture origin)** — Peebles & Xie, *ICCV 2023* · [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)

---

## Boundary

- 同軸對手 Veo (Google) → [`./veo.md`](./veo.md)
- 同軸但 robotics 微調 → [`../foundation-physics-models/cosmos-wfm.md`](../foundation-physics-models/cosmos-wfm.md)
- 不同 axis (latent rollout) → [`../latent-world-models/v-jepa-2.md`](../latent-world-models/v-jepa-2.md) · [`../latent-world-models/dreamer-v4.md`](../latent-world-models/dreamer-v4.md)
- 不同 axis (sim-in-loop) → [`../differentiable-simulators/genesis.md`](../differentiable-simulators/genesis.md)
- 不同 axis (neural surrogate) → [`../neural-surrogates/fno.md`](../neural-surrogates/fno.md) · [`../neural-surrogates/graphcast.md`](../neural-surrogates/graphcast.md)
- 與 VLA 接口 → `bridge-to-vla/video-wm-to-action.md`
- 與 5 axis 全景 → `cheat-sheet/ontology.md`

---

## ✍️ 維護者註（v0.5 → v1 升級清單）

本 v0.5 基於 product blog + system card + PhyWorld + 社區拆解。下次升 v1 時補：

1. ⏳ VBench-2.0 Physics 確切分數（Sora-480p / Sora-2）
2. ⏳ Sora-2 referee model 的訓練資料 / reward signal 細節（如果 OpenAI 之後披露）
3. ⏳ Veo-3 audio launch timing vs Sora-2 對照
4. ⏳ $15M/day inference cost 二手來源驗證（目前 single-sourced）
5. ⏳ Sora API 退役後（2026-09-24）的替代品 landscape（pred §7.1 #2）
6. ⏳ PhyGenBench 具體分數
7. ⏳ Open-Sora 2.0 vs Sora-1 / Sora-2 在 PhyWorld OOD 上的對照
8. ⏳ Status v0.5 → v1，刪本節

---

[← Back to Video World Models](./overview.md)

Sources:
- [OpenAI Sora 1.0 tech report](https://openai.com/index/video-generation-models-as-world-simulators/)
- [OpenAI Sora 2 announcement](https://openai.com/index/sora-2/)
- [PhyWorld arXiv 2411.02385](https://arxiv.org/abs/2411.02385)
- [Open-Sora 2.0 arXiv 2503.09642](https://arxiv.org/abs/2503.09642)
- [VBench-2.0 arXiv 2503.21755](https://arxiv.org/html/2503.21755v1)
