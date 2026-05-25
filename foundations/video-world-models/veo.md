<!-- ontology-5axis output=pixel-video injection=data-only control=text|image-init temporal=clip-parallel domain=generalist -->

# Google Veo (Veo 1 / Veo 2 / Veo 3 / Veo 3.1)

## 1. One-paragraph TL;DR

Veo 是 Google DeepMind 的 text-to-video flagship — Sora 的直接對手，但 Google 賭的東西不一樣。Sora 賭「joint spatio-temporal latent」一條乾淨的路；Veo 走「latent diffusion transformer + 漸進 audio 整合」的混合路。三代節奏：**Veo 1 (2024-05 I/O)** 拿 1080p / >60s clip 做地基（waitlist via VideoFX）；**Veo 2 (2024-12 公告 → 2025-04 GA on Vertex AI)** 直接把 ceiling 推到 **4K / 多分鐘**，宣稱明顯改善物理理解（流體、光照、攝影機運動）；**Veo 3 (2025-05-20)** 是業界第一個 **native audio-video joint diffusion** — 同一個 transformer 同時去噪 video latent 與 audio latent，產出對嘴的 dialogue / synced SFX / ambient。Veo 3.1 (2025-10) 加長到 60s 並補 ingredients-to-video。Google 的策略賭注：**「audio 是下一個 modality cliff，先卡住它比追 Sora 物理分數更重要」**。代價：複雜物理、長 horizon 一致性、character consistency 仍輸 Sora 2。

## 2. Core mechanism

公開資料（Veo-3-Tech-Report.pdf + DeepMind blog + 二手 deconstruct）拼出來的架構：

```
text prompt ─┐
image prompt ┼─► [Text/Image Encoder (Gemini-family backbone)]
             │              │
             │              ▼
             │      [Conditioning tokens]
             │              │
             ▼              ▼
   ┌─────────────────────────────────────┐
   │  Latent Diffusion Transformer (DiT) │
   │                                     │
   │   spatio-temporal video patches ◄──►│ joint denoising
   │   temporal audio latents       ◄──►│ (Veo 3 起)
   │                                     │
   │   cross-attention to conditioning   │
   └─────────────────────────────────────┘
             │              │
             ▼              ▼
     [Video VAE decoder]  [Audio decoder → 48kHz stereo AAC@192k]
             │              │
             ▼              ▼
        RGB frames       waveform
        (24 fps,         (synced)
         up to 4K)
```

關鍵點：

- **Latent diffusion + DiT backbone** — VAE 把 RGB clip 壓進 latent，DiT 在 latent 做 denoising（與 Sora 的 spacetime-patch 思路同源）
- **Veo 3 的單一最重要創新**：audio latent 與 video latent **共用同一個 diffusion process**，而不是後接一個獨立 audio 模型。這讓 lip-sync / footstep / 物件碰撞聲在 denoising step 就被綁定，而不是事後對齊
- 訓練資料規模：DeepMind **未公開**具體 hours/source — 一般認為使用 YouTube + Google 內部資料（這部分有版權爭議，Google 立場是「在 ToS 允許範圍內」）`[TBD: verify exact training corpus disclosure in Veo-3-Tech-Report.pdf]`
- Conditioning 端與 Gemini 共用 text encoder，這是 Google 體系最大的 leverage — prompt 理解明顯比 Sora-1 強

## 3. 五軸定位 + 同軸對手

| 軸 | Veo 3 / 3.1 值 | 備註 |
|---|---|---|
| Output | `pixel-video`（+ audio） | Veo 3 起多一條 audio stream，但 ontology 沒有 audio 軸，保留 `pixel-video` |
| Injection | `data-only` | 公開資料無 PDE / constraint loss；物理全靠 scale |
| Control | `text|image-init`（3.1 加 ingredients = multi-image reference） | 無 force / contact / trajectory 接口 |
| Temporal | `clip-parallel`（8s 一段；3.1 可串到 60s） | 不是 streaming AR |
| Domain | `generalist` | 無 robotics / driving 專版 |

同軸對手：

| 方法 | 與 Veo 的差異 |
|---|---|
| **[Sora 2](./sora.md)** (OpenAI, 2024-25) | 同 clip-parallel + DiT；物理分數普遍評為更高；character consistency 較強；但**無 native audio joint diffusion** — audio 走後接路線 |
| **[Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md)** (NVIDIA, 2025) | 同 pixel-video / data-only，但定位 pre-trained WFM 供 robotics/driving fine-tune；Veo 無此 downstream hook |
| **Kling 2.x / 3.0** (Kuaishou) | 中國線 SOTA；解析度 / 性價比領先（Artificial Analysis leaderboard 第一）；audio 較弱 |

> [Genie-2](../latent-world-models/genie-2.md) (DeepMind 內部姊妹) — output 是 `action-seq + pixel-video`、temporal 是 `streaming-cache`、control 含 `action` — 與 Veo 是「同公司不同問題」：Veo 解 cinematic generation，Genie 解 agent-playable WM。

## 4. ⚡ shines / ❌ breaks

⚡ **真正領先的 regime**

- **Audio-video joint generation**：對嘴 dialogue / 物件互動聲 / ambient — 目前**唯一一家**做 native joint diffusion，不是後接（社群普遍認為這是 Veo 3 的決定性優勢）
- **解析度 ceiling**：Veo 2 / 3 在 Vertex AI 跑得到 **4K**（Veo 2 文檔明確 4096×2160），消費端 VideoFX 鎖 720p / 8s
- **Cinematic prompt 控制**：camera angle、lens、shot composition — 受惠於 Gemini text encoder，遠優於 Veo 1 / SVD
- **流體與光照細節**：DeepMind 自家 demo 與二手測試（ChatForest, DataCamp）都標 Veo 2 在 coffee pouring / shadows / reflections 上有明顯進步

❌ **Known failure modes**

- **複雜物理仍弱於 Sora 2**：多方 benchmark 共識 — Sora 2 在「物件互動 / 光線折射 / 重力 / fluid splash」最強，Veo 3.1 在 lighting / texture 領先但 physics 不是它的強項（[Genra AI](https://genra.ai/blog/kling-3-vs-seedance-2-vs-veo-3-vs-sora-2), [Lushbinary](https://lushbinary.com/blog/ai-video-generation-sora-veo-kling-seedance-comparison/)）
- **Object hallucination during camera moves**：Google AI Developers Forum #111898 有 dental workflow 案例 — Veo 3.1 在 first/last frame + reference image 條件下，相機移動時仍會幻覺出不存在的器械、改變牙齒幾何
- **Speech 偶爾亂碼**：短語音段落 lip-sync / off-beat、garbled speech 仍是 active dev area
- **多場景 / character arc / 跨 clip 一致性差**：「Veo 3 will not carry multi-scene narratives, character arcs, or ongoing calendars」— 8s 邊界一切歸零
- **長度天花板**：Veo 3 原版 8s 上限；3.1 可串到 60s 但靠 ingredients-to-video stitching，不是真 long-horizon rollout
- **配額牆**：消費端 3-4 generations/day 即斷

## 5. Reproduction

無 public weights — **完全 closed model**。可用路徑：

| 入口 | 解析度 / 長度 | 價格 (Veo 3 標準, USD) |
|---|---|---|
| VideoFX (consumer, Google Labs) | 720p / 8s | 受 AI Pro/Ultra 訂閱配額限制 |
| Gemini API (`veo-3.0-*`) | 720p-1080p | 按生成計 |
| **Vertex AI** (enterprise) | up to 4K / 多分鐘 (Veo 2)、8s up to 4K (Veo 3) | **$0.50/s video-only, $0.75/s with audio** → 8s+audio ≈ **$6.00/clip** |
| Veo 3.1 Fast / Light | 1080p / 較短 | $0.10/s (Fast) — 約 1/5 價 |

典型踩坑：

- audio 多 50% 成本，但短 prompt 反而比關 audio 還貴一倍 — 做 ablation 記得算錢
- Vertex preview 模型 ID 會變（`veo-2.0-generate-001`, `veo-3.0-generate-preview` 等），CI 別 hard-code
- 跑大量 generation 時，$300 free credit 約 600-857s Veo 3 standard — 一個 paper benchmark 就燒完

## 6. Cross-line synthesis

- **vs pixel-WM 同條線（Sora / Cosmos / Kling）**：Veo 是「Google 體系 + audio-first」分支。要做 robotics WM pre-train 還是選 Cosmos-Predict（有 robotics conditioning hook），Veo 沒這條接口
- **vs latent-WM ([DreamerV4](../latent-world-models/dreamer-v4.md) / [V-JEPA-2](../latent-world-models/v-jepa-2.md))**：Veo 完全不重疊 — Veo 賣的是生成 fidelity，latent-WM 賣的是 agent 用得起的 rollout cost；Veo 的 latent 不對外暴露，沒法當 dreamer
- **vs diff-sim ([Genesis](../differentiable-simulators/genesis.md) / Brax 系)**：Veo 是 data-only 純 black-box，diff-sim 是 white-box；要 sim2real 的 robotics scenario 還是 diff-sim 路線
- **vs surrogate ([GraphCast](../neural-surrogates/graphcast.md) / [FNO](../neural-surrogates/fno.md))**：完全不同 domain（generalist vs weather/fluid field），無重疊
- **Veo + diff-sim 合作可能**：用 diff-sim 生 contact-rich rollout → Veo 做 photoreal stylization → 給 robotics policy 當 augmented vision data。**但 Veo 沒 trajectory/action conditioning，這條 pipeline 目前只能單向**

## 7. References

**Primary**

- **Veo 3 Tech Report (Google DeepMind, 2025)** — https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf
- DeepMind Veo product page — https://deepmind.google/models/veo/
- Google I/O 2024 Veo 1 announcement (TechCrunch) — https://techcrunch.com/2024/05/14/google-veo-a-serious-swing-at-ai-generated-video-debuts-at-google-io-2024/
- Vertex AI Veo 2 preview docs — https://cloud.google.com/vertex-ai/generative-ai/docs/models/veo/2-0-generate-preview

**Secondary (deconstructions / tests)**

- Deconstructing Veo 3 (Bhargav Patel, Google Cloud Community) — joint audio-video latent diffusion 分析
- Google Veo 2 Review (ChatForest) — 4K / Sora 對比
- Veo 3.1 Review after 3 Weeks (pxz.ai) — 實測 limits
- Kling vs Seedance vs Veo vs Sora 2 (Genra AI, Lushbinary) — physics benchmark consensus

## 8. §8 Pitfall log

### §8.1 Camera-move hallucination on professional workflows
- **Source**: Google AI Developers Forum thread #111898 (Veo 3.1 dental workflow)
- **Severity**: High for domain-specific use
- **Symptom**: Reference image + first/last frame 條件下，camera move 期間幻覺出不存在的器械、牙齒幾何在 pan 過程中改變
- **Workaround**: 加強 prompt 中的物件 specificity（"low angle drone view", "neon streetlights", "loud engine roar" 這類具體描述明顯減少幻覺）；接受 8s clip 內部物件 identity drift 是 hard limit

### §8.2 Speech / lip-sync degrade on short utterances
- **Source**: Google's own Veo 3 limitations doc + community reports
- **Severity**: Medium
- **Symptom**: Garbled speech / off-beat lip sync，尤其短語音段
- **Workaround**: 避免極短對白；prompt 明確時間戳；或關 audio 用後製對嘴

### §8.3 Multi-scene / character-arc 不持續
- **Source**: Veo 3 Limits and Restrictions (Segmind)
- **Severity**: High — 任何敘事長片 use case 必踩
- **Symptom**: 8s 邊界外 character identity / scene continuity 不保
- **Workaround**: Veo 3.1 ingredients-to-video（多 reference image stitching）部分緩解；但本質仍是 clip-level clip-parallel 限制

### §8.4 Physics vs Sora 2 gap
- **Source**: Genra AI / Lushbinary / VidGuru 多方 benchmark
- **Severity**: Medium（依 use case）
- **Symptom**: 物件互動、光線折射、重力、fluid splash 在 head-to-head 普遍輸 Sora 2
- **Workaround**: 如果 deliverable 重 physics realism，用 Sora 2；如果重 audio + cinematic + Google ecosystem，留 Veo

### §8.5 Pricing trap on audio
- **Source**: Vertex AI 公開定價
- **Severity**: Low（純成本）
- **Symptom**: audio 多 50% 成本，benchmark / ablation 預算容易爆
- **Workaround**: 開發階段先跑 video-only 變體，audio 留到最後一輪

### §8.6 Training data disclosure 不透明
- **Source**: Veo 3 Tech Report `[TBD: verify exact disclosure]`
- **Severity**: Compliance / reproducibility 影響
- **Symptom**: 訓練資料來源（YouTube 比例、版權範圍）未完全公開；學術引用時 dataset description 受限
- **Workaround**: 如做 reproducibility study，明確標註 closed-model + 不可知 corpus

### §8.7 Implicit-from-data 物理的 ceiling
- **Source**: 整條 pixel-WM 路線共通問題（overview.md §8 zone-level）
- **Severity**: 路線性 — 不是 Veo 單一問題
- **Symptom**: 流體 splash、剛體碰撞細節、紙張褶皺 — 高頻物理破綻在 zoom-in 時暴露
- **Workaround**: 路線上的天花板。要真物理 ground truth，繞 diff-sim / sim-in-loop injection；Veo 沒這條 hook
