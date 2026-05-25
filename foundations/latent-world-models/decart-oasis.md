<!-- ontology-5axis output=pixel-video injection=data-only control=action|image-init temporal=streaming-cache domain=robotics -->

# Decart Oasis — A Universe in a Transformer (Decart × Etched, Oct 2024)

> **Domain tag 註**：v2 Check 9c 不允許 `domain=generalist`（白名單只給 Sora/Veo/Cosmos）。Oasis 實際 domain 是「Minecraft gaming」，這在 v2 ontology 沒有原生 value；最接近的代理是 `robotics`（agent-in-environment, action-conditioned, sim-like loop）— 待 v2.x 加 `gaming` sub-tag 後重簽。

## 1. One-paragraph TL;DR

Oasis 是 Decart AI（以色列）與 Etched（矽谷晶片新創）2024-10-31 共同發表的「first playable, real-time, AI-generated world」— 給定一張 Minecraft 起始幀 + 即時鍵鼠 input，模型以 **20 fps、360–720p** 串流生成下一幀，全程沒有 Minecraft engine 也沒有 game logic，純粹從數百萬小時的 Minecraft gameplay 影片裡 implicit 學出物理與規則。Prior gap = 「Sora 級 video diffusion + Genie 1 級 action conditioning + **真正能 real-time 給人玩**」這三件之前沒人同時做到 — Sora 一秒影片要算 10–20 秒，Genie 1 限平台 2D，Decart 把 DiT autoregressive 跑進 47ms / frame。它跟 Genie 2 是同代 bet — 都壓 `streaming-cache` + action-conditioning，但 Oasis 賭硬體（Etched Sohu ASIC）能解 inference 瓶頸，Genie 2 賭 DeepMind 的 scale + distill。發表 1 個月後 Decart 拿到 $32M A 輪。

## 2. Core mechanism

公開資料來自 Decart blog + open-oasis-500M weights（HF / GitHub）：

```
              ┌──────────────────────────┐
 image init ─▶│  ViT VAE encoder         │── z_init (latent frame)
              └──────────────────────────┘
                          │
                          ▼
   ┌──────────────── KV cache (sliding window, prev N latent frames) ────────────────┐
   │                      │                                                          │
   │   ┌──────────────────┴──────────────────┐                                       │
   │   │  DiT backbone                       │                                       │
   │   │  - spatial attention (intra-frame)  │                                       │
   │   │  - temporal attention (inter-frame) │  ──▶ ε̂  (predicted noise)             │
   │   │  - interleaved blocks               │       │                               │
   │   └──────────────────▲──────────────────┘       │                               │
   │                      │                          │                               │
 action token ────────────┘                          │                               │
 (kb/mouse, per-frame)                               ▼                               │
                                  Diffusion Forcing denoising loop                   │
                                  (per-token independent noise level,                │
                                   dynamic noise schedule)                           │
                                                     │                               │
                                                     ▼                               │
                                              z_{t+1} (next latent) ─────────────────┘
                                                     │
                                                     ▼
                                          ViT VAE decoder ──▶ pixel frame
```

要點：

- **DiT backbone + ViT VAE**：跟 Sora / Mochi-1 同 family（Peebles & Xie DiT），但跑 autoregressive 而非 clip-parallel — Oasis 同時 inherit DiT 視覺品質 + Genie 1 的 frame-by-frame action interface。
- **Diffusion Forcing (Chen et al., NeurIPS 2024)**：每個 token / 幀有獨立 noise level，訓練時不要求整 sequence 共用 noise；inference 時可以「乾淨幀 condition 髒幀」，這是讓 autoregressive 不 collapse 的關鍵 trick。
- **Spatial-temporal interleaved attention**：spatial block 處理單幀內部，temporal block 跨幀看 KV cache 裡的歷史 latents — 後者就是 `streaming-cache` 的工程實現。
- **Dynamic noising at inference**：前幾步 diffusion 注更多 noise 抑制 error accumulation，後幾步抽掉讓高頻細節穩定。這是 Oasis 的反 drift 主要 trick（vs Genie 2 用 classifier-free guidance）。
- **47 ms / frame on H100**：blog 自報數字；不算 VAE decode 也不算 first-token latency。實測 demo 是 20 fps（50 ms / frame budget），跟 47 ms 對得上。
- **Etched Sohu hardware bet**：Sohu 是 transformer-only ASIC（TSMC 4nm, 144GB HBM3E），claim 8× Sohu server ≈ 160× H100（[TBD] benchmark 從未獨立驗證，2026-03 仍未出貨）。Oasis 的「4K future」完全壓在 Sohu 能不能 ship 上。
- **參數**：開源 weights 是 500M downscaled 版（HF: `Etched/oasis-500m`）；live demo 用更大的 checkpoint，size 未揭露 [TBD]。

## 3. 五軸定位 + 同軸對手

| Axis | Value | 說明 |
|---|---|---|
| Output | `pixel-video` | ViT VAE decode 出 360–720p RGB；不是 pure latent rollout |
| Injection | `data-only` | 數百萬小時 Minecraft 影片，無 physics loss、無 sim-in-loop |
| Control | `action\|image-init` | per-frame 鍵鼠 token + 起始幀 image |
| Temporal | `streaming-cache` | DiT + sliding KV cache + Diffusion Forcing per-frame |
| Domain | `robotics`（proxy，實為 gaming/Minecraft）| v2 沒 `gaming` value；Minecraft 視為 agent-environment loop |

同軸對手（皆 action-conditioned interactive WM）：

| 競爭方法 | 五軸差異 | 對撞點 |
|---|---|---|
| **[Genie-2](./genie-2.md)** (DeepMind, Dec 2024) | domain=`generalist`（多場景）· output=`latent\|pixel-video` · distilled real-time | 同 streaming-cache 同 action-conditioning；Genie 2 通用視覺多，Oasis 速度快 + 開源 500M — generality vs reproducibility 對撞 |
| **[DreamerV4](./dreamer-v4.md)** (2025) | output=`latent-tokens` · temporal=`latent-rollout` · 配 actor-critic | Dreamer 為 RL agent，Oasis 為人玩 + demo — 不 decode vs 必 decode 對撞 |
| **[V-JEPA-2](./v-jepa-2.md)** (Meta, Jun 2025) | output=`latent-tokens` · injection=`data-only` · domain=`robotics` | V-JEPA-2 純 representation 不生像素，Oasis 必須生像素給人看 |
| **[Sora](../video-world-models/sora.md)** (OpenAI, Feb 2024) | temporal=`clip-parallel` · 不收 action · 10-20s/sec inference | Sora 視覺最強但不收 action 也不 real-time，Oasis 走相反端 |
| **MineWorld** (arxiv 2504.08388, Apr 2025) | open-source · 同 Minecraft domain | MineWorld 是 Oasis 的學術開源替代品；架構更小但 full reproducible |

## 4. ⚡ Shines / ❌ Breaks

### ⚡ Shines

- **真正 real-time interactive (20 fps)**：Sora/Veo 量級的視覺品質下，第一個能讓人 keyboard 進去玩的 — 47ms/frame 是 SOTA video gen 100× 加速。
- **公開 web 試玩**：oasis.decart.ai 提供瀏覽器 demo，無 API key 無排隊（vs Genie 2 只有 DeepMind closed preview）。
- **500M 開源 + HuggingFace weights**：`Etched/oasis-500m` + `etched-ai/open-oasis` — 唯一這個量級的 action-conditioned WM 有可下載 weights（Genie 2、Genie 3 全閉）。
- **Diffusion Forcing 範式驗證**：把 NeurIPS 2024 的 paper 立刻 productionize；後續 MineWorld、Memory Forcing (arxiv 2510.03198) 都沿用同一條 trick。
- **Hardware-software co-design 故事**：Etched Sohu 的存在不只是 PR — 它是把「transformer-only inference」當作晶片設計 thesis，跟 Oasis 構成「我們會買最該被加速的 workload」的閉環。

### ❌ Breaks

- **5-min hard session limit**：Wikipedia 引述官方 demo 規則；超過要 reset。原因是 KV cache 滑動窗口 + drift 累積，無法持久。
- **空間 drift / dementia 感**：媒體普遍評語是「dream-like」「hallucinatory」— 走幾秒後回頭，牆換顏色、洞變樹。官方自承 limitation 為 "fuzzy video in the distance, temporal consistency of uncertain objects"。
- **Inventory / 數值狀態不一致**：放一塊 dirt 可能 spawn 整個新地形；UI 數字（生命、物品數）會 morph — 因為純像素 prediction 不知道有 discrete 狀態這回事。
- **單一 domain (Minecraft-only)**：訓練資料窄；OOD prompt（非 Minecraft texture 的起始幀）失敗率 [TBD: no systematic OOD eval published]。
- **Etched 硬體依賴未兌現**：blog 賣的 4K future 壓在 Sohu 上；2026-03 仍 0 ship 0 independent benchmark — 整條「Oasis × Sohu vertical」是 paper plan。
- **無聲音、無物理 query API**：純 visual sandbox，沒法當 agent training env（不能讀 state、不能 reward）— 跟 DeepMind 把 Genie 2 包成 "agent training environment" 的姿勢相反。
- **完整 arxiv paper 缺**：只有 blog post + 500M reference code；training set details、large model size、training compute、systematic eval 全未公開 [TBD]。

## 5. Reproduction notes

**官方狀態**：500M weights 開源（HF + GitHub `etched-ai/open-oasis`）；large demo checkpoint、training code、dataset 閉源。

**Public 可用替代**：

- **open-oasis 500M**：reference inference code（`dit.py`, `vae.py`, `attention.py`, `rotary_embedding_torch.py`），可在單張 H100 跑；demo 品質明顯比 live demo 差。
- **MineWorld** (arxiv 2504.08388, Apr 2025)：完整開源，同 Minecraft domain，可當 ablation 平台。
- **Memory Forcing** (arxiv 2510.03198, Oct 2025)：在 Oasis-style 架構上加 spatio-temporal memory 解 §8.2 drift；reference 實作公開。

**最小可跑 Oasis-style setup（500M scale）**：

- 1× H100 (80GB) for inference (47ms/frame)；training scale unknown but probably 32–128 × H100 cluster [TBD]
- 資料：Minecraft gameplay screen recordings + keyboard/mouse log 同步；Decart 用 "millions of hours"，社區 reproduction 通常 1k–10k h
- 架構：ViT VAE（compress 384px → 8× downsampled latent）+ DiT backbone（spatial/temporal interleave）
- 訓練：Diffusion Forcing — 每 latent frame 獨立 noise level；不要照 standard video diffusion 共用 noise
- 典型踩坑：
  1. **Autoregressive exposure bias** — 用 standard diffusion loss 訓，inference 時前幾秒還行，之後爆。Diffusion Forcing 的 per-frame noise schedule 是必須。
  2. **KV cache 滑動窗口長度** — 太短 = 物件 5s 後消失；太長 = inference 變慢 + 訓練 memory 爆。Oasis 公開 spec 沒給；社區實測 16–32 frames 之間 [TBD]。
  3. **VAE decode 是 latency bottleneck** — pure DiT step 已經 ≤ 47ms，但 ViT VAE decode 可能再吃 20-30ms，必須 distill 或 pipeline。
  4. **Action token embedding 維度** — 一個 frame action 是 ~10 keys (WASD + space + shift + click + mouse Δx/Δy)；錯誤展平成 single token 會丟 mouse 精度。

## 6. Cross-line synthesis

| 路線 | 怎麼接 |
|---|---|
| **Pixel-WM** ([Sora](../video-world-models/sora.md) / [Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md)) | Cosmos-Predict 同 `data-only` injection 但 `temporal=clip-parallel`，Oasis 走 `streaming-cache`。可把 Cosmos 拿來當「offline 高品質 batch generator」生 training data，Oasis 當「online interactive proxy」— 兩者 architecture 都是 DiT backbone，weight 部分可 share 或 distill。 |
| **Latent-WM** ([Genie-2](./genie-2.md) / [DreamerV4](./dreamer-v4.md)) | Genie 2 是直接同代對手（DeepMind Dec 2024 vs Decart Oct 2024）；DreamerV4 走 latent rollout 給 RL agent，Oasis 走 pixel rollout 給人。可以把 Oasis 當 environment、Dreamer agent 當 policy — 但 Oasis 沒 state API，需要先包一層 visual classifier 抽 state。 |
| **Diff-sim** ([Genesis](../differentiable-simulators/genesis.md) / [MuJoCo MJX](../differentiable-simulators/mujoco-mjx.md)) | 完全不接 — Oasis 走 implicit visual physics，diff-sim 走 explicit ground-truth physics；要硬接，把 diff-sim 當 "physics oracle" 給 Oasis 做 [TBD: PhysGen-style] post-hoc consistency filter，但目前無公開實作。 |
| **Neural surrogate** ([FNO](../neural-surrogates/fno.md) / [GraphCast](../neural-surrogates/graphcast.md)) | 不接；surrogate 走場域物理，Oasis 走 generalist visual — 哲學距離最遠。 |

Composition pattern：**Oasis-as-real-time-renderer + 外掛 state engine** — Oasis 視覺強但無 state，可在外面套一層 game state engine（Minecraft 本身 state 規則簡單）對 visual prediction 做硬約束 — 即 hybrid neural-symbolic interactive WM。**Streaming-cache wave** = Genie-2 + Decart-Oasis + Genie-3 共同代表 2024Q4–2025Q3 的 action-conditioned interactive WM 浪潮，本質是把 LLM streaming inference 那套 KV cache 工程搬到 video generation 上。

## 7. References

**Canonical**：

1. [Decart AI — "Oasis: A Universe in a Transformer" (官方 blog, 2024-10-31)](https://decart.ai/publications/oasis-interactive-ai-video-game-model)
2. [Oasis model card site](https://oasis-model.github.io/) — 唯一的「semi-technical」資料來源
3. [etched-ai/open-oasis GitHub](https://github.com/etched-ai/open-oasis) — 500M reference inference code
4. [Etched/oasis-500m HuggingFace weights](https://huggingface.co/Etched/oasis-500m)

**Secondary**：

5. [TechCrunch — "Decart's AI simulates a real-time, playable version of Minecraft" (2024-10-31)](https://techcrunch.com/2024/10/31/decarts-ai-simulates-a-real-time-playable-version-of-minecraft/)
6. [InfoQ — "Decart and Etched Release Oasis" (2024-11)](https://www.infoq.com/news/2024/11/decart-etched-oasis/)
7. [Wikipedia — Oasis (Minecraft clone)](https://en.wikipedia.org/wiki/Oasis_(Minecraft_clone))
8. [Tom's Hardware — Sohu AI chip 20× H100 claim](https://www.tomshardware.com/tech-industry/artificial-intelligence/sohu-ai-chip-claimed-to-run-models-20x-faster-and-cheaper-than-nvidia-h100-gpus)
9. [Chen et al., "Diffusion Forcing" NeurIPS 2024](https://arxiv.org/abs/2407.01392) — Oasis 訓練 trick 原 paper
10. [Peebles & Xie, "Scalable Diffusion Models with Transformers (DiT)" ICCV 2023](https://arxiv.org/abs/2212.09748)
11. [MineWorld (arxiv 2504.08388, Apr 2025)](https://arxiv.org/abs/2504.08388) — 開源 Minecraft WM 替代品
12. [Memory Forcing (arxiv 2510.03198, Oct 2025)](https://arxiv.org/abs/2510.03198) — Oasis drift 後續解決方案
13. [SiliconANGLE — "Decart reels in $32M" (2024-12-19)](https://siliconangle.com/2024/12/19/ai-world-model-startup-decart-reels-32m/)

## 8. §8 Pitfall log

> 因 Oasis 無 arxiv paper、large model 閉源，無正式 GitHub issue tracker 對應 live demo bug；以下 pitfalls 以官方 blog 自承 + 媒體實測 + open-oasis-500M 社區回報為主。

### §8.1 Spatial drift — 5-min hard session limit

- **Source**：Wikipedia 引官方 demo 規則「5 minutes per play session before restart required」；Decart blog 自列 "difficulties over long contexts"
- **Severity**：HIGH — 是商品化的直接上限
- **Mechanism**：sliding KV cache 滑出窗口的歷史幀完全丟失；autoregressive + data-only 雙重 drift 放大
- **Workaround**：Memory Forcing (arxiv 2510.03198) 加 spatio-temporal explicit memory；或外掛 retrieval memory bank 把舊 latent 召回

### §8.2 Object identity flip / dementia-like teleport

- **Source**：Tom's Guide / TechCrunch 媒體實測 "could not maintain coherent logic"；Decart blog "temporal consistency of uncertain objects"
- **Severity**：HIGH — 直接破壞遊戲性
- **Mechanism**：純像素 prediction 不知道「牆是牆」；走幾步回頭，pixel 上重 generate 一面牆，顏色/紋理可能換
- **Workaround**：3D-explicit scene memory（World Labs gen-3D + Oasis hybrid）；或 retrieval-augmented latent token

### §8.3 Inventory / discrete state morph

- **Source**：InfoQ 引 "placing dirt blocks spawns entirely new environments"；Decart blog "precise control over inventories"
- **Severity**：HIGH for gameplay; LOW for tech demo
- **Mechanism**：discrete game state（生命值、物品數、座標）被當連續像素學，無 symbolic 護欄
- **Workaround**：hybrid neural-symbolic — 把 inventory state 從 pixel decode 出來再 hard-overwrite 回去；或乾脆讓 Oasis 只生環境、UI 用 traditional engine 覆蓋

### §8.4 OOD prompt → catastrophic failure

- **Source**：[TBD: no systematic OOD eval published]；社區社交 anecdote 為主
- **Severity**：MEDIUM — 對研究選型重要，對 Minecraft demo 不重要
- **Mechanism**：訓練資料 Minecraft-only，非 Minecraft texture 的 init image 會 collapse 回 Minecraft-like 視覺或直接 noise out
- **Workaround**：跨 domain 重訓；或承認 single-domain WM 定位（vs Genie 2 的 generalist 賣點）

### §8.5 Etched Sohu hardware bet 未兌現

- **Source**：blog 標榜「4K with Sohu」；Sohu 2024-06 announce，2026-03 仍未獨立 benchmark 也未量產 (Tom's Hardware, Spheron Network)
- **Severity**：MEDIUM — 不影響當前 demo，但影響「Oasis × Sohu vertical」的長線敘事
- **Mechanism**：Sohu 是 transformer-only ASIC，hard-wired computation graph；任何 architecture deviation（diffusion 不算 pure transformer、MoE、SSM）都不支援 — Oasis 走 DiT 屬「transformer + diffusion noise loop」，是否完全契合 Sohu 的 critical path 未公開驗證
- **Workaround**：把 Oasis 視為 H100-native 產品來評估；忽略 4K future story

### §8.6 無 arxiv paper / training details 全閉

- **Source**：blog 只給 inference 數字；training compute、dataset size、large checkpoint size 從未公開
- **Severity**：HIGH for academic reproduction；MEDIUM for industry（500M weights + MineWorld 可替代）
- **Workaround**：用 open-oasis-500M + Diffusion Forcing paper + DiT paper 三件套重建工程細節；MineWorld 當 receipt

### §8.7 無聲音、無 state API — 不能直接當 agent training env

- **Source**：blog 完全不提 agent training；對比 DeepMind Genie 2 blog 自己定位為 "endless variety of action-controllable environments for training and evaluating embodied agents"
- **Severity**：MEDIUM — 限制下游 use case，不影響 PR
- **Workaround**：外掛 visual classifier 抽 state；或選 Genie 2 / MineWorld 替代
