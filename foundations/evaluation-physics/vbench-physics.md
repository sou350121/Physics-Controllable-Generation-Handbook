<!-- ontology-5axis output=N/A injection=N/A control=text temporal=clip-parallel domain=N/A -->

# VBench / VBench-2.0 / PhysBench — Eval Suite Landscape

> Eval benchmarks 在五軸 ontology 裡 `output/injection` 都 N/A — 它們不生成，但**它們決定誰被當成「物理對」的生成器**。對 handbook 而言比任何單一 paper 都重要。

---

## 1. One-paragraph TL;DR

2024 年以前 video generation 評估幾乎被 FVD / IS / CLIP-score 三件套壟斷 — 這些 metric 對「像不像影片」很敏感，但對「物理對不對」幾乎零訊號。**VBench** (arxiv 2311.17982, CVPR 2024 Highlight) 第一次把「video generation quality」拆成 16 個 disentangled 子維度，並附 human preference annotation 校準；它讓 [Sora](../video-world-models/sora.md) / Gen-3 / Kling / Pika 有了同一張比分表。但 VBench v1 的 16 維大多仍屬 **perceptual quality** —— 物理直覺只能從 "motion smoothness" / "dynamic degree" 兩個間接維度推測。**VBench-2.0** (arxiv 2503.21755, Mar 2025) 才正式新增 `Physics` 與 `Commonsense` 兩個 top-level dimension，定義為 "intrinsic faithfulness" 範疇。並行地，**PhysBench** (arxiv 2501.16411, ICLR 2025) 走另一條路 —— 它不評生成模型，而是評 VLM 對物理世界的**理解**能力（10,002 條 video-image-text，4 domains × 19 sub-classes）。三者合起來組成 2026 年看一個「物理可控生成」模型的標準三角形：VBench 看 surface quality / VBench-2.0 看 intrinsic physics / PhysBench 看 evaluator 自己懂不懂物理。

---

## 2. Core mechanism

### VBench v1 的 16 維

```
┌──────────────── VBench v1 (16 dims) ──────────────────┐
│ Quality (7)         Semantic (9)                       │
│  - subject cons.     - object class                    │
│  - background cons.  - multiple objects                │
│  - temporal flicker  - human action                    │
│  - motion smooth     - color                           │
│  - dynamic degree    - spatial relationship            │
│  - aesthetic         - scene                           │
│  - imaging quality   - temporal style                  │
│                      - appearance style                │
│                      - overall consistency             │
└────────────────────────────────────────────────────────┘
   Total Score = weighted_avg(Quality, Semantic)
```

每個維度有 **量身打造的 prompt 子集** + 對應的 specialist scorer（CLIP / DINO / RAFT / 動作分類器 / etc.），最後做 0-1 正規化。Human preference annotation 用來 calibrate weight。

### VBench-2.0 的 5 大領域

```
Human Fidelity | Controllability | Creativity | Physics | Commonsense
     ↑              ↑                 ↑           ↑          ↑
   anatomy       prompt obey      novelty    PHY laws    causality
```

論文宣稱共 **18 fine-grained capabilities**。`Physics` 與 `Commonsense` 為新增大類，scorer 改採 **generalist + specialist 混合** — generalist 是 SOTA VLM/LLM（GPT-4o / Qwen-VL），specialist 是針對 anomaly 的視覺偵測器。各維度分數正規化到 0.3–0.8 共同尺度（per-dim normalization 是 VBench-2.0 一個有爭議的設計，後 §8 詳）。Physics sub-dim 具體列表 paper 公開頁尚未列全，需直接讀 PDF 或等 leaderboard 開源 `[TBD: 確認 Physics 5 sub-dim 名稱 — 從 arxiv 2503.21755 v1 §3 表抓]`。

### PhysBench 結構

VLM-only 評估，不評生成器：
- 10,002 條 video-image-text interleaved 題目（200 val + 10k test）
- 4 domains: object properties / object relationships / scene understanding / physics-driven dynamics
- 19 sub-classes，跨 8（一說 10）capability dims `[TBD: paper 表 1 同時出現 8/10 兩個數字 — 以 ICLR camera-ready §3.2 為準]`
- 75 個 VLM 實測，best zero-shot 仍離 human 25+ 分；作者另外 release `PhysAgent` (agent + 物理先驗 + 專家模型) 提升 GPT-4o 18.4%

---

## 3. 五軸定位 + 同軸對手

**Header 解釋**：`output=N/A injection=N/A` —— eval suite 不生成、不注入物理；`control=text` —— 對 T2V 模型 evaluate；`temporal=clip-parallel` —— 評估目標主要是一次性 clip；`domain=generalist` —— 不限定場域（這也是缺點，見 §4）。

| Benchmark | Physics 軸涵蓋 | 評估對象 | 開源 leaderboard |
|---|---|---|---|
| **VBench v1** | 弱（僅 motion smooth / dyn degree 間接） | T2V 生成 | ✅ HF Space |
| **VBench-2.0** | 中（5 大類含 Physics + Commonsense） | T2V 生成 | ✅ |
| **PhysBench** | 強（4 domains × 19 sub） | VLM 理解 | ✅ |
| **PhyGenBench** | 強（27 physical laws × 4 domains: mechanics/optics/thermal/material） | T2V 生成 | ✅ (OpenGVLab, ICML 2025) |
| **PhyWorldBench** | 強（多場景 physical realism） | T2V 生成 | ✅ (arxiv 2507.13428, 2025) |
| **WorldModelEval** | 中（agent control 視角） | WM rollout policy | partial |

**Composition gap**：沒有任何單一 benchmark 同時測「生成 + 物理 + downstream policy success」。Handbook §3 (downstream-task-eval) 將補這塊。

---

## 4. ⚡ shines / ❌ breaks

### ⚡ Where VBench shines
- **跨模型可比**：Sora / [Veo](../video-world-models/veo.md) / Kling / [Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md) / Wan2.2 / CogVideo 都在一張表上，這在 2024-04 之前根本辦不到
- **disentangled**：拿到一個模型 "subject inconsistency 暴跌" 比 "FVD 92" 可操作得多
- **public + reproducible**：`pip install vbench` 一行，prompts / generated videos / human anno 全開源
- **PhysBench**：對 VLM-as-judge 場景特別關鍵 —— 如果你打算用 GPT-4o 評生成影片，先在 PhysBench 上跑一輪確認它至少懂這個物理 sub-domain

### ❌ Where it breaks

**(a) Goodhart 已經來了**。Issue [#71](https://github.com/Vchitect/VBench/issues/71) 明白指出 subject/background/motion-smoothness 三維在 leaderboard 上已經高度飽和（top 模型差距 < 0.5%），失去 discriminability。這通常是 "模型開始為 metric 而生" 的早期訊號。

**(b) Prompt 是否標準化爭議**。Issue [#77](https://github.com/Vchitect/VBench/issues/77) 揭露：leaderboard 上有些模型用 GPT-rewrite prompt、有些用原 prompt，分數不可比。Issue [#104](https://github.com/Vchitect/VBench/issues/104) 同樣指出解析度/時長未強制統一 — 直接影響 motion-smoothness 等時間相關指標。

**(c) Reproducibility 痛點**。Issue [#42](https://github.com/Vchitect/VBench/issues/42) 一直未解：使用者跑出來的 Total/Quality/Semantic Score 跟官方 leaderboard 對不上。Issue [#202](https://github.com/Vchitect/VBench/issues/202) (Wan2.2) 也報同款問題。

**(d) Domain blind spot**。三個 benchmark 全部 `domain=generalist`：沒有 robotics manipulation contact-rich、沒有 fluid CFD validation、沒有 driving long-horizon。對 handbook 的 5 個 domain（robotics / driving / fluid / rigid / soft）幾乎都不直接支援。

**(e) VBench-2.0 的 0.3–0.8 正規化**已被 ML 社群質疑會壓縮真正的 outlier（最強和最弱模型差距人為縮小），但目前無 follow-up 量化 critique `[TBD: 找 reddit/twitter critique 引用]`。

**(f) PhysBench ≠ generation eval**。常見誤用：把 PhysBench 分數當「我的 VLM 可以評生成影片物理對錯」的證明 —— 但 PhysBench 的 task 多為**判別**（multi-choice / VQA），不是 open-ended 評分。

---

## 5. Reproduction notes

```bash
# VBench v1 - 最簡 setup
pip install vbench                  # 或 git clone
# 注意：detectron2 只支援 CUDA 11.x / 12.1，新卡需降版

# 跑 16 維（單卡 A100 約 4-8h on 1000 videos）
vbench evaluate --videos_path ./gen_videos --dimension all

# VBench-2.0 - 需另外 clone Vchitect/VBench-2.0 repo (3 月 release)
# scorer 需 GPT-4o API key（generalist 評估呼叫 OpenAI）— 預算注意

# PhysBench
git clone https://github.com/USC-GVL/PhysBench
# val 200 條可離線跑；test 10k 需提交到 official server
```

**踩坑**：
- VBench 對 video resolution / fps / duration 敏感；生成側請確保跟 leaderboard top 模型同規格再比
- VBench-2.0 用 VLM 當 scorer → 同一份 video 跑兩次分數可能差 1-3%（GPT-4o sampling temperature 即使設 0 仍非 deterministic）
- PhysBench 評 VLM 時 prompt template 強烈影響分數 — paper 提供官方 template 必須照用
- Detectron2 / CUDA 不匹配是 90% 安裝失敗原因

---

## 6. Cross-line synthesis

對 handbook 4 條技術線的覆蓋：

| 路線 | VBench v1 | VBench-2.0 | PhysBench | 缺口 |
|---|---|---|---|---|
| **pixel-WM** (Sora/Veo/Cosmos) | ✅ 直評 | ✅ 直評含 physics | △ 評 evaluator | downstream-task |
| **latent-WM** ([DreamerV4](../latent-world-models/dreamer-v4.md)/[V-JEPA-2](../latent-world-models/v-jepa-2.md)) | ❌ 需先 decode | ❌ | △ | latent-space metric |
| **diff-sim** ([Genesis](../differentiable-simulators/genesis.md)/Brax-render) | ❌ 不適用 | ❌ | ❌ | physics-fidelity-vs-ref |
| **neural surrogate** ([FNO](../neural-surrogates/fno.md)/[GraphCast](../neural-surrogates/graphcast.md)) | ❌ | ❌ | ❌ | PDEBench / 守恆律 |

**實務 stack**：top 模型（Sora, Veo, Kling, Cosmos-Predict, Wan2.2）的 VBench / VBench-2.0 公開分目前是「mass-market T2V 模型誰行」的最強 signal；但對 robotics / driving 應用，**真正可靠的 ground truth 是 downstream policy success**（生成的影片去訓 VLA，看 task success rate）—— 這條 evaluation route 還沒有公開 benchmark，是 handbook §3 wishlist 中最大缺口。

複合用法：用 PhyGenBench 篩 candidate 模型 → VBench-2.0 看 commonsense / physics 是否同時不崩 → 內部 downstream-task eval 看是否真的有用。

---

## 7. References

**Canonical papers**:
- VBench: Huang et al., "VBench: Comprehensive Benchmark Suite for Video Generative Models", arxiv 2311.17982, CVPR 2024 Highlight
- VBench++ (extended): arxiv 2411.13503 (2024-11)
- VBench-2.0: Zheng et al., "VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness", arxiv 2503.21755, Mar 2025
- PhysBench: Chow et al., "PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding", arxiv 2501.16411, ICLR 2025
- PhyGenBench: Meng et al., "Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation", arxiv 2410.05363, ICML 2025
- PhyWorldBench (newer alt.): arxiv 2507.13428, 2025

**Code / leaderboards**:
- https://github.com/Vchitect/VBench
- https://vchitect.github.io/VBench-2.0-project/
- https://github.com/USC-GVL/PhysBench
- https://github.com/OpenGVLab/PhyGenBench
- https://huggingface.co/spaces/Vchitect/VBench_Leaderboard

**Secondary critique**:
- VBench Issue #71 (discriminability) / #77 (prompt) / #104 (resolution) / #42 (reproducibility) / #202 (Wan2.2)
- PhyGenBench paper §5：best Gen-3 模型只拿 0.51，揭示頂級 T2V 的物理理解尚遠未解

---

## 8. §8 Pitfall log

| # | Issue / Source | Severity | 摘錄 | Workaround |
|---|---|---|---|---|
| 8.1 | [Vchitect/VBench #71](https://github.com/Vchitect/VBench/issues/71) — discriminability | High | "subject consistency, background consistency, motion smoothness have relatively high scores on the leaderboard" → 飽和失能 | 看 dynamic-degree / temporal-flicker 等未飽和維度；別只看 Total Score |
| 8.2 | [Vchitect/VBench #77](https://github.com/Vchitect/VBench/issues/77) — prompt 不一致 | High | leaderboard 上有些用 GPT-rewrite prompt、有些用原 prompt | 自評時固定 prompt source；對比 leaderboard 時聲明 prompt 處理 |
| 8.3 | [Vchitect/VBench #104](https://github.com/Vchitect/VBench/issues/104) — resolution/duration 未強制 | Med | 不同模型 leaderboard 上未必同規格 | 評自己模型時務必匹配比較對象的 res/dur |
| 8.4 | [Vchitect/VBench #42 / #202](https://github.com/Vchitect/VBench/issues/42) — reproducibility | High | 本地跑分對不上官方 | 用 official Docker / 官方 prompts 子集 ；若仍差距>2%，視為 unsolvable noise |
| 8.5 | VBench-2.0 0.3–0.8 normalization | Med (推測) | per-dim 線性 rescale 會壓縮 outlier | 看原始 raw 分而非 normalized 分 |
| 8.6 | VBench-2.0 用 GPT-4o 當 generalist scorer | Med | 非 deterministic + cost ↑ | 多跑 3 seed 取 median；預算限 1k videos |
| 8.7 | PhysBench 誤用為「生成評分器」 | High | 它是 **VLM-understanding** benchmark, 非 generation eval | 若要用 VLM 當 judge，先檢查它在 PhysBench 對應 sub-class 達 60+% |
| 8.8 | 三個 benchmark 全 `domain=generalist` | High | robotics contact-rich / fluid 完全沒覆蓋 | domain-specific 評估需要自建（handbook §3 downstream-task） |
| 8.9 | Best PhyGenBench score = 0.51 (Gen-3) | Critical | 頂級閉源模型仍未解物理 commonsense | 別輕信 "Sora 已是 world model" 行銷敘事 |

---

> 寫作日期 2026-05-25。VBench-2.0 公布 ~14 個月，二手實測仍少；本文 §2 對 Physics 5 sub-dim 細節留 `[TBD]`，待 paper 表抓全或 leaderboard 開源後補。
