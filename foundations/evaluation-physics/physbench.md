<!-- ontology-5axis output=N/A injection=N/A control=text temporal=clip-parallel domain=N/A -->

# PhysBench — Benchmarking VLMs for Physical-World Understanding

> 姊妹 anchor：[VBench / VBench-2.0 / PhysBench eval-suite landscape](./vbench-physics.md)。本篇單獨拆 **PhysBench**（USC-GVL / ICLR 2025），與 [VBench-Physics](./vbench-physics.md) 互補：VBench 評 **生成器**，PhysBench 評 **理解者**（即未來要當 judge 的 VLM）。

---

## 1. One-paragraph TL;DR

當 2024–25 的「物理可控生成」社群開始用 VLM-as-judge（GPT-4o / Qwen-VL）當 evaluator 時，一個被長期忽略的問題浮上來：**這些 VLM 自己懂物理嗎？** PhysBench (Chow et al., arxiv 2501.16411, ICLR 2025) 不評生成模型、不打分影片質量；它出一份「物理世界閱讀考卷」給 VLM 自己考 —— 10,002 條 interleaved video-image-text 題目、4 大 domain、19 sub-class、多選 QA 形式。它跟 [VBench-Physics](./vbench-physics.md) 是 dual：VBench-Physics 看 generator 輸出像不像物理對，PhysBench 看 evaluator 心裡懂不懂物理。為什麼這個方向重要？因為 2025 年起，幾乎所有 large-scale T2V evaluation（VBench-2.0 / PhyGenBench）都把 VLM 拉進 scorer pipeline 裡 —— 如果 VLM 自己在 PhysBench 上不及格（best zero-shot 51.94%，人類 95.87%），那它對生成影片的物理判斷就是有偏的 prior，不是 ground truth。論文額外 release **PhysAgent**（VLM + 物理先驗 + 專家視覺模型 ensemble），證明用 agent + 工具至少可以在 GPT-4o 上爭回 18.4 個百分點。

---

## 2. Core mechanism

### 結構

```
PhysBench (10,002 entries, ICLR 2025)
│
├─ 4 Domains
│   ├─ D1 Physical Object Properties     (mass, friction, elasticity, ...)
│   ├─ D2 Physical Object Relationships  (motion, speed, spatial, ...)
│   ├─ D3 Physical Scene Understanding   (light, viewpoint, temperature)
│   └─ D4 Physics-based Dynamics         (collision, throwing, fluid, explosion)
│
├─ 19 sub-classes (~5 per domain)
├─ "8 (paper) / 10 (README) capability dimensions"  ← Pitfall 8.1
│
├─ Splits
│   ├─ val: 200 (open-ended, 公開答案)
│   └─ test: 10,002 全集；submit to EvalAI for grading
│       ├─ general VLM split: 10,002 (含 interleaved video+image)
│       └─ image / video-only sub-split: 8,099 (拆掉 interleaved)
│
└─ Format: multiple-choice (variable option count)
   Score = top-1 accuracy
```

### 評估流程

1. 每題給 VLM **{video frames (optional) + image(s) (optional) + text question + N choices}**
2. VLM 回 1 個 option index
3. 跨 19 sub-class 算 accuracy → 加權回 4 domain → 加權回 overall

**為何用 multi-choice 而非 open-ended**：作者明示 open-ended 對 VLM 不公平（評分需另一個 LLM judge，回到雞生蛋）。MCQ 直接拿 GT label 對。代價：陷入 Goodhart（見 §4d）。

### PhysAgent —— 不是 model，是 agent

```
┌──────────── PhysAgent (paper §5) ──────────────┐
│ Step 1: VLM (GPT-4o) reads scene               │
│ Step 2: dispatch to specialist tools           │
│         ├─ optical-flow estimator              │
│         ├─ depth predictor                     │
│         └─ symbolic physics prior (mass rule…) │
│ Step 3: VLM aggregates tool outputs → answer   │
└────────────────────────────────────────────────┘
  +18.4% over GPT-4o naive  /  also tested on MOKA embodied agent
```

PhysAgent 證明 "VLM 不懂物理" 不是 model capacity 問題，是 **接觸不到物理先驗**；給它工具就追平大半 gap。

---

## 3. 五軸定位 + 同軸對手

**Header 解釋**：與 [VBench-Physics 姊妹篇](./vbench-physics.md) 同 — eval suite `output=N/A injection=N/A`；PhysBench 接受 text question + 多模態 stimulus，故 `control=text`；目標素材是 video clip 答題，標 `temporal=clip-parallel`；`domain=N/A` 因 benchmark 跨多 domain 但不對應 generation-side domain coupling。

### 同軸對手 — 都是 "physics-understanding / physics-fidelity benchmark"

| Benchmark | 評估對象 | 物理粒度 | 任務形式 | 開源 leaderboard | Anchor |
|---|---|---|---|---|---|
| **PhysBench** | VLM（理解者） | 4 domain × 19 sub | MCQ on video+image | ✅ EvalAI | 本篇 |
| [**VBench-Physics**](./vbench-physics.md) | T2V generator | 5 sub-dim (VBench-2.0) | 16+18 disentangled scorer | ✅ HF Space | 姊妹 |
| **PhyGenBench** | T2V generator | 27 physical laws (mechanics/optics/thermal/material) | LLM-as-judge + specialist | ✅ OpenGVLab | 待 anchor |
| **PhyWorld / PhyWorldBench** (arxiv 2411.02385, ICML 2025) | T2V generator (controlled 2D sim) | within / OOD / combinatorial generalization | physical-law adherence on synthetic sim | ✅ | 待 anchor |

**Triangulation 規則**：要 claim 一個 T2V 模型「物理對」，至少要 (a) VBench-Physics 過關、(b) PhyGenBench score 不墊底、(c) 若用 VLM-as-judge 評，VLM 本身 PhysBench 不能太低。三角任一條缺，物理 claim 都打折。

**為什麼跟 PhyWorld 是同軸卻不同陣營**：PhyWorld 用受控 2D sim 測 generator 的 OOD generalization（會不會學 case-based shortcut：色 > 大小 > 速度 > 形狀），是 _generation-side_ 的 mechanism probe；PhysBench 是 _understanding-side_ 的多模態題庫。兩者剛好夾住 "VLM 看影片懂物理嗎" 跟 "video model 拍影片守物理嗎" 兩條路 — 一頭一尾。

---

## 4. ⚡ Where it shines / ❌ where it breaks

### ⚡ shines

- **唯一在 VLM-as-judge 鏈條最前段做 audit 的 benchmark**：2025 後評估 pipeline 充滿 "用 GPT-4o 判物理對錯" 的設計，PhysBench 直接量化這個 judge 自己的物理底子
- **disentangled 4×19 結構**：拿到一個 VLM 不只看 overall 51%，可以看「它在 fluid dynamics sub 是 30%、在 rigid collision 是 65%」，挑 judge 變可行
- **PhysAgent 提供 "如何把 VLM 變得稍微懂物理" 的可重現 recipe**（agent + 工具）—— 對想做 generation eval pipeline 的人，這是直接可用的 evaluator 升級套件
- **公開 75 VLM 實測 + 36 後補**：大型 baseline panel 在 2025 年算 dense

### ❌ breaks

**(a) Capability dimension 數字自相矛盾**。Paper 寫 **8 dimensions**、GitHub README 寫 **10 dimensions**、project page 也說 10。我們對 paper PDF 與 README 雙向驗證 — 推測 8 是 paper-camera-ready 早期表 1 的 grouping，10 是 dataset release 時細拆。實務影響：寫 paper 引用要寫「8 (paper) / 10 (release)」並聲明 ICLR camera-ready 版本以哪個為準。

**(b) MCQ-Goodhart 已可預期**。MCQ 給 fixed N options，VLM 即使對物理沒概念，靠語言先驗 + option elimination 也能踩到 25–40%。一個過 60% 的 VLM 不代表它「懂物理」，可能只是 "懂 PhysBench 的 distractor 風格"。對比 human 95.87% — gap 大不只因為模型笨，也因為 ceiling 接近天花板，模型一旦學會 distractor pattern 就會跳。

**(c) 跟 generation 的 disconnect**。PhysBench 沒測 "VLM 對一段 _被故意製造物理錯誤_ 的合成影片能否找出錯處"。它測的是真實/合理影片上的 VQA。誤用案例：拿 PhysBench 高分 VLM 直接當 VBench-2.0 的 physics scorer，假設它能抓 Sora 的物理 hallucination — 完全沒被驗證過。

**(d) Domain blind spot**。4 個 domain 全是 generalist 視覺場景（家居 / 戶外 / 簡單實驗），完全不涉及 robotics manipulation contact-rich、CFD fluid validation、driving long-horizon。對 handbook 5 個 domain（robotics / driving / fluid / rigid / soft）幾乎零覆蓋 — 跟 [VBench-Physics 姊妹篇](./vbench-physics.md) §4d 是同一個結構性缺口。

**(e) Test set 只能上 EvalAI 評**。val 只有 200，自評容易過擬合；要拿真分必須 submit test。Issue [#12](https://github.com/USC-GVL/PhysBench/issues/12) 報告 EvalAI 提交流程 2026-01 後一度失效，使 active leaderboard 凍結。對在做 model selection 的人是阻塞。

**(f) PhysAgent 不開源完全**。Issue [#10](https://github.com/USC-GVL/PhysBench/issues/10) reports PhysAgent 跑不起來；issue [#7](https://github.com/USC-GVL/PhysBench/issues/7) reports 訓練 data / agent 細節不清。換言之「+18.4%」結果可信度比 paper benchmark score 低一檔。

---

## 5. Reproduction notes

```bash
git clone https://github.com/USC-GVL/PhysBench
cd PhysBench
# Apache-2.0; Python 99.6%

# val (200) — 公開答案，可本地評
python eval/run_val.py --model_name <hf_model> --output ./out_val.json

# test (10,002) — 必須上傳 EvalAI
# 1. 跑 inference 產 predictions.json
# 2. submit 到 EvalAI 對應 challenge
# 注意：2026-01 起有提交鏈路問題（Issue #12, open）
```

**踩坑**（從 issue tracker 萃取）：

- **#5 / #6 evaluation error**：early-2025 多人報 InternVL2 / 通用 eval script 跑不通；都是 path/model checkpoint 配置不齊 → 看 #4 (`eval/models/checkpoints`) 那條 thread 的更新
- **video frame extraction**：image VLM 跑 video split 必須先 frame-sample；論文未強制 frame rate，不同抽法分數差 1–3%
- **prompt template 強依賴**：MCQ 的 question template / option order 一變，分數可以晃 2–5%（與 [VBench-Physics §5](./vbench-physics.md#5-reproduction-notes) 的同類問題對齊）
- **GPT-4o 一致性**：當 baseline 跑 GPT-4o 時，OpenAI API 端輸出隨時間 drift —— paper 公佈的 49.49% 是 2024 末抓的，2026 重跑可能差 1–2pp
- **PhysAgent 跑不起來**：issue #10 open；建議先跳過 PhysAgent，只用 PhysBench score 作 evaluator audit

---

## 6. Cross-line synthesis

PhysBench 跟 handbook 4 條技術線的耦合，主要在「誰拿來當 evaluator」這個次層：

| 路線 | 是否拿 VLM 當 evaluator | PhysBench 介入位置 |
|---|---|---|
| **pixel-WM** (Sora / [Veo](../video-world-models/veo.md) / [Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md)) | VBench-2.0 用 GPT-4o 當 generalist scorer | **前置 audit**：先在 PhysBench 看 GPT-4o 在你 care 的 sub-class 是否過 60% |
| **latent-WM** ([DreamerV4](../latent-world-models/dreamer-v4.md) / [V-JEPA-2](../latent-world-models/v-jepa-2.md)) | decode 後才能用 VLM 評 | 直接用 PhysBench 評不適用；要走 downstream policy success |
| **diff-sim** ([Genesis](../differentiable-simulators/genesis.md)) | sim 對 sim 不需 VLM | 不適用 |
| **neural surrogate** ([FNO](../neural-surrogates/fno.md) / [GraphCast](../neural-surrogates/graphcast.md)) | PDE residual 是 ground truth | 不適用 |

**與 [crossing/conservation-violation-atlas](../../crossing/conservation-violation-atlas/)（待寫）的接口**：PhysBench 4 domain 的 D4 "Physics-based Dynamics" 子題涵蓋 collision / fluid / explosion，正好對應 conservation-violation atlas 裡的「動量守恆」「質量守恆」「能量守恆」三條主線。換言之 PhysBench D4 可當 atlas 的 **proxy reading test**：要先確定 VLM 看到守恆律違反能識別，才有資格進 atlas 的自動化打標 pipeline。

**Composition 路徑**：用 PhysBench → 選一個過 60% 的 VLM → 用它做 VBench-Physics 的補強 scorer → 再用 downstream VLA success 做 ground truth 校正。三段都過才能 claim 一個物理生成模型「在這個 domain 可信」。

---

## 7. References

**Canonical paper**:
- Chow, Mao, Li, Seita, Guizilini, Wang, *"PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding"*, **arxiv 2501.16411**, **ICLR 2025**

**Repo / leaderboard**:
- https://github.com/USC-GVL/PhysBench  （Apache-2.0; ~90 stars, 5 forks as of 2026-05）
- https://physbench.github.io/  （project page + leaderboard）
- HuggingFace dataset + EvalAI submission portal

**Sibling benchmarks** (本 handbook 已 / 待 anchor):
- VBench / VBench-2.0 → [vbench-physics.md](./vbench-physics.md)
- PhyGenBench (arxiv 2410.05363, ICML 2025)
- PhyWorld (arxiv 2411.02385, ICML 2025) — generation-side OOD probe

**Secondary signals**:
- USC-GVL/PhysBench issues #5, #6, #10, #11, #12（見 §8）
- Leaderboard 2026-05 snapshot：InternVL2.5-38B 51.94% / InternVL2.5-78B 51.16% / GPT-4o 49.49% / Human 95.87%

---

## 8. §8 Pitfall log

| # | Issue / Source | Severity | 摘錄 | Workaround |
|---|---|---|---|---|
| 8.1 | Paper says **8** capability dimensions, README/project page say **10** | High | 兩處 dataset card 直接矛盾，影響引用 | 寫文時雙標「8 (paper) / 10 (release)」並引 ICLR camera-ready；自評時跑全 19 sub-class，dim grouping 自己重算 |
| 8.2 | [USC-GVL/PhysBench #12](https://github.com/USC-GVL/PhysBench/issues/12) — "Eval cannot process submission anymore" (open, 2026-01) | High | test set 必過 EvalAI；提交鏈路 2026-01 起壞 | 暫時改用 val 200 算 proxy；或等官方修；或自行算 unofficial score |
| 8.3 | [USC-GVL/PhysBench #10](https://github.com/USC-GVL/PhysBench/issues/10) — "PhysAgent error" (open, 2025-06) | High | PhysAgent 跑不起來；+18.4% 結果無法獨立復現 | 引 PhysAgent 結果時聲明 "as reported, not re-verified"；做 evaluator 升級先試 simpler tool-augmented baseline |
| 8.4 | [USC-GVL/PhysBench #11](https://github.com/USC-GVL/PhysBench/issues/11) — "Image & Video Mode Performance" (open, 2025-07) | Med | image-only split vs video-only split 分數差異成因未明 | 報分時拆 image / video sub-split；別只報 overall |
| 8.5 | [USC-GVL/PhysBench #5, #6](https://github.com/USC-GVL/PhysBench/issues/5) — "Code error for evaluation" / "InternVL2 evaluation error" (closed) | Med | 早期 eval script 對 third-party model 不穩 | 用 #4 thread 修補後的 checkpoint config；不要直接 fork 老 commit |
| 8.6 | MCQ Goodhart risk | High | 多選題形式 + 固定 distractor pattern → VLM 可靠 language prior + option elimination 過 baseline | 看 sub-class 分佈而非 overall；對 evaluator 要求至少在你 care 的 sub-class 過 60% |
| 8.7 | 誤用為 generation evaluator | Critical | "我 VLM PhysBench 60% → 它能評生成影片物理對錯" 是 **未被驗證的推論** | PhysBench 評的是 _understanding on real-ish video_；對 _hallucinated generated video_ 的辨識能力需另測 |
| 8.8 | `domain=N/A` 但實際全 generalist 視覺 | High | robotics contact-rich / CFD / driving 0 覆蓋 | 對 domain-specific generation 評估，PhysBench 結果僅作 weak prior |
| 8.9 | Leaderboard top 51.94% vs Human 95.87% | Critical | 2025 末 SOTA VLM 還離 human 44 個百分點 | 對 "VLM 已可當物理 judge" 的敘事保持懷疑；近期 evaluator pipeline 仍需 specialist 補強（PhysAgent / 物理工具） |

---

> 寫作日期 2026-05-26。三個資訊點仍待二次驗證：(1) paper §3.2 表 1 是否確為 8 dim（vs README 10）— ICLR camera-ready PDF 直讀；(2) issue #12 (EvalAI 提交斷線) 是否已在 2026-05 後修復；(3) PhysAgent 開源完整度（issue #10 thread 進度）。三點任一變動會推一篇 patch。
