<!-- ontology-5axis
output=pixel-video
injection=data-only|sim-in-loop-infer
control=text|image-init|trajectory|action|camera
temporal=clip-parallel|hierarchical
domain=generalist
ref=../../cheat-sheet/ontology.md
-->

# NVIDIA Cosmos World Foundation Model 解構（Cosmos WFM Dissection）

> **發布時間**：2025-01 · arXiv [2501.03575](https://arxiv.org/abs/2501.03575)（Predict1）· 後續 [2503.15558](https://arxiv.org/abs/2503.15558)（Reason1, 2025-03）· [2601.16163](https://arxiv.org/abs/2601.16163)（Policy, 2026-01）
> **論文**：*Cosmos World Foundation Model Platform for Physical AI*
> **作者**：NVIDIA Cosmos team（含 Sanja Fidler 組 + Toronto AI Lab + NVIDIA Research Robotics / AV）
> **核心定位**：第一個 **open-weight、generalist 預訓練、顯式支援多模 conditioning（text / image / video / trajectory / action / depth+seg）、配套 Reason-VLM + Tokenizer + Transfer multi-controlnet** 的 video FM stack。在 v2 ontology 上 anchor `output=pixel-video × injection=data-only|sim-in-loop-infer × control=multi-modal × temporal=clip-parallel|hierarchical × domain=generalist`。

**Status:** v0.5 — 解構基於 Predict1 paper 摘要、後續官方 release notes、社群 reproduction 報告與 NVIDIA blog。Predict2 / Predict2.5 / Cosmos-Drive-Dreams 的 arxiv ID 與部分 benchmark 數字待維護者升 v1 時補。
**TL;DR:** Cosmos 把 video FM 的天價 pre-training cost（**10K H100 × 3 個月**、**~20M 小時 video → ~10^8 clips**）一次燒掉，下游 robotics / driving 團隊只要幾百到幾千 GPU-hour 做 post-train 就能拿到一個可掛 prompt + image + trajectory + action conditioning 的 world simulator。它的 USP 不在「比 Sora 更會做夢」，而在 **(a) open weight + (b) 顯式 control axes + (c) 配套 Reason1 reasoning VLM + Transfer multi-controlnet + Tokenize1**。最關鍵實證：**Cosmos-Policy 從 Predict2-2B 單階段 post-train 即在 LIBERO / RoboCasa 超越從零訓的 diffusion policy 與 VLA baseline**，且論文明說「無架構修改」是 unlock。

**X-Ray.** Cosmos 在 v2 ontology 上佔據 `pixel-video × data-only|sim-in-loop-infer × multi-control × generalist` 這個 **anchor 格子**——它不是「另一個 Sora」，是 NVIDIA 把 Sora 路線變成 **physical-AI 開發者可組裝的 pipeline**。它解掉了三個結構性 prior gap：(a) Sora / Veo 是 closed-weight 且不收 action / trajectory conditioning；(b) GAIA-1/2 是 Wayve 內部 driving-only；(c) V-JEPA-2 是 latent representation 不出 pixel —— 沒有一條能直接餵 VLA 訓練或做 photoreal 數據增廣。Cosmos 用「**tokenizer + 兩個 base model（Diffusion-7B/14B + AR-4B/12B）+ 一系列 multi-variant post-train**」這套 modular 設計把上述空缺一次填滿，並把 Reason1 + Predict 拆成 slow-plan / fast-rollout 的 hierarchical stack。**它打不開的 envelope 也很明確**：contact-rich physics（force / 接觸不可微）、>8s long-horizon drift（重力違反 / object morph）、3D consistency（環繞鏡頭物件 morph）—— 這三條是 pixel-video 路線的結構性 break，不會因 scale up 自動解。對 physics-gen handbook 讀者意義：**Cosmos 是 anchor，不是 ceiling**——理解它在五軸上的位置，才能判斷你的下游任務該直接 fine-tune Cosmos、還是接 diff-sim 補 axis 2、還是換 latent-WM 補 long-horizon。

## 📍 研究全景時間線

```ascii
   2023        2024              2025-01           2025-10                2026?
   Sora ──────► Cosmos beta ────► Predict1 ──────► Predict2.5 ─────────► closed-loop
   pre-cursor   internal NVIDIA    YOU ARE HERE     + Reason1 as text     AV / robot
   (closed)     pipeline           ★ 7B/14B Diff    encoder, T2W/I2W/V2W   data engine
                                   ★ 4B/12B AR      合成 single flow      (?)
                                   ★ Tokenize1      Drive-Dreams + 
                                   ★ open weight    Transfer2.5
                                                    Policy (2026-01)
   └─ closed FM ──────────────► open pipeline ─► hierarchical stack ─► sim-in-loop?
                                                  (Reason × Predict)
```

★ = 主要新點：open weight + multi-control + Reason+Predict 分層 stack。**仍未解：contact-rich physics、long-horizon drift > 8s、3D consistency**（pixel-video 路線結構性 break，留給 latent-WM / diff-sim 補）。

---

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 同軸對手

| 維度 | Sora 2 | Veo 3 | Wayve GAIA-2 | **Cosmos** |
|---|---|---|---|---|
| **License** | closed | closed | closed (internal) | **open-weight (NVIDIA Open Model License)** |
| **Conditioning** | text + image | text + image | ego-traj + agent config + road semantics | **text / image / video / trajectory / action / depth+seg / edge** |
| **Pre-train scale** | 內部未披露 | 內部未披露 | driving-only | **20M hr → 10^8 clips, 10K H100 × 3 mo** |
| **Post-train kit** | ❌ | ❌ | ❌ (internal) | **✅ Drive / Drive-Dreams / Policy / Transfer / Transfer2.5** |
| **Reasoning VLM 配套** | ❌ | ❌ | ❌ | **✅ Reason1-7B (Qwen2.5-VL based)** |
| **Tokenizer standalone** | ❌ | ❌ | ❌ | **✅ Tokenize1 (CV/DV, up to 2048× spatio-temporal)** |
| **Domain** | generalist | generalist | driving-only | **generalist → robotics / driving fine-tune** |

### 1.2 ⚡ Eureka Moment

> **「Tokenize1 + 兩個 base (Diff + AR) + 一系列 multi-variant post-train」是 modular 設計賭注** —— Cosmos 不押 single model size，押 **"foundation × specialization" decoupling**：pre-train 一次燒掉 20M hr / 10K H100 / 3 mo，post-train 用幾百~幾千 GPU-hour 出 Drive / Policy / Transfer 各個 vertical。賭注核心：pixel-video FM 的 **implicit physics + 下游 sim-in-loop reward** 比 hard PDE 路線更 scalable。

對比之下 Sora / Veo 是 "one model fits all" 黑盒，GAIA-2 是 "one model fits driving" 內部產線。Cosmos 是第一個把 video FM 當作 **可被分眾 post-train 的 backbone** 來釋出的 — 這就是「Cosmos 不是 model，是 platform」這句話的工程實質。

### 1.3 信息流（架構圖）

```ascii
┌─────────────────────────────────────────────────────────────────┐
│  RAW VIDEO (~20M hr) ─► Curation pipeline (shot detect, filter, │
│                          caption, dedup) ─► ~10^8 clips         │
└────────────────────────────────┬────────────────────────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │ Cosmos-Tokenize1             │
                  │  CV (continuous): 4×8×8 /    │
                  │     8×8×8 / 8×16×16          │
                  │  DV (discrete):  4×8×8 /     │
                  │     8×8×8 / 8×16×16          │
                  │  → 最高 2048× spatio-temporal│
                  └─────────┬────────────────────┘
                            ▼
       ┌────────────────────┴────────────────────┐
       ▼                                         ▼
┌────────────────────┐                  ┌────────────────────┐
│ DIFFUSION branch   │                  │ AUTOREGRESSIVE     │
│ Cosmos-1.0-        │                  │ Cosmos-1.0-AR-4B   │
│  Diffusion-7B /    │                  │  AR-12B  (base)    │
│  14B  (T2W, V2W)   │                  │  AR-5B / 13B V2W   │
│ Latent diffusion   │                  │ Llama3-style GPT,  │
│ + DiT, prompt-     │                  │ tokens from DV     │
│ upsampler 12B      │                  │ tokenizer          │
└─────────┬──────────┘                  └─────────┬──────────┘
          ▼                                       ▼
     ┌─────────────────────────────────────────────────┐
     │  POST-TRAINING  (downstream specialization)     │
     │  ├─ Cosmos-Drive (multi-cam, traj conditioning) │
     │  ├─ Cosmos-Drive-Dreams (long-tail AV scenes)   │
     │  ├─ Cosmos-Policy (visuomotor head)             │
     │  │     LIBERO / RoboCasa, single-stage SFT      │
     │  ├─ Cosmos-Transfer / Transfer2.5 (multi-CN:    │
     │  │     RGB + depth + seg + edge → video)        │
     │  └─ Cosmos-Reason1-7B (Qwen2.5-VL-based         │
     │      reasoning VLM, SFT + RL on physical CoT)   │
     └─────────────────────────────────────────────────┘
                            ▼
                ┌──────────────────────────┐
                │  HIERARCHICAL ROLLOUT    │
                │  Reason1 (slow, plan)    │
                │   → CoT + action desc    │
                │   → Predict (fast, frame)│
                └──────────────────────────┘
```

Predict2.5（2025-10）把 T2W / I2W / V2W 三條合成單一 flow-based 主幹，並用 **Cosmos-Reason1 當 text encoder**（取代純 T5/CLIP）—— 這是「自舉式 stack」的明顯設計：自家 VLM 餵自家 video FM。

---

## §2 · 數學層

### 📌 Napkin Formula

```
   Cosmos-Tokenize1 (DV path):
   
      video ∈ ℝ^(T×H×W×3)  ──tokenize──►  z ∈ ℤ^(T/8 × H/16 × W/16 × 1)  (DV8x16x16)
   
      compression =  (8 × 16 × 16 × 3 bytes) / (log₂(V) bits)
                  ≈  2048×  spatio-temporal           ← 最高設定
   
   Pre-train cost (Predict1, paper-stated):
   
      ~20M hours video → ~10^8 clips (~5s each)
      10K H100 × 3 months  ≈  21.6M H100-hours
      
   Post-train cost (Cosmos-Policy from Predict2-2B):
   
      8× A100/H100 × ~1 night  ≈  100-200 GPU-hours      ← 5 orders of mag less
```

**直覺**：壓縮比 2048× 是 trick 所在 —— 把 raw video 壓到 token-level 後，diffusion 與 AR 兩條 branch 才跑得起 long sequence。但這也是 **8.9 (Tokenize1 在 DV8x16x16 細紋丟失) 的根因**：壓越狠，下游細節恢復越難，texture detail 不可逆。Pre-train vs post-train 的 5 orders-of-magnitude cost gap 就是「foundation × specialization」decoupling 賭注的算術依據。

### 2.x Loss / 訓練細節

- **Diffusion branch**：latent diffusion + DiT，prompt-upsampler 12B 改寫 caption 增強 spatial relation；Predict1 是這條的主力。
- **AR branch**：Llama3-style GPT，token 來自 DV tokenizer；輸出需經 `Cosmos-1.0-Diffusion-7B-Decoder-DV8x16x16ToCV8x8x8` 後處理才能拿到「乾淨」pixel（社群常忘這步）。
- **Predict2.5 flow-based 主幹**：T2W / I2W / V2W 合成單模型 + Reason1 當 text encoder。
- **Cosmos-Policy SFT**：原模型 token 預測頭直接做 visuomotor，**不加 action head / diffusion head**（加了反而退化，paper §method ablation 明列）。

---

## §3 · 數據層 / 訓練 scale

| 階段 | 規模 | 來源 |
|---|---|---|
| Raw video | **~20M 小時** | curated internet video + 授權來源 |
| Curated clips | **~10^8 個** (~5s each) | shot detect → filter → caption → dedup pipeline |
| Pre-train compute | **10K H100 × 3 個月** ≈ 21.6M H100-hours | NVIDIA 內部 cluster |
| Post-train (Policy) | **hundreds of demos** (LIBERO scale) | 8× A100/H100 × 1 晚 |

**Scale 對比**：Sora / Veo 內部數據規模未披露；GAIA-2 是 driving-only 千小時級；V-JEPA-2 是 internet video 但 latent-only。Cosmos 是公開可知最大規模的 **video FM + open-weight** 組合。

---

## §4 · 代碼層

| 項 | 狀態 |
|---|---|
| Repo | [github.com/nvidia-cosmos](https://github.com/nvidia-cosmos)（predict1, predict2, predict2.5, transfer2.5, reason1 各 sub-repo） |
| Checkpoint | HF：`nvidia/Cosmos-Predict1-{7B,14B}-Text2World`、`Cosmos-1.0-AR-{4B,12B}`、`Cosmos-Reason1-7B`、`Cosmos-Tokenize1-CV8x8x8-720p` / `DV8x16x16-720p` |
| License | **NVIDIA Open Model License**（permissive，但 HF 下載需 accept → CI 自動化卡點，見 §6） |
| Inference GPU (7B T2W) | 1× H100 80GB；fp8/bf16 ~50GB；單 clip 5s 推理 ~2-4 min |
| Inference GPU (14B T2W) | H100/H200 80GB (fp8) 或 2× A100 sequence parallel；社群在 A100 80GB 常 OOM |
| Inference GPU (Reason1-7B) | 1× H100 / 雙 A100；vLLM 推理；NIM endpoint 可 API |
| Streaming | ❌（clip-parallel diffusion + AR；hierarchical via Reason1 reset prompt） |
| Metric scale | N/A（pixel output，沒 metric scale 概念，但物理 violations 是 §6 主訴） |

最小可跑 setup（2026-05 狀態，**Tokenize1 standalone 可用** 做 video embedding / VAE 替代品很好用）。

---

## §5 · 評測 / Benchmark

| Benchmark | Metric | 前 SOTA | Cosmos | Δ |
|---|---|---|---|---|
| **LIBERO** | Visuomotor success rate | from-scratch diffusion policy / VLA baseline | **Cosmos-Policy 超越（single-stage post-train from Predict2-2B）** | ⚡ 顯著 |
| **RoboCasa** | Visuomotor success | from-scratch baseline | **Cosmos-Policy 超越** | ⚡ 顯著 |
| Long-horizon stability | object permanence / gravity | n/a | **>8s 後 motion instability 顯著** | ❌ 自述 limitation |
| Spatial relation (Predict1) | "left of" / "behind" prompt fidelity | baseline T5/CLIP | Predict1 弱 → Predict2.5（Reason1 text encoder）大幅改善 | 🔧 fix-by-iteration |
| Cosmos-Drive-Dreams | Long-tail AV scene coverage | data augmentation baseline | Wayve 採用 Cosmos backbone | ⚡ production |

具體數字 `UNVERIFIED` — 待維護者升 v1 時從 arxiv 2501.03575 §eval + 2601.16163 §results 補完整 table。

**解讀**：LIBERO / RoboCasa 的 Δ 是真 capability —— **「video FM 直接當 policy backbone」首個 clean evidence**，不需要設計 action head。但 Sora-style 通用 generation benchmark（VBench 等）Cosmos 並未領跑，這是設計取捨：它把資源花在 multi-control + post-train kit 上，不在「最會做夢」上。

---

## §6 · Issues & Limitations

### 6.1 論文自述 limitations

- **Long-horizon drift > 8s**：object disappearing / deforming / 重力違反 / motion instability（arxiv 2501.03575 §Limitations 明列）
- **Contact-rich physics 不可微**：抓取 / 布料 / 流體 visually 合理但 force / contact phase 崩
- **Prompt fidelity (Predict1)**：spatial relation 弱，需 PromptUpsampler-12B
- **3D / multi-view inconsistency**：環繞鏡頭物件 morph，data-only 路線通病
- **GPU 門檻**：14B Text2World 需 H100/H200 80GB；4B AR 可 ≤40GB 但品質差距大

### 6.2 Hidden Assumptions（我們從架構推斷）

- **Pixel-video FM 的 implicit physics 夠用**：賭注是「資料規模 + capacity 自動湧現物理」，但 contact / 流體 / 長時穩定性實證打臉
- **Tokenize1 的 2048× 壓縮 detail loss 在下游 task 可接受**：細紋 / 紋理敏感 task（醫療 / 微觀流體）不成立
- **Reason1 + Predict 解耦訓練 OK**：兩者 alignment 靠 caption + CoT，沒有 end-to-end fine-tune；長 horizon 一致性受限
- **Single-stage Policy SFT 無需架構修改**：paper 強調是 unlock，但這意味著 **action 是 text-like token**，連續控制可能受限
- **Open weight 不等於 reproducibility**：pre-train 數據 curation pipeline 未公開，社群無法從零復現

### 6.3 GitHub / community-validated 失敗模式（§8 pitfall log 摘要）

完整 pitfall log 見下方 §8。重點：8.1（long-horizon drift）/ 8.2（contact-rich silent failure）/ 8.3（3D inconsistency）是**結構性 break**，不會因 scale up 自動解；8.5-8.8 是 ops 工程坑可繞過。

**Maintainer 響應度**：NVIDIA Cosmos 系列 repo 活躍維護（Predict1 → Predict2 → Predict2.5 連續 release，2025-01 → 2025-10 不到一年三代），社群 issue 多在 1-2 週內有 staff 回應。**這是與 Meta FAIR 「release 但不維護」路線的顯著差異** —— Cosmos 走的是 NVIDIA SDK 路線，更接近 production support。

---

## §7 · 比較 & 面試 Tip

| 同軸對手 | Axis 1 (output) | Axis 2 (injection) | Axis 3 (control) | Axis 4 (temporal) | Axis 5 (domain) | Open? |
|---|---|---|---|---|---|---|
| **Cosmos** | pixel-video (+ latent via DV) | data-only + sim-in-loop | text/image/video/traj/action/depth+seg | clip-parallel / AR / hierarchical | generalist → robotics/AV | ✅ |
| Sora 2 | pixel-video | data-only | text + image | clip-parallel | generalist | ❌ |
| Veo 3 | pixel-video | data-only | text + image | clip-parallel | generalist | ❌ |
| GAIA-2 | multi-cam pixel | data-only + structured cond | ego-traj + agent + road | clip-parallel multi-view | driving-only | ❌ |
| V-JEPA-2 | **latent** | data-only | action (zero-shot planning) | latent-rollout | generalist embodied | partial |
| DreamerV4 | latent | data-only | action | latent-rollout | RL embodied | ✅ |

> **🎤 Interview Tip.** 「我們要不要從 Cosmos-Predict 起步做 robotics-data-gen？」**正確答**：「**看你卡在哪條 axis**。如果是 axis 3 (control) 與 axis 5 (open + post-train kit) 卡你，Cosmos 是 anchor，直接起步 —— LIBERO/RoboCasa 上 Cosmos-Policy 是 first clean evidence『video FM 當 policy backbone』。如果卡你的是 contact-rich physics（grasp / 布料）或 long-horizon > 8s 一致性，Cosmos 解不了——這是 pixel-video 路線結構性 break，要接 diff-sim (Genesis / MJX) 補 axis 2，或退到 latent-WM (V-JEPA-2 / DreamerV4)。如果你做 AV，Cosmos-Drive-Dreams + Transfer2.5 是現成 production pattern（sim renderer 出粗糙→Transfer 貼皮）。」**錯答**：「Cosmos 是最強 video FM，所以從它起步」—— 沒看清你的 envelope。axis 1 = pixel-video 這條本身就限制了一堆下游用法。

### 7.1 Falsifiable predictions

1. **2026-12 前**：第一篇「Cosmos-Predict × diff-sim contact label co-training」會出現 —— 把 Genesis / MJX 的 force 信號當 auxiliary loss 餵 Cosmos post-train，補 §8.2 contact-rich silent failure。理由：production 已經在做 Genesis → Transfer2.5 兩段式 pipeline，下一步就是 end-to-end co-train。
2. **2027-06 前**：Cosmos-Predict3 會加 3D-aware conditioning（3DGS / NeRF prior 或 multi-view consistency loss），收 §8.3 環繞鏡頭 morph 問題。理由：GAIA-2 已證明 multi-view consistency 是 closed-loop 必須，NVIDIA 內部 Isaac/Omniverse 有完整 3D 資產可接。
3. **2027-12 前不會發生**：Cosmos 系列成為 closed-loop drone / 高速車控制器的 in-loop simulator —— 即使 long-horizon drift 用 hierarchical 解了一半，contact-rich + force-fidelity 仍是 pixel-video 路線結構性 break，10ms 級 closed-loop 仍要 diff-sim。

---

## §8 · For the Reader（按 persona 分流）

- **VLA / robot policy 工程師** —— Cosmos-Policy 是 first clean evidence「video FM 直接當 policy backbone」。從 Predict2-2B 單階段 SFT，8× A100/H100 一晚跑通；**不要加 action head / diffusion head**（paper §method ablation 明列退化）。grasp / 接觸 task 不要單獨用 Cosmos rollout 當訓練資料 —— §8.2 silent failure 已被多筆社群 reproduction 報告。
- **自駕 closed-loop 工程師** —— 用 **Cosmos-Drive-Dreams + Transfer2.5** 做 long-tail 資料增廣（遮擋行人、異常車輛），是 Wayve / NVIDIA Isaac 採用的 production pattern。但**不要拿來當 in-loop simulator 跑 PID/MPC** —— long-horizon drift + 3D inconsistency 會讓控制環學歪。Cosmos-Drive 是 **out-of-loop data engine**，不是 in-loop sim。
- **影片生成工程師** —— 如果只追 prompt fidelity / 視覺品質，Sora 2 / Veo 3 仍領先。Cosmos 的價值在 **open weight + multi-controlnet (Transfer2.5)** —— RGB+depth+seg+edge 同時 condition，是 production data engine 用法。Predict2.5 用 Reason1 當 text encoder 後 spatial relation 大幅改善，值得從 Predict1 升上去。
- **神經 PDE / surrogate 研究者** —— Cosmos 與 GraphCast / FNO 不直接對接（一邊是 pixel，一邊是 field）。組合方式只在 scientific viz：surrogate 算流場 → renderer → Cosmos refine 視覺。**不要把 Cosmos 當作流體 / 接觸 surrogate** —— implicit physics 在這層完全失效。
- **物理 conditioning 研究者** —— Cosmos 是 `injection=data-only|sim-in-loop-infer` 的 anchor。要加 hard constraint 必須走 sim-in-loop（Genesis / MJX 做 ground-truth → Cosmos-Transfer 貼皮）。**這是 axis 2 從 data-only 跳到 sim-in-loop 的 cleanest production pattern**，比 hard PDE 路線更 scalable，但 force-fidelity 仍受 pixel 層限制。
- **Research 學生** —— Cosmos 是當前 anchor，但**不是 ceiling**。注意 §7.1 三條預測。可研究方向：(a) Cosmos × diff-sim contact co-training；(b) DV tokenizer 當 latent space 跑 latent dynamics（AR-12B 可視為 latent dynamics model，但官方未鋪這條路，社群實驗少）；(c) Reason1 + Predict 的 end-to-end alignment（目前是解耦訓練）。Tokenize1 standalone 可用做 video embedding / VAE 替代品，是免費的 starting point。

---

## §9 · §8 Pitfall log（GitHub / community-validated 失敗模式）

| # | Severity | Issue | Source | Workaround |
|---|---|---|---|---|
| 9.1 | 🔴 High | Long-horizon drift > 8s：object morph / 重力違反 / motion instability | arxiv 2501.03575 §Limitations + HF blog community report | 切短 clip（5-8s）+ hierarchical rollout（Reason1 重置 prompt） |
| 9.2 | 🔴 High | Contact-rich silent failure：grasp / 接觸視覺合理但 force phase 崩 | community VLA reproduction 報告（多筆社群） | 不要單獨用 Cosmos rollout 當 VLA train data；與 diff-sim contact label 對齊 |
| 9.3 | 🟠 Medium | 3D / multi-view inconsistency：物件在環繞鏡頭中 morph | 對比 GAIA-2 paper §1 motivation | driving 場景用 GAIA-2 / 等 Predict3 多 view 版；其他場景接 Transfer + 3DGS prior |
| 9.4 | 🟠 Medium | Predict1 prompt fidelity 弱（spatial relation） | arxiv 2501.03575 § eval；Predict2.5 換 Reason1 text encoder 即是修正 | 升級 Predict2.5；或先過 PromptUpsampler-12B |
| 9.5 | 🟠 Medium | 14B inference OOM in A100 80GB；社群 reproduction 痛點 | HF discussions、cosmos-predict1 README 性能表 | fp8 量化；或退 7B；或 sequence parallel 多卡切 |
| 9.6 | 🟡 Low | AR 變體 decode 需額外 `Diffusion-7B-Decoder-DV8x16x16ToCV8x8x8`，常被遺漏 | docs/predict1/autoregressive/reference | 比對 README 推理 pipeline，勿跳 decoder 步驟 |
| 9.7 | 🟡 Low | HF 下載需接受 NVIDIA Open Model License — CI 自動化卡點 | HF model card | 用 HF token + `huggingface-cli login` 並預先 accept；或本地鏡像 |
| 9.8 | 🟡 Low | Predict2.5 強制同時 load Reason1 當 text encoder → VRAM 峰值升高 | docs/cosmos/latest/predict2 release notes | 接受更高 VRAM；或拆服務（Reason1 在另一卡 / NIM 端） |
| 9.9 | 🟠 Medium | Tokenize1 在低 bitrate（DV8x16x16，total ~2048×）細紋丟失，texture detail 不可逆 | arxiv 2501.03575 tokenizer eval | 細節敏感任務改 CV4x8x8 或 CV8x8x8（compression 較低） |
| 9.10 | 🟡 Low | Policy post-train 加 action head / diffusion head 反而退化 | arxiv 2601.16163 §method ablation | 遵守「無架構修改」原則，只 SFT 原模型 token 預測頭 |

> Pitfall 觀察重點：9.1 / 9.2 / 9.3 是 **pixel-video FM 路線結構性 break** — 不會因 scale up 自動解決；要靠 axis 2 補 `sim-in-loop` 或 axis 1 換 `3d-scene / latent` 才能根治。9.5–9.8 是 ops 層級可繞過的工程坑。

---

## §10 · Cross-line synthesis

- **× diff-sim ([Genesis](../differentiable-simulators/genesis.md) / [MJX](../differentiable-simulators/mujoco-mjx.md))**：Cosmos `data-only` × diff-sim `hard-constraint / sim-in-loop` —— 組合方式：Genesis 出 ground-truth physics rollout（粗糙渲染）→ Cosmos-Transfer 2.5 用 depth+seg 當 control 貼 photoreal 皮。**production 標準 pattern**，Wayve / NVIDIA Isaac 均採。
- **× neural surrogate ([GraphCast](../neural-surrogates/graphcast.md) / [FNO](../neural-surrogates/fno.md))**：不直接對接（field vs pixel）；只在 scientific viz 場景組合。
- **× 3D-aware (World Labs / 3DGS)**：Cosmos 缺顯式 3D，組合方式：3DGS 出 multi-view → Cosmos-Transfer 做時序補洞 + photoreal 化。長期看 Predict 系列加 3D-aware conditioning 是 roadmap 明顯空缺（見 §7.1 prediction 2）。
- **× VLA**：Cosmos-Policy 已是 direct fork —— video FM backbone + action token head。VLA pre-training 可改為「video FM weights → freeze partial → action SFT」取代 from-scratch transformer。對 [`bridge-to-vla/`](../../bridge-to-vla/) 是 anchor case。
- **× latent-WM ([DreamerV4](../latent-world-models/dreamer-v4.md), [V-JEPA-2](../latent-world-models/v-jepa-2.md))**：Cosmos 的 DV tokenizer 本身就提供 latent space；理論上 AR-12B 可視為 latent dynamics model。但官方未鋪這條路，社群實驗少 —— 留給 research 學生（§8）。

---

## References

- **Cosmos Predict1** — NVIDIA Cosmos team. *Cosmos World Foundation Model Platform for Physical AI*. 2025-01 · [arXiv:2501.03575](https://arxiv.org/abs/2501.03575)
- **Cosmos Reason1** — *From Physical Common Sense To Embodied Reasoning*. 2025-03 · [arXiv:2503.15558](https://arxiv.org/abs/2503.15558)
- **Cosmos Policy** — *Fine-Tuning Video Models for Visuomotor Control and Planning*. 2026-01 · [arXiv:2601.16163](https://arxiv.org/abs/2601.16163)
- NVIDIA blog 公告：<https://blogs.nvidia.com/blog/cosmos-world-foundation-models/>
- NVIDIA docs (Predict2.5 / Transfer2.5 cookbook, 2025-10)：<https://docs.nvidia.com/cosmos/latest/>
- GitHub orgs：<https://github.com/nvidia-cosmos>（predict1, predict2, predict2.5, transfer2.5, reason1）
- HuggingFace blog *Topic 24: Cosmos WFM Platform* (Kseniase) — failure mode summary
- LearnOpenCV — *Cosmos-Reason VLM for Video VQA*（reproduction notes）
- Wayve × NVIDIA collab announcement（Cosmos backbone for GAIA-related work）

---

## Boundary

- 同軸 closed-weight 對手 → [`../video-world-models/sora.md`](../video-world-models/sora.md) · [`../video-world-models/veo.md`](../video-world-models/veo.md)
- 同軸 driving-only 對手 → [`../video-world-models/gaia-2.md`](../video-world-models/gaia-2.md)
- 同軸 latent 對手 → [`../latent-world-models/v-jepa-2.md`](../latent-world-models/v-jepa-2.md) · [`../latent-world-models/dreamer-v4.md`](../latent-world-models/dreamer-v4.md)
- 跨 zone 組合（diff-sim 補 axis 2） → [`../differentiable-simulators/genesis.md`](../differentiable-simulators/genesis.md) · [`../differentiable-simulators/mujoco-mjx.md`](../differentiable-simulators/mujoco-mjx.md)
- VLA 接口 → [`../../bridge-to-vla/`](../../bridge-to-vla/)
- 五軸全景 → [`../../cheat-sheet/ontology.md`](../../cheat-sheet/ontology.md)

---

## ✍️ 維護者註（v0.5 → v1 升級清單）

本 v0.5 基於 Predict1 paper 摘要 + 後續 release notes + NVIDIA blog + 社群 reproduction。下次升 v1 時補：

1. ⏳ **Cosmos-Drive-Dreams arxiv ID** —— NVIDIA blog 2025-Q1 出現但無 standalone preprint 確認
2. ⏳ **Predict2 / Predict2.5 完整 arxiv ID + author list**（目前僅有 docs / release notes）
3. ⏳ **完整 benchmark 數字**：LIBERO / RoboCasa Cosmos-Policy 具體 success rate vs baseline，VBench-style 通用 metric 數字
4. ⏳ **Tokenize1 完整 compression ratio table**（CV vs DV 各 setting 的 PSNR / LPIPS）
5. ⏳ **Curation pipeline 細節**：shot detect / filter / caption / dedup 各步具體模型 + 過濾率
6. ⏳ **Pre-train 數據組成**：20M hr 的 domain 分布（driving / robotics / generic）—— 影響下游 fine-tune 效率
7. ⏳ **Cosmos-Policy 在 RoboCasa 以外的真機部署數據**（目前都是 sim benchmark）
8. ⏳ Status v0.5 → v1，刪本節

---

[← Back to Foundation Physics Models](./overview.md)

Sources:
- [Cosmos Predict1 arXiv 2501.03575](https://arxiv.org/abs/2501.03575)
- [Cosmos Reason1 arXiv 2503.15558](https://arxiv.org/abs/2503.15558)
- [Cosmos Policy arXiv 2601.16163](https://arxiv.org/abs/2601.16163)
- [NVIDIA Cosmos GitHub org](https://github.com/nvidia-cosmos)
- [NVIDIA Cosmos docs](https://docs.nvidia.com/cosmos/latest/)
- [NVIDIA blog announcement](https://blogs.nvidia.com/blog/cosmos-world-foundation-models/)
