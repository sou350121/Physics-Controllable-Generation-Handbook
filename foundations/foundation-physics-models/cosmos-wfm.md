<!-- ontology-5axis output=pixel-video injection=data-only|sim-in-loop-infer control=text|image-init|trajectory|action|camera temporal=clip-parallel|hierarchical domain=generalist -->

# NVIDIA Cosmos World Foundation Model

> Anchor dissection · 涵蓋 Cosmos-Predict1 (Jan 2025) → Predict2 → Predict2.5 (Oct 2025), Cosmos-Transfer / Cosmos-Reason1 / Cosmos-Drive-Dreams / Cosmos-Policy 子線。

## 1. One-paragraph TL;DR

Cosmos 不是「另一個 [Sora](../video-world-models/sora.md)」— 是 NVIDIA 為 **physical AI 開發者** 鋪的 open-weight pipeline：把 video FM 的 pre-training cost (10K H100 × 3 個月、~20M hours of video → ~10^8 clips) 一次燒掉，下游 robotics / driving 團隊用幾百~幾千 GPU-hour 做 post-train 就能拿到一個帶有 prompt + image + 軌跡 conditioning 的 world simulator。Prior gap 很具體：Sora / [Veo](../video-world-models/veo.md) 是 closed weight 且不收 action / trajectory；GAIA-1/[GAIA-2](../video-world-models/gaia-2.md) 是 Wayve 內部 driving-only；[V-JEPA-2](../latent-world-models/v-jepa-2.md) 是 latent representation 不出 pixel。Cosmos 是第一個 **(a) open-weight, (b) generalist pre-train, (c) 顯式支援 image2world / video2world / 多模 control, (d) 配套 Reason-VLM + Tokenizer + Transfer multi-controlnet** 的 stack。它的賭注是：pixel-video FM 的 implicit physics + 下游 sim-in-loop reward 比 hard PDE 路線更 scalable。

## 2. Core mechanism

兩條主路徑 (Predict1, 2025-01) — Diffusion 與 Autoregressive，後續 Predict2.5 (2025-10) 收斂到單一 flow-based 模型。

```
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
│ DIFFUSION branch    │                 │ AUTOREGRESSIVE     │
│ Cosmos-1.0-         │                 │ Cosmos-1.0-AR-4B   │
│  Diffusion-7B /     │                 │  AR-12B  (base)    │
│  14B  (T2W, V2W)    │                 │  AR-5B / 13B V2W   │
│ Latent diffusion    │                 │ Llama3-style GPT,  │
│ + DiT, prompt-      │                 │ tokens from DV     │
│ upsampler 12B       │                 │ tokenizer          │
└─────────┬───────────┘                 └─────────┬──────────┘
          ▼                                       ▼
     ┌─────────────────────────────────────────────────┐
     │  POST-TRAINING  (downstream specialization)     │
     │  ├─ Cosmos-Drive (multi-cam, traj conditioning) │
     │  ├─ Cosmos-Drive-Dreams (long-tail AV scenes)   │
     │  ├─ Cosmos-Policy (visuomotor head, LIBERO/RoboCasa)│
     │  ├─ Cosmos-Transfer / Transfer2.5 (multi-CN:    │
     │  │     RGB + depth + seg + edge → video)        │
     │  └─ Cosmos-Reason1-7B (Qwen2.5-VL-based         │
     │      reasoning VLM, SFT + RL on physical CoT)   │
     └─────────────────────────────────────────────────┘
```

Hierarchical 用法是 **Reason1 (slow, plan) → Predict (fast, rollout)** — VLM 輸出 chain-of-thought + 行動描述，Predict 拿來當 caption + 起始幀 condition 出未來 video。這是「Cosmos 不只是 video gen」的關鍵：layered architecture 讓 LLM 端的 reasoning 與 video FM 的 rollout 可以解耦訓練。

Predict2.5 進一步把 T2W / I2W / V2W 三條合成單一 flow-based 主幹，並用 Cosmos-Reason1 當 text encoder（取代純 T5/CLIP），是「自舉式 stack」的明顯設計。

## 3. 五軸定位 + 同軸對手

| Axis | Cosmos | Sora 2 | Veo 3 | Wayve GAIA-2 | V-JEPA-2 |
|---|---|---|---|---|---|
| Output | pixel-video（+ latent via DV tokenizer） | pixel-video | pixel-video | multi-cam pixel-video | latent |
| Injection | data-only + sim-in-loop（post-train 可掛 Genesis / Isaac） | implicit | implicit | implicit + structured cond | implicit |
| Control | text / image / video / **trajectory（Drive）** / **action（Policy）** / depth+seg（Transfer） | text + image | text + image | ego-traj + agent config + road semantics | action（zero-shot planning） |
| Temporal | clip-parallel（diffusion）/ AR；hierarchical（Reason + Predict） | clip-parallel | clip-parallel | clip-parallel, multi-view consistent | latent-rollout |
| Domain | generalist → robotics / driving fine-tune | generalist | generalist | driving-only | generalist embodied |

要點：Cosmos 與 Sora/Veo 在 axis 1/2/4 接近，**真正的 USP 在 axis 3（control 多元）+ axis 5（顯式 multi-domain post-train kit）+ open weight**。對比 GAIA-2，Cosmos 更通用但 driving 多 camera consistency 與 ego-dynamics 顯式 conditioning 弱；對比 V-JEPA-2，Cosmos 出 pixel 可直接喂 VLA／可視化，但 latent rollout 速度與 long-horizon 穩定度差一截。

## 4. Where it shines / where it breaks

⚡ **Shines**
- **Robotics post-train**: Cosmos-Policy（從 Predict2-2B 單階段 post-train）在 LIBERO / RoboCasa 上超過從零訓的 diffusion policy 與 VLA baseline — single-stage、無需設計 action head，這是「video FM 直接當 policy backbone」首個 clean evidence。
- **Driving data augmentation**: Cosmos-Drive-Dreams 對 long-tail（遮擋行人、異常車輛）的可控生成，補 AV 真實資料覆蓋空白，Wayve 也採用 Cosmos backbone。
- **Multi-controlnet (Transfer 2.5)**: RGB + depth + seg + edge 同時 condition — 真實 production data engine 用法（先用 sim renderer 出粗糙視訊，再用 Transfer 「貼皮」到 photoreal）。
- **Open weight + permissive license**: 唯一可 audit 物理感的同級 FM。HF/GitHub 完整 release（Predict1-7B/14B、AR-4B/12B、Reason1-7B、Transfer 2.5、Tokenize1 全套）。

❌ **Breaks**
- **Long-horizon drift**: 官方 paper 明列「objects disappearing or deforming, violations of physics like implausible movements or ignoring gravity」— autoregressive 變體 >8s 後 motion instability 顯著。
- **Contact-rich physics**: 與所有 pixel-video FM 一樣，抓取碰撞、布料折疊、流體混合 visually 看起來合理但 force/contact 不可微 — 不能拿來當 closed-loop diff-sim 替代品。實測社群在 grasp 任務上把 Cosmos rollout 餵 VLA 訓練會出現 silent failure（policy 學到視覺 cue 但實機 contact phase 崩）。
- **Object permanence / 3D consistency**: 鏡頭環繞或長平移時物件 morph，這是 data-only 路線的通病；GAIA-2 多 camera consistency 設計就是針對這點。
- **Prompt fidelity (Predict1)**: 14B Text2World 在 fine-grained spatial relation（"left of", "behind"）依賴 PromptUpsampler-12B 改寫；Predict2.5 用 Reason1 當 text encoder 才大幅改善。
- **GPU 門檻**: 14B Text2World 推理需 H100/H200 等級單卡 80GB；4B AR 可 ≤40GB 但品質差距大；post-train 7B 需 multi-node。社群 reproduction 在 A100 80GB 跑 14B 常 OOM，需量化或切 14B → 7B。

## 5. Reproduction notes

最小可跑 setup（2026-05 狀態）：

- **Inference 7B Text2World**: 1× H100 80GB；HuggingFace `nvidia/Cosmos-Predict1-7B-Text2World`；fp8 / bf16 模式 ~50GB VRAM；單 clip 5s 推理 ~2-4 min。
- **Inference 14B**: H100/H200 80GB（fp8）或 2× A100 切；社群常 OOM，需用 sequence parallel。
- **Cosmos-Reason1-7B**: 單 H100 / 雙 A100 即可，vLLM 推理；NIM endpoint 可直接 API。
- **Cosmos-Policy post-train**: 從 Predict2-2B base，single-stage SFT on demo trajectories（LIBERO 規模 ~hundreds of demos），8× A100/H100 一晚跑通。
- **Tokenizer 獨立可用**: `Cosmos-Tokenize1-CV8x8x8-720p` / `DV8x16x16-720p` HF 上有 standalone weight，做 video embedding / VAE 替代品很好用。

典型踩坑：
- HF 下載要接受 license（NVIDIA Open Model License）— 自動化 CI 會卡。
- Predict2.5 強制要 Reason1 在 GPU 上同時 load 當 text encoder，VRAM peak 比 Predict1 高。
- AR 變體輸出需經 `Cosmos-1.0-Diffusion-7B-Decoder-DV8x16x16ToCV8x8x8` 後處理才能拿到「乾淨」pixel；很多人忘了這步直接看 DV decode 結果就嫌畫質差。
- 訓 Policy 時不要改 action head 結構，paper 明說「無架構修改」是 unlock — 加了 diffusion action head 反而退化。

## 6. Cross-line synthesis

- **× diff-sim ([Genesis](../differentiable-simulators/genesis.md) / [MJX](../differentiable-simulators/mujoco-mjx.md))**: Cosmos 在 axis 2 = `data-only`，diff-sim = `hard-constraint / sim-in-loop`。組合方式：Genesis 出 ground-truth physics rollout（粗糙渲染）→ Cosmos-Transfer 2.5 用 depth+seg 當 control 貼 photoreal 皮。這是 **「physics 由 sim 保證、視覺由 FM 補」** 的標準 production pattern，Wayve / NVIDIA Isaac 均採。
- **× neural surrogate ([GraphCast](../neural-surrogates/graphcast.md) / [FNO](../neural-surrogates/fno.md))**: 不直接對接 — surrogate 是 field output，Cosmos 是 pixel；只在 scientific viz 場景組合（surrogate 算流場 → renderer → Cosmos refine 視覺）。
- **× 3D-aware (World Labs / Gaussian splatting)**: Cosmos 缺顯式 3D，組合方式是 3DGS 出 multi-view → Cosmos-Transfer 做時序補洞 + photoreal 化。長期看，Predict 系列加 3D-aware conditioning 是 roadmap 上的明顯空缺。
- **× VLA**: Cosmos-Policy 已是直接 fork — video FM backbone + action token head。意義是：VLA pre-training 可以用「video FM weights → freeze partial → action SFT」取代「from-scratch transformer」。對 `bridge-to-vla/` 來說這是 anchor case。
- **× latent-WM ([DreamerV4](../latent-world-models/dreamer-v4.md), [V-JEPA-2](../latent-world-models/v-jepa-2.md))**: Cosmos 的 DV tokenizer 本身就提供 latent space；理論上可把 AR-12B 視為 latent dynamics model。但官方未鋪這條路；社群實驗少。

## 7. References

- Canonical: NVIDIA Cosmos team, *Cosmos World Foundation Model Platform for Physical AI*, arxiv:2501.03575 (Jan 7, 2025).
- Cosmos-Reason1: arxiv:2503.15558 (Mar 2025), *Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning*.
- Cosmos-Policy: arxiv:2601.16163 (2026-01), *Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning*.
- NVIDIA blog: <https://blogs.nvidia.com/blog/cosmos-world-foundation-models/> (release announcement).
- NVIDIA docs: <https://docs.nvidia.com/cosmos/latest/> (Predict2.5 / Transfer2.5 cookbook, Oct 2025).
- GitHub orgs: <https://github.com/nvidia-cosmos> (predict1, predict2, predict2.5, transfer2.5, reason1).
- 二手實測:
  - HuggingFace blog *Topic 24: Cosmos WFM Platform* (Kseniase) — failure mode summary.
  - LearnOpenCV — *Cosmos-Reason VLM for Video VQA* (reproduction notes).
  - Wayve × NVIDIA collab announcement (Cosmos backbone for GAIA-related work).
  - [TBD: verify Cosmos-Drive-Dreams arxiv ID — appears in NVIDIA blog 2025-Q1 but no standalone preprint confirmed].

## 8. §8 Pitfall log

| # | Severity | Issue | Source | Workaround |
|---|---|---|---|---|
| 8.1 | High | Long-horizon drift > 8s：object morph / 重力違反 / motion instability | arxiv 2501.03575 §Limitations + HF blog community report | 切短 clip（5-8s）+ hierarchical rollout（Reason1 重置 prompt） |
| 8.2 | High | Contact-rich silent failure：grasp / 接觸視覺合理但 force phase 崩 | community VLA reproduction 報告（多筆社群） | 不要單獨用 Cosmos rollout 當 VLA train data；與 diff-sim contact label 對齊 |
| 8.3 | Medium | 3D / multi-view inconsistency：物件在環繞鏡頭中 morph | 對比 GAIA-2 paper §1 motivation | driving 場景用 GAIA-2 / 等 Predict3 多 view 版；其他場景接 Transfer + 3DGS prior |
| 8.4 | Medium | Predict1 prompt fidelity 弱（spatial relation） | arxiv 2501.03575 § eval；Predict2.5 換 Reason1 text encoder 即是修正 | 升級 Predict2.5；或先過 PromptUpsampler-12B |
| 8.5 | Medium | 14B inference OOM in A100 80GB；社群 reproduction 痛點 | HF discussions、cosmos-predict1 README 性能表 | fp8 量化；或退 7B；或 sequence parallel 多卡切 |
| 8.6 | Low | AR 變體 decode 需額外 `Diffusion-7B-Decoder-DV8x16x16ToCV8x8x8`，常被遺漏 | docs/predict1/autoregressive/reference | 比對 README 推理 pipeline，勿跳 decoder 步驟 |
| 8.7 | Low | HF 下載需接受 NVIDIA Open Model License — CI 自動化卡點 | HF model card | 用 HF token + `huggingface-cli login` 並預先 accept；或本地鏡像 |
| 8.8 | Low | Predict2.5 強制同時 load Reason1 當 text encoder → VRAM 峰值升高 | docs/cosmos/latest/predict2 release notes | 接受更高 VRAM；或拆服務（Reason1 在另一卡 / NIM 端） |
| 8.9 | Medium | Tokenize1 在低 bitrate（DV8x16x16，total ~2048×）細紋丟失，texture detail 不可逆 | arxiv 2501.03575 tokenizer eval | 細節敏感任務改 CV4x8x8 或 CV8x8x8（compression 較低） |
| 8.10 | Low | Policy post-train 加 action head / diffusion head 反而退化 | arxiv 2601.16163 §method ablation | 遵守「無架構修改」原則，只 SFT 原模型 token 預測頭 |

> Pitfall 觀察重點：8.1 / 8.2 / 8.3 是「pixel-video FM 路線的結構性 break」— 不會因 scale up 自動解決；要靠 axis 2 補 `sim-in-loop` 或 axis 1 換 `3d-scene / latent` 才能根治。8.5–8.8 是 ops 層級可繞過的工程坑。
