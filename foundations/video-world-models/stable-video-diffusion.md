<!-- ontology-5axis output=pixel-video injection=data-only control=text|image-init|camera temporal=clip-parallel domain=rigid -->

# Stable Video Diffusion (SVD / SVD-XT)

## 1. One-paragraph TL;DR

Stable Video Diffusion（Blattmann et al., Stability AI, [arxiv 2311.15127](https://arxiv.org/abs/2311.15127), 2023-11-25；模型權重 2023-11-21 釋出）是 **第一個 open-weight、能在單卡 A100 上 reproduce 的 latent-video-diffusion baseline**。它不是 SOTA — Sora 2024 一出就把它在 fidelity 上甩開兩個身位 — 但**整個社群的 image-to-video 生態（ComfyUI workflows、AnimateDiff 對照、PhysGen 上游 backbone、I2V LoRA 蒸餾、camera-conditioned post-training）幾乎全是疊在 SVD 之上**，因為 Sora/Veo/Cosmos 不是真正開源（Cosmos 開的是 6B/12B 的後續一代，但社群實驗最容易的「拿來改 backbone」的仍然是 SVD）。本 handbook 把 SVD 標為 **v2 ontology `control=image-init` 的 canonical anchor**（ontology v2 line 100）— 它示範了 image-to-video 作為 "initial-condition conditioning"（而非真正物理控制）這條最低門檻路徑的價值與天花板。它的價值對 reader：(a) 你要 finetune 一個 image-to-video model，這仍是 2025-2026 開源 baseline 的首選；(b) 你要理解 "data-only physics" 為什麼會在 4 秒以內就崩；(c) 你要把 SVD 當 backbone 接 PhysGen / Force Prompting / ControlNet — 必須先理解它的固有 motion bias 與 conditioning 失效模式。

## 2. Core mechanism

SVD 把 Stable Diffusion 2.1 的 spatial U-Net 加 temporal layers，分三階段訓練：

```
Stage 1: Text-to-Image pretrain (SD 2.1)
   │  spatial U-Net, 大圖庫
   ▼
Stage 2: Video pretrain (LVD-F, 152M clips)
   │  insert temporal conv + temporal attention layers
   │  low-res (320×576), 14 frames
   ▼
Stage 3: High-quality video finetune
   │  curated subset (~1M)
   │  high-res (576×1024 for SVD-XT)
   │  14 frames (SVD) or 25 frames (SVD-XT, 4s @ 6fps)
   ▼
[Per-frame VAE encode]  ← 與 Sora 不同：VAE 只壓空間，時間留給 U-Net
   │
   ▼
latent_t (z_1, z_2, ..., z_T)
   │  conditioning:
   │    - image embedding (CLIP) of init frame
   │    - VAE latent of init frame (concat to each frame)
   │    - micro-conditioning: fps_id, motion_bucket_id, cond_aug
   ▼
[Temporal U-Net]  EDM-style continuous noise schedule
   │  spatial attention (per-frame) ↔ temporal attention (across frames)
   ▼
denoised latents → [VAE decoder, per-frame] → pixel video
```

關鍵設計（vs Sora / Cosmos / AnimateDiff）：

- **Per-frame VAE，無時間壓縮**：與 Sora 的 spacetime VAE 不同；SVD 的 latent 是 frame-wise stack，時間 coherence 全靠 U-Net 內部 temporal attention。這是 SVD finetune 友善的關鍵（spatial 部分可直接重用 SD2.1 LoRA），但也是長度天花板（25 frames 之後 attention quadratic 爆）。
- **Micro-conditioning 三件套**：`fps_id`（控制生成幀率）、`motion_bucket_id`（0-255，越大運動越大；訓練時對每段 clip 計算 optical-flow magnitude 落到 bucket）、`cond_aug` / `noise_aug_strength`（加在初始幀的 latent 噪聲；越大「離初始幀越遠」）。三者皆 inference 可調，但**沒有可微學習 loss 保證 motion 真的對應 bucket** — 純資料 implicit 學。
- **LVD curation pipeline**：原始 580M 視頻對 → 三道機械過濾（optical flow magnitude > θ、OCR text density < θ、CLIP aesthetic > θ、synthetic caption coverage）→ LVD-F 152M。論文最 重要的實驗 finding 是 **curation > raw scale**：在 10M curated subset 訓出的模型勝過在 50M raw 訓的。
- **Clip-parallel rollout**：一次 denoise 整段 14/25 frames，無 KV cache 串流。長片必須用 chained generation（用上段最後一幀做下段 init frame），drift 嚴重。

## 3. 五軸定位 + 同軸對手

```
output     = pixel-video           ← per-frame VAE，最終 decode 像素
injection  = data-only             ← 無 physics loss / 無 sim / 無 arch bias
control    = text|image-init|camera ← text 在 SVD 上其實很弱（base model 沒做 T2V finetune）；
                                     主用 image-init；後續 SVD-MV / camera-LoRA 加 camera
temporal   = clip-parallel         ← 14/25 frames 一次 denoise
domain     = rigid                 ← 訓練集偏自然影片 + 攝影機運動（people / objects / scenes）
                                     按 Check 9c，SVD 不在 generalist 白名單；最常 surface 的
                                     失效域是流體 splash / 軟體布料 / 顆粒 — 都崩。剛體與小幅
                                     攝影機 pan / dolly 是它表現相對最穩的子域，故標 rigid
```

> **Check 9c 註記**：v2 generalist 白名單僅含 Sora · Veo · Cosmos-Predict · Cosmos-WFM。SVD 雖訓練集亦覆蓋通用網路影片，但模型行為——特別是常被報告的 "tends to be static / slow camera pan / fails fluid & cloth"——表現得遠不如真正的 generalist foundation。把它釘在 `rigid` 子域是誠實的：它最能跑通的工況是剛體 + 小幅攝影機，這也是社群下游（PhysGen 等）選它做 backbone 的真實原因。

同軸對手（皆 `pixel-video × data-only × image-init × clip-parallel`）：

- **[Sora](./sora.md)**：閉源，scale 與 fidelity 領先一代；對比是「open vs closed」與「DiT spacetime VAE vs U-Net per-frame VAE」兩條設計分歧
- **[Veo / Veo-2 / Veo-3](./veo.md)**：Google 線，閉源；強 text→video，物理感隨版本漸進加 audio + camera control
- **[Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md)**：NVIDIA 開源（2025），更新一代 backbone，6B/12B；針對 robotics / driving 微調而設計；SVD 仍在「最容易上手 finetune」這格佔優
- **CogVideoX**（智譜，2024-08，開源 5B）：DiT 架構，2024 之後的開源 SOTA；許多原本基於 SVD 的研究在 2025 切到 CogVideoX
- **Open-Sora** / **Open-Sora-Plan**（2024-2025）：嘗試 reproduce Sora 的開源項目；架構接近 DiT，但訓練資源遠不及；SVD 是它們的對照 baseline
- **[GAIA-2](./gaia-2.md)**：Wayve 駕駛專用；非通用，加 layout/trajectory 多模態 conditioning

## 4. Where it shines / where it breaks

### ⚡ Shines

- **Open weights + finetune-friendly**：SD2.1 backbone 讓 spatial-side LoRA、ControlNet、IP-Adapter 等大量 image-domain 工具可直接遷移到 video 端
- **單 A100 可跑** inference：SVD-XT ~180s 出 25 frames，是 2023-2024 開源 video model 中門檻最低的
- **生態系**：ComfyUI、AUTOMATIC1111、diffusers 都有一階公民 pipeline；社群 finetune（SVD-Reverse、SVD-MV、CameraCtrl SVD-LoRA）持續產出
- **作為下游研究 backbone**：[PhysGen](../physics-conditioning/physgen.md) 把剛體 sim 結果做 image2video 的接管者就是 SVD；許多 "controllable video gen" 論文（Drag-A-Video、Boximator、MotionCtrl）首版實作都基於 SVD

### ❌ Breaks

- **Lower fidelity than Sora / Veo**：HF 模型卡自承 "may generate videos without motion or very slow camera pans"、"does not achieve perfect photorealism"、"faces and people may not generate properly"
- **Short clips, hard ceiling**：25 frames @ 6 fps ≈ 4 秒；chained generation 接長片 drift 嚴重，identity 漂移、第二段重新「想像」場景
- **Text conditioning 形同無效**：HF 模型卡明寫 "Cannot be controlled through text"；SVD 是 i2v 為主，text 在 base SVD 上沒做真正 fine-tune
- **Motion bucket 行為非單調**：`motion_bucket_id=180` 可能比 `220` 動更多（[Issue #237](https://github.com/Stability-AI/generative-models/issues/237) 用戶報告意義不清）— bucket 是訓練時 optical-flow 統計分桶，與用戶語意「我要多少運動」沒有 calibration
- **License 非完全開源商用**：早期社群誤以為 CreativeML OpenRAIL++-M；實際 HF repo 用 `stable-video-diffusion-community` license，商用需走 Stability AI 商用授權

## 5. Reproduction notes

最小可跑 setup（2026-05 仍有效）：

```python
# diffusers >= 0.25
from diffusers import StableVideoDiffusionPipeline
import torch

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16, variant="fp16",
)
pipe.enable_model_cpu_offload()              # 降到 <8GB VRAM
pipe.unet.enable_forward_chunking()
frames = pipe(
    init_image,                              # PIL Image, 1024×576
    decode_chunk_size=2,                     # 不要忘 — 預設會 OOM
    motion_bucket_id=127,                    # default; 越大越動 (非單調)
    noise_aug_strength=0.02,                 # default; 越大越偏離初始幀
    num_frames=25,
    num_inference_steps=25,
).frames[0]
```

預算：
- **VRAM**：搭配 `enable_model_cpu_offload` + `enable_forward_chunking` + `decode_chunk_size=2`，可降到 <8GB。沒設定 → 40GB+，常見 OOM（[diffusers #6385](https://github.com/huggingface/diffusers/issues/6385)）
- **時間**：A100 80GB SVD-XT ~3 min；4090 ~5-7 min（@ fp16, CPU offload）
- **訓練成本**：論文報 ~200K A100-80GB hours（不是個人能 reproduce 的量級，只能 finetune）

典型踩坑：
- 忘了 `decode_chunk_size` → VAE decode 階段 OOM
- `motion_bucket_id` 拉到 200+ 反而 motion 變糟 → 不要當 monotonic dial
- 用 fp32 在 V100-32GB 試 → 最後 sampling step 仍 OOM（[Issue #180](https://github.com/Stability-AI/generative-models/issues/180)）
- ComfyUI workflow 模板：[thecooltechguy/ComfyUI-Stable-Video-Diffusion](https://github.com/thecooltechguy/ComfyUI-Stable-Video-Diffusion)
- 訓練 / finetune code：[pixeli99/SVD_Xtend](https://github.com/pixeli99/SVD_Xtend) 是事實上的社群標準

## 6. Cross-line synthesis

SVD 在本 handbook 五軸 ontology 上的策略價值，**不在它自己物理感多強**（很弱），而在它作為一個 stable open backbone 可以被「掛物理」：

- **× diff-sim**：[PhysGen](../physics-conditioning/physgen.md) (Liu et al., ECCV 2024) 用 rigid-body sim 算出剛體軌跡 → 投影成 control map → 餵 SVD 做 image2video render。SVD 提供 "把 sim 結果 photoreal 化" 的能力，sim 提供 SVD 缺的物理一致性 — 是 `pixel-video × sim-in-loop-train` 的典型 composition
- **× neural-surrogate**：尚未見到 mature 案例；理論上 FNO/GraphCast 算出的場可以 colormap → image2video，但時序對齊與 latent space 失真會吃掉大半 signal
- **× latent-WM**：SVD latent 不是 planning-friendly latent（per-frame，無 dynamics-aware bottleneck），Dreamer / V-JEPA 路線不會把它當 backbone；要做 latent-WM 應該選 spacetime VAE 系（Cosmos, Open-Sora）
- **× VLA / action-conditioning**：SVD 沒原生 action input；後續 ATM / AVDC 等工作把 action 透過 ControlNet-style adapter 接到 SVD，但 control fidelity 不如真正 action-native 模型（Genie / DreamerV4）

**v2 ontology 定位**：SVD 是 `control=image-init` canonical anchor（ontology v2 line 100）。它示範了一個簡單事實 — image-init 是「給條件」中最便宜、最 robust 的一種（pixel domain 就在那、不需學 embedding 對齊），這也是為什麼 2024-2026 所有 i2v 模型（Cosmos-img2vid、Wan、Kling i2v、CogVideoX-i2v）幾乎都把 image-init 當主入口。

## 7. References

Canonical paper：
- Blattmann, A. et al. "[Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127)". arXiv:2311.15127, 2023-11-25. Stability AI.

Model releases：
- [stabilityai/stable-video-diffusion-img2vid](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) — 14 frames base, 2023-11-21 release
- [stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) — 25 frames extended (SVD-XT), 2023-11
- SVD 1.1 — Stability AI 商用迭代，2024-02 發布；更一致的攝影機運動與 motion bucket 行為
- License：`stable-video-diffusion-community` (商用需 [Stability AI 商用授權](https://stability.ai/license))

二手實測 / 拆解：
- [Stability AI launches SVD 1.1, VentureBeat](https://venturebeat.com/ai/stability-ai-launches-svd-1-1-a-diffusion-model-for-more-consistent-ai-videos) — 1.1 改進總結
- [InfoQ — Stability AI Open-Sources SVD](https://www.infoq.com/news/2023/12/stable-video-diffusion/) — 釋出時的社群反應
- [HuggingFace diffusers SVD docs](https://huggingface.co/docs/diffusers/en/using-diffusers/svd) — 參數語意與 memory 配置的 canonical guide
- [pixeli99/SVD_Xtend](https://github.com/pixeli99/SVD_Xtend) — 社群事實標準 training / finetune code

## 8. §8 Pitfall log

> **§8.0 Check 9b cross-axis 自查**：`output=pixel-video × injection=data-only` 是 v2 相容矩陣中合法 cell（✓，line 163）。SVD 不嘗試在像素空間做 hard constraint，因此不觸發 Check 9b 例外解釋條款。

### §8.1 motion_bucket_id 非單調且語意未校準 — [Issue #237](https://github.com/Stability-AI/generative-models/issues/237)

- **原文摘錄**：用戶 KyriaAnnwyn 2023-12-08：「Could you please add some clarification about the meaning of the parameters? cond_aug — is it ... amount of noise added to each frame? What does 127 variants of motion_bucket_id mean?」issue 至今 no maintainer response。
- **Severity**：Medium（影響 reproducibility / 下游 ablation）
- **Workaround**：不要假設 bucket 與感知運動量線性對應；做 sweep（30/60/127/180）找 task-specific sweet spot；report 中註明 fps_id × motion_bucket_id 組合。

### §8.2 VRAM 文檔誤導 — [diffusers #6385](https://github.com/huggingface/diffusers/issues/6385)

- **原文摘錄**：用戶照官方文檔 "follow low-memory recommendations to limit VRAM to 8GB" 操作，仍遇 CUDA OOM 需要 40GB+。
- **Severity**：High（影響首次 reproduction）
- **Workaround**：三件套必須 **同時** 開：`enable_model_cpu_offload()` + `unet.enable_forward_chunking()` + `decode_chunk_size=2`；遺漏任一都會在 VAE decode 階段或 attention 階段炸顯存。

### §8.3 V100-32GB 在最後 sampling step 才 OOM — [Issue #180](https://github.com/Stability-AI/generative-models/issues/180)

- **原文摘錄**：「OOM at the last sampling step ... seen examples on 24GB or smaller GPUs」
- **Severity**：High（V100 是學術環境常見硬體）
- **Workaround**：強制 `decode_chunk_size=1` 或先 sample latent、再單獨 chunked decode；fp16 是必要前提；最終手段切到 A100 / 4090 / 6000-Ada。

### §8.4 forward chunking 降質 — [diffusers #6258](https://github.com/huggingface/diffusers/issues/6258)

- **原文摘錄**：「Stable Video Diffusion Pipeline: forward chunking degrades image quality」
- **Severity**：Medium（memory↔quality tradeoff）
- **Workaround**：chunking 是時序 attention 分塊，會破壞長距 frame attention；若有足夠 VRAM，關掉 chunking 換更高一致性；若不行，至少把 chunk 開大（不要 1，試 4-8）。

### §8.5 第一幀後幾乎不動（"static video" 模式） — HF 模型卡 known limitation

- **原文摘錄**：「may generate videos without motion or very slow camera pans」
- **Severity**：High（最常被新手回報；以為模型壞了）
- **Workaround**：(a) `motion_bucket_id` 上調至 150-180；(b) `noise_aug_strength` 上調至 0.1-0.2（會減弱對 init image 的對齊但增加運動）；(c) 換初始幀 — 過於對稱、低紋理、近肖像構圖會觸發 static 模式。

### §8.6 Chained long-form 出現 identity drift

- **原文摘錄**：社群實測（[Facebook SD group](https://www.facebook.com/groups/stablediffusion/posts/1306537613345044/) 等）；用 SVD 接超過 4 秒長片，第二段用第一段最後一幀做 init，會出現主體變形、背景重建。
- **Severity**：High（限制商用案例）
- **Workaround**：(a) 接 [VideoCrafter / Wan / CogVideoX-i2v](https://github.com/THUDM/CogVideo) 做長段；(b) 引入 IP-Adapter 鎖 identity；(c) 加 ControlNet (depth/pose) 做 across-chunk 軌跡 anchor；(d) 接受短片定位，把 SVD 用在「4 秒以內 cinemagraph」這個它真正穩的工況。

### §8.7 [Descriptive] Injection × Temporal — `data-only × clip-parallel` 的特性

SVD 不屬於需要 `sim-in-loop-infer` 的迭代範式，因此沒有 PhysDiff 那類「每步 denoising 插 sim」的合法接口；要把物理掛回去只能走 `aux-loss` finetune 或 backbone-on-top（如 PhysGen 路線）。這也解釋了為什麼 2024-2025 物理可控視頻的主戰場移到 DiT 系（Sora / Cosmos）— per-frame VAE 對 sim-in-loop-infer 不友好。

### §8.8 [Descriptive] Control × Domain — text 形同無效，audit 應退到 image-init 主軸

雖然 v2 control 軸允許多值 `text|image-init|camera`，但對 SVD 而言 `text` 是繼承自 SD2.1 的退化通道（HF 模型卡明寫 cannot be controlled through text）。下游若需 text-driven，應該選 CogVideoX / Wan / Cosmos 而非 SVD。
