<!-- ontology-5axis output=pixel-video injection=data-only control=text|image-init|trajectory|camera temporal=clip-parallel domain=rigid -->

# AnimateDiff (+ ControlNet-Video / SVD-ControlNet line)

*Canonical anchor for the controllability-mechanisms zone — the "freeze the base, learn a temporal sidecar" pattern that the entire 2024 community video stack inherited.*

## 1. One-paragraph TL;DR

AnimateDiff (Guo et al., ICLR 2024 **Spotlight**, arxiv [2307.04725](https://arxiv.org/abs/2307.04725)) is the canonical demonstration that **temporal coherence can be bolted onto any frozen T2I checkpoint** by inserting a small "motion module" — a temporal-attention block — between the existing 2D U-Net layers. Base SD1.5 / SDXL / personalized DreamBooth/LoRA weights are untouched; the motion module is trained **once** on WebVid-style video, and at inference time slots into any compatible checkpoint. Pre-AnimateDiff, video diffusion either retrained the full backbone (expensive, destroys personalization) or stitched frames post-hoc (no temporal model). The same pattern then propagated to **SparseCtrl** (sparse-frame conditioning), **ControlNet-Video** (per-frame structural conditioning replicated across the clip), and the **SVD-ControlNet** line for trajectory / pose / depth — every 2024-2026 "video ControlNet" in ComfyUI is in this lineage.

## 2. Core mechanism

The motion module is inserted **after every existing 2D self-attention / cross-attention block** of the SD U-Net. It reshapes the `(B·F, C, H, W)` tensor to `(B·H·W, F, C)` and runs **1D self-attention over the frame axis** — i.e. each spatial location attends to itself across all F frames. Sinusoidal **temporal positional encoding** is added; `temporal_position_encoding_max_len` defaults to 24 (architectural cap that bounds the practical clip length).

```
        SD U-Net block (frozen ×N)
        ┌─────────────────────────────────────┐
F frames│  ┌──────────┐  ┌────────────┐       │
 ─────▶ │  │ 2D self- │─▶│ 2D cross-  │──┐    │
batched │  │  attn    │  │  attn      │  │    │
        │  └──────────┘  └────────────┘  ▼    │
        │           ┌─────────────────────┐   │  ◀── trainable
        │           │  MOTION MODULE      │   │       (~417M params,
        │           │  reshape→(BHW,F,C)  │   │        SD1.5 v2)
        │           │  + temporal pos-enc │   │
        │           │  + 1D temporal attn │   │
        │           │  reshape back       │   │
        │           └─────────────────────┘   │
        └─────────────────────────────────────┘
                       (×N blocks)
                              │
                              ▼
                   F coherent latent frames
```

Training: ε-prediction diffusion loss on WebVid-10M-style clips, **base frozen**, only motion modules updated. **MotionLoRA** fine-tunes the motion module with a small LoRA for shot-type adaptation (zoom-in, pan-left) — same parameter-efficient trick on the temporal axis.

**v3 SparseCtrl** generalizes the recipe: a separate ControlNet-style encoder takes sparse RGB / scribble keyframes (1, 2, or N) and injects into the motion-augmented U-Net. **ControlNet-Video** and the **SVD-ControlNet** line apply per-frame ControlNet conditioning (depth/pose/canny) replicated across F frames, optionally with temporal smoothing in the conditioning branch.

## 3. 五軸定位 + 同軸對手

| Axis | Value | Reason |
|---|---|---|
| Output | `pixel-video` | VAE decode to RGB frames |
| Injection | `data-only` | No physics loss / sim / constraint — motion comes from WebVid statistics |
| Control | `text \| image-init \| trajectory \| camera` | text via base CLIP cross-attn; v3 SparseCtrl adds image-init / scribble; SVD-ControlNet line adds trajectory & camera |
| Temporal | `clip-parallel` | 16-frame clip generated jointly per denoising step; `temporal_position_encoding_max_len=24` is the hard cap |
| Domain | `rigid` | Most demonstrations are general human / object motion; not generalist (Check 9c white-list excludes it) |

**Same-axis competitors:**

- **[Stable Video Diffusion](../video-world-models/stable-video-diffusion.md)** — same `pixel-video` + `data-only` + `clip-parallel`, but a **single fused image-to-video backbone** rather than a sidecar. Trades the ecosystem (no DreamBooth/LoRA reuse) for stronger motion priors. SVD-ControlNet community ports are the "trajectory/pose" siblings of AnimateDiff v3 SparseCtrl.
- **CogVideoX trajectory conditioning** — `pixel-video` + `data-only` + `trajectory`, but trajectory is fused into a **DiT-style** end-to-end trained backbone (no frozen base). Better long-range, worse customization.
- **[Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md)** — `text|camera|trajectory`, generalist domain, full retrain. AnimateDiff is the "poor person's Cosmos" — same axis values minus `camera`, minus the scale.

**Shared blind spot:** all four are `injection=data-only`. None know what a contact, force, or conservation law is. This is the reason `physics-conditioning/` zone exists.

## 4. ⚡ shines / ❌ breaks

**⚡ Shines**

- **Personalized animation** — any SD1.5 / SDXL DreamBooth or LoRA gets video for free; the RealisticVision + AnimateDiff demo defines the sweet spot.
- **Compositional with 2D ControlNet** — per-frame canny/depth/openpose replicates trivially across the clip; basis of every "vid2vid" ComfyUI workflow.
- **Cheap** — motion module ~417M params, retrainable on 8×A100 in days. Inference ~13 GB VRAM @ 16 frames / 512² fp16, runs on a 4090.

**❌ Breaks**

- **Long clips drift hard.** `temporal_position_encoding_max_len=24` is a real wall; beyond ~30-60 frames coherence collapses (issue #430 below). Identity drift on human faces is the dominant failure.
- **Off-distribution checkpoints.** Motion module trained on SD1.5-base transfers poorly to heavily-tuned anime checkpoints — flicker and washed-out colors; only fix is retraining the motion module.
- **No physics whatsoever.** Objects pass through each other, gravity is a suggestion, contact is hallucinated. `data-only` at its purest — why Force Prompting / PhysGen exist as the next-axis answer.
- **77-token CLIP ceiling** (issue #66) — long prompts silently truncated.
- **SDXL VRAM cliff** at 1024² + 16 frames OOMs 16 GB cards (issue #224); the `(B·H·W, F, C)` reshape is quadratic.

## 5. Reproduction notes

- **Repo:** [`guoyww/AnimateDiff`](https://github.com/guoyww/AnimateDiff) — v1 (Jul 2023) / v2 (Nov 2023, default) / v3 (Dec 2023, with SparseCtrl). Checkpoints `mm_sd_v15_v2.ckpt` (recommended), `v3_sd15_mm.ckpt`, SDXL beta `mm_sdxl_v10_beta.ckpt` on HuggingFace `guoyww/animatediff`.
- **Min GPU:** ~13 GB VRAM @ 16 frames / 512² / fp16 / SD1.5; SDXL ≥24 GB.
- **ComfyUI:** `ComfyUI-AnimateDiff-Evolved` (Kosinkadink) is the canonical node pack — v1/v2/v3, MotionLoRA, SparseCtrl, sliding-window context for >16 frames.
- **Typical pitfalls:** (1) version-mismatch motion module on wrong base → silent garbage; (2) sliding-window context re-introduces seam flicker every 16 frames, partially mitigated by `context_overlap=4`; (3) CFG ≥10 amplifies flicker, sweet spot 7-8.

## 6. Cross-line synthesis

- **pixel-WM line:** Direct ancestor of every plug-in video stack ([SVD](../video-world-models/stable-video-diffusion.md), [GAIA-2](../video-world-models/gaia-2.md), [Veo](../video-world-models/veo.md) all share the temporal-attention pattern). The motion module **is** the temporal half of a pixel world model.
- **latent-WM line:** Not directly compatible — AnimateDiff is pixel-decoded; latent WMs ([V-JEPA-2](../latent-world-models/v-jepa-2.md), Dreamer) don't decode.
- **diff-sim line:** No contact / constraint — pairs naturally with a `sim-in-loop-infer` wrapper (PhysDiff-style per-step projection). Open territory.
- **surrogate line:** Orthogonal; surrogates output `field`, AnimateDiff outputs `pixel-video`.
- **physics-conditioning line:** AnimateDiff v3 + SparseCtrl is the **sister-wedge** to the `text-action-trajectory-spectrum` (see `crossing/text-action-trajectory-spectrum.md` when written) — trajectory conditioning plumbed onto a `data-only` backbone, the cheap end of the spectrum. Force Prompting / NewtonGen sit at the expensive end.

**Canonical 2025 composition:** AnimateDiff motion module (frozen) + v3 SparseCtrl (trajectory keyframes) + 2D ControlNet (depth/pose, replicated per-frame) + LoRA-personalized base = full-stack controllable video on a single GPU. The recipe every "AI music video" pipeline reduces to.

## 7. References

**Canonical**

- Guo, Yang, Rao, Liang, Wang, Qiao, Agrawala, Lin, Dai. *AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning.* ICLR 2024 **Spotlight**. [arxiv 2307.04725](https://arxiv.org/abs/2307.04725) · [OpenReview Fx2SbBgcte](https://openreview.net/forum?id=Fx2SbBgcte) · [ICLR poster 19044](https://iclr.cc/virtual/2024/poster/19044).
- Guo et al. *SparseCtrl: Adding Sparse Controls to Text-to-Video Diffusion Models.* arxiv 2311.16933 (companion to v3).

**Secondary / community measurement**

- `guoyww/AnimateDiff` GitHub — official repo, v1/v2/v3 checkpoints + MotionLoRA.
- `Kosinkadink/ComfyUI-AnimateDiff-Evolved` — de-facto reference implementation; the context-window / sliding-window extension lives here, not upstream.
- `continue-revolution/sd-webui-animatediff` — A1111 port; large bug-report surface for SparseCtrl edge cases (issues #522, #536).
- `huggingface/diffusers` — `MotionAdapter` + `SparseControlNetModel` classes; cleanest reading of the v3 architecture.

## 8. §8 Pitfall log

| # | Source | Severity | Issue | Workaround |
|---|---|---|---|---|
| 8.1 | [guoyww/AnimateDiff#430](https://github.com/guoyww/AnimateDiff/issues/430) | high | Motion module trained on small (~20-video) datasets fails to learn temporal coherence at all — frames are disjoint slideshow. No maintainer reply; confirms the recipe is **data-hungry**, not just compute-hungry. | Use the pretrained v2/v3 motion module; only fine-tune via MotionLoRA on small data, never train motion module from scratch with <100k clips. |
| 8.2 | [guoyww/AnimateDiff#66](https://github.com/guoyww/AnimateDiff/issues/66) | medium | CLIP 77-token cap silently truncates long prompts (`400 > 77` warning). Unresolved upstream. | Use compel / long-prompt-weighting community extension; or split prompt across SparseCtrl keyframes. |
| 8.3 | [guoyww/AnimateDiff#224](https://github.com/guoyww/AnimateDiff/issues/224) | medium | SDXL motion module OOMs at 1024² on 16 GB cards — the `(B·H·W, F, C)` temporal reshape is quadratic in spatial × frames. | Tile to 768²; or fall back to SD1.5 v2 motion module + upscale. |
| 8.4 | Community-consensus (sandner.art, vidmodel.ai write-ups; no single tracked issue) | high | Identity / character drift on long clips — `temporal_position_encoding_max_len=24` is the architectural ceiling; sliding-window extensions in Evolved partially mask but never eliminate seam flicker every ~16 frames. | Cap clip length at 16-24 frames; for longer outputs, generate overlapping clips and blend in editing — not in the model. The 2025 successor patterns (Genie-2's `streaming-cache`, SVD-XT) are the real fix. |
| 8.5 | Repo discussions + SparseCtrl HF model card | medium | Off-distribution base checkpoints (heavily-tuned anime LoRAs, niche realism models) cause flicker and color drift even with v2 motion module. | Match motion module training distribution; or retrain motion module against your checkpoint family — community has done this for several anime base models. |

**Cross-axis descriptive note (per Ontology v2 §9b/9c):**
- Output × Injection: `pixel-video × data-only` is the most populated cell in the v2 compatibility matrix and the **default for the entire community video stack**. AnimateDiff is its purest expression — no physics, no constraint, just frozen-base + temporal sidecar trained on WebVid.
- Control × Domain: `trajectory` + `camera` controls would naturally pair with `driving` (GAIA-2 territory); AnimateDiff using them on `rigid` general scenes is why trajectory adherence is weaker than GAIA-2 / Cosmos-Drive on the same benchmarks. Domain mismatch, not architectural inferiority.
- `domain=generalist` was **not** used here (Check 9c — only Sora / Veo / Cosmos-Predict / Cosmos-WFM may claim that tag); `rigid` is the honest label given the WebVid training distribution.
