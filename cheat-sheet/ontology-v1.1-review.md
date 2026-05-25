# Ontology v1.1 Review (date: 2026-05-25)

> ## 🚧 STATUS: PROPOSAL — NOT YET ADOPTED
>
> 這是 v1.1 的 **expert review 提案**，不是當前 ontology spec。
> 當前 active spec = [`cheat-sheet/ontology.md`](./ontology.md) (v1)。
>
> 採納時程：等 30+ dissection 後評估；若採納，14 篇 dissection header 都需同步重簽（最大改動：Axis 2 rename 與 Axis 3 `*-prompt → *-init` rename）。

---

Reviewer voice: engineer, critical. Inputs: Cosmos (2501.03575), LeCun JEPA position paper (2022 OpenReview / Courant), V-JEPA 2 (2506.09985), GAIA-2 (2503.20523), DreamerV3 (Hafner 2023 / Nature 2025), Genie / Genie-2 (2402.15391 + DeepMind blog), GraphCast (Science 2023), Pangu-Weather (2211.02556), FourCastNet (2202.11214), FNO/Geo-FNO (2207.05209), PhysGen (2409.18964), PhysDiff (2212.02500), Force Prompting (2505.19386), NewtonGen (2509.21309), Morpheus benchmark (2504.02918), VJEPA-2-reward (2510.21840), NeurIPS 2025 World-Models / Video-Generation workshops, AlphaFold 3 (Nature 2024).

The TL;DR: the 5-axis frame is structurally sound and worth keeping. Axes 1, 4, 5 are mostly fine. Axis 2 has a real category-confusion problem (`score-conditioned` vs `constraint-loss` overlap) that the v1 TODO already flags. Axis 3 leaks scope. Below are per-axis critiques and a concrete diff.

---

## Axis 1: Output space

**Validation.** Values are *almost* mutually exclusive but not quite. `3d-scene` is defined as "explicit 3D (3DGS / mesh / occ / SDF)" then `mesh` is broken out separately — `mesh` is a strict subset of `3d-scene`. `latent` is an *encoding*, not a space (every diffusion model has a latent; the question is whether the *final delivered* output is latent or decoded pixels). For world-model literature (V-JEPA 2, DreamerV3) the distinction matters: V-JEPA 2 deliberately never decodes — that *is* the output for planning. Cosmos-Predict and GAIA-2 are latent-diffusion *internally* but deliver pixel video.

**Critique.**
- The `3d-scene` / `mesh` overlap. Either fold `mesh` into `3d-scene` with a sub-tag, or split `3d-scene` into `3d-gaussian` / `mesh` / `sdf-occ`. The 2025 GS-world-model wave (Generative Gaussian Splatting 2503.13272, Visionary 2512.08478) makes the GS-vs-mesh distinction load-bearing.
- `latent` is doing two jobs: (a) "model emits latent tokens consumed by downstream policy" (V-JEPA 2, DreamerV3) and (b) "latent diffusion that will be decoded" (Stable Video Diffusion, Cosmos). Only (a) belongs as an output-space value. Re-tag (b) as `pixel-video` because that's what the user sees.
- Missing: `implicit-field` (NeRF / occupancy / SDF as the *output*, distinct from 3DGS) — relevant for the AlphaFold-3-style diffusion-on-coordinates case (which is technically `particle` of atoms, but anchoring it explicitly avoids debate).
- Missing: `motion` (joint angles / SMPL params / human motion sequences). PhysDiff 2212.02500 sits awkwardly between `action-seq` and `pixel-video` today.

**Proposed v1.1 changes.**
- Merge `mesh` into a renamed `3d-explicit` (covers 3DGS, mesh, point cloud); add `3d-implicit` (SDF/NeRF/occupancy).
- Rename `latent` → `latent-tokens` and restrict to non-decoded planning latents (V-JEPA, DreamerV3 wm output, MuZero).
- Add `motion` (skeletal / articulated body pose sequences).

**Canonical anchors per value.**
- `pixel-video`: Cosmos 2501.03575 (NVIDIA, Jan 2025).
- `latent-tokens`: V-JEPA 2 2506.09985 (Meta, Jun 2025); DreamerV3 (Hafner et al., Nature 2025).
- `3d-explicit`: Generative Gaussian Splatting 2503.13272 (Mar 2025).
- `3d-implicit`: DeepSDF (Park et al. 2019, arxiv 1901.05103) — old but canonical.
- `particle`: Neural Material Point Method 2408.15753 (Aug 2024).
- `field`: GraphCast (Lam et al., Science 2023); FNO (Li et al. 2010.08895, 2020).
- `action-seq`: Genie 2402.15391 (Feb 2024).
- `motion` (new): PhysDiff 2212.02500.

---

## Axis 2: Physics injection

This is the USP axis and also the weakest. The v1 TODO already asks the right question ("is `score-conditioned` a subset of `constraint-loss`?"). Answer from the literature: **no, but the boundary is fuzzy and needs an explicit decision rule.**

**Validation.** Six values, not mutually exclusive. PhysDiff 2212.02500 inserts a physics simulator *inside the diffusion denoising loop* — that is simultaneously `sim-in-loop` and `score-conditioned` under the current definitions. Force Prompting 2505.19386 has zero physics in the architecture or loss — it's pure `implicit-from-data` with a fancy conditioning channel. NewtonGen 2509.21309 trains a *neural Newtonian dynamics* head — neither `hard-PDE` (no exact constraint) nor pure `constraint-loss` (architecture is biased). VJEPA-2-as-reward (2510.21840) uses physics as a *reward at inference search time*, not training — there is no v1 bucket for this.

**Critique (strongest pushback).**
- The `constraint-loss` vs `score-conditioned` distinction collapses in practice: classifier-guided diffusion (Dhariwal/Nichol 2105.05233) adds a gradient at inference; PINNs (Raissi 2017) add the same kind of gradient at training. They differ in **when** physics enters (train vs test), not **how**. Better split axis: `train-time` vs `inference-time` physics, orthogonal to the loss-vs-architecture-vs-simulator distinction.
- `sim-in-loop` is overloaded. It currently covers (i) differentiable simulator providing gradients (Genesis-style), (ii) non-differentiable simulator providing rewards (V-JEPA-2 reward search), (iii) simulator providing ground-truth data (Cosmos post-training). These have very different engineering profiles.
- `energy-based` is rarely *the* physics injection mechanism today; it's been subsumed by score-based models. LeCun's JEPA position paper explicitly frames JEPA as a latent-variable EBM, but the *physics* in V-JEPA 2 is implicit-from-data, not the EBM structure. Risk: this value is mostly aspirational and gets misapplied.
- Missing: `architectural-bias-soft` — networks with physics-flavored inductive bias that does *not* guarantee conservation (NewtonGen's dynamics head, neural ODEs without symplectic structure, MeshGraphNet's message-passing). Different from `hard-PDE` which guarantees exact conservation/equivariance.

**Proposed v1.1 changes.**
- Replace single Axis 2 with **two sub-axes**: `injection-mechanism` ∈ {`data-only`, `aux-loss`, `architecture-bias`, `hard-constraint`, `simulator-coupled`, `guidance-gradient`} and `injection-time` ∈ {`train`, `inference`, `both`}. Acknowledge this is a v2 move; for v1.1, minimum viable fix:
  - Rename `implicit-from-data` → `data-only` (shorter, sharper).
  - Split `sim-in-loop` → `sim-in-loop-train` and `sim-in-loop-inference` (PhysDiff is the latter; Cosmos-Reason-rollout-eval is also the latter; Genesis-train is the former).
  - Merge `score-conditioned` into a renamed `guidance-gradient` (covers both classifier-guided diffusion and PINN-style auxiliary gradient at any time).
  - Demote `energy-based` to a sub-tag, or note "use only when EBM is the *primary* physics structure, not when it's incidental".
  - Add `architecture-bias-soft` (NewtonGen, MeshGraphNet).
  - Keep `hard-PDE` (rename to `hard-constraint`).

**Canonical anchors.**
- `data-only`: Cosmos 2501.03575; Sora technical report (OpenAI 2024, no arxiv).
- `aux-loss` (was `constraint-loss`): PINN (Raissi et al. 1711.10561, Nov 2017).
- `simulator-coupled` train: Genesis (Zhou et al. 2412.18608, Dec 2024); inference: PhysDiff 2212.02500.
- `guidance-gradient`: Classifier-Free Guidance 2207.12598; PhysDiff (also fits here — confirms overlap problem).
- `architecture-bias-soft` (new): NewtonGen 2509.21309 (Sep 2025); MeshGraphNet (Pfaff et al. 2010.03409).
- `hard-constraint`: E(3)-equivariant DeepH 2210.13955 (Oct 2022); Hamiltonian NN (Greydanus 1906.01563).

---

## Axis 3: Controllability input

**Validation.** Reasonable, but the `multi` value is a hack — it's already addressable by the `|` separator. Drop it.

**Critique.**
- `image-prompt` and `3d-prompt` are *initial-condition* inputs, not really controls. GAIA-2 distinguishes "scene init" from "ego-action conditioning" — a useful axis-split.
- Missing: `camera` (camera pose / trajectory as a separate conditioning channel — major in 2025 video gen; Cosmos, GAIA-2, generative-GS all expose it).
- Missing: `language-structured` (BEV layout, scene-graph, semantic map) — distinct from free `text`. GAIA-2 takes structured road layouts; Cosmos-Drive takes BEV. Lumping them under `text` loses signal.
- `physical-param` is rare and could be folded into a broader `param` (mass/stiffness/viscosity/friction).
- `action` vs `trajectory` vs `force` is fine; these are the agent-physics conditioning ladder (low → high physical specificity).

**Proposed v1.1 changes.**
- Remove `multi`; rely on `|` in headers.
- Rename `image-prompt` → `image-init`; `3d-prompt` → `3d-init`. Add a separate flag `camera` for camera-pose conditioning.
- Add `layout` for structured scene/BEV/road-graph conditioning.
- Keep `physical-param` (rename `param` for brevity).

**Canonical anchors.**
- `text`: Cosmos 2501.03575.
- `action`: Genie 2402.15391; DreamerV3 (Nature 2025).
- `trajectory`: GAIA-2 2503.20523.
- `force`: Force Prompting 2505.19386 (May 2025).
- `contact`: PhysDiff 2212.02500 (motion-contact); ContactNets (Pfrommer 2009.11193).
- `image-init`: Stable Video Diffusion (Blattmann 2311.15127).
- `3d-init`: Generative Gaussian Splatting 2503.13272.
- `camera` (new): Cosmos 2501.03575 (camera-conditioned post-training).
- `layout` (new): GAIA-2 2503.20523.
- `param`: NewtonGen 2509.21309.

---

## Axis 4: Temporal paradigm

**Validation.** Six values, mostly well-defined. Two issues.

**Critique.**
- `joint-rollout` vs `temporal-transformer-rolling` is mostly an architectural detail; both produce N frames in one forward pass. The real semantic split is **fixed-window** vs **streaming-window-with-cache**.
- `hierarchical` is a structural property orthogonal to the others — DreamerV3 is `latent-rollout` *and* arguably hierarchical via the world-model/actor split. Cosmos-Reason (planner) + Cosmos-Predict (renderer) is hierarchical *over* two underlying paradigms. Either keep as a separate flag or define it strictly as "two-rate temporal hierarchy".
- `single-frame` is fine but tag it as "exclude from temporal evaluation" in audit.

**Proposed v1.1 changes.**
- Rename `joint-rollout` → `clip-parallel` (clearer about what's happening).
- Rename `temporal-transformer-rolling` → `streaming-cache` (more honest).
- Keep `hierarchical` but document it as compositional / can co-tag with another paradigm in `|`.

**Canonical anchors.**
- `single-frame`: PhysGen 2409.18964 (image→video bootstrapped from one frame is a borderline case; for pure single-frame use any T2I).
- `autoregressive`: original VideoGPT (Yan 2104.10157).
- `clip-parallel`: Stable Video Diffusion 2311.15127; Cosmos-Predict 2501.03575.
- `latent-rollout`: DreamerV3 (Nature 2025); V-JEPA 2 2506.09985.
- `hierarchical`: TECO (Yan 2210.02396); Cosmos (planner + predictor split).
- `streaming-cache`: Genie / Genie-2 (DeepMind blog Dec 2024); Decart Oasis.

---

## Axis 5: Domain coupling

**Validation.** Cleanest axis. Mutually exclusive in spirit. Issue: `fluid` and `weather` overlap conceptually (atmospheric fluid). The rationale (different communities, different benchmarks) is defensible but should be explicit.

**Critique.**
- `bio` is overloaded: protein structure (AlphaFold 3, static-ish), molecular dynamics (rollout), cell shapes, neural-tissue. Consider sub-tags later; for v1.1 leave as `bio` with a note.
- Missing: `astro` (astrophysics / N-body / cosmology surrogates) — small literature but real (e.g., CAMELS).
- Missing: `medical` if you ever want to tag deformable-organ simulators (overlaps `soft`).
- `generalist` is a fallback bin. Fine, but enforce via audit that ≥80% of dissections tag a specific domain when possible — `generalist` is meant for Cosmos/Sora/Veo only.

**Proposed v1.1 changes.**
- Add `astro` (optional, low-priority).
- Document `fluid` ⊃ liquid/gas CFD, `weather` = global atmospheric/oceanic forecasting; overlap is intentional.
- Keep everything else.

**Canonical anchors.**
- `generalist`: Sora (OpenAI tech report 2024); Cosmos 2501.03575.
- `robotics`: V-JEPA 2 2506.09985 (Franka deploy); RT-2 (Brohan 2307.15818).
- `driving`: GAIA-2 2503.20523.
- `fluid`: FNO 2010.08895; Neural-MPM-fluid 2505.18926.
- `rigid`: PhysGen 2409.18964.
- `soft`: PhysWorld 2510.21447 (deformable objects, Oct 2025).
- `granular`: Inverse-granular-GNS 2401.13695.
- `bio`: AlphaFold 3 (Abramson et al. Nature 2024, no canonical arxiv); ESMFold (Lin 2206.13517) as a hard anchor.
- `weather`: GraphCast (Science 2023); Pangu-Weather 2211.02556.

---

## Cross-axis issues

1. **Injection × Output coupling.** `hard-constraint` (Axis 2) almost forces a specific Output: equivariant nets ship `particle`/`field`/`motion`, not `pixel-video`. Worth a compatibility matrix in `crossing/`. A pixel-video model cannot realistically be `hard-constraint`.

2. **Injection × Temporal coupling.** `sim-in-loop-inference` only makes sense for paradigms with iterative refinement (`autoregressive`, `latent-rollout`, `streaming-cache`) — not pure `clip-parallel`. PhysDiff abuses this by treating each diffusion denoising step as the iteration, which is a third axis (denoising-step rollout) the ontology currently does not capture. Either accept the abuse or add a 6th axis `refinement-loop` (vote: do not — keep 5 axes; just note PhysDiff's idiosyncrasy in dissection).

3. **Control × Domain coupling.** `force`/`contact` controls are virtually only used in `robotics` / `rigid` / `soft`. `layout`/`trajectory` are mostly `driving`. Document expected control vocabularies per domain in the cheat sheet to make audit catch obvious mis-tags.

4. **Output × Temporal coupling.** `field` outputs almost always pair with `clip-parallel` or `autoregressive` (one-step-ahead surrogates: GraphCast does 6h autoregressive). `action-seq` pairs with `streaming-cache` or `latent-rollout`. This is descriptive, not prescriptive — but flag suspicious combos in audit (e.g., `field` + `streaming-cache` is rare; verify).

5. **Score-based / EBM conceptual leak.** Per LeCun JEPA position paper, JEPA is framed as a latent EBM. By the v1 ontology that would suggest `injection=energy-based`. But V-JEPA 2 has no physics in the energy structure — its physics is `data-only`. Resolution: tag based on the *physics mechanism*, not the loss family. Make this explicit in axis-2 docstring.

---

## Proposed v1.1 diff (concrete edit instructions)

Apply to `cheat-sheet/ontology.md`:

**Axis 1 — Output space**
- Remove row `mesh`.
- Rename row `3d-scene` → `3d-explicit`; update description: "explicit 3D (3DGS / mesh / point cloud)".
- Add row `3d-implicit` (SDF / NeRF / occupancy) — anchor DeepSDF 1901.05103.
- Rename row `latent` → `latent-tokens`; restrict description to non-decoded planning latents; remove "may be decoded" implication. Move Cosmos-Predict's anchor out of this row.
- Add row `motion` (skeletal / articulated body pose sequences) — anchor PhysDiff 2212.02500.

**Axis 2 — Physics injection**
- Rename `implicit-from-data` → `data-only`.
- Rename `constraint-loss` → `aux-loss` (keep PINN as anchor).
- Split `sim-in-loop` into `sim-in-loop-train` (Genesis 2412.18608) and `sim-in-loop-infer` (PhysDiff 2212.02500).
- Rename `score-conditioned` → `guidance-gradient`; broaden description to "physics gradient injected at training (PINN-style) or inference (classifier guidance, PhysDiff projection)".
- Demote `energy-based` to a footnote; remove as a primary value (tag with `hard-constraint` or `data-only` depending on dominant mechanism).
- Add row `architecture-bias-soft` — anchor NewtonGen 2509.21309.
- Rename `hard-PDE` → `hard-constraint`; broaden to include exact equivariance and exact conservation.
- Add explicit docstring: "tag by *physics mechanism*, not loss family. JEPA's EBM structure does not imply `hard-constraint`."

**Axis 3 — Controllability input**
- Remove `multi` (use `|` separator).
- Rename `image-prompt` → `image-init`; rename `3d-prompt` → `3d-init`; rename `physical-param` → `param`.
- Add `camera` (camera-pose trajectory) — anchor Cosmos 2501.03575.
- Add `layout` (structured BEV/scene-graph/road-graph) — anchor GAIA-2 2503.20523.

**Axis 4 — Temporal paradigm**
- Rename `joint-rollout` → `clip-parallel`.
- Rename `temporal-transformer-rolling` → `streaming-cache`.
- Document `hierarchical` as compositional (allowed to co-tag via `|`).

**Axis 5 — Domain coupling**
- Add optional `astro`.
- Document `fluid`/`weather` overlap explicitly (intentional split by community).
- Audit rule: `generalist` reserved for foundation video models (Cosmos / Sora / Veo); everything else must claim a specific domain.

**Audit (handbook_audit.py Check 9)**
- Update enum sets to match above.
- Add Check 9b: cross-axis sanity (Output × Injection compatibility matrix).
- Add Check 9c: `generalist` allowlist (Cosmos/Sora/Veo/Cosmos-Predict only by default).

**Header convention**
- Keep `|` for multi-value within an axis. Disallow `multi` keyword.
- Drop "TODO v1.1" section once applied; promote unresolved items to GitHub issues.

---

Word count ~1700. End of review.
