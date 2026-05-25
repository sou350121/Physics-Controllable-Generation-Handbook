<!-- ontology-5axis output=motion|3d-explicit injection=guidance-gradient|aux-loss control=contact|text temporal=clip-parallel domain=rigid|robotics -->

# ContactGen — Contact-Guided Interactive 3D Human Generation for Partners

> Ku, Oh, Yoo, Choi. **AAAI 2024**. arXiv [2401.17212](https://arxiv.org/abs/2401.17212). Project: [dongjunku.github.io/contactgen](https://dongjunku.github.io/contactgen/). Code: [dongjunKu/ContactGen](https://github.com/dongjunKu/ContactGen).

> **Naming disambiguation**：本篇取 AAAI 2024 版本（diffusion-based, human-human interaction），與同名 [ICCV 2023 ContactGen (Liu et al, 2310.03740)](https://arxiv.org/abs/2310.03740)（CVAE-based grasp generation）是兩篇獨立工作。AAAI'24 版採用 **guided diffusion + contact prediction module**，直接落在 `diffusion-physics` zone；ICCV'23 版是 conditional VAE + model-based optimization，屬 `physics-conditioning`/non-diffusion 範式。當文獻引用「ContactGen」並提及 partner / interaction / guided diffusion 時，預設指 AAAI'24；提及 grasp / object-centric / contact map+part map+direction map 時，指 ICCV'23。本倉 diffusion-physics 線只 anchor AAAI'24 版。

## 1. TL;DR

PhysDiff 用 simulator 當 inference-time projector 把 humanoid 動作拉回物理 manifold（→ [`../physics-conditioning/physdiff.md`](../physics-conditioning/physdiff.md)）— 但它解的是 **單一人體 vs 地面**這種「自我接觸 + 重力」的退化情形。當 task 變成兩個人互動（hug / push / dance / fight），單體投影完全不夠 — **兩個 SMPL 之間的接觸區域既要符合語義（hug 的接觸在背 + 肩、push 在手掌 + 胸）又要符合幾何（不穿插、力學閉合）**，而 simulator-projection 在這裡幫不上忙（imitator 不知道「該接觸哪裡」）。

ContactGen (AAAI'24) 的 insight：**把「該接觸哪裡」訓練成一個獨立模塊**，再用它的輸出當 guided diffusion 的 conditioning signal。具體分兩階段：先訓 ContactNet（給定 partner pose + interaction label，預測 self body 每個 vertex 的接觸機率分布）；再訓 ContactDiff（標準 motion / pose diffusion model），denoising 時把 ContactNet 預測的 contact region 當作 **classifier-style guidance**，每步微調 score。等效於 `injection=guidance-gradient + aux-loss` 雙標記 —— guidance 在 inference 走 score gradient，aux loss 在 training-time penalize predicted-vs-target contact mismatch。

對應 v2 ontology Axis 3 `control=contact` 的 **canonical anchor**：把「接觸」從輔助訊號升級為一級 controllability input，並且首次在 human-human 多體 setting 跑通 diffusion-time 注入。對於想做 robotics dual-arm / human-robot handover / multi-agent video generation 的工程師，ContactGen 是 contact-as-control 的最近 reference。

## 2. Core mechanism

兩階段架構，第一階段是 contact 預測器，第二階段是 contact-guided diffusion。

**Stage 1 — ContactNet**：輸入是 partner SMPL pose $p_{\text{partner}}$ 與 interaction label $\ell \in \{\text{hug, push, hold, lift, ...}\}$（CHI3D 8 類），輸出是 self body 上每個 SMPL vertex 的接觸機率 $c \in [0, 1]^{N_v}$（$N_v = 6890$ for SMPL）。這個模塊單獨用 BCE loss 訓在 CHI3D 標註的 ground-truth contact region 上 — 屬於 **aux-loss 訓練階段的物理偏好注入**。

**Stage 2 — ContactDiff**：標準 SMPL pose diffusion（denoise 在 axis-angle / 6D rotation 空間上），condition 走 cross-attention 進 partner pose + label。關鍵是 denoising 每一步 t 都附加一個 **contact guidance term**：

$$
\hat{x}_{t-1} = \text{denoise}(x_t, t) - \lambda_t \nabla_{x_t} \mathcal{L}_{\text{contact}}(x_t, c_{\text{target}})
$$

其中 $\mathcal{L}_{\text{contact}}$ 是「當前去噪 sample 對應的 SMPL vertex 是否落在 $c_{\text{target}}$ 高機率區」的可微距離（典型實作：SDF-based mesh distance + Chamfer between contact-positive vertices and partner surface）。$\lambda_t$ 是 guidance scale，隨 t schedule。

```
partner SMPL p_partner ──┐
                         ▼
   interaction label ℓ ─►ContactNet (Stage 1, BCE-trained)
                         │
                         ▼
                    c_target ∈ [0,1]^6890   (target contact prob per vertex)
                         │
                         ▼
   x_T (noise) ───►┌──────────────┐  ──► x̂_0     ┌─────────────────────┐
                   │ ContactDiff  │              │ contact loss        │
                   │ denoise(t)   │              │ L = ||vert(x̂_0) -   │
                   │ cross-attn   │              │   partner.surf||·c  │
                   │   p_partner  │              └────────┬────────────┘
                   │   ℓ          │                       │
                   └──────┬───────┘                       │
                          │       ◄──── λ_t · ∇_x L ──────┘
                          ▼
                       x_{t-1}
                       loop t=T→0
```

實作細節：guidance 從 t = T 用到 t = 0（與 PhysDiff 的 "End 4 Space 1" 末段策略相反 — ContactGen 全程注入，因為 contact 訊號比 simulator rollout 廉價，且早期 noise 階段也有 informative gradient 因為 SMPL vertex 距離在低頻空間平滑）。Repo 確認兩個獨立 training entry point：`train_diffusion.py` 訓 ContactDiff，`train_guidenet.py` 訓 ContactNet — 兩階段是 decoupled，**ContactNet 可以單獨 swap**（後續可換成 LLM-driven contact predictor，這是 follow-up 工作的開放方向）。

## 3. 五軸定位 + 同軸對手

| Axis | ContactGen (AAAI'24) |
|---|---|
| Output | `motion`（SMPL pose seq, 兩人）+ `3d-explicit`（rigged human mesh） |
| Injection | **`guidance-gradient`**（inference score guidance）+ **`aux-loss`**（ContactNet BCE pretraining） |
| Control | **`contact`**（target contact region，主控訊號）+ `text`/label（interaction class） |
| Temporal | `clip-parallel`（一次生成 pose / 短 clip；非 long-horizon） |
| Domain | `robotics` / `rigid`（剛體連桿 humanoid） |

**同軸對手**：

- **[PhysDiff](../physics-conditioning/physdiff.md)** — 同樣 motion diffusion + 物理注入，但 PhysDiff 用 **simulator rollout** 當投影器（`sim-in-loop-infer`），ContactGen 用 **contact prediction module** 當 score guidance（`guidance-gradient`）。PhysDiff 解單體 vs ground 的接觸 / 穿插 / floating 三類底層 artifact；ContactGen 解兩體之間的 semantic contact placement。**互補不替代**：理想的 dual-human 系統會疊兩條 — ContactNet 決定該接觸哪裡、UHC simulator 保證不穿插。AAAI'24 paper 沒做這個 combo（複現難度高 + compute 加倍），是 open extension。

- **[PhysGen](../physics-conditioning/physgen.md)** — rigid-body pipeline，static image → physics-grounded video。Injection 是 `sim-in-loop-train`，control 上沒有 contact axis。ContactGen vs PhysGen 的對比凸顯 `contact` control 在 motion 領域是一級訊號、在 video / rigid-body 領域目前還是隱式（PhysGen 從 image 推 mass / friction，但不直接 condition contact map）。

- **ContactNets** (Pfrommer et al, [arxiv 2009.11193](https://arxiv.org/abs/2009.11193), CoRL 2020) — v2 ontology Axis 3 `contact` 的 **第二 canonical anchor**（與 ContactGen 並列；列在 `hard-constraint` line）。ContactNets 不是生成模型，是 **dynamics learning**：用 implicit signed distance + 接觸 frame Jacobian 從 60 秒真實數據學 contact-induced 的不連續動力學，借 complementarity + maximum dissipation 原理當 loss。injection 是 `aux-loss + architecture-bias-soft`（implicit SDF 表示接觸不連續）。ContactGen vs ContactNets：**前者把 contact 當作生成端的 control input**，**後者把 contact 當作 dynamics 學習目標**。兩者代表「接觸」進入 ML pipeline 的兩種徹底不同入口，handbook 把它們同列於 Axis 3，因為都是 contact-first 範式的源頭。

- **ICCV'23 同名 ContactGen** (Liu et al, [2310.03740](https://arxiv.org/abs/2310.03740)) — grasp generation。架構是 CVAE（contact map → part map → direction map sequential VAE）+ MANO 模型轉 piecewise SDF + model-based optimization 從接觸表徵反推手部 pose。**完全不是 diffusion**，所以不在本 zone；如未來開 `physics-conditioning/grasp-modeling/` 子目錄會把它移過去。值得提的是 ICCV'23 ContactGen 的「contact + part + direction」三維度物件中心表徵比 AAAI'24 ContactGen 的單 vertex 機率更細，是後續 cross-fertilize 的候選。

- **CHOIS (Li 2023, [arxiv 2312.03913](https://arxiv.org/abs/2312.03913))** — 人物物互動生成（human-object interaction），同樣 diffusion-based，但 control 是 object trajectory + text 而非 contact。CHOIS 的 contact 是 emergent，ContactGen 的 contact 是 prescriptive，差異等於 PhysGen-style 隱式物理 vs PhysDiff-style 顯式物理。

## 4. ⚡ shines / ❌ breaks

⚡ **真正領先的 regime**：

- **Human-human interaction 是 ContactGen 設計的甜蜜點** — CHI3D 8 種互動類別（hug / kick / push / lift / hold / posing / handshake / dance）覆蓋的「對等多體接觸」，目前沒有更乾淨的 baseline。其他人 dyadic motion 工作（InterGen, InterHuman 等）多走 text-driven 路線，幾乎不顯式 condition contact region，semantic contact 經常擺錯地方（hug 變成手肘戳腰）。
- **Contact module 可獨立替換** — Stage 1 / Stage 2 decoupled 是這篇最 actionable 的工程價值。ContactNet 可以 swap 成更強的 contact predictor（LLM-grounded、video-derived、physics-simulated）而不重訓 ContactDiff；類比 ControlNet 之於 SD 的 condition adapter 思路。
- **CHI3D 上量化提升明確** — interpenetration volume 與 contact F1 同時改善；但注意 paper 主表是 CHI3D-only，未在 InterHuman 等更大 dataset 驗證泛化（→ §8）。

❌ **Known failure modes**：

- **Contact prediction 失準時 guidance 反向誤導** — diffusion-physics overview 的共通 pitfall 在 ContactGen 尤其嚴重，因為 contact 是主控訊號（不像 PhysDiff 的 simulator 至少能保證物理可行性，ContactNet 預測錯地方 ContactDiff 就把人推去錯地方接觸）。CHI3D 訓練分布外的互動（如 ballroom dance 或 wrestling）幾乎沒有 graceful fallback。
- **單 frame / 短 clip only，long-horizon dynamics 不在範圍** — paper 主要 evaluate 在 single pose 或 短 motion clip 上，沒做 minute-scale interaction trajectory；對「dance choreography」這類 1-2 分鐘級任務直接超綱。
- **Guidance scale $\lambda_t$ tuning 脆弱** — 過大採樣崩塌（人體扭曲為遷就接觸區），過小 contact constraint 失效。Repo 的 default 在 CHI3D 上調好，換 dataset 要重 grid search 5-10 個值 × 訓練 3-5 個 model 才能確定 Pareto。
- **CHI3D scale 太小** — 8 類 × ~370 sequences，總 motion frame 數 ~10⁵ 級別，相較於 HumanML3D / Motion-X 的 10⁶+ 小一個量級。ContactDiff 的 base diffusion 因此泛化能力受限，跨 dataset 表現未公佈。
- **沒整合 collision-avoidance hard constraint** — ContactGen guidance 鼓勵接觸區符合，但不主動排斥非接觸區的穿插。極端 interaction label（如 piggyback）容易導致 mesh interpenetration。需要再疊一層 SDF-based penalty 或事後做 simulator projection（這就是 PhysDiff + ContactGen 的合理 combo 方向）。
- **`output=motion + injection=guidance-gradient` 在 v2 Check 9b 是合法 cell**，但因為 motion 空間維度高（pose × frames），guidance gradient 的 noise 累積較 vertex-level 3D 更明顯，sampling 不穩定的根因之一。

## 5. Reproduction notes

**官方代碼狀態**：[dongjunKu/ContactGen](https://github.com/dongjunKu/ContactGen) 公開，含 `train_diffusion.py` / `train_guidenet.py` / `test_*.py` / `sample.py` 與 Google Drive 預訓練 checkpoint，docker image `dongjunku/humaninter:cu11`。README 沒列 GPU memory 與 wallclock — 從架構推估 ContactDiff（標準 SMPL pose UNet）+ ContactNet（vertex-level MLP / GCN）大致 12-16 GB VRAM 就能跑 CHI3D 規模 training，inference 單卡 4090 數秒級。

**典型踩坑**：

1. **CHI3D 取得需要申請** — 不是 public download，要簽 license，等批准週期 1-2 週。沒有 CHI3D 沒法完整重現，社群 fork 多半止步於 inference 預訓練 checkpoint。
2. **SMPL-X 與 SMPL 切換** — repo 文檔示意用 SMPL 但 CHI3D 標註是 SMPL-X，vertex 數對不上（6890 vs 10475）要對應改 ContactNet 輸出維度。容易在 BCE loss shape mismatch 階段卡住。
3. **Docker `cu11`** — CUDA 11.x docker image 在新 GPU（H100 / RTX 50 系）上會 driver mismatch；要自建 cu12 image，PyTorch 對應 2.1+，並 verify 所有 ops 在新 CUDA 下行為一致。
4. **ContactNet pretrain 收斂判斷** — BCE loss 容易在 imbalance（contact-positive vertex << contact-negative）下「假收斂」到 trivial 0 預測。要監控 contact F1 / IoU 而非 raw BCE。
5. **Guidance schedule** — 預設 constant $\lambda$ 在 CHI3D OK，換 dataset 後常需要 cosine schedule（早期低、中期高、末期低）來避免 sampling collapse。

## 6. Cross-line synthesis

**ContactGen 在 diffusion-physics zone 的角色**：與 PhysDiff 並列為 **physics injection 入口** 的兩個樣本。
- PhysDiff = `sim-in-loop-infer + guidance-gradient`，物理進入靠 simulator black box。
- ContactGen = `guidance-gradient + aux-loss`，物理進入靠**可微 contact predictor**。

可微 predictor 路線的優勢：**training-time 把物理偏好內化進 module**，inference 是純 forward + gradient，沒有不可微 sim。劣勢：predictor 自身能表達的「物理」上限就是訓練資料品質（CHI3D 標的接觸區），不像 simulator 是 closed-form 力學 — predictor 不會 generalize 到完全沒見過的 interaction 類別，simulator 至少能保證守恆律。

**對 video generation（Sora-line）的延伸**：Sora 系列已知缺乏接觸理解（foot sliding, object passing through hands），ContactGen 思路提示一個 partial fix：訓練一個 video-level ContactNet（給定兩個物體的 bounding region，預測接觸概率 heatmap），把 heatmap 當 ControlNet 條件餵 video diffusion。這條路線比 PhysDiff-video（需要 video-mesh 閉環 simulator）門檻低得多，且不需要可微 sim — 是 [`crossing/conservation-violation-atlas/`](../../crossing/conservation-violation-atlas/) 中 contact-related violation 子表的可行近期解。

**與 ContactNets（dynamics learning）的綜合**：
- ContactNets 學 **接觸後的 dynamics**（impulse / friction / stiction）。
- ContactGen 學 **該接觸哪裡**（spatial contact prior）。
- 完整 pipeline：ContactGen 決定接觸區 → 物理 simulator 或 ContactNets 推演接觸後動力學 → 結果回饋微調 ContactGen guidance。這個閉環目前沒人做，是 dyadic motion + manipulation 的 open frontier。

**Cross-axis 註記（v2 Check 9b/descriptive notes）**：
- `output=motion + injection=guidance-gradient` 在 Check 9b 表中是 ✓ cell。
- `control=contact + domain=rigid|robotics` 符合 descriptive note（contact 通常配 robotics/rigid），無 audit 警告。
- 同時標 `injection=aux-loss` 是因為 ContactNet 的 BCE pretraining 也屬物理偏好注入；嚴格說兩 stage 不同階段，但 v2 ontology 允許多 injection 共標。

## 7. References

**Canonical**：
- Ku, Oh, Yoo, Choi. *ContactGen: Contact-Guided Interactive 3D Human Generation for Partners*. **AAAI 2024**. arXiv [2401.17212](https://arxiv.org/abs/2401.17212). Project: [dongjunku.github.io/contactgen](https://dongjunku.github.io/contactgen/). Code: [dongjunKu/ContactGen](https://github.com/dongjunKu/ContactGen).

**Axis 3 `contact` 同列 anchor**：
- Pfrommer, Halm, Posa. *ContactNets: Learning Discontinuous Contact Dynamics with Smooth, Implicit Representations*. **CoRL 2020**. arXiv [2009.11193](https://arxiv.org/abs/2009.11193). DAIR Lab UPenn page: [dair.seas.upenn.edu](https://dair.seas.upenn.edu/assets/pdf/Pfrommer2020.pdf).

**Naming disambiguation 引用（非本 zone anchor）**：
- Liu, Zhou, Yang, Gupta, Wang. *ContactGen: Generative Contact Modeling for Grasp Generation*. **ICCV 2023**. arXiv [2310.03740](https://arxiv.org/abs/2310.03740). Project: [stevenlsw.github.io/contactgen](https://stevenlsw.github.io/contactgen/). Code: [stevenlsw/contactgen](https://github.com/stevenlsw/contactgen). CVAE + model-based optimization，非 diffusion。

**直接相關 / 對照組**：
- Yuan et al. *PhysDiff*. ICCV 2023. arXiv [2212.02500](https://arxiv.org/abs/2212.02500) — `sim-in-loop-infer` 對照組（本倉 [physdiff dissection](../physics-conditioning/physdiff.md)）。
- Tevet et al. *Human Motion Diffusion Model (MDM)*. ICLR 2023. arXiv [2209.14916](https://arxiv.org/abs/2209.14916) — base diffusion 思路源頭。
- Li et al. *CHOIS: Controllable Human-Object Interaction Synthesis*. arXiv [2312.03913](https://arxiv.org/abs/2312.03913) — emergent contact 對照組。

**CHI3D 資料集**：
- Fieraru et al. *CHI3D — Three-dimensional reconstruction of human interactions*. CVPR 2020.

## 8. §8 Pitfall log

1. **Wrong-paper trap**：搜尋「ContactGen」會同時拉到 ICCV'23 grasp paper 與 AAAI'24 interaction paper。下游引用者經常混用，導致 ablation 對照組張冠李戴。**Severity: high (research-correctness)**, workaround: 引用時必附 arxiv id（2401.17212 vs 2310.03740）。
2. **ContactNet 預測失準 → ContactDiff 偏離**：因 contact 是主控訊號（不像 PhysDiff 的 simulator 有物理 floor），ContactNet 預測錯位（如 push 互動把接觸區猜在背而非胸）會被 ContactDiff guidance 放大 — 生成結果整個錯位。**Severity: high**, workaround: 在 ContactNet 輸出加 confidence threshold gate，confidence < τ 時 fallback 到 raw diffusion sampling（paper 未做）。
3. **Guidance scale 雙刃**：$\lambda_t$ 過大 → 採樣 collapse（人體姿態被強拉到 contact-positive 區）；過小 → 接觸區散失 controllability。CHI3D default 在新 dataset 失效；需要每個新 setting 重 grid-search 5-10 個 $\lambda$。**Severity: medium**, workaround: cosine schedule（early low → mid high → late low），但需 ablation 驗證每 dataset。
4. **CHI3D 申請 + SMPL/SMPL-X 不一致**：dataset 取得卡 1-2 週 license；CHI3D 是 SMPL-X 但 code 部分 hard-code SMPL（vertex 數 6890 vs 10475），不改會在 BCE shape mismatch。**Severity: medium (engineering)**, workaround: fork 並改 ContactNet 輸出維度 + SMPL-X parametrization，需審慎驗證 contact map align。
5. **無 collision-avoidance hard constraint**：guidance 鼓勵接觸區符合，但不主動防 mesh interpenetration — piggyback / wrestling 等高重疊互動容易穿插。**Severity: high (visual quality)**, workaround: 後處理疊 SDF-based collision penalty 或接 PhysDiff-style simulator projection（複現難度倍增）。
6. **CHI3D scale 限制泛化**：~370 sequences × 8 類，比 HumanML3D 小一個量級。跨 dataset（InterHuman / dyadic InterX）未公開實驗，paper claim 不能直接遷移。**Severity: medium (generalization)**, workaround: 自己在 InterHuman pretrain ContactDiff，再 fine-tune ContactNet 到 CHI3D；訓練成本 5-10× original。
7. **Docker `cu11` 老舊**：repo 提供 `dongjunku/humaninter:cu11`，在 H100 / RTX 50 系新 GPU 上 CUDA driver mismatch 無法啟動。**Severity: low (engineering)**, workaround: 自建 cu12 image，PyTorch 2.1+，verify SMPL ops 與 contact loss 在新 CUDA 下數值一致。
8. **Conflation with ContactNets**：v2 ontology Axis 3 `contact` 同時 anchor ContactGen 與 ContactNets，但兩者是 generative-side vs dynamics-side 兩條完全不同 line。新人讀者經常以為「ContactGen 內部用 ContactNets」 — **不是**，兩者沒任何代碼/數據共享。**Severity: medium (educational)**, workaround: 本 dissection §3 / §7 顯式分離。

---

**Cross-refs**: [`./overview.md`](./overview.md) (zone overview) · [`../physics-conditioning/physdiff.md`](../physics-conditioning/physdiff.md) (sister motion-diffusion, `sim-in-loop-infer` 對照) · [`../physics-conditioning/physgen.md`](../physics-conditioning/physgen.md) (rigid-body pipeline 對照) · [`../physics-conditioning/force-prompting.md`](../physics-conditioning/force-prompting.md) (Axis 3 `force` 姊妹軸) · [`../../crossing/conservation-violation-atlas/`](../../crossing/conservation-violation-atlas/) (contact-related violations 子表).
