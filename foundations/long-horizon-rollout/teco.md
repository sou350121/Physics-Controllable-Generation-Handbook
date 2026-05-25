<!-- ontology-5axis output=pixel-video|latent-tokens injection=data-only control=action|image-init temporal=hierarchical|latent-rollout domain=robotics -->

# TECO — Temporally Consistent Transformers for Video Generation

> Yan, Hafner, James, Abbeel. **ICML 2023** (PMLR 202: 39062–39098). arxiv [2210.02396](https://arxiv.org/abs/2210.02396). Code: [`wilson1yan/teco`](https://github.com/wilson1yan/teco) (JAX).

注：用戶 brief 標 ICLR 2023，實為 **ICML 2023**；arxiv v1 2022-10-05、v2 2023-05-31。原 title「Temporally Consistent Video Transformer for Long-Term Video Prediction」（v1）改為「Temporally Consistent Transformers for Video Generation」（v2 / ICML 版）。本 dissection 以 ICML camera-ready 版為準。

---

## 1. One-paragraph TL;DR

TECO 解的是 **長序列 video prediction 的雙重瓶頸**：(a) AR 在像素或 token 序列上 rollout 到 100+ 幀必然 drift / identity 漂移（物體離開視野再回來就變內容）；(b) joint clip-parallel 模型把長度寫死、跨 clip 銜接靠 image-init 不穩。TECO 的答案是 **hierarchical latent rollout**：先 VQ 把每幀壓成空間 token grid → 再壓一次（cross-frame spatial pool）成「temporally consistent embedding」→ temporal transformer 只在這個極稀疏的 high-level latent 上做長 horizon AR → 最後用 MaskGit 在 spatial 維度展開回 VQ token grid 並 decode 像素。結果是用單一模型在 DMLab / Minecraft / Habitat / Kinetics-600 上 **conditioned-on-36-frames、預測 300 幀** 並維持 identity 連貫。它是 **Cosmos Reason+Predict 兩層架構** 的學術祖先 — 雖然當時還沒「foundation WM」這個詞。

## 2. Core mechanism

三階段：**(i) per-frame VQ encode + cross-frame spatial compression → (ii) temporal transformer on compressed latent → (iii) MaskGit spatial expansion + VQ decode**。Loss 是標準 VQ-VAE reconstruction + temporal AR cross-entropy + MaskGit masked-token loss。沒有任何物理 prior、沒有 sim loop、沒有 force / contact channel — Axis 2 是純 `data-only`。

```
                 ┌────── HIGH LEVEL (low temporal rate, low spatial rate) ──────┐
                 │                                                              │
   action a_t ──►│   Temporal Transformer (AR over compressed latent z_t)       │
                 │       z_1 → z_2 → z_3 → ... → z_300                           │
                 │         ▲                                                    │
                 │         │ (compress: spatial pool over VQ grid)              │
                 └─────────┼────────────────────────────────────────────────────┘
                           │
                 ┌─────────┴───── LOW LEVEL (per-frame, full spatial grid) ────┐
                 │                                                              │
   x_t (frame) ──► VQ-VAE encoder ─► token grid e_t (H'×W' codes)               │
                                            ▲              │                    │
                                            │              ▼                    │
                                            │      MaskGit spatial prior        │
                                            │      (parallel iter, conditioned  │
                                            │       on z_t from high level)     │
                                            │              │                    │
                                            └───◄── VQ decoder ◄────── x̂_{t+1}  │
                 └──────────────────────────────────────────────────────────────┘
```

關鍵：**high-level temporal transformer 看不到 VQ token grid**，只看 compressed embedding；這讓 100+ 幀的 attention 可行（compute 從 O((HWT)²) 降到 O(T²) + per-frame MaskGit）。MaskGit 取代 raster-scan AR decode → 像素重建快數十倍。

## 3. 五軸定位 + 同軸對手

| Axis | TECO 值 | 註 |
|---|---|---|
| Output | `pixel-video` + `latent-tokens` | 最終 decode 像素，但 high-level z_t 也是合法 planning latent（與 Dreamer 同位） |
| Injection | `data-only` | 純資料學習，無物理 prior |
| Control | `action` + `image-init` | DMLab/Minecraft 帶動作；初始 36 幀作 conditioning |
| Temporal | `hierarchical` + `latent-rollout` | v2 ontology 允許 `|` compositional |
| Domain | `robotics` | 主 benchmark DMLab/Minecraft/Habitat 都是 agent 視角；Kinetics 是次要 |

同軸對手：

- **[DreamerV4](../latent-world-models/dreamer-v4.md)** — 同樣 latent-rollout，但 single-scale RSSM；TECO 多一層 spatial MaskGit decode。Dreamer 為 RL planning 服務，TECO 為 video prediction 服務 — 同個 latent 概念兩種下游。
- **[V-JEPA-2](../latent-world-models/v-jepa-2.md)** — Output 只到 `latent-tokens` 不 decode 像素（masked feature prediction），所以 TECO ⊃ V-JEPA 在 generation 端。Injection 都 `data-only`，差別在 V-JEPA 沒有 hierarchical temporal — 它把長 horizon 推給下游 reward / policy。
- **[Genie-2](../latent-world-models/genie-2.md)** — temporal=`streaming-cache`（sliding KV window）vs TECO 的 `hierarchical`。Genie 不壓時間，靠 cache 滾；TECO 不滾，靠壓。兩種都解 long-horizon 但機制正交：Genie sliding 對 unbounded length 友善但會「忘」，TECO hierarchical 對 fixed-length 300 幀內 identity 連貫但展不開到無窮長。
- **[Cosmos-WFM](../foundation-physics-models/cosmos-wfm.md)** — Cosmos 的 **Reason1 + Predict1** 兩層架構在精神上是 TECO 的 foundation-scale 升級：Reason1 = 高層慢時間 planner（接近 TECO 的 temporal transformer，但加了 chain-of-thought 與 sim-in-loop-infer），Predict1 = 細節 video diffusion renderer（接近 MaskGit spatial decode，但換成 latent diffusion）。**這是 TECO 在 2025 foundation WM 時代留下的最重要遺產**。

## 4. ⚡ Where it shines / ❌ where it breaks

### ⚡ Shines

- **300 幀 conditioned-on-36 的 identity 連貫**：DMLab 走出視野的牆面顏色再走回來仍然一致 — 這是 AR pixel transformer / SVD / FitVid 全部敗給 TECO 的關鍵 demo。
- **首個用一個 spec 同時 cover DMLab/Minecraft/Habitat/Kinetics 的 long video**：之前 CW-VAE / FitVid 多半 dataset-specific tuning。
- **Compute scaling**：把長 horizon 從「attention 二次方在 HWT 上」降到「在 T 上」— 學界第一次給出 hierarchical latent rollout 可 scale 的工程證據。後面 Cosmos / Genie / Decart 的 sliding-window 都建立在 TECO 給的證明上。
- **MaskGit 旁路 raster-scan**：parallel decode 比 VideoGPT 快數十倍，這在 2022 是大進展。

### ❌ Breaks

- **VQ-VAE 量化 artifact**：codebook 大小固定，紋理細節（草地、樹葉、Minecraft sand）容易閃爍；**codebook collapse** 在長 rollout 下是已知問題（少量 token 被反覆使用，多樣性塌陷）。
- **Planner-renderer drift**：high-level z_t 是抽象 embedding，MaskGit spatial decode 沒有明確 contract 保證「z_t 表示的高層狀態 ↔ 解出的 token grid」一一對應；當 step size 大（如 video 跳 5 幀）兩層會錯位、出現 hallucinated 物體。這是後來 Cosmos Reason+Predict 用 explicit chain-of-thought + sim eval 來補的洞。
- **Fixed-length 訓練**：300 幀寫死在訓練 pipeline；要展到 1000+ 幀沒有原生 streaming 機制（需要 sliding strategy 接補丁，與 Genie 思路混血）。
- **無物理 prior**：carbody collision / soft-body deformation / fluid 通通不保證 — 純資料學習 → 在 robotics 上接 VLA 評估時，contact dynamics 失效率高。這也是為什麼 V-JEPA-2 / Cosmos-Reason 後來加了 reward / sim-in-loop-infer。
- **JAX 鎖**：codebase 是 JAX + TPU 為主，PyTorch 社群 reproduce 需要自己搬（見 §5）。
- **VQGAN 預訓練不在 repo**：作者 README 明說「VQGAN training code is absent; users must convert checkpoints from the original PyTorch repository」— 重訓 from scratch 需要先解這個依賴。

## 5. Reproduction notes

- **Repo**: [`wilson1yan/teco`](https://github.com/wilson1yan/teco)（已驗證存在，BAIR open research commons, 2022-10-06 deposit；無顯式 LICENSE 檔，使用前自行確認授權）。
- **Framework**: JAX + Flax；CUDA 11.3 / cuDNN 或 TPU。
- **Dataset 規模**:
  - DMLab：40k trajectories, 300 frames @ 64×64（~54 GB）— 可單機 GPU 訓。
  - Minecraft：200k trajectories, 300 frames @ 128×128（~210 GB）— 需要 model-parallel。
  - Habitat：作者 README 標「coming soon」（截至 2023 中）— 重現 Habitat 結果有資料缺口風險。
  - Kinetics-600：用作評估，100 幀預測。
- **GPU 預算**:
  - DMLab / Minecraft 64×64 可在 4-8× A100 訓；
  - Habitat / Kinetics 128×128 在 ~32 TPU-v3 規模做（作者實驗），單機重現要 model-parallel + bf16。
- **典型踩坑**:
  1. **VQGAN checkpoint 缺失** — 必須去原 PyTorch VQGAN repo 抓 + 轉成 JAX numpy；轉換腳本不在 repo。
  2. **JAX pmap + Flax 版本鎖** — 用作者指定版本，新版 jax-flax 會在 sharding API 改名後爆炸。
  3. **MaskGit decode 步數** — paper 用 8 步，調小到 4 步會掉 FVD 但快 2×；可調 trade-off。
  4. **動作 conditioning 對齊** — DMLab/Minecraft 是 (s_t, a_t) → s_{t+1} 的 frame-aligned 慣例；換到非標準 dataset 時要小心 off-by-one。

## 6. Cross-line synthesis

跨四條技術路線 TECO 怎麼接：

- **× Pixel-WM（[Sora](../video-world-models/sora.md)）**：Sora 走 `clip-parallel` 一次生 cliplet 並用 spatial-temporal diffusion；長 horizon 靠 image-init 銜接。TECO 是 Sora 的「正交解」— 用 hierarchical 換 clip-parallel；2024-25 的 trend 是兩條合流（Cosmos-Predict 是 clip-parallel diffusion，但加 Cosmos-Reason 高層做 TECO 式 hierarchical planning）。
- **× Latent-WM（DreamerV4 / V-JEPA-2）**：TECO 的 high-level z_t 在概念上是 Dreamer 的 RSSM state、是 V-JEPA 的 embedding；差別是 TECO 把它 decode 回像素，DreamerV4 / V-JEPA 不 decode（後者直接給 policy）。Generation 端 ↔ Perception 端的 mirror。
- **× Diff-sim**：TECO 完全不接 — 沒有可微 sim、沒有 contact loss。如果要把 TECO 拿來做 contact-rich robotics，路線是「在 high-level z_t 上接 sim-in-loop-infer（PhysDiff 式 denoising loop in latent）」或「在 low-level VQ token 上接 ContactNets 殘差」— 兩條都還沒人做。
- **× Neural surrogate（[GraphCast](../neural-surrogates/graphcast.md)）**：完全異質 — surrogate 在 field 空間求 PDE，TECO 在 pixel/latent 空間做 generation。但 hierarchical 思想是共通的：GraphCast 的 6h autoregressive step 也是「高層慢時間」變體。

**最重要的歷史座標**：TECO（2022 末）→ Cosmos-Reason+Predict（2025）的 lineage 比 Sora→Cosmos-Predict 更深 — Cosmos 把 reasoning 與 prediction 分開兩個模型訓練、再 compose，這個 architectural commit 直接對應 TECO 的 temporal-transformer / MaskGit 分層。後人引用 TECO 不一定多，但架構血脈在。

## 7. References

- **Canonical**: Yan, Hafner, James, Abbeel. *Temporally Consistent Transformers for Video Generation*. ICML 2023, PMLR 202: 39062–39098. arxiv [2210.02396](https://arxiv.org/abs/2210.02396).
- **Code**: [`wilson1yan/teco`](https://github.com/wilson1yan/teco) — JAX/Flax；BAIR open research commons deposit 2022-10-06。
- **Project page**: [wilsonyan.com/teco](https://wilsonyan.com/teco/)（含 300-frame qualitative demos）；mirror [wilson1yan.github.io/teco](https://wilson1yan.github.io/teco/)。
- **PMLR**: [proceedings.mlr.press/v202/yan23b.html](https://proceedings.mlr.press/v202/yan23b.html)。
- **Danijar Hafner project page**: [danijar.com/project/teco/](https://danijar.com/project/teco/) — 補充 talk slides。
- **HuggingFace request**: [huggingface/transformers#27752](https://github.com/huggingface/transformers/issues/27752) — 社群要求 port TECO 到 PyTorch transformers（截至 2024 未合併，間接證明 reproduce 門檻）。
- **Baselines 比較**：CW-VAE（Saxena et al. NeurIPS 2021）· FitVid（Babaeizadeh et al. 2021）· Latent FDM（Harvey et al. 2022）· Perceiver AR（Hawthorne et al. ICML 2022）— 在 paper Table 1 全敗。

## 8. §8 Pitfall log

> 注：TECO repo 上次大規模更新在 2023 中，open issue 數少（截至 fetch 時 ~2 個），下列 pitfalls 主要來自 paper limitations、社群 reproduce 報告與通用 VQ-hierarchical 失敗模式；GitHub issue # 未對得上時用通用敘述。

- **§8.1 VQ codebook collapse on long rollout** — severity: **high**。長 horizon AR 在 VQ token 序列上會偏向少數高頻 code，視覺上表現為「紋理凍結」或同色塊週期出現。Workaround：擴大 codebook（4096 → 8192）+ commitment loss weight 調高 + EMA codebook update（Esser et al. VQGAN 標準藥方）。
- **§8.2 Planner-renderer drift @ large action step** — severity: **medium-high**。high-level z_t 與 MaskGit spatial token 不是 bijective；大跳步（每 step 對應 5+ 物理幀）時 MaskGit decode 出來的物體位置會與 z_t 想表達的「高層狀態」錯位。Workaround：縮小 step size、或在 z_t 加 explicit spatial anchor（後人 Cosmos-Reason 的 chain-of-thought 是進化版解法）。
- **§8.3 Fixed 300-frame training horizon** — severity: **medium**。展到 500+ 幀 inference 沒有 native streaming 機制；硬展會看到 identity 漂移回到 baseline 水平。Workaround：sliding-window inference（每 300 幀重新用最後 36 幀做 image-init），但會在 window 接縫產生 perceptual gap — 與 Genie sliding cache 相同的縫合問題。
- **§8.4 VQGAN dependency missing in repo** — severity: **medium**。reproduce 必須先取得 PyTorch VQGAN checkpoint 並轉換 — README 明寫但很多人重訓時卡這。Workaround：使用 CompVis/taming-transformers 預訓 checkpoint 並寫 numpy 轉換腳本。
- **§8.5 Habitat dataset link incomplete** — severity: **low-medium**。截至 v2 commit Habitat 資料連結標「coming soon」；Habitat scan 結果重現性受限。Workaround：用 HM3D / Replica 自行 render 替代，但 metric 無法對齊 paper Table。
- **§8.6 No physics prior at all** — severity: **inherent**。物理一致性靠 data；contact / collision / fluid 不保證。**Axis 2 cross-check（v2 Check 9b）**：本條 output=`pixel-video|latent-tokens` + injection=`data-only` 是矩陣中合法 cell，無需額外解釋；但**這也是 hierarchical 路線通病**——把長 horizon 解了，物理一致還沒解；Cosmos-Reason 後面用 sim-in-loop-infer 補。
- **§8.7 JAX/Flax 版本鎖** — severity: **low**。新版 JAX sharding API 與 repo 寫死的 pmap 慣用法不相容；2024 後 reproduce 需 pin 舊版（jax≤0.4.13 區間）。Workaround：建 conda env 鎖版本。
- **§8.8 Descriptive note (cross-axis)** — temporal=`hierarchical|latent-rollout` 是 v2 文件化的 compositional 用法；TECO 是該軸 canonical anchor（見 ontology.md Axis 4 表 TECO 2210.02396 entry）。後續 Cosmos-Reason+Predict 沿用此標法。

---

> **Bridge**：本篇 cross-link 進 [`overview.md` hierarchical 段](./overview.md#三類解法)，並作 Axis 4 `hierarchical` 的 anchor 之一（另一 anchor 為 Cosmos-Reason+Predict）。
