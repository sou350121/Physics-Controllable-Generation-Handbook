<!-- ontology-5axis output=latent-tokens injection=data-only control=action|image-init temporal=latent-rollout domain=robotics -->

# V-JEPA / V-JEPA 2 (with action conditioning)

> Meta FAIR · V-JEPA (Bardes et al., 2024-02) → V-JEPA 2 + V-JEPA 2-AC (Assran/Ballas/LeCun et al., 2025-06)。LeCun "latent prediction beats pixel reconstruction" thesis 的旗艦實證。

## 1. TL;DR

**Prior gap**：pixel-level video prediction（MAE-style, diffusion video, Cosmos-Predict）燒 compute 去重建 RGB 細節，而下游 agent control 真正需要的只是「物理上合理的下一步 latent」。LeCun 多年公開立場（"generative reconstruction is doomed"）認為應該在 representation 空間預測 representation，不要在 pixel 空間預測 pixel。

**V-JEPA (2024)** 是這條 thesis 的 video 版實作：mask 部分時空 patch，要求網路在 **encoder 輸出的 latent feature** 空間預測被遮住的 patch 之 feature（不是還原 pixel）。在 1M+ 小時 internet video 上做 self-supervised pretrain，得到 ViT-g/16 backbone。

**V-JEPA 2 (2025-06)** 是 scale-up：>1M 小時、ViT-g 級 encoder、在 Something-Something v2 motion understanding 拿到 77.3% top-1、Epic-Kitchens-100 action anticipation 39.7 recall-at-5。

**V-JEPA 2-AC** 是把 action conditioning 釘到 V-JEPA 2 frozen encoder 上的 300M 參數 transformer：只用 **62 小時 Droid robot video（unlabeled）** post-train，就能在 **沒見過的 Franka arm + 沒做 task-specific 訓練 + 純 image goal planning** 下做 pick-and-place（two labs，65–75% 級別）。這是 latent-WM 派第一次把 "pretrain on internet video + 少量機器人 video → zero-shot 真機" 跑通到能講故事的程度。

語意：V-JEPA 系列的價值不在 SOTA 數字，而在 **這條路徑成立**——latent 預測 + 少量 action 後訓 = 可用的 robot WM。

## 2. Core mechanism

### V-JEPA loss（無 action）

```
            video clip
               │
        ┌──────┴───────┐
        ▼              ▼
    context x_c     target x_t  (3D multi-block mask)
        │              │
   ┌────┴─────┐   ┌────┴─────┐
   │ Encoder  │   │ EMA-     │  ← stop-grad on target side
   │ E_θ      │   │ Encoder  │
   └────┬─────┘   │ E_ξ      │
        │         └────┬─────┘
        ▼              ▼
    z_c=E(x_c)     z_t=E(x_t)
        │              │
        ▼              │
   ┌─────────┐         │
   │Predictor│         │
   │ P_φ(z_c,│         │
   │  mask)  │         │
   └────┬────┘         │
        │              │
        ▼              ▼
        ŷ ──── L1 ──── z_t      ← loss in latent, NOT pixel
```

- Encoder：ViT-L 或 ViT-g/16（16×16 spatial patch + temporal patch）
- Target encoder：EMA copy，stop-gradient，避免 collapse（雙塔 self-distillation）
- Predictor：較小的 transformer，吃 context tokens + 被 mask 位置的 query
- Loss：predicted feature 與 EMA target feature 之間的 L1（重點：**不重建像素**）
- 3D multi-block masking：時空一起遮，比 random patch mask 更逼網路學動態

### V-JEPA 2-AC action conditioning

frozen V-JEPA 2 encoder（ViT-g）→ 16 frames @ 256×256, 4 fps → token grid。

action-conditioned predictor：
- 24 layers, 16 heads, hidden=1024 (~300M params)
- **block-causal attention**：t 時刻的 patch 只 attend 到 t 與 t-1, t-2, ... 的 token
- **3D RoPE**：temporal / H / W 三軸獨立 rotary positional encoding
- action / end-effector state / feature 各有獨立 learnable affine 投影到 hidden dim
- 訓練 loss：teacher-forced next-feature prediction + 2-step rollout loss（autoregressive 穩定性）

### Planning（推理時）

energy-based MPC：在 latent 空間 rollout 多個 action sequence，選 L1(imagined_z_t, goal_z) 最小的；Cross-Entropy Method（CEM）取樣；receding horizon，每一步 replan。每個動作約 **16 秒** wall-clock（Cosmos pixel diffusion 同等場景 ~4 分鐘）。

## 3. 五軸定位 + 同軸對手

| 軸 | V-JEPA 2(-AC) | [DreamerV4](./dreamer-v4.md) | [Genie-2](./genie-2.md) | [Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md) |
|---|---|---|---|---|
| Output | `latent` (action-free) / `latent` rollout (AC) | `latent` (RSSM) | `latent` + decoded pixel | `pixel-video` |
| Injection | `data-only` | `data-only` | `data-only` | `data-only` |
| Control | `image-init`(goal) + `action`(AC) | `action` | `action` (latent action token) | `text` + `image-init` |
| Temporal | `latent-rollout` (block-causal AR) | `latent-rollout` (RSSM step) | `streaming-cache` | `clip-parallel` (diffusion clip) |
| Domain | `generalist` → `robotics` (AC) | `generalist`/`robotics` | `generalist`(game) | `generalist`/`driving` |

**同軸對手細看**：
- **DreamerV4**：先做 RSSM (recurrent state-space) 而不是 transformer；action 是一等公民從頭就在訓練 loop 裡（不是 post-hoc）；V-JEPA-2-AC 賭的是「先大規模 unsupervised pretrain，action 後訓很便宜」這條 vs Dreamer「joint train」這條。
- **Genie-2**：用 latent action token 做 inverse dynamics 自己發明動作；V-JEPA-2-AC 則直接吃真實 7-DoF action vector，沒有 inverse dynamics 步驟。
- **Cosmos-Predict**：pixel diffusion 路線，視覺品質更高但 compute 重 ~15× 且需要 text；不是同一場戰爭，但常被 robotics 領域擺在一起比。詳見 `crossing/pixel-vs-latent-physics/`。

## 4. ⚡ shines / ❌ breaks

### ⚡

- **Sample efficiency on robot side**：62 小時 unlabeled robot video → zero-shot 兩個新實驗室的 Franka。對比 Octo（full Droid behavior clone，1000+ 小時）grasp 15% 而 V-JEPA-2-AC 65%。差距巨大但 caveat 多（見 §8）。
- **Inference 比 pixel WM 快一個量級**：16s/action vs Cosmos 4min/action（latent 空間 rollout 不解碼）。
- **Pretrain 可遷移**：同一 V-JEPA 2 encoder 同時拿到 SSv2 motion 77.3%、Epic-Kitchens action anticipation 39.7、video-QA 上 ~84/76（與 8B LLM 對齊後）。一條 backbone 三種能力。
- **沒有 pixel reconstruction loss**：避開 "畫面好看但物理錯" 的 pixel-WM 通病——loss 直接押在 latent dynamics 是否一致。

### ❌

- **JEPA 家族 mode collapse 史**：encoder/predictor 對偶很容易塌到 trivial constant solution（所有輸入映到同一 feature）。V-JEPA 用 EMA + stop-grad 緩解，但 **training loss 與下游 accuracy 不相關**，需要 RankMe / LiDAR / alpha-ReQ 等 proxy metric 來選 checkpoint。hyperparameter（EMA momentum schedule、stop-grad 位置）非常 brittle。VICReg variant 在有 static 背景時會塌到「最慢變的 feature」這條已知 global minimum。
- **Camera sensitivity (V-JEPA 2-AC)**：靠 monocular RGB 隱式推斷 action axes，需要人工擺相機位置。換 viewpoint 直接掉。
- **Autoregressive drift**：block-causal AR 預測 long-horizon 退化，所以 planning 只能短視距 + replan；不適合長 horizon task。
- **Goal 必須是圖**：current AC 版本不接 language goal，只能 image goal——對比 Octo / RT-2 一類 VLA 的 instruction following 是劣勢。
- **Pixel grounding loss**：latent rollout 看不見，debug 時要另外 decode；decode 自己會引入誤差，遇到 "rollout 對了但 decode 不像" 或 "rollout 已飄但 decode 還像" 很難辨。
- **OOD action / object**：62h Droid 後訓的分布相對窄，Droid 之外的 gripper 形狀 / object 物性沒有保證——pick cup 70%、pick box 30% 的差距已說明物件 OOD 敏感。

## 5. Reproduction

- **Weights**：`facebook/vjepa2-vitl-fpc64-256`, `facebook/vjepa2-vitg-fpc64-384`, `facebook/vjepa2-vitg-fpc64-384-ssv2` 等，HuggingFace `facebook/v-jepa-2` collection；transformers 4.4x+ 支援 `AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-384")`。
- **License**：主體 MIT（部分組件其他 license，需逐檔確認）。
- **Code**：`github.com/facebookresearch/vjepa2`（PyTorch）。
- **GPU 預算（推理）**：ViT-L 級單卡 24GB 可推，ViT-g 級需 40–80GB 級或多卡分片。
- **GPU 預算（pretrain）**：1M+ 小時 video 是 hyperscaler-only，重現不現實；可行重現範圍是 fine-tune / probe / 下游 eval。
- **V-JEPA 2-AC 復現**：需 Droid dataset（公開）+ frozen ViT-g + 自己訓 300M predictor。Droid 約 800h，paper 只用 62h，可控；但真機驗證需要 Franka + 鏡頭擺位 + CEM planner stack。
- **典型踩坑**：
  - 看 training loss 沒意義，靠 linear probe / k-NN / RankMe 選 checkpoint
  - EMA momentum schedule 抄錯會 collapse
  - 動作座標系與相機外參的隱式假設沒對齊 → 真機掉到隨機水平
  - tokenizer/patch size 與 fps 改動會破壞 RoPE 預設，downstream 全部要重 tune

## 6. Cross-line synthesis

- **vs pixel WM (Cosmos / Sora-style)**：核心對撞點，詳見 `crossing/pixel-vs-latent-physics/`。V-JEPA-2-AC 賭 "latent 空間夠用且便宜"，Cosmos 賭 "pixel 才能跨任務 transfer + 給人看 + 給 VLM 看"。兩條路線目前各自證明了一塊版圖：pixel 派 fidelity 高、language goal 好接；latent 派 sample/inference 效率高、與 robot planning loop 融合便宜。
- **× diff-sim**：latent WM 可以給 differentiable simulator（如 [MuJoCo MJX](../differentiable-simulators/mujoco-mjx.md) / [Genesis](../differentiable-simulators/genesis.md)）提供 representation prior（用 encoder 做 perception module），但 V-JEPA 系列目前沒這條接口；可組合方向見 `crossing/sim-vs-gen-data/`。
- **× VLA**：V-JEPA-2-AC 對應的是 VLA 中的 "world model 模組"（給 action 預測未來），不是 policy 本體。可作為 RT-2 / Octo / π0 一類 VLA 的輔助 critic 或 imagination module；目前公開資料中沒看到 production 級組合。
- **× surrogate ([FNO](../neural-surrogates/fno.md)/[GraphCast](../neural-surrogates/graphcast.md))**：兩條路線 domain coupling 不重疊（V-JEPA 是 robotics/generalist，surrogate 是 fluid/weather），唯一交點是 "latent dynamics + PDE residual 是否能 hybrid"——未見實證。

## 7. References

**Canonical**

- Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations from Video* (V-JEPA), arXiv **2404.08471**, 2024-02-15.
- Assran, Bardes, Ballas, LeCun et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv **2506.09985**, 2025-06-11.
- LeCun, *A Path Towards Autonomous Machine Intelligence* (position paper), OpenReview, 2022（JEPA 概念原始 position paper）。

**Secondary / community**

- HuggingFace `facebook/v-jepa-2` collection（weights + config 可下載）。
- `github.com/facebookresearch/vjepa2`（官方 code, MIT）。
- Sapunov, *V-JEPA 2: Scaling V-JEPA*, Gonzo ML newsletter, 2025（架構解讀）。
- *V-JEPA-2-AC: Video World Modeling for Robotics*, emergentmind.com（robotics 段落整理）。
- Connecting JEPA with Contrastive Self-supervised Learning, arXiv 2410.19560（理論連結 VICReg ↔ JEPA collapse）。
- Video Representation Learning with JEPA, arXiv 2412.10925（後續 video JEPA 變體比較）。
- *LeWorldModel: Stable End-to-End JEPA*, arXiv 2603.19312（收斂穩定性後續工作）。

## 8. §8 Pitfall log

| # | 議題 | severity | 來源 | workaround |
|---|---|---|---|---|
| 8.1 | JEPA mode collapse — encoder 與 predictor 退化到 constant feature | **high** | LeWorldModel paper, VICReg-JEPA 研究 | EMA + stop-grad（V-JEPA 預設）+ RankMe/LiDAR proxy 選 checkpoint；切勿用 train loss |
| 8.2 | Training loss 與下游 accuracy 不相關 | high | V-JEPA 2 paper acknowledged | 用 linear probe / k-NN / RankMe 驗證；多次 ckpt sweep |
| 8.3 | V-JEPA 2-AC camera sensitivity（隱式推 action axes） | high（真機） | V-JEPA 2 paper appendix | 人工擺相機；未來需 explicit camera token / 多視角訓 |
| 8.4 | Autoregressive drift over long horizon | medium | V-JEPA 2 paper limitations | receding-horizon replan；不要 plan 超過 2-step rollout 經驗範圍 |
| 8.5 | Object OOD（box 30% vs cup 70% grasp） | medium | V-JEPA 2 Table 2 | 擴充後訓 Droid hours；或加 task-specific small fine-tune |
| 8.6 | No language goal | medium | V-JEPA 2-AC scope | 暫接 image goal pipeline；等待 LLM-conditioned 後續變體 |
| 8.7 | VICReg-JEPA 在 static 背景下塌到「最慢變 feature」 | low-medium（V-JEPA 主路線用 EMA 而非 VICReg） | VICReg-JEPA collapse analysis | V-JEPA 主路線避開；若改 VICReg 變體要加 augmentation 打破 static distractor |
| 8.8 | Latent debug 困難（看不見） | low | 經驗 | 訓 decoder side-head 做 visualization；接受 decode error |

[TBD: verify V-JEPA 2 確切總參數量上限（paper 報告多 size，最大 ViT-g 級約 1B+ encoder，但 paper 內未集中給「總參數量」單一數字）]
[TBD: verify V-JEPA 2-AC 在 Cosmos baseline 上的「pick-and-place」精確分數對比表]
[TBD: verify HuggingFace weights 是否所有 checkpoint 都 MIT，或部分 robot-specific checkpoint 另有 license（paper repo README 提及「portions under separate terms」但未列舉）]
