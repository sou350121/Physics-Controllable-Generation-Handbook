<!-- ontology-5axis
output: latent-tokens
injection: data-only
control: action | image-init
temporal: latent-rollout (block-causal AR)
domain: generalist → robotics
ref: ../../cheat-sheet/ontology.md §3 (latent-world-models)
-->

# V-JEPA 2 解構（V-JEPA 2 / V-JEPA 2-AC Dissection）

> **發布時間**：2025-06 · arXiv [2506.09985](https://arxiv.org/abs/2506.09985)（V-JEPA 2 / 2-AC）· 前作 arXiv [2404.08471](https://arxiv.org/abs/2404.08471)（V-JEPA, 2024-02）
> **論文**：*V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*
> **作者**：Assran, Bardes, Ballas, LeCun et al.（Meta FAIR）
> **核心定位**：LeCun 多年「latent prediction beats pixel reconstruction」thesis 的旗艦實證。V-JEPA 2 是 1M+ 小時 internet video 的 latent 預訓 backbone；V-JEPA 2-AC 把 **62 小時 Droid robot video** 釘到 frozen encoder 上，成為 latent-WM 派第一次跑通「pretrain on internet video + 少量 robot video → zero-shot 真機 Franka」的故事。

**Status:** v0.5 — 解構基於 paper 全文 + HuggingFace weights + 二手分析（Gonzo ML、emergentmind）。`UNVERIFIED` 標記待維護者升 v1 補。
**TL;DR:** ① 把 LeCun 2022 position paper（"generative reconstruction is doomed"）grounded 到能控真機的程度；② 核心 trick 是 **block-causal attention + 3D RoPE** 讓 latent prediction 真的吃到 video 時序結構，而不是退化成 still-frame embedding；③ 對 Physics-Gen 讀者意義 ★：這是 latent-WM 派與 pixel-WM 派（Cosmos/Sora）對撞點上**少數真機落地的證據**；④ 關鍵數字：Octo（full Droid behavior clone, 1000+h）grasp 15% vs V-JEPA-2-AC（62h, zero-shot）65–75%，inference 16s/action vs Cosmos 4min/action（~15× 快）。

**X-Ray.** LeCun 2022 的 position paper 是一張**架構上的賭注書**——他賭 representation 空間預測 representation 才是路徑，pixel 重建是 dead-end。但 position paper 沒有可用代碼、沒有 robot demo，整個 latent-WM 派（DreamerV1-3、IRIS、TWM 等）一直被 pixel diffusion 派（Cosmos、Sora、Genie）壓著打——「你們做的東西沒人看得到，怎麼證明 latent 學到了物理？」V-JEPA 2 是把這條 thesis grounded 到 action 的關鍵一步：encoder 不是只給 video-QA 跑分，而是被當作 frozen perception module，接 300M action-conditioned predictor + CEM planner，在兩個未見過的實驗室、未見過的 Franka 上做 pick-and-place。但這條路徑的 **OOD ceiling 是 closed action vocabulary**——62h Droid 後訓的 gripper / object / camera 分布之外，整個 stack 沒有保證；它不是 generalist WM，是「Droid-shaped WM」。對 Physics-Gen handbook 讀者：這篇該寫 anchor，因為它是 [`crossing/pixel-vs-latent-physics/`](../../crossing/pixel-vs-latent-physics/overview.md) wedge 上 latent 派**唯一一篇有真機 demo 撐場的**——其他全是 video benchmark 跑分。

## 📍 研究全景時間線

```ascii
   2022                  2024-02            2025-06              2026?
   LeCun JEPA ─────────► V-JEPA ──────────► YOU ARE HERE ──────► action+language
   position paper        feature pred        V-JEPA 2 / 2-AC      joint world model
   (no code,             on video            1M+ h pretrain       (LLM-conditioned
    no demo)             ViT-g/16            + 62h Droid AC       AC, multi-cam,
                         SSv2 72.2%          + Franka demo ★      open vocab action)
                         K400 81.9%
   ────────────────────► thesis-only ─────► grounded to action ─► open action space
                          (latent-WM 派      "latent 夠用"        (still missing)
                           被 pixel-WM 派
                           壓著打)
```

★ = 主要新點：把 LeCun 2022 thesis **第一次跑出真機 zero-shot demo**（兩個 lab、未見過 Franka、pure image goal）。**仍未解：closed action vocabulary、no language goal、camera 敏感**——下一代要做。

---

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs V-JEPA (2024) / 同軸對手

| 維度 | V-JEPA (2024-02) | **V-JEPA 2 / 2-AC (2025-06)** |
|---|---|---|
| **Pretrain scale** | ~2M video clips | **1M+ 小時 internet video（>10×）** |
| **Encoder** | ViT-L/16 | **ViT-g/16, fpc64-384**（~1B+ params） |
| **Action conditioning** | ❌ 無 | ✅ **300M block-causal predictor（V-JEPA 2-AC）** |
| **Positional encoding** | learned spatial + temporal | **3D RoPE（temporal / H / W 三軸獨立 rotary）** |
| **Temporal attention** | full self-attn | **block-causal**：t 只 attend ≤ t（autoregressive in latent） |
| **真機 demo** | ❌ video benchmark only | ✅ **2 個 lab × Franka × zero-shot pick-and-place** |
| **Robot post-train cost** | n/a | **62 小時 Droid unlabeled video** |
| **SSv2 motion top-1** | ~72% | **77.3%** |
| **K400 top-1** | 81.9% (probe) | 維持/略升 |

### 1.2 ⚡ Eureka Moment

> **block-causal attention + 3D RoPE = 把 latent prediction grounded 到 video temporal structure** —— 不是 still-frame embedding 拼接，而是強制 t 時刻 token 只能看 ≤ t 的歷史，配 3D RoPE 把 (時間, H, W) 三軸位置編碼分開。**結果**：predictor 學到的是「下一個 latent 怎麼從歷史 latent 演化」，而不是「這幀畫面的全局摘要」。

這是 V-JEPA 1 → 2 的真正分水嶺。V-JEPA 1 的 multi-block mask 已經試圖逼網路學動態，但 full self-attention 讓網路偷吃「未來 patch leak」。block-causal 是 inverse dynamics 派（Dreamer / IRIS）老 trick 換到 ViT scale 上跑——簡單但 critical：**沒有 block-causal，V-JEPA 2-AC 的 AR rollout 直接退化成隨機猜**。

### 1.3 信息流（架構圖）

```ascii
        V-JEPA (2024)                          V-JEPA 2-AC (2025-06)
   ──────────────────────────              ──────────────────────────────
                                                                          
   x_c ───► E_θ ───► z_c ──┐               x_{0..t} ──► E_ξ (frozen ViT-g)
                            │                              │              
   x_t ───► E_ξ ───► z_t ◄──┼───◄ L1        ▼ token grid (16×16×T)        
        (EMA, stop-grad)    │              ┌──────────────────┐           
                            │              │  Block-causal AR │           
                       ┌────┴───┐           │  predictor       │           
                       │Predictor│          │  24 layers, 16 h │           
                       │ P_φ    │          │  hidden=1024     │           
                       │(no action)         │  3D RoPE         │           
                       └────────┘          │  ← action_t      │           
                                            │  ← ee_state_t    │           
                                            └────────┬─────────┘           
                                                     ▼                    
                                              ẑ_{t+1}, ẑ_{t+2}  (rollout) 
                                                     │                    
                                                     ▼                    
                                            ┌──────────────────┐           
                                            │  CEM planner     │           
                                            │  L1(ẑ_T, z_goal) │ ← image goal
                                            │  16s/action      │           
                                            └──────────────────┘           
                                                                          
        Loss in latent (L1 vs EMA)             teacher-forced + 2-step    
        No action, no plan                     rollout loss               
```

對比 [`dreamer-v4`](./dreamer-v4.md): Dreamer 用 RSSM（recurrent state-space）+ joint train action，V-JEPA 2-AC 賭「先 unsupervised pretrain encoder，action 後訓便宜」這條路。

對比 [`genie-2`](./genie-2.md): Genie 用 latent action token + inverse dynamics 自己發明動作；V-JEPA 2-AC 直接吃真實 7-DoF action vector（closed vocabulary）。

對比 [`cosmos-wfm`](../foundation-physics-models/cosmos-wfm.md): Cosmos 在 pixel 空間 diffusion rollout，視覺品質高但 compute ~15× 重，且 4min/action 不能進控制環。

---

## §2 · 數學層

### 📌 Napkin Formula

```
   V-JEPA loss (no action):
   
      L_jepa  =  || P_φ( E_θ(x_c), mask )  −  sg[ E_ξ(x_t) ] ||_1
                       ▲                          ▲
                       │                          └─ EMA target (no grad)
                       └─ predictor in latent, NOT pixel
   
   V-JEPA 2-AC (action-conditioned):
   
      ẑ_{t+1}  =  P_φ( z_{≤t},  a_t,  s_t^{ee} )       ← block-causal AR
      
      L_ac    =  || ẑ_{t+1}  −  sg[ E_ξ(x_{t+1}) ] ||_1
                  +  λ_2 · || ẑ_{t+2}  −  sg[ E_ξ(x_{t+2}) ] ||_1   ← 2-step rollout
   
   Planning (CEM, latent MPC):
   
      a*_{0..H}  =  argmin_{a_{0..H}}  || ẑ_H  −  z_goal ||_1
   
   Sample efficiency:
      pixel reconstruct:  L_pix = || decode(ẑ) − x_pix ||  ← burns FLOPs on RGB detail
      latent feature:     L_lat = || ẑ − z_target ||      ← only physics-relevant signal
      → ~15× cheaper inference (16s vs 4min per action vs Cosmos)
```

**直覺**：pixel reconstruction loss 把網路逼去**畫對每個 pixel**——大部分 capacity 浪費在背景紋理、光照、皮膚細節。latent prediction loss 只懲罰**representation 不一致**——網路被允許忘記「桌布顏色」但必須記得「手臂下一刻在哪」。對 robot control 這正是 signal/noise 比最好的訊號。**代價**：latent 看不見，debug 要訓 side-decoder（§6.2）。

### 2.x Loss / 訓練細節

- **EMA momentum schedule**：target encoder E_ξ 是 E_θ 的 EMA copy，momentum 從 0.996 → 1.0 cosine。抄錯 → collapse。
- **3D multi-block masking**：時空一起遮，比 random patch mask 更逼網路學動態（V-JEPA 1 已驗證）。
- **2-step rollout loss**：teacher-forced 單步 + free-running 2-step；不能超過 2 步，否則 AR drift 把 gradient 弄飛。
- **無 VICReg 變體**：V-JEPA 主路線靠 EMA + stop-grad 防 collapse，**不**走 VICReg（VICReg-JEPA 在 static 背景下會塌到「最慢變 feature」，§6.2 第 7 條）。

---

## §3 · 數據層 / 訓練 scale

| 階段 | 數據 | 規模 | 標註 |
|---|---|---|---|
| V-JEPA 2 pretrain | internet video（YT/HowTo100M/Ego4D 等混合）| **1M+ 小時** | 無（self-supervised）|
| V-JEPA 2-AC post-train | **Droid** robot video | **62 小時**（Droid 全集 ~800h 的 ~8%）| **unlabeled**（無 task label，只有 action + image）|
| Eval（understanding）| SSv2, K400, Epic-Kitchens-100 | benchmark | 標準 |
| Eval（planning）| 2 labs × Franka × pick-and-place | **zero-shot** | image goal only |

**對比 Octo**：Octo full behavior clone Droid 1000+ 小時 → grasp 15%；V-JEPA 2-AC 62 小時 zero-shot → grasp 65–75%。差距巨大但 **caveat 多**——Octo 跑的是 instruction-conditioned 多任務 vs V-JEPA 2-AC 只跑 image-goal pick-and-place；不是 apples-to-apples。但 sample efficiency 的數量級對比真實。

**訓練 scale 的押注**：1M+ 小時 pretrain 是 hyperscaler-only，重現不現實——這條路徑的**重現門檻被 Meta 鎖死**，社群只能 fine-tune / probe（§4）。

---

## §4 · 代碼層

| 項 | 狀態 |
|---|---|
| Repo | [github.com/facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) |
| Checkpoint | `facebook/vjepa2-vitl-fpc64-256`, `facebook/vjepa2-vitg-fpc64-384`, `facebook/vjepa2-vitg-fpc64-384-ssv2`（HF collection `facebook/v-jepa-2`）|
| License | 主體 MIT（部分組件其他 license，需逐檔確認，`UNVERIFIED`）|
| Inference GPU | ViT-L 級 24GB 單卡；ViT-g 級需 40–80GB 或多卡分片 |
| Pretrain GPU | hyperscaler-only（1M+ 小時 video, ViT-g）|
| Streaming | ❌（CEM planner 是 receding-horizon batch replan）|
| Metric scale | ❌（latent 空間 rollout，無 metric depth/pose 輸出）|
| Transformers 支援 | 4.4x+ 可 `AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-384")` |

**典型踩坑**：
- 看 training loss 沒意義，靠 linear probe / k-NN / RankMe 選 checkpoint（§6.2）
- EMA momentum schedule 抄錯會 collapse
- 動作座標系 ↔ 相機外參的隱式假設沒對齊 → 真機掉到隨機水平
- tokenizer/patch size 與 fps 改動會破壞 3D RoPE 預設，downstream 全部要重 tune

---

## §5 · 評測 / Benchmark

| Benchmark | Metric | 前 SOTA / V-JEPA 1 | **V-JEPA 2 / 2-AC** | Δ |
|---|---|---|---|---|
| **K400** | top-1 (linear probe) | 81.9% (V-JEPA 1) | 維持/略升 `UNVERIFIED` | — |
| **SSv2** | top-1 motion | 72.2% (V-JEPA 1) | **77.3%** | +5.1 pp |
| **Epic-Kitchens-100** | action anticipation recall@5 | baseline | **39.7** | new SOTA |
| **Video-QA**（with 8B LLM）| accuracy | — | ~84/76 | new |
| **Franka pick-and-place**（zero-shot）| success | Octo 15% (full Droid) | **65–75%**（two labs）| ~50pp 但 caveat 多 |
| **Inference latency**（per action）| wall-clock | Cosmos ~4 min | **~16s** | ~15× |

**解讀**：SSv2 +5.1pp 是真 capability（motion understanding 是 JEPA 派強項，block-causal + 3D RoPE 直接吃這條）。**Franka 65–75% 不能拿來跟 Octo 直接比**——任務範圍窄（pick-and-place only）、image goal only（不需 language understanding）、camera 是人工擺位（§6.2 第 3 條）。但**對 latent-WM 派的歷史意義巨大**：在這之前，整個 latent 派沒有任何「兩個 lab + 未見 robot + zero-shot」的證據。Sample efficiency 對比是真的，但不是 benchmark Goodhart——是「latent 路線終於有 demo 撐場」的政治意義。

---

## §6 · Issues & Limitations

### 6.1 論文自述 limitations

- **Camera sensitivity（V-JEPA 2-AC）**：靠 monocular RGB 隱式推斷 action axes，需要人工擺相機；換 viewpoint 直接掉。
- **Autoregressive drift**：block-causal AR 長 horizon 退化；planning 只能短視距 + replan，不適合 long-horizon task。
- **Object OOD**：62h Droid 後訓分布窄，pick cup 70% vs pick box 30%。
- **No language goal**：current AC 只接 image goal，instruction following 是劣勢 vs Octo / RT-2。
- **Training loss 與 downstream accuracy 不相關**：需 RankMe / LiDAR / linear probe 選 checkpoint。

### 6.2 Hidden Assumptions（隱含假設）

1. **JEPA mode collapse 派系遺傳病**（high severity）—— encoder/predictor 對偶易塌到 trivial constant solution。EMA + stop-grad 緩解但 brittle。VICReg variant 在 static 背景下會塌到「最慢變 feature」這條已知 global minimum（見 arXiv 2410.19560、LeWorldModel arXiv 2603.19312）。
2. **Closed action vocabulary**（high）—— 7-DoF Franka action vector 是 V-JEPA 2-AC 的硬上限；換 gripper / dexterous hand / mobile base，整個 action affine projection 重訓。
3. **Pixel grounding loss**（medium）—— latent rollout 看不見，debug 要訓 decoder side-head；「rollout 對了但 decode 不像」vs「rollout 已飄但 decode 還像」很難辨。
4. **Frozen encoder 假設**（medium）—— V-JEPA 2-AC 把 ViT-g 凍住只訓 predictor；如果 internet pretrain 缺某類 robot scene（特殊光照、wet object、deformable），predictor 救不回來。
5. **Image goal 是 well-posed 假設**（low-medium）—— current scope 假設 goal image 與 current scene 同視角同光照；換 lighting / mild distractor 表現未測。

### 6.3 §8 Pitfall log（保留原表，collapse-related citations 完整）

| # | 議題 | severity | 來源 | workaround |
|---|---|---|---|---|
| 8.1 | JEPA mode collapse — encoder/predictor 退化到 constant feature | **high** | LeWorldModel paper [arXiv:2603.19312](https://arxiv.org/abs/2603.19312), VICReg-JEPA [arXiv:2410.19560](https://arxiv.org/abs/2410.19560) | EMA + stop-grad（V-JEPA 預設）+ RankMe/LiDAR proxy 選 checkpoint；切勿用 train loss |
| 8.2 | Training loss 與下游 accuracy 不相關 | high | V-JEPA 2 paper acknowledged | linear probe / k-NN / RankMe 驗證；多次 ckpt sweep |
| 8.3 | V-JEPA 2-AC camera sensitivity（隱式推 action axes）| high（真機）| V-JEPA 2 paper appendix | 人工擺相機；未來需 explicit camera token / 多視角訓 |
| 8.4 | Autoregressive drift over long horizon | medium | V-JEPA 2 paper limitations | receding-horizon replan；不要 plan 超過 2-step rollout 經驗範圍 |
| 8.5 | Object OOD（box 30% vs cup 70% grasp）| medium | V-JEPA 2 Table 2 | 擴充後訓 Droid hours；或加 task-specific small fine-tune |
| 8.6 | No language goal | medium | V-JEPA 2-AC scope | 暫接 image goal pipeline；等待 LLM-conditioned 後續變體 |
| 8.7 | VICReg-JEPA 在 static 背景下塌到「最慢變 feature」| low-medium（V-JEPA 主路線用 EMA 而非 VICReg）| [arXiv:2410.19560](https://arxiv.org/abs/2410.19560) | V-JEPA 主路線避開；若改 VICReg 變體要加 augmentation 打破 static distractor |
| 8.8 | Latent debug 困難（看不見）| low | 經驗 | 訓 decoder side-head 做 visualization；接受 decode error |

### 6.x GitHub-validated 失敗模式

`UNVERIFIED` — `facebookresearch/vjepa2` repo issue tracker 待維護者掃 atlas 升 v1。Meta FAIR 開源節奏一貫：code release 但 issue 不答；預期類似 [`vggt-omega`](https://github.com/facebookresearch/vggt-omega) 模式（15 open / 0 closed 比例）。

---

## §7 · 比較 & 面試 Tip

| 同軸對手 | Output (Axis 1) | Injection (Axis 2) | Control (Axis 3) | Temporal (Axis 4) | Streaming | Open? | Status |
|---|---|---|---|---|---|---|---|
| **V-JEPA 2-AC** | latent | data-only | action + image-init | block-causal AR | ❌ | ✅ MIT | shipped 2025-06 |
| [DreamerV4](./dreamer-v4.md) | latent (RSSM) | data-only | action | RSSM step | ✅ (online) | ✅ | shipped |
| [Genie-2](./genie-2.md) | latent + pixel | data-only | latent action token | streaming-cache | ✅ | ❌ closed | demo only |
| [Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md) | pixel-video | data-only | text + image-init | clip-parallel diffusion | ❌ | ✅ OpenRAIL++ | shipped |
| Octo (VLA) | action | n/a (policy) | language + image | action chunk | ✅ | ✅ | shipped |

> **🎤 Interview Tip.** 「我們要在 latent (V-JEPA) 還是 pixel (Cosmos/Sora) 訓 robotics WM？」**正確答**：「**先問你的下游 loop 是 control 還是 data generation**。control loop（VLA imagination / MPC critic）→ latent 派贏，V-JEPA-2-AC 已經有 zero-shot Franka demo + 16s/action 進得了 receding-horizon；data generation（sim2real video augmentation / VLM training data）→ pixel 派贏，Cosmos 4min/action 可接受、視覺品質 + language goal + 給 VLM 看是 latent 派目前沒有的。**這不是同一場戰爭**——latent 是 perception module，pixel 是 data factory。」**錯答**：「latent 比較省 compute 所以全面贏」——錯，因為你會丟掉 language conditioning / 給 VLM 訓練 / 給人 debug 三條 affordance；或「pixel 比較真實所以全面贏」——錯，因為你 4min/action 進不了 control 環、且 pixel-WM 對 "畫面好看但物理錯" 沒有結構性防線。

### 7.1 Falsifiable predictions

連到 [`crossing/pixel-vs-latent-physics/`](../../crossing/pixel-vs-latent-physics/overview.md)：

1. **2026-12 前**：第一篇「V-JEPA encoder + LLM action head」（language-conditioned AC）會出現——open action vocabulary + image goal → language goal 是下一步必爭。
2. **2027-06 前**：第一篇「latent WM × diff-sim hybrid」會把 V-JEPA-class encoder 當 perception module，接 [MuJoCo MJX](../differentiable-simulators/mujoco-mjx.md) / [Genesis](../differentiable-simulators/genesis.md) 的 dynamics residual——latent 學 perception、diff-sim 學 physics、兩邊 gradient 互相 regularize。
3. **2027-12 前不會發生**：V-JEPA 系列在 dexterous manipulation（5 指 hand）或 deformable / fluid 任務上達到 Franka pick-and-place 同等成功率——closed action vocabulary + frozen encoder + 62h Droid 分布外 prior 完全缺失。要打開這 envelope 需要的不是 scale，是 architecture 改動。

---

## §8 · For the Reader（按 persona 分流）

- **VLA / robot policy 工程師** —— V-JEPA 2-AC 是 latent imagination module 的最強 baseline。如果你在做 RT-2 / Octo / π0 一類 instruction-conditioned policy，**把 frozen ViT-g encoder 當 visual backbone 試試**；imagination 用它做 short-horizon critic，policy 主幹仍是 language-conditioned。**不要**直接把 V-JEPA 2-AC 當 policy——它沒 language。
- **自駕 closed-loop 工程師** —— pretrain 沒看過 driving distribution；想做 latent driving WM 等下一代或自己 fine-tune Ego4D 類 driving subset。短期繼續用 Cosmos-Predict / domain-specific driving WM。
- **影片生成工程師** —— **不要用 V-JEPA**——它 by-design 沒 decoder，輸出不是給人看的。你要的是 [Cosmos](../foundation-physics-models/cosmos-wfm.md) / Sora 那條路。但**讀這篇可以幫你理解**為什麼 pixel WM 燒這麼多 compute——大部分 capacity 浪費在 RGB detail，這是 V-JEPA 派攻擊 pixel 派的核心論點。
- **神經 PDE / surrogate 研究者** —— domain coupling 不重疊（V-JEPA robotics/generalist vs surrogate fluid/weather），但 **latent dynamics + PDE residual hybrid** 是開放 wedge——目前無實證，見 §7.1 預測 2。
- **物理 conditioning 研究者** —— V-JEPA 2-AC 是「action conditioning 釘在 frozen encoder 上」的範本。如果你想做 contact force conditioning / language conditioning / multi-modal conditioning，**架構模板可抄**（小 predictor + affine projection + block-causal AR）。`UNVERIFIED` 是否能擴到 non-action conditioning，待後續工作。
- **Research 學生** —— **必讀** §6 的 collapse 文獻（[2410.19560](https://arxiv.org/abs/2410.19560), [2603.19312](https://arxiv.org/abs/2603.19312)）+ LeCun 2022 position paper。如果你想做 latent-WM 方向，**collapse 是入場費**——你看 train loss 沒意義這件事，比 hyperparameter 重要 10 倍。

---

## References

**Canonical**

- **V-JEPA 2** — Assran, Bardes, Ballas, LeCun et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv [2506.09985](https://arxiv.org/abs/2506.09985), 2025-06.
- **V-JEPA** — Bardes et al. *Revisiting Feature Prediction for Learning Visual Representations from Video*, arXiv [2404.08471](https://arxiv.org/abs/2404.08471), 2024-02.
- **LeCun JEPA position paper** — *A Path Towards Autonomous Machine Intelligence*, OpenReview, 2022（JEPA 概念原始 thesis）。

**Collapse / theory（§6.3 引用）**

- *Connecting JEPA with Contrastive Self-supervised Learning*, arXiv [2410.19560](https://arxiv.org/abs/2410.19560)（VICReg ↔ JEPA collapse 理論連結）。
- *Video Representation Learning with JEPA*, arXiv [2412.10925](https://arxiv.org/abs/2412.10925)（後續 video JEPA 變體比較）。
- *LeWorldModel: Stable End-to-End JEPA*, arXiv [2603.19312](https://arxiv.org/abs/2603.19312)（收斂穩定性後續工作）。

**Secondary / community**

- HuggingFace [`facebook/v-jepa-2`](https://huggingface.co/collections/facebook/v-jepa-2) collection（weights + config）。
- [`github.com/facebookresearch/vjepa2`](https://github.com/facebookresearch/vjepa2)（官方 code, MIT）。
- Sapunov, *V-JEPA 2: Scaling V-JEPA*, Gonzo ML newsletter, 2025（架構解讀）。
- *V-JEPA-2-AC: Video World Modeling for Robotics*, emergentmind.com（robotics 段落整理）。

---

## Boundary

- 與 [DreamerV4](./dreamer-v4.md) 的 RSSM joint-train vs V-JEPA pretrain-then-AC 對比 → 本檔 §1.3
- 與 [Genie-2](./genie-2.md) 的 latent action token vs explicit 7-DoF action 對比 → 本檔 §7
- 與 [Cosmos-WFM](../foundation-physics-models/cosmos-wfm.md) 的 pixel diffusion 對撞 → [`crossing/pixel-vs-latent-physics/`](../../crossing/pixel-vs-latent-physics/overview.md)
- 與 VLA imagination module 接口 → [`bridge-to-vla/world-model-as-policy.md`](../../bridge-to-vla/world-model-as-policy.md)
- 與 video pretraining for action 接口 → [`bridge-to-vla/video-pretraining-for-action.md`](../../bridge-to-vla/video-pretraining-for-action.md)
- 與 5 axis 全景 → [`cheat-sheet/ontology.md`](../../cheat-sheet/ontology.md) §3

---

## ✍️ 維護者註（v0.5 → v1 升級清單）

本 v0.5 基於 paper 全文 + HuggingFace weights + 二手分析。下次升 v1 時補：

1. ⏳ V-JEPA 2 確切總參數量上限（paper 報告多 size，最大 ViT-g 級約 1B+ encoder，paper 內未集中給「總參數量」單一數字）
2. ⏳ V-JEPA 2-AC 在 Cosmos baseline 上的「pick-and-place」精確分數對比表
3. ⏳ HuggingFace weights 是否所有 checkpoint 都 MIT，或部分 robot-specific checkpoint 另有 license（paper repo README 提及「portions under separate terms」但未列舉）
4. ⏳ K400 / Epic-Kitchens / Video-QA 完整 benchmark 表（vs V-JEPA 1 + InternVideo + VideoMAE）
5. ⏳ `facebookresearch/vjepa2` GitHub issue tracker atlas 掃描（§6.x）
6. ⏳ 1M+ 小時 pretrain 的具體數據來源 list（YT / HowTo100M / Ego4D 比例）
7. ⏳ EMA momentum schedule 精確值 + 3D RoPE 實作細節（patch size / RoPE base frequency）
8. ⏳ Status v0.5 → v1，刪本節

---

[← Back to Latent World Models](./overview.md)

Sources:
- [V-JEPA 2 arXiv 2506.09985](https://arxiv.org/abs/2506.09985)
- [V-JEPA arXiv 2404.08471](https://arxiv.org/abs/2404.08471)
- [LeCun 2022 JEPA position paper (OpenReview)](https://openreview.net/forum?id=BZ5a1r-kVsf)
- [HuggingFace facebook/v-jepa-2 collection](https://huggingface.co/collections/facebook/v-jepa-2)
- [facebookresearch/vjepa2 (GitHub, MIT)](https://github.com/facebookresearch/vjepa2)
- [Connecting JEPA with Contrastive SSL — arXiv 2410.19560](https://arxiv.org/abs/2410.19560)
- [LeWorldModel: Stable E2E JEPA — arXiv 2603.19312](https://arxiv.org/abs/2603.19312)
