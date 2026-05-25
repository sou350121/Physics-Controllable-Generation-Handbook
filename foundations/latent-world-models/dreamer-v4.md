<!-- ontology-5axis output=latent injection=implicit-from-data control=action temporal=latent-rollout domain=robotics|generalist -->

# DreamerV3 → DreamerV4 (Hafner et al.)

> 一條從 RSSM 走到 block-causal transformer 的 latent WM 主線。V3 把 RL 中的 model-based 流派救回桌上、V4 把 latent WM 帶到 offline + scalable 的層級。本篇同時 cover 兩代，並把演化的取捨講清楚。

## 1. One-paragraph TL;DR

DreamerV3（Hafner et al., 2023）的核心 thesis：用一個共用的 latent world model（RSSM）把高維觀察壓成 discrete + continuous 混合 latent，actor 與 critic 全在 latent imagination 中訓練 — 完全不再回 pixel 空間取 reward signal。一組 fixed hyperparams 就跨 150+ tasks（Atari, DMC, Crafter, Minecraft），首次無人類示範拿到 Minecraft 鑽石。**DreamerV4**（Hafner, Yan, Lillicrap, 2025, arXiv 2509.24527）把 RSSM 換掉，改用 block-causal transformer + causal tokenizer + 新的 **shortcut forcing** 訓練目標，並把 latent rollout 推到 9.6 秒 context、單 H100 ≥20 FPS 的互動 inference，**第一次純 offline data（VPT contractor 2.5K hr，只 ~100 hr action-labeled）就拿到 Minecraft 鑽石**。Prior gap：(a) V3 之前 model-based RL 在 sparse-reward、long-horizon 場景一直被 model-free 海量 sample 打趴；(b) 2024-2025 的 Genie/Oasis/MineWorld 等 pixel-WM 拿 Minecraft 做 demo 但 temporal consistency 崩、不能做 RL；V4 用 latent rollout + offline imagination 同時解掉 fidelity 與 efficiency。

## 2. Core mechanism

### V3 — RSSM + symlog + actor-critic on imagination

```
o_t ──Encoder──► z_t (32 cat × 32 cls, straight-through)
                  │
                  ├──► Decoder ──► ô_t   (recon loss, symlog)
                  │
h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})  ← deterministic recurrent
z_t ~ q(z_t | h_t, o_t)               ← posterior (encoder)
ẑ_t ~ p(ẑ_t | h_t)                    ← prior (dynamics)
                  │
                  └──► Reward head, Continue head
Actor π(a | h_t, z_t), Critic v(h_t, z_t)
       └── trained on latent rollouts (imagination horizon H≈15)
```

關鍵 trick：**symlog** 對 reward / value / decoder target 做對稱對數壓縮 → 跨域 reward scale 無需 hp tune；**KL balancing** + **free bits** 防 posterior collapse；**twohot** critic 把 value regression 改成分布 head。

### V4 — Causal tokenizer + block-causal transformer + shortcut forcing

```
Frames ──Causal Tokenizer──► token grid (per frame, masked-AE pretrain, MSE+LPIPS)
                                    │
       Block-causal Transformer ◄───┤
         · axial attn (space-only / time-only 分層)
         · sparse temporal attn (every 4 layers)
         · GQA + register tokens
         · alternating batch lengths（length generalization）
                                    │
       Shortcut-forcing objective ──┤  ← x-prediction + ramp-loss
         · 由 diffusion-forcing + shortcut models 擴展而來
         · 4-step sampling/frame，比 diffusion-forcing 64-step 快 16×
                                    │
       Action-conditioned rollout ──┤  ← 多數 frame 無 action label，
                                       仍能從 ~100 hr labeled 學 action 控制
       Imagination Policy  ◄────────┘  (offline RL inside WM)
```

Context length 撐到 9.6 秒（前代約 1.5 秒），單 H100 互動 ≥20 FPS。整條 pipeline 仍然是「latent imagination 上做 actor-critic / policy improvement」的 Dreamer 範式 — 換的是 backbone 與訓練目標，不是哲學。

## 3. 五軸定位 + 同軸對手

- `output=latent`（V4 另出 pixel 解碼用於 visualization，但 RL 全在 latent 跑）
- `injection=implicit-from-data`（無 PDE / 守恆 / contact loss；物理是從 VPT video distribution 隱式長出來的）
- `control=action`（V4 也支援 image-prompt 做 rollout 開頭）
- `temporal=latent-rollout`（V3 GRU、V4 block-causal transformer，兩代都不在 pixel 上 autoregress）
- `domain=robotics|generalist`（V3 跨 150 tasks 含 DMC robot、V4 主打 Minecraft 但 architecture 是 generalist）

| 同軸對手 | 差異 |
|---|---|
| **TD-MPC2** | latent dynamics + MPC planning（無 actor），更 sample-efficient、但 long-horizon planning cost 線性放大；Dreamer 用 amortized policy 換更便宜的 deploy |
| **[V-JEPA-2](./v-jepa-2.md)** (Meta, 2025) | self-supervised latent 預測 + 後續 action head；強在 representation transfer、弱在 closed-loop control（Dreamer 從 day-1 就 close loop） |
| **[Genie-2](./genie-2.md)** (DeepMind, 2024) | latent action token + autoregressive 對 pixel/latent token，主打 interactive playable WM；不做 RL 內訓練，無 actor-critic |
| **MuZero / EfficientZero-v2** | latent dynamics + MCTS，仍是 latent-WM 家族但 discrete tree search；Dreamer 用 stochastic latent + gradient-based policy improvement |

## 4. ⚡ shines / ❌ breaks

**⚡ Where it shines**

- **V3：Minecraft Diamond from scratch**（無 demo / curriculum），跨 150+ tasks 同一組 hp — 把「model-based RL 不通用」這條 prior 直接打破。
- **V3：DMC、Atari、Crafter** 上對 model-free SOTA (PPO, Rainbow, DrQ-v2) 多數 task data-efficiency 高 5-100×。
- **V4：Offline Minecraft Diamond**（無環境互動）— 完整 20K+ keyboard/mouse action sequence，0.7% 成功率（VPT 等 baseline 更低）；stone pickaxe >90%、iron pickaxe 29%。
- **V4：100× 資料效率**（vs VPT 用 YouTube 全集）+ 單 H100 即時互動。
- 跨方法 superiority：Oasis / Lucid-v1 / MineWorld 等 pixel-WM 在同一 Minecraft setup 上「fail to maintain temporal consistency or hallucinate structures」（V4 paper 自述）。

**❌ Where it breaks**

- **Latent rollout drift**：V3 imagination horizon 設 ~15 步是有原因 — 超過 30 步後 prior dynamics 偏移已嚴重（社群實測，issue 討論常出現「longer horizon hurts」）。V4 雖把 context 推到 9.6 s，但作者明說「context limited to ~9.6 seconds restricts very long-horizon consistency」。
- **Decoder artifacts**：V3 用 MSE-style decoder，UI 文字 / inventory 數字復現模糊（issue #186 vFf0621 問「應該先會玩還是先會 reconstruct？」反映復現順序對使用者 unclear）；V4 自述「inventory UI elements can be unclear or change over time」。
- **Sparse-reward exploration**：V3 Diamond 雖能拿但 yield 仍低；V4 把 Diamond 從 online 移到 offline 但成功率只有 0.7% / 60-min episode — 作者直言未拆解「main bottlenecks (planning, exploration, credit assignment)」。
- **Reward / model exploitation**：V4 paper 明說「potential model exploitation and reward hacking not systematically studied」— 在 latent WM 內 imagine 出高 reward 但真實環境不認的 trajectory 是已知風險（同樣的問題 MuZero / Dreamer 系列都被指出過）。
- **複現工程坑**：JAX + CUDA matrix（issue #188 segfault on RTX 6000 Ada / JAX 0.4.33 / CUDA 12.4），docker build failures（issue #210），訓練 OOM-kill（issue #203）。

## 5. Reproduction notes

**V3（`danijar/dreamerv3`，MIT, JAX, 3.3k stars）**

- Python 3.11+，JAX with GPU/TPU；CPU 路徑保留但只用來 smoke test。
- `pip install` 後 `python dreamerv3/main.py --configs crafter size50m`；config block 可組合（size12m / size50m / size200m / size400m）。
- **Scaling law verified**：作者強調「larger models consistently increase both final performance and data-efficiency」— 與 model-free 通常 saturate 形成對比。
- GPU 預算：Atari size50m 約 1× A100 一天可跑到 reasonable level；Minecraft Diamond 原 paper 約 17 天 1× A100（社群多次驗證）。`--run.train_ratio 32` 是預設。
- 典型踩坑：(a) CUDA / cuDNN 對 JAX 版本敏感，issue #188 直接 segfault → 先 `--batch_size 1` 驗 OOM 還是 driver；(b) docker build 在新 base image 上常壞（#210）；(c) reward / observation scale 若極端，雖然 symlog 號稱免調，仍偶爾要看 KL loss 曲線。

**V4（無官方 release at write-time）**

- 官方 code 尚未開源（截至 2026-05，arXiv 2509.24527）。
- **Unofficial PyTorch 復現**：`nicklashansen/dreamer4`（HF dataset 同名）— 重要參考但**作者非 Hafner**；功能完整度需自行驗。
- **Minecraft 特化復現**：`IamCreateAI/Dreamerv4-MC` — 聚焦 Minecraft pipeline，含 tokenizer + transformer + shortcut-forcing 三段；上手快但與 paper 細節對齊度 [TBD: verify against official release]。
- 訓練 GPU days：paper 數字 [TBD: verify exact GPU·days from §experiments — Nature/arxiv full text]。inference 已驗證 ≥20 FPS / 1× H100。

## 6. Cross-line synthesis

**vs pixel-WM（[Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md) / [Sora](../video-world-models/sora.md)-style / [Genie-2](./genie-2.md) / Oasis）**

- Pixel-WM 強在 visualization 與人類驗證；latent-WM 強在 RL 內訓練的 compute efficiency。Dreamer 系列證明「latent rollout 就夠 agent control 用」— 但 V4 paper 自己也指出同期 pixel-WM 在 Minecraft 上 temporal 崩、所以 latent 暫時是 agent-control 主路線。詳見 `crossing/pixel-vs-latent-physics/`（TODO）。

**vs MPC-on-WM（TD-MPC2, DINO-WM）**

- 二者共用 latent dynamics，但 Dreamer 把 planning amortize 進 actor，TD-MPC 每步重新 plan。trade-off：amortize 部署快、re-plan 對 distribution shift 更 robust。Dreamer 在「inference cheap、policy 多次重用」場景勝；MPC 在「task 多變、policy 不好預訓」場景勝。

**vs neural-surrogate（[FNO](../neural-surrogates/fno.md) / [GraphCast](../neural-surrogates/graphcast.md) / MeshGraphNet）**

- Surrogate 解的是「給定 PDE，預測下一狀態」— 沒有 action / reward / policy 概念，injection 是 hard-PDE 或 constraint-loss。Dreamer 是 implicit-from-data 的 agentic WM。兩條線可組：surrogate 當 simulator → Dreamer 在其上訓 policy（類似 PhysGen × RL，文獻仍少）。

**vs [V-JEPA-2](./v-jepa-2.md)（self-supervised latent + action head）**

- V-JEPA-2 押注 representation transfer（看大量 unlabeled video 學表示，再 attach action head）；Dreamer-V4 同樣吃 unlabeled video（VPT 2.5K hr，只 100 hr labeled），但 latent 是「為 imagination rollout 與 control 而生」、不是純 representation。兩者在「offline video → controllable agent」這個 2026 主戰場正面對撞，**diff 在 latent 是否一開始就為 dynamics 設計**。

## 7. References

**Canonical**

1. Hafner, Pasukonis, Ba, Lillicrap. "Mastering Diverse Domains through World Models." arXiv:2301.04104 (Jan 2023). Nature 版：*Nature* 2025, "Mastering diverse control tasks through world models" (https://www.nature.com/articles/s41586-025-08744-2).
2. Hafner, Yan, Lillicrap. "Training Agents Inside of Scalable World Models." arXiv:2509.24527 (Sep 2025).

**Secondary / 實測 / 復現**

3. `danijar/dreamerv3` (MIT, JAX) — 官方 V3 復現，3.3k stars / 539 forks. https://github.com/danijar/dreamerv3
4. `nicklashansen/dreamer4` — 非官方 PyTorch V4 復現 + HuggingFace dataset.
5. `IamCreateAI/Dreamerv4-MC` — Minecraft 特化 V4 復現.
6. emergentmind 對 V4 的 technical breakdown (tokenizer / shortcut-forcing / 對比表). https://www.emergentmind.com/papers/2509.24527
7. TechXplore (Oct 2025) 對 V4 的科普報導：https://techxplore.com/news/2025-10-deepmind-ai-agent-tasks-scalable.html
8. `TransDreamerV3` (arXiv 2506.17103) — 把 V3 的 GRU 換 Transformer 的中間世代研究，可視為 V3 → V4 的橋梁實驗.

## 8. Pitfall log

GitHub-validated 已知問題（來源：`danijar/dreamerv3` issues，截至 2026-05）：

| # | 問題 | Severity | 摘錄 / 狀態 | Workaround |
|---|---|---|---|---|
| **#188** | Segfault on `main.py` even with minimal run | **High** | JAX 0.4.33 + CUDA 12.4 + RTX 6000 Ada 直接 segfault；open 2025-06 | Pin 較舊 JAX (0.4.28-0.4.30) + CUDA 12.1；或 docker-only 部署 |
| **#203** | The training process was directly killed by the system | High | OOM-kill；open 2025-12 | 先 `--batch_size 1` 確認，下調 `train_ratio`、size50m → size12m |
| **#210** | Docker Build Failed With Dockerfile | Med | base image / JAX wheel 不相容；open 2026-03 | 改用 NVIDIA NGC JAX image，或自 build CUDA 12.1 base |
| **#187** | Change in dyn loss between initial and latest version | Med | 不同 commit 之間 dyn loss 行為改變；作者尚未公開回覆；open 2025-05 | Pin commit hash；reproducing paper 數據時優先用 paper 發布當週的 SHA |
| **#181** | Bootstrapped λ-returns 細節問題 | Low (correctness) | λ-return 邊界 / discount 細節 unclear；open 2025-04 | 對照 `train.py` 中 `lambda_return` 實作，與 V2 dreamerv2 對比 |
| **#186** | Video reconstruction | Low (UX) | 用戶不確定「先會玩還是先會 reconstruct」— 反映 decoder 與 policy 訓練順序的文檔不足；open 2025-05 | Reconstruction quality ≠ policy quality；訓練中前段 decoder 經常糊但 reward curve 已上升 |
| **#212** | World Model Open-Loop Predictions | Low (design) | 截斷 WM↔policy gradient 是否仍 long-horizon 預測；open 2026-05 | Dreamer 設計上 WM 與 policy gradient 是分離 (WM 用 recon + KL loss、policy 用 imagined returns)，截斷預期可行 |
| #201 | Debug `dreamerv3.Agent.train()` | Low | 一般使用問題；open 2025-11 | 用 `jax.debug.print` + `disable_jit` |
| #191 | README links to outdated paper version | Trivial | open 2025-08；應該指向 Nature 2025 版 | 直接看 arXiv 2301.04104 v3 + Nature 連結 |

**V4 特有的 known limitations**（paper §discussion 自述，非 GitHub）：

- Diamond 成功率 0.7% / 60-min — bottleneck 未拆解（planning vs exploration vs credit assignment 三者影響未分離）
- 9.6 s context → 真正 long-horizon (10 min+) consistency 仍未解
- Inventory UI 元素 latent decoding 不穩定
- 無 uncertainty estimation / risk-aware optimization
- Model exploitation / reward hacking 尚未系統性檢驗

**跨代共同 pitfall**（與 `overview.md` §8 對齊）

- Latent dynamics mode collapse：V3 用 discrete (32×32) latent + KL balancing + free bits 緩解；V4 改 continuous + tanh，靠 masked-AE pretrain 加大資料量壓住，未獨立報告 collapse 數據 [TBD: verify V4 collapse metrics]
- Action conditioning OOD：訓練見過的 action 分布外（e.g. Minecraft 中 inventory 操作極短時間多按鍵組合）成功率明顯下降，與 #186 / V4 inventory artefact 是同一根因的兩面

---

> 寫於 2026-05-25；V4 paper 仍是 arXiv preprint（無官方 code release），所有 V4 reproduction notes 以非官方實作為準；GitHub issue 引用以 `danijar/dreamerv3` 為準。
