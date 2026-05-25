<!-- ontology-5axis output=pixel-video injection=implicit-from-data control=text|image-prompt temporal=joint-rollout domain=generalist -->

# OpenAI Sora (1.0 / Sora-2)

## 1. One-paragraph TL;DR

Sora 1.0（OpenAI tech report "Video generation models as world simulators", 2024-02-15）是 video-gen 的 GPT-moment：第一個對外公開的、能 joint-rollout 出 ~60s、1080p 的 text-to-video。它的賭注很單一 — **scale + DiT + 變長 spacetime patches**，物理規律完全靠資料 implicit 學，不掛任何 PDE / contact / 守恆 loss。它的價值對本 handbook 不在「物理感真的解了」(沒有)，而在於它**把 implicit-from-data 路線推到一個極限**：用足夠大的 latent diffusion transformer + 變形/長度不對齊的訓練 corpus，能撐出什麼樣的 emergent physics — 以及哪些 physics 它即使在巔峰時也仍然崩。Sora-2（2025-09-30 發布）加 audio + 自家的 "referee model" 物理檢核回路；OpenAI 已宣布 Sora app 於 2026-04-26 關閉、API 2026-09-24 退役，整個品牌處於 sunset 狀態 — 但作為公開最完整的 implicit-from-data 案例研究，價值不會過期。

## 2. Core mechanism

公開資訊（OpenAI tech report 不含訓練細節 / 完整架構）：

```
raw video (任意 H, W, T, aspect)
   │
   ▼
[Video VAE encoder]            ← 自家從頭訓練；空間+時間同時壓縮
   │
   ▼
latent volume (h, w, t)
   │ patchify into spacetime patches
   ▼
sequence of patch tokens (variable length)
   │
   ▼
[Diffusion Transformer (DiT)]  ← U-Net 被 Transformer 取代
   │   text cond: re-captioned prompts (內部用 GPT 改寫)
   │   joint training on images (T=1) + videos
   ▼
denoised latent → [VAE decoder] → pixel video
```

關鍵設計（vs 早期 VDM / Imagen Video）：

- **Spacetime patches 而非固定 frame 數**：原生支援變長/變解析度/變 aspect ratio 訓練與生成（report 點名這條讓 framing 和 composition 更穩）
- **VAE 同時做空間+時間壓縮**：不像 SVD 走 per-frame VAE，Sora 的 latent 帶 temporal axis，讓 DiT 能在 latent 中做真正的時空 attention
- **Joint-rollout**：整段 clip 一次性 denoise；無 KV-cache 式 AR — 因此長度受 attention compute 約束，而非 drift 累積
- **Re-captioning**：訓練 prompts 由 DALL·E-3 風格 GPT 改寫得更密 — 推理時也會自動 expand 用戶 prompt

Sora-2 公開資訊更少，社區拆解（skywork、CineD、A2E）總結三點：

1. 內建 **referee model** 抓 floating objects / penetration / 不自然運動，反饋給 re-training（OpenAI 沒給 paper 細節，看起來像 RLHF-for-physics 或 reward-model-filtered SFT）
2. **同管線 audio**：dialogue + SFX + 環境聲音與 video token 共生成（細節未公開；推測類似 unified multimodal token stream）
3. Visual encoder 仍是 3D spacetime patch；report 描述「inherits Sora 1 framework」

## 3. 五軸定位 + 同軸對手

```
output     = pixel-video
injection  = implicit-from-data       ← Sora-2 加 referee 但仍非顯式 loss
control    = text | image-prompt
temporal   = joint-rollout            ← 不是 AR
domain     = generalist
```

同軸對手（皆 pixel-video / implicit-from-data / joint-rollout / generalist）：

| 對手 | 出處 / 年 | 差異點 |
|---|---|---|
| **[Google Veo / Veo-2 / Veo-3](./veo.md)** | 2024–2025 | 物理感後來居上；Veo-3 加 audio 早於 Sora-2 [TBD: verify Veo-3 audio release timing vs Sora-2] |
| **Kling**（快手） | 2024+ | 中國線 SOTA；長 clip 與 motion 表現一度被認為超 Sora |
| **Hunyuan-Video**（騰訊） | 2024-12 | 13B 開源，被 Open-Sora 2.0 / Open-Sora-Plan 視為對標 |
| **Wan**（阿里） | 2024–2025 | 中國線；開源權重路線 |
| **[Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md)**（NVIDIA） | 2025-01 | 同樣 DiT 但對 robotics / driving 微調；附 World Foundation Model 標籤 |

關鍵分歧：Cosmos 公開權重給下游接 robotics conditioning，Sora 完全閉源；Veo 走 Google 線（image-prompt 與 inpaint 更強）；Hunyuan/Wan 開源權重，社區實測物理 case 與 Sora 已可一戰。

## 4. ⚡ shines / ❌ breaks

**⚡ 領先 regime**

- **長 clip 的時空一致性** — 60s clip 中角色身分、背景、相機運動的一致性比同期 AR 模型（如 Phenaki / EMU-Video）穩太多。Joint-rollout 賭對了
- **Text following** — 受惠於 GPT re-captioning，多物件、多動詞、相機指令的服從度顯著高於前代
- **Composition / framing** — 變長變 aspect 訓練讓豎屏短影音、電影感橫屏都能撐
- **Implicit affordance** — 桌面、街景、liquid pour 等高頻 daily-life prior 在絕大多數 case 已過視覺圖靈測試

**❌ 已記錄的物理 failure（OpenAI report 自己 + 社區）**

OpenAI 在原 report "Limitations" 段就**自己貼了示例**：

- 玻璃杯掉地不會碎、會像橡膠彈
- 咬一口餅乾後，餅乾上沒有缺口
- 物體穿越彼此 / 桌面（"objects passing through each other"）
- 椅子被人提起時形變不正確

二手實測（Liu/Kang et al. PhyWorld, arxiv 2411.02385, ICML 2025）給了結構性結論：

- **In-distribution**：完美外推
- **Combinatorial**：scaling 有 measurable 改善
- **OOD（換新初始條件）**：全面失敗
- **Feature-priority hierarchy when generalizing**：`color > size > velocity > shape` — 模型擇近鄰 case 的順序揭示它根本沒在學 dynamics，只在學「哪個訓練片最像」（"case-based generalization"）

Sora-2 改善但**不均衡**：基礎人體運動 / 球類碰撞 / 玻璃破裂明顯好；流體 / 煙 / 布料、需要 volumetric simulation 的場景（如 "steam rising from a cup"）仍崩。

VBench-2.0（2026-03）Physics 維度 — 大多數當前 model 仍 < 60% — Sora 的具體分數需從 leaderboard 查 [TBD: pull exact VBench-2.0 Physics score for Sora-480p vs Sora-2 if reported].

## 5. Reproduction notes

**無公開權重，無 paper-grade 細節**（連 model size 都沒給）。社區重建走兩條：

- **Open-Sora (hpcaitech/Open-Sora)** — v2.0 (2025-03) 號稱 ~$200k 訓出對標 Hunyuan-Video / Runway Gen-3 的模型；v1.2 加 3D-VAE + rectified flow；v1.3 升 VAE + Transformer。代碼 / 權重 / data pipeline 全開
- **Open-Sora-Plan (PKU-YuanGroup)** — 平行路線，v1.5 用 8B + 40M video 達 Hunyuan-class
- **CogVideoX**（智譜） — 另一條開源 DiT-video，1.5 版重點改 motion 連續性，社區常作 Sora-class baseline 對照
- **HunyuanVideo (Tencent)** — 13B DiT-video，2024-12 開源，物理 case 實測社區常拿來與 Sora 對打

典型踩坑：

- 3D-VAE 訓不穩 — 直接 reuse SD VAE 的 per-frame 路線會犧牲 temporal smoothness；要重訓
- 變長 / 變 aspect 訓練要 dynamic batching + 自製 collator
- 沒有足夠 caption 密度時，DiT 的 text following 會明顯退化 — re-caption pipeline 是必補
- 200k USD 是 v2.0 的最低估算，data 與工程人月不計 — 真實 cost 高得多

## 6. Cross-line synthesis

- **vs latent-WM（[V-JEPA](../latent-world-models/v-jepa-2.md) / [DreamerV4](../latent-world-models/dreamer-v4.md)）**：Sora 在 pixel 空間直接 rollout，compute 量級遠高於 latent rollout；latent-WM 賭的是「不需要每個 pixel 都正確，只要 affordance / dynamics 對」，更適合 VLA / planning。Sora 路線適合 content / media，不適合 closed-loop agent
- **vs 3D-aware (World Labs / NVIDIA Edify3D)**：Sora 沒有顯式 3D 表徵，所以視差 / occlusion / 物件 identity 在大相機運動下會漏；3D-aware 路線用 3DGS / mesh 顯式 anchor object identity — fidelity 與 controllability 不同 trade-off
- **vs diff-sim / sim-in-loop（[Genesis](../differentiable-simulators/genesis.md), Cosmos-Reason rollout）**：Sora 完全沒 sim 接口；要做 robotics data-gen 必須外掛 — [Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md) 就是把 Sora-class backbone 接 NVIDIA sim ecosystem 的範例
- **vs neural surrogate ([FNO](../neural-surrogates/fno.md), [GraphCast](../neural-surrogates/graphcast.md))**：完全不同問題；surrogate 走 field/PDE，Sora 走 RGB pixel — 兩條路在 `bridge-to-vla` 才會合流（影片產資料 → policy 學動作 vs surrogate 提供 force/contact 標註）

對本 handbook 的 USP：Sora 是 **Axis-2 implicit-from-data 的 reference point** — 任何宣稱「我加了 constraint-loss / sim-in-loop / hard-PDE」的方法，benchmark 都該回答「比 Sora-class scale 的純資料路線好多少」。

## 7. References

1. OpenAI, "Video generation models as world simulators", 2024-02-15. https://openai.com/index/video-generation-models-as-world-simulators/
2. OpenAI, "Sora 2 is here", 2025-09-30. https://openai.com/index/sora-2/
3. OpenAI, "Sora 2 System Card", 2025. https://openai.com/index/sora-2-system-card/
4. Kang/Yue/Lu et al., "How Far is Video Generation from World Model: A Physical Law Perspective", arxiv 2411.02385 (ICML 2025). https://arxiv.org/abs/2411.02385 / https://phyworld.github.io/
5. Liu et al., "Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models", arxiv 2402.17177. https://arxiv.org/abs/2402.17177
6. "Sora as a World Model? A Complete Survey on Text-to-Video Generation", arxiv 2403.05131. https://arxiv.org/abs/2403.05131
7. Open-Sora (hpcaitech): https://github.com/hpcaitech/Open-Sora ; Open-Sora 2.0 paper arxiv 2503.09642
8. PhyGenBench / VBench-2.0 (intrinsic faithfulness, 2026-03): https://arxiv.org/html/2503.21755v1

## 8. §8 Pitfall log

| # | 來源 | 失效 | Severity | Workaround |
|---|---|---|---|---|
| 8.1 | OpenAI report (官方) | 玻璃杯落地不碎、像橡膠彈 | high | 無 — implicit-from-data 路線在剛體破裂 / brittle fracture 上沒有 inductive bias |
| 8.2 | OpenAI report | 咬餅乾無缺口、liquid splash 失真 | high | 同上 |
| 8.3 | OpenAI report | 物體互相穿透 / 穿桌 | high | Sora-2 referee model 部分緩解；fluid/cloth 仍崩 |
| 8.4 | PhyWorld (arxiv 2411.02385) | OOD 全面失敗；case-based 而非 principled | structural | scaling 不解；需引入 inductive bias（constraint-loss / sim-in-loop） |
| 8.5 | PhyWorld | 泛化特徵優先序 `color > size > velocity > shape` | structural | 訓練資料要 deliberately decorrelate；或加 physics-aware data augmentation |
| 8.6 | 社區實測（Sora-2） | steam / smoke / 布料仍崩 | medium | 等 volumetric-aware backbone；現階段切換 sim-in-loop |
| 8.7 | 商業層 | Sora app 2026-04-26 關閉、API 2026-09-24 退役；峰值後下載量降 ~66%、報告日燒 $15M inference cost | product | 工程依賴需切 Veo / Kling / Hunyuan / Open-Sora 等替代 [TBD: verify $15M/day figure beyond single secondary source] |
| 8.8 | 透明度 | OpenAI 從未公開 model size / 訓練資料來源 / 訓練 FLOPs | research | 學術 reproduction 全靠 Open-Sora / Open-Sora-Plan / Hunyuan 等代理 |

---

**TBDs**: Veo-3 audio launch timing vs Sora-2; exact VBench-2.0 Physics scores per Sora variant; OpenAI $15M/day inference figure (二手轉述, single-sourced).
