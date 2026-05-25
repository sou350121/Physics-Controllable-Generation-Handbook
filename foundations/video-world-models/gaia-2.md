<!-- ontology-5axis output=pixel-video injection=data-only control=text|trajectory|action|image-init|camera|layout temporal=clip-parallel domain=driving -->

# Wayve GAIA-1 / GAIA-2 (Driving World Model)

## 1. One-paragraph TL;DR

Wayve 的 GAIA 系列是「driving-domain 專用 video world model」的 anchor 線。**GAIA-1**（arXiv 2309.17080, 2023-09-29）把駕駛影片建模當成 unsupervised sequence modeling：image / text / action 各自 tokenize，丟給一個 6.5B autoregressive transformer 預測下一個 image token，再由 video diffusion decoder 還原像素；總參數 9B，於 ~4,700 小時 UK 駕駛資料上訓練。**GAIA-2**（2025-03-26 release, technical report arXiv 2503.20523）完全切到 **latent diffusion** 架構，原生支援 multi-camera surround-view，conditioning 從 GAIA-1 的 text + scalar action 擴張到「ego-action（speed/steering curvature）+ environment（weather/time-of-day）+ road semantics（lanes/speed limits/intersections）+ external latent embeddings（來自自家 driving model）」。**GAIA-3**（2025-12-02 announced）把訓練資料再 ×10、參數 ×2 到 15B，並從「合成 footage」轉向「offline 評估工具」，把 WM 當成 policy validation harness。三代的 prior gap 一致：通用 video WM（Sora/Veo）缺駕駛專屬幾何先驗、多視角同步、ego-controllability，Wayve 賭駕駛資料 × 駕駛條件 conditioning 才能撐 closed-loop。

## 2. Core mechanism

### GAIA-1（token-based, autoregressive）

```
video frames ──► image tokenizer (VQ) ──┐
text prompt  ──► text tokenizer        ─┤
action seq   ──► action tokenizer      ─┴─► autoregressive Transformer (6.5B)
                                            │  predict next image token
                                            ▼
                                       video diffusion decoder ──► pixel video
```

訓練分兩階段：(1) 各 modality 的 tokenizer 預訓練；(2) world model 在 image / text / action token 的混合序列上做 next-token prediction（cross-modal autoregression）。decoder 是 video diffusion，把離散 image token 映回 RGB clip。**9B total = 6.5B WM + tokenizer + decoder**；64×A100 訓 15 天。

### GAIA-2（latent diffusion, multi-view）

切換到 **latent diffusion world model**。結構性 conditioning 取代了 GAIA-1 的 token concat：

- ego-vehicle dynamics（speed, steering curvature）
- agent configuration（其他車輛/行人）
- environmental factors（weather, time-of-day）
- road semantics（lane count, speed limit, pedestrian crossing, intersection）
- **external latent embeddings**（feature from Wayve 自家 driving model — 等同 distill 一個 policy 的 representation 進 WM）

關鍵差異：joint diffusion over multi-camera 確保 spatial-temporal coherence 跨 surround view；GAIA-1 是單視角 frame-by-frame token 生成，無天然 multi-cam 一致性。Wayve blog 明說 GAIA-2 消除了 GAIA-1 的 "temporal discontinuities" 與 motion 不連貫。具體 camera 數 / resolution / fps / clip length 在 public material 未揭露 `[TBD: verify camera count + resolution from full technical report]`。

## 3. 五軸定位 + 同軸對手

| Axis | GAIA-1 | GAIA-2 |
|---|---|---|
| Output | `pixel-video` | `pixel-video`（multi-view joint） |
| Injection | `data-only` | `data-only`（資料 + 結構化 conditioning，但無 PDE/constraint loss） |
| Control | `text \| action \| image-init` | `text \| trajectory \| action \| image-init \| param`（speed/curvature/weather scalar） |
| Temporal | `autoregressive`（token-level） | `clip-parallel`（latent diffusion clip） |
| Domain | `driving` | `driving`（多國：UK/US/DE） |

**同軸對手**：

- **NVIDIA Cosmos-Drive**（2025）—— 同樣 driving pixel-WM，但開源 checkpoint，走 [Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md) 底座 + trajectory ControlNet；GAIA 全閉源。trade-off：Cosmos-Drive 可 fine-tune，GAIA 資料 / control 維度更密。
- **DriveDreamer / DriveDreamer-2**（2024，學術線）—— 同樣 latent diffusion + 結構化 layout（HD map / 3D box）conditioning，但訓練資料是公開 nuScenes/Waymo，scale 數量級小於 GAIA-2 的 multi-country 自採。
- **Tesla world model**（公開資訊有限）—— 不走 pixel video，走 **occupancy / 3D scene** output（屬 ontology Axis 1 的 `3d-scene`），跟 GAIA 同樣 driving domain 但 output space 不同；trade-off：occupancy 對 planning 更直接，pixel video 對 perception data augmentation 更直接。
- **OmniRe / DriveGS / Street Gaussians**（2024-25）—— `3d-scene` output（3DGS-based driving sim），靠 reconstruction 從 real log 重演而非 generate；ego trajectory editing 受限但物理一致性最強。GAIA 反之：高 diversity，幾何精度低。

要點：GAIA 在「driving-specific scale + 結構化 condition 密度」一軸領先；幾何精度與可微 closed-loop 不如 3DGS / occupancy 線。

## 4. Where it shines / where it breaks

### ⚡ 真正領先的 regime

- **稀有 / 危險場景合成**：emergency maneuver, pre-collision, U-turn, 多國天氣 / 時段組合 — Wayve blog 強調 GAIA-2 用於 safety-critical scenario synthesis 與 OOD robustness testing。
- **Ego-controllability**：speed + steering curvature 作為直接 scalar control，能對同一場景做 counterfactual rollout（"if the ego had braked here"）；通用 Sora/Veo 沒這條控制軸。
- **Multi-geography conditioning**：UK / US / Germany 同模型，靠 geographic conditioning token 切換 → 訓練 perception 模型時的 augmentation 工具。
- **Multi-camera spatial coherence**（GAIA-2）—— 是 GAIA-1 顯著缺乏的能力。

### ❌ Known failure modes

- **Long-horizon drift**（GAIA-1 已被作者承認）—— autoregressive token rollout 超過 ~10s 後場景結構漂移；GAIA-2 latent diffusion clip 長度受限（具體 `[TBD: verify GAIA-2 max clip length]`），跨 clip 銜接仍是公開難題。
- **Multi-camera consistency 在 sharp maneuver 下脆**：Wayve 自己 demo 仍多為 forward-facing；社群觀察（matt3r.ai blog "Realism Gap"）指出 surround view 在劇烈轉向 / 交會時 inter-view geometry 偶有破綻 `[TBD: verify with specific GAIA-2 demo timestamps]`。
- **Rare-object 物理破綻**：動物、施工錐、異常車型 — data-only injection 的通病；訓練資料分布外時 object identity / 物理運動皆會崩。GAIA-2 paper 自己列為 limitation `[TBD: verify exact wording in §Limitations of arXiv 2503.20523]`。
- **No explicit physics / contact model**：碰撞瞬間是「畫出來」的，不是「算出來」的 — 對 safety validation 是 realism gap 的根源（見 matt3r blog）。
- **Closed-loop 速度與 latency**：latent diffusion clip-by-clip 生成在當前 paper 中沒給即時 fps；要做真正 closed-loop policy training 仍卡 inference cost `[TBD: verify GAIA-2 inference latency or fps]`。
- **Closed model**：無 checkpoint / 無 inference API 對外，第三方無法獨立驗證；所有 failure mode 來自 Wayve 自己選的 demo + 二手 blog 分析。

## 5. Reproduction notes

**Wayve 完全閉源**：沒有 weights、沒有 inference API、沒有訓練代碼。只有：

- GAIA-1 arXiv 2309.17080 + Wayve "Scaling GAIA-1" blog（給出 9B / 6.5B / 64×A100×15d / 4,700h UK 資料）
- GAIA-2 technical report arXiv 2503.20523 + Wayve "GAIA-2" blog
- GAIA-3 press release（2025-12-02, 15B params, 10× data, "redesigned video tokenizer"）

**最接近的 open reproduction path**：

- DriveDreamer-2（GitHub `f1yfisher/DriveDreamer2`）— latent diffusion + layout conditioning，nuScenes scale
- Vista（OpenDriveLab, 2024）— 公開 driving WM，autoregressive video diffusion
- Cosmos-Predict + trajectory ControlNet → 自己 fine-tune 駕駛資料

**GPU 預算（GAIA-1 復現級）**：64×A100 80GB × 15d ≈ 23k A100-hours；GAIA-2 latent diffusion 通常省 ~3-5×，但 multi-cam joint training 額外貴 `[TBD: verify GAIA-2 training compute]`。對個人 / 中小 lab 不可行；最務實是用 Vista / DriveDreamer-2 作 baseline，targeting GAIA-2 是商用 scale 的問題。

## 6. Cross-line synthesis

- **× pixel-WM 通用線（[Sora](./sora.md) / [Veo](./veo.md) / [Cosmos-Predict](../foundation-physics-models/cosmos-wfm.md)）**：GAIA = 駕駛 domain-adapt 的版本。Cosmos-Predict 底座 + 駕駛 fine-tune 是 NVIDIA 對 GAIA-2 的對位答案；trade-off：通用底座更廣，GAIA 駕駛 prior 更深。
- **× latent-WM（[DreamerV4](../latent-world-models/dreamer-v4.md) / [V-JEPA-2](../latent-world-models/v-jepa-2.md)）**：GAIA 沒走「latent rollout + decoder」分離結構，GAIA-2 latent diffusion 仍是 pixel-output。若接 latent-WM，可能能解 long-horizon（latent space rollout 較穩）但 Wayve 押寶 pixel realism。
- **× diff-sim**：GAIA 完全沒 physics simulator-in-loop；補 diff-sim 的方向是 "GAIA + CARLA bridge" 之類 hybrid — 但 Wayve 公開資訊不顯示走這條。
- **× surrogate net (3DGS / occupancy)**：與 OmniRe / DriveGS 是 **互補** 而非替代。3DGS 提供幾何 ground-truth 與可微 closed-loop，GAIA 提供 diversity 與條件控制。產線上會兩者混用（real log → 3DGS replay 做 regression test；GAIA → 合成 long-tail 做 augmentation / OOD）。
- **× VLA（cross-handbook）**：GAIA-2 的 "external latent embeddings from a proprietary driving model" 暗示 distill 一個 driving policy 的 feature 進 WM，這恰恰是 Pixel-WM ↔ VLA bridge 的工程接口（見 sister `VLA-Handbook`：world model rollout 提供 negative example for policy distillation）。

## 7. References

**Canonical**：

1. Hu, Russell, Yeo, Murez, Fedoseev, Kendall, Shotton, Corrado. *GAIA-1: A Generative World Model for Autonomous Driving*. arXiv:2309.17080, 2023-09-29. <https://arxiv.org/abs/2309.17080>
2. Wayve. *GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving*. arXiv:2503.20523, 2025-03-26. <https://arxiv.org/abs/2503.20523>
3. Wayve press: "Wayve Unveils GAIA-2…" 2025-03-26. <https://wayve.ai/press/wayve-unveils-gaia2/>
4. Wayve press: "Wayve launches GAIA-3, advancing world models from simulation to evaluation" 2025-12-02. <https://wayve.ai/press/wayve-launches-gaia3/>

**二手實測 / 分析**：

5. Wayve blog: "Scaling GAIA-1: 9-billion parameter generative world model" — 2023 details on params / compute / data hours. <https://wayve.ai/thinking/scaling-gaia-1/>
6. matt3r.ai: "Generative Models in Autonomous Driving: GAIA-1 to GAIA-2 and the Realism Gap" — 第三方分析合成資料的 realism gap 與 safety validation 限制。<https://matt3r.ai/blogs/our-latest-thoughts/gaia-2-synthetic-data-autonomous-driving>
7. Wayve "Thinking" page: GAIA-2 — claims on multi-geography conditioning + action-conditioned generation. <https://wayve.ai/thinking/gaia-2/>

## 8. §8 Pitfall log

> Wayve 全閉源 → 沒有 GitHub issue tracker。本節 pitfall 來自 (a) 作者 paper 自承的 limitation、(b) Wayve 自家 blog demo 的可觀察破綻、(c) 第三方分析 blog（matt3r 等）。Severity 評級：H=阻止商用 / M=工程可繞 / L=cosmetic。

| # | 來源 | 摘錄 / 觀察 | Severity | Workaround |
|---|---|---|---|---|
| §8.1 | GAIA-1 paper §Limitation | Autoregressive rollout 在 long horizon 出現 scene structure drift | H | GAIA-2 改 latent diffusion clip 已部分緩解；單 clip 長度仍受限 |
| §8.2 | GAIA-2 paper（待 verify 確切字句）`[TBD]` | Rare objects（animals, construction cones, atypical vehicles）OOD 時 identity 漂 | M | 用 structured agent conditioning 注入；不依賴 pure text |
| §8.3 | matt3r.ai blog "Realism Gap" | 合成 footage 在 safety validation 上仍有 "realism gap"，pixel realism ≠ behavior realism | H | 不要單獨用 GAIA output 做 closed-loop training；混 real log + 3DGS replay |
| §8.4 | Wayve demo 觀察 | Multi-camera 在 sharp maneuver / 交會 / 鏡像物體下 inter-view geometry 偶有破綻 `[TBD: 特定 demo timestamp]` | M | 限制使用情境到緩動 ego；或下游模型加 cross-view consistency loss |
| §8.5 | 缺 open weight / API | 第三方無法 benchmark；所有 metric 來自 Wayve 自選 demo | H | 用 Vista / DriveDreamer-2 / Cosmos-Drive 做 open baseline，把 GAIA 當 reference target |
| §8.6 | 物理 injection = `data-only` | 碰撞瞬間是「畫出來」，沒有 contact dynamics — 不可用於 fine-grained collision physics validation | H | 加 sim-in-loop（CARLA / Genesis）混合；或 surrogate force model 後處理 |
| §8.7 | Inference cost `[TBD]` | Latent diffusion multi-cam joint rollout 推理延遲未公開；closed-loop policy training 還是 batch offline 模式 | M | 目前定位是 offline data augmentation / evaluation harness（GAIA-3 明確走這方向），不是 online closed-loop |
| §8.8 | GAIA-3 evaluation 主張 | "Synthetic-test rejection rates reduced fivefold" 與 "closely mirrors real-world results" 來自 Wayve 自家 study，無第三方覆核 `[TBD: 等獨立 benchmark]` | M | 視為 vendor claim；自家先小規模對比 real road test |
