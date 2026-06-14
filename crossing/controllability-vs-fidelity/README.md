# Controllability-vs-Fidelity

> **Thesis**：可控性與保真度不是同一條軸上的兩端，而是**爭奪同一個生成預算的兩股力**——每加一道 conditioning，模型就被往「服從」拉得越緊，留給「自然分佈采樣」的自由度越少；當 conditioning 把預測推出訓練分佈邊界，保真度就以過飽和、結構崩壞、細節退化的形式付帳。會不會掉保真度，取決於**注入方式有沒有給模型留逃逸空間**，而不是條件數量本身。

這是一個跨路線的對比 wedge，不是 paper summary。它橫切 [video-world-models](../../foundations/video-world-models/overview.md)（text/image 條件）、[latent-world-models](../../foundations/latent-world-models/overview.md)（action 條件）、[physics-conditioning](../../foundations/physics-conditioning/overview.md)（force/contact 條件）三條路線，問同一個問題：**為什麼控制信號越強、越多，畫面越容易壞**。

---

## 一、核心機制：classifier-free guidance 是保真度的第一手帳單

最乾淨、量化最透徹的「控制換保真度」案例就是 classifier-free guidance（CFG，Ho & Salimans, arXiv [2207.12598](https://arxiv.org/abs/2207.12598)）。CFG 不改架構，只在采樣時把條件得分外推：

```
ε̃ = (1 + w)·ε_cond − w·ε_uncond
```

`w` 就是「服從度旋鈕」。`w` 越大，輸出越貼合條件（IS/alignment 升），但越偏離真實圖像分佈（FID 升、多樣性降）。Ho & Salimans 在 ImageNet 上給出的曲線**直接證明這是非單調的權衡**，不是線性 trade：

ImageNet 128×128（CFG 原文 Table 2，T=256）：

```
 w     FID      IS         讀法
0.0    7.27     82.45   無 guidance：分佈準但對齊弱
0.3    2.43    158.47   FID 最佳點（保真度甜蜜點）
1.0    7.86    297.98   IS 還在升，FID 已回彈到 0.0 的水準
4.0   21.53    421.03   IS 翻倍，FID 崩壞 9 倍
```

ImageNet 64×64 同向（best FID 1.55 @ w=0.1，到 w=4.0 時 FID 26.22 / IS 260.2）。**關鍵讀法**：FID 在 `w≈0.1–0.3` 觸底後立刻回升，而 IS 一路狂飆——也就是說，在最有用的對齊區間之後，每多一分控制都是純付帳，**畫面更「像條件要的樣子」但更不像真實照片**。

機制上，Imagen（Saharia et al., arXiv [2205.11487](https://arxiv.org/abs/2205.11487)）把這帳算到了像素級：大 guidance weight 使 x-prediction 超出訓練資料的 `[-1, 1]` 邊界，而 diffusion 是把自己的輸出反覆迭代的——一旦越界，誤差被放大、累積，產生**過飽和、不自然、甚至發散**的圖。Imagen 稱之為 train-test mismatch，並用 **dynamic thresholding** 硬把越界像素拉回，才敢用比前人更大的 guidance。換句話說：**conditioning 強到一定程度，模型不是「畫得更好」而是被推出它見過的世界**。這就是「加 text → 細節走樣」的底層原因。

> 這條曲線是後面所有 conditioning 形式的母帳單：text、trajectory、force 不管怎麼注入，最終都化約成「往某個方向外推得分」，都吃同一種越界懲罰。

---

## 二、Pareto frontier：方法類型 × (controllability, fidelity) × 掉保真度的機制

| 方法類型（錨點） | Controllability | Fidelity | 為什麼加這個 condition 會掉保真度 |
|---|---|---|---|
| Text-only（[Sora](../../foundations/video-world-models/sora.md) / Veo） | 中 | 高 | 條件稀疏（一句話），外推方向粗，CFG 開大才聽話 → 過飽和；但條件本身不約束像素，fidelity 地板高 |
| Text+image（SVD / Cosmos-img2vid） | 中-高 | 高 | image-init 把首幀釘死，後續是自然 rollout；幾乎不外推像素分佈，**保真度幾乎不掉**（成本在時間一致性而非單幀） |
| Action-only（[Genie-2](../../foundations/latent-world-models/genie-2.md)） | 高 | 中 | action 是 latent 條件，逐幀強約束動態；latent 預算被「服從動作」吃掉，紋理/細節讓位 |
| Trajectory-cond（Cosmos-Drive/Transfer） | 高 | 中-高 | 軌跡是稠密空間約束，前景被釘住 → 前景多樣性塌；fluid/陰影等次級細節在被約束區退化（見 §3-c） |
| Force/contact（[Force Prompting](../../foundations/physics-conditioning/force-prompting.md) / [PhysGen](../../foundations/physics-conditioning/physgen.md)） | 高 | 中 | 力/接觸與 video prior 直接衝突，模型在「服從力」和「保持先驗外觀」間二選一 → 局部結構崩（見 §3-b） |
| Multi（text+action+force 同時硬加） | 最高（理論） | 最低（**樸素實作**） | 多條件各自外推方向不一致 → 互相壓制 + CFG 被稀釋；**但若用自適應 fusion 可逆轉**（見 §4） |

注意最後一列的限定詞：**「樸素實作」**。下文會證明，多條件未必比單條件差——壞的是**把每道條件都開到最大、均勻疊加**這種樸素做法。

---

## 三、三個 anchor 的失效實測（皆有引用）

### (a) Guidance scale：對齊與保真度的非單調帳（CFG / Imagen）

如 §1，CFG 原文 Table 2 的 ImageNet 128 數據是最硬的實測：FID 從 `w=0.3` 的 2.43 一路惡化到 `w=4.0` 的 21.53，**同一個模型、同一份資料，只動一個旋鈕，保真度差 9 倍**。Imagen 進一步指出這對 text-to-image 是普遍現象——「increasing the classifier-free guidance weight improves image-text alignment, but damages image fidelity producing highly saturated and unnatural images」（[2205.11487](https://arxiv.org/abs/2205.11487)）。這不是調參失誤，是 CFG 機制的內生稅。

### (b) Force-conditioning：video prior 與力意圖的直接衝突（Force Prompting）

Force Prompting（arXiv [2505.19386](https://arxiv.org/abs/2505.19386), NeurIPS 2025）把力向量當 conditioning channel 注入 pretrained video diffusion。作者明確報告失效模式：**「Failures in visual fidelity or physical realism occur when the video prior conflicts with the force prompt's intent」**——具體例子是吹頭髮場景中，**臉會隨風向重新轉向**（face reorients based on wind direction）。這是一個教科書級的 multi-conditioning 干擾：text/image prior 要「臉朝前」，force 要「順風形變」，兩股得分在同一區域對撞，模型既沒畫對物理也沒守住外觀。Force Prompting 同時點名根因之一是**力-視頻配對資料稀缺**（真實世界拿不到力信號，合成資料的視覺品質又受 simulator 限制）——資料分佈越窄，條件越容易把采樣推出 prior，保真度越脆。

### (c) Trajectory / 多模態空間條件：被約束區的細節退化（Cosmos-Transfer1）

Cosmos-Transfer1（NVIDIA, arXiv [2503.14492](https://arxiv.org/abs/2503.14492)）做多模態空間控制（edge / depth / segmentation / blur）。它的 Table 1 把「單條件強約束的代價」量化得很清楚：

- **單一模態把自己那項對齊指標頂滿，但多樣性塌**——例如 blur-visual 拿到最高 Blur SSIM（0.96）卻是最低多樣性；edge 拿最好 Edge F1（0.28）但同樣壓死生成自由度。
- 這正是「軌跡/邊緣這類**稠密空間條件**把前景像素釘死 → 被釘區域沒有空間生成自然細節」的實測：可控性指標漂亮，但**那一塊的多樣性/真實感被換掉了**。

換言之，trajectory-cond 之所以「物體走對軌跡但物理感/細節弱」，是因為強空間條件在被約束區把生成預算榨乾，模型只能複述條件、無餘力補次級物理細節（流體、陰影、接觸形變）。`UNVERIFIED`：原 stub 寫「Cosmos-Predict 加 trajectory 後 fluid 細節品質下降」的**具體 fluid FID 數字**未在公開資料核到；上述以 Cosmos-Transfer1 Table 1 的「強單模態 = 多樣性塌」作為同機制的可引用替代。

---

## 四、Fusion 機制對比：哪種最不損保真度（把 open question 變有據討論）

把原 open question 落地：multi-conditioning 的 **fusion 機制**才是決定掉不掉保真度的真變數，不是條件數量。四類典型注入方式：

| Fusion 機制 | 代表 | 對保真度的影響 | 機制原因 |
|---|---|---|---|
| **均勻疊加（token/branch 直加）** | ControlNet 多分支 | 中-高風險 | 各條件得分方向不對齊時互相外推，且**會稀釋甚至抵消 CFG**（見下） |
| **per-stage / 解析度加權** | ControlNet CFG-Resolution-Weighting | 中 | 按 feature map 解析度給不同連接權重，恢復被條件吃掉的 CFG 引導 |
| **cross-attention 注入** | 主流 video diffusion 的 text/action | 中-低 | 條件只調 attention，不直接覆寫像素分佈，外推較溫和 |
| **自適應時空加權（spatiotemporal control map）** | Cosmos-Transfer1 | **最低** | 每個時空位置只讓最相關的模態主導，其餘讓出自由度 → 不在無關區硬約束 |

幾個關鍵實證，澄清一個常見誤解：

1. **ControlNet 原文宣稱多條件「直接相加、無需額外加權」**（"No extra weighting or linear interpolation is necessary"，[2302.05543](https://arxiv.org/abs/2302.05543)）。但同一篇承認**加了 conditioning 後 CFG 會被破壞**：「when no prompts are given, adding it to both [ε_uc 與 ε_c] will completely remove CFG guidance」，因此被迫發明 **CFG Resolution Weighting**。這正說明：**條件注入本身就在跟 guidance 搶帳**，樸素疊加之所以「看起來沒事」，是因為單張靜態圖容錯高；到了 video / 強物理條件就藏不住。

2. **Cosmos-Transfer1 證明好的 fusion 能逆轉「多 = 差」**：它的自適應加權模型在多控制輸入下拿到**最高 Quality Score（8.54）與最佳 depth 重建**，前景用 edge+vis 低自由度、背景用 depth+seg 高自由度，前景多樣性 LPIPS 從 0.01 提到 0.12 而品質不掉。結論很反直覺但有據：**多條件不是保真度殺手，「在每個位置都把每道條件開到最大」才是**。自適應 fusion 透過「只在該約束的地方約束」把生成預算還給模型。

3. **衝突感知降權**：SmartControl（arXiv [2404.06451](https://arxiv.org/abs/2404.06451)）用 control-scale predictor 偵測「條件與 prompt 衝突的局部區域」並調低該區條件權重，讓衝突區回去聽 prompt。這是 §3-b 那種「臉 vs 風向」對撞的工程解：**不是更聽話，而是學會在哪裡少聽一點**。

**小結**：損保真度最少的順序大致是 自適應時空加權 < cross-attn < 解析度加權 < 均勻硬疊加。共同原理是**留逃逸空間**——只在必要的時空位置施加必要強度的條件。

---

## 五、有沒有「fidelity-preserving controllability」？

有，但都不是「免費」，而是**把外推稅挪到別處付**。已被驗證的幾條路：

- **採樣端校正（最便宜）**：dynamic thresholding（Imagen, [2205.11487](https://arxiv.org/abs/2205.11487)）把越界像素拉回 `[-1,1]`，讓大 guidance 不過飽和。代價是輕微的對比/細節犧牲，但換到「敢開大 guidance 而不崩」。同類還有對 CFG 平行分量降權（只留正交分量提質）等後續工作（如 [2410.02416](https://arxiv.org/abs/2410.02416)）。
- **空間預算分配（最有效）**：Cosmos-Transfer1 的 spatiotemporal control map——可控性與保真度不再全域競爭，而是**逐位置分權**，把「該控的地方控死、該自由的地方放開」做成一階設計變數。
- **條件物理化而非像素化（最治本）**：把控制信號注入到**物理狀態 / 動態 solver** 而不是像素得分。如 PhysGen（[2409.18964](https://arxiv.org/abs/2409.18964)）把 force/torque 餵給 2D rigid solver 產生 trajectory，再讓 diffusion 只負責「把已定運動 lift 回 photoreal」——diffusion 不被迫直接服從力，外推被前移到物理層，像素層只做它擅長的渲染。NewtonGen（neural ODE → optical flow → noise warping）同理。**這是目前「物理可控 + 視覺保真」共存最有結構的方向**。
- **衝突感知控制**（SmartControl）：把「哪裡該降低服從度」變成可學習的，從根上避開 §3-b 的硬對撞。

一句話：**fidelity-preserving controllability ≠ 不付外推稅，而是把稅挪到采樣校正 / 空間分權 / 物理層**，讓像素生成器永遠在它見過的分佈內工作。

```
conditioning 強度  低 ───────────────────────────────► 高
保真度            高 ●──●──●                              
                        \   FID 甜蜜點（CFG w≈0.3）       
                          ●─●                             
                              \  過飽和 / 越界 / 細節崩    
                                ●──●──●──● ◄ 樸素硬疊加落點
                                          
                  ▲ 自適應 fusion / 物理化條件 把整條曲線往右上抬：
                    同樣控制強度下保真度更高（虛線）
            高 ····●····●····●····●····●  ◄ fidelity-preserving 設計
```

---

## 六、與相鄰 wedge 的關係

- 條件**種類光譜**（text→action→trajectory→force 各自能控什麼）見 [text-action-trajectory-spectrum](../text-action-trajectory-spectrum/overview.md)；本 wedge 只問「加了之後保真度怎麼掉」。
- 條件強約束導致的**物理守恆破壞**（不只是視覺保真）見 [conservation-violation-atlas](../conservation-violation-atlas/overview.md)。
- 在 **pixel vs latent** 學物理的取捨見 [pixel-vs-latent-physics](../pixel-vs-latent-physics/overview.md)——latent 條件（Genie action）與 pixel 條件（edge/depth）吃保真度的方式不同。

---

## 七、Open question（已縮小）

- 自適應時空加權目前靠人或啟發式設定 control map；能不能**讓模型自己學會「哪裡該讓出自由度」**（衝突感知 + 預算分配端到端可微）？SmartControl 是局部解，全域版本未見。
- 物理化條件（PhysGen 線）把外推前移到 solver，但 solver 本身的覆蓋（剛體 only / 2D）就是新瓶頸——**solver 表達力**與**像素保真度**之間是否又是一條新 Pareto？
- 跨模態 force+text+action 同時硬加、且各自得分方向衝突時，是否存在**理論上的保真度下界**（類似 rate-distortion 的 controllability-distortion bound）？尚無形式化結果。`UNVERIFIED`

---

## 參考（arXiv）

- Ho & Salimans, *Classifier-Free Diffusion Guidance*, arXiv [2207.12598](https://arxiv.org/abs/2207.12598)（FID/IS vs guidance scale 量化曲線）
- Saharia et al. (Imagen), *Photorealistic Text-to-Image Diffusion Models*, arXiv [2205.11487](https://arxiv.org/abs/2205.11487)（高 guidance 過飽和 / train-test mismatch / dynamic thresholding）
- Zhang, Rao, Agrawala, *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet), arXiv [2302.05543](https://arxiv.org/abs/2302.05543)（多條件疊加 / CFG Resolution Weighting）
- NVIDIA, *Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control*, arXiv [2503.14492](https://arxiv.org/abs/2503.14492)（spatiotemporal control map / 單模態 vs 自適應 fusion Quality Score）
- Gillman et al., *Force Prompting*, arXiv [2505.19386](https://arxiv.org/abs/2505.19386)（video prior 與力意圖衝突的失效實測）
- Liu et al., *PhysGen*, arXiv [2409.18964](https://arxiv.org/abs/2409.18964)（把條件物理化：solver-in-loop 保住像素保真度）
- *Eliminating Oversaturation and Artifacts of High Guidance Scales*, arXiv [2410.02416](https://arxiv.org/abs/2410.02416)（CFG 平行分量降權）
- *SmartControl*, arXiv [2404.06451](https://arxiv.org/abs/2404.06451)（衝突感知 local control-scale）
