# World Model Benchmarks

> 量測 world model 的預測保真度、長程 rollout 一致性、以及下游 control/planning 收益的 benchmark 目錄（latent WM 與 pixel WM 通吃）。

## 這類在測什麼

World model (WM) 評測有三個層次，常被混為一談：

1. **預測保真度 (prediction fidelity)** — 給定 context，模型預測的下一段 frame/latent 與真實 ground truth 有多接近（pixel WM 看 FVD/PSNR，latent WM 看表徵相似度）。
2. **長程一致性 (long-horizon consistency)** — autoregressive rollout 跑久了會不會 drift、塌掉、違反物理（守恆、permanence）。這是目前最弱、最缺公認 metric 的一環。
3. **下游 control/planning 收益** — 把 WM 拿去做 model-based RL 或 planning，最終的 policy return / 任務成功率才是真正的 ground truth。重建漂亮不等於能拿來規劃。

好的 WM 評測要能把這三層拆開講；多數 benchmark 只測得到第 1 層，用它代理第 2、3 層就會被 game。

## Benchmark 全景

| Benchmark | 測什麼 | 主要 metric | 已知缺陷 / 可被 game | 出處 |
|---|---|---|---|---|
| WorldModelBench | 把 video-generation 模型當 WM 來判：commonsense / instruction-following / physics-adherence 三維 | 微調 2B multimodal judger 評分（自報 judger 比 GPT-4o 高 8.6% violation 預測準確率）；350 condition pairs / 7 domains / 56 subdomains / 67K human labels / 14 模型 | clip 短、judger 本身會錯（學到偏好而非物理）；instruction-following 與物理糾纏；只覆蓋 7 個應用域 | arXiv 2502.20694 |
| LikePhys | video diffusion WM 的 intuitive physics（valid vs impossible 配對） | Plausibility Preference Error (PPE)，用 denoising ELBO 當 likelihood surrogate；12 場景 / 4 物理域，training-free | likelihood≠真懂物理；配對需精心策劃；chaotic dynamics 仍崩 | arXiv 2510.11512 |
| IntPhys 2 | 複雜合成場景的 intuitive physics（permanence / immutability / continuity / solidity） | violation-of-expectation：plausible vs implausible 二分；多數模型停在 chance（約 50%），人類近滿分 | 合成場景 domain gap；二分任務可被淺層線索矇；不測 control 收益 | arXiv 2506.09849 |
| IntPhys (原版) | 視覺 intuitive physics 框架（4 原則隔離測試） | 整段影片的 physical plausibility score；possible vs impossible 區分 | 僅合成；早期、場景簡單；只測判別不測生成 | arXiv 1803.07616 |
| GRASP | video MLLM 的 language grounding + situated physics 兩層 | 兩階：grounding（顏色/形狀）+ intuitive physics（permanence/continuity 等）；模型多在 chance（約 50%），人類約 80% | grounding 嚴重依賴 prompt；Unity 合成；測理解非測 rollout | arXiv 2311.09048 |
| V-JEPA 2 / V-JEPA-2-AC | latent video WM 的 understanding / prediction / planning + zero-shot 機械控制 | SSv2 motion（自報 77.3 top-1）、EK100 anticipation（自報 39.7 R@5）；V-JEPA-2-AC 用 image-goal planning 在 Franka 上 zero-shot pick-and-place | 控制評測規模小（單臂、少數任務）；成功率高方差；non-action-free 部分需互動資料 | arXiv 2506.09985 |
| DreamerV3 (model-based RL) | WM 拿去 RL 的 sample-efficiency + return | 跨 150+ 任務 / 8 域，含 **Atari100k**（離散密 reward）、**DMC Vision**（連續稀 reward）、**Crafter**、BSuite、DMLab；固定超參 | Atari100k 只給 100k 步、方差大、易調參過擬合；單一 seed 數量少時排名不穩；return 是真 ground truth 但訓練貴 | arXiv 2301.04104 |

備註：Atari100k / DMC / Crafter 本身是「環境套件 + 評測協議」（sample efficiency + 平均 return / score），不是單一論文 benchmark；DreamerV3 是 model-based RL WM 在這些套件上的代表性評測。

## 怎麼誠實讀分數

- **short-horizon-only 偏差**：絕大多數 video WM benchmark（WorldModelBench、LikePhys）只評幾秒短 clip，分數高不代表長 rollout 不 drift。短片段保真 ≠ 長程一致。
- **replay ≠ predict**：teacher-forcing / 給足 context 的「重建」分數，和真正 free-running autoregressive rollout 是兩回事；後者才暴露誤差累積。報分時要問是哪一種。
- **reconstruction quality ≠ planning value**：FVD/PSNR 漂亮的 WM 不保證能拿來 planning；判別式 intuitive-physics 過關也不等於生成式 rollout 守物理。
- **downstream return 才是 ground truth**：DreamerV3 那條路（policy return / 任務成功率）是唯一直接量「WM 有沒有用」的，但它最貴、方差最大、且把 WM 與 policy/explorer 糾纏在一起，不易單獨歸因到 WM。

## 現況與缺口

- **長程一致性沒有公認 metric**：autoregressive WM 跑久了會 drift（誤差累積、distributional shift → 模糊、kinematic drift、結構違規），但「FVD-over-time / drift 曲線」尚無標準協議與報告慣例。具體的時間閾值自報（例如 Cosmos 類模型「超過某秒數開始 drift」）多散見於各家技術報告，**缺乏統一可比的 ">Ns drift" 量測**（具體秒數 UNVERIFIED）。
- **互動 WM 無公開協議**：Genie 2 / Genie 3 這類 real-time interactive WM 只有官方 blog，自報「可保持一致性數分鐘 / 720p / 24fps」，但**無公開 benchmark 協議、無可重現的 eval set**（UNVERIFIED）。互動可控性 + 長程記憶的客觀評測是最大空白。
- **三層各管各的**：fidelity（WorldModelBench/LikePhys）、physics-understanding（IntPhys/GRASP）、planning-value（DreamerV3/V-JEPA-2-AC）各有 benchmark，但缺少把三者串起來、能歸因「保真→一致→可規劃」因果鏈的整合評測。

## 連回

- [benchmarks/ overview](../overview.md)
- [foundations/evaluation-physics](../../foundations/evaluation-physics/overview.md)
- [foundations/latent-world-models — DreamerV4](../../foundations/latent-world-models/dreamer-v4.md) · [V-JEPA 2](../../foundations/latent-world-models/v-jepa-2.md) · [Genie 2](../../foundations/latent-world-models/genie-2.md)
- [foundations/long-horizon-rollout](../../foundations/long-horizon-rollout/overview.md)
- [crossing/pixel-vs-latent-physics](../../crossing/pixel-vs-latent-physics/overview.md)

## 參考

- WorldModelBench: Judging Video Generation Models As World Models — arXiv 2502.20694
- LikePhys: Evaluating Intuitive Physics Understanding in Video Diffusion Models via Likelihood Preference — arXiv 2510.11512
- IntPhys 2: Benchmarking Intuitive Physics Understanding In Complex Synthetic Environments — arXiv 2506.09849
- IntPhys: A Framework and Benchmark for Visual Intuitive Physics Reasoning — arXiv 1803.07616
- GRASP: A novel benchmark for evaluating language GRounding And Situated Physics understanding — arXiv 2311.09048
- V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning — arXiv 2506.09985
- DreamerV3: Mastering Diverse Domains through World Models — arXiv 2301.04104
- Genie 2 / Genie 3 (Google DeepMind, blog-only, 無公開 benchmark 協議 — UNVERIFIED)
- Cosmos World Foundation Model Platform for Physical AI — arXiv 2501.03575（long-horizon drift 具體秒數 self-report UNVERIFIED）
