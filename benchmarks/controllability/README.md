# Controllability Benchmarks

> 量「給定 control 信號，輸出服從多少、又損失多少保真度」的 Pareto：可控性與保真度搶同一份生成預算。

## 這類在測什麼

可控生成有兩個相互拉扯的指標。**Controllability（可控性）**問：輸出有多服從 condition（軌跡、edge map、depth、segmentation、instruction、ego 軌跡）？通常用「條件還原誤差」量——把生成結果丟回同一個 detector，看抽出的 condition 與輸入差多少。**Fidelity（保真度）**問：撇開服不服從，畫面本身的視覺/物理品質、時序穩定度、與真實分佈的距離（FID/FVD）如何？

兩者搶同一份生成預算：把 condition 壓得越死（高 guidance、強 ControlNet 權重），輸出越服從，但越容易僵硬、失真、多樣性崩塌；放鬆 condition 則畫面自然但飄離指令。這條取捨曲線就是 controllability-fidelity Pareto——而幾乎沒有 benchmark 把兩軸畫在同一張圖上。

## Benchmark 全景

| Benchmark / 指標 | 測什麼 | 主要 metric | 已知缺陷 / 可被 game | 出處 |
|---|---|---|---|---|
| **ObjMC**（MotionCtrl / DragNUWA） | 軌跡條件視頻：物體是否沿給定軌跡走 | 預測軌跡與 GT 軌跡的 Euclidean 距離（越低越好；DragNUWA ObjMC≈326.5） | 只看軌跡點對齊，不管物理合理性；軌跡 tracker 本身有誤差；GT 軌跡來自同一 detector 易自我循環 | [2306.14435 (MotionCtrl)](https://arxiv.org/abs/2306.14435) · [2308.08089 (DragNUWA)](https://arxiv.org/abs/2308.08089) |
| **TSE / CSE / FVD**（Cosmos-Drive-Dreams） | 軌跡條件駕駛視頻：ego 軌跡、相機是否對齊 layout | Trajectory Spatial Error、Camera Spatial Error、FVD | TSE/CSE 量空間對齊，FVD 量分佈品質——兩者分報，不合成單一 Pareto；多視角一致性另計 | [2506.09042](https://arxiv.org/abs/2506.09042) |
| **Edge F1 / Blur SSIM / Depth si-RMSE / Mask mIoU**（Cosmos-Transfer1） | 空間 condition 還原度：生成結果重抽 edge/depth/seg 與輸入差多少 | Canny F1（edge=0.28↑）、模糊後 SSIM（vis=0.96↑）、Depth si-RMSE（=0.49↓）、Mask mIoU（seg=0.68↑） | 條件還原度高≠品質高（Quality Score 另計）；Edge F1 絕對值低（0.28），單看會誤判「不可控」 | [2503.14492](https://arxiv.org/abs/2503.14492) |
| **Quality Score（DOVER-technical）**（Cosmos-Transfer1） | 保真度側：撇開服從度的整體視覺品質 | DOVER technical score（multimodal uniform≈8.54 ↑ vs 單模態 5.48–6.51） | 與上面的對齊 metric 分開報——服從度高的單模態品質反而低，正是 Pareto 但未連線 | [2503.14492](https://arxiv.org/abs/2503.14492) |
| **ControlNet 條件 fidelity**（多 condition） | edge/depth/seg/pose 等空間條件能否被加上且互不干擾 | 各 condition 還原一致性（任務各異）；多 condition 同時加 | 多 condition 互相干擾（multi-condition interference）少有系統量化；論文示範多 condition 但無干擾基準 | [2302.05543](https://arxiv.org/abs/2302.05543) |
| **VBench 維度**（控制相關子集） | 16 維裡哪些反映「服從」 | Quality 側：subject/background consistency、temporal flickering、motion smoothness；Semantic 側：object class、spatial relationship、color、human action（prompt 服從） | 量的是 prompt/時序服從，非顯式空間 condition；各維獨立，不出 Pareto；維度可被偏科 game | [2311.17982](https://arxiv.org/abs/2311.17982) |
| **CFG guidance 曲線**（核心 Pareto 旋鈕） | 提高 guidance scale ω：服從↑、保真↓ | 沿 ω 掃出 IS↑/FID↑（或 precision↑/recall↓）的取捨曲線 | 這是「唯一」公認的 controllability-fidelity 取捨量法，但只對 text/class condition，不涵蓋空間/軌跡 condition | [2207.12598](https://arxiv.org/abs/2207.12598) |
| **Edit/Instruction 服從**（Emu Edit / InstructPix2Pix） | 指令編輯：改對了沒、又保留原圖多少 | CLIP directional similarity（指令服從）vs CLIP-img/DINO（內容保留），沿 edit strength 掃 | 「改得多」與「保留多」天生競爭，是另一條 Pareto；CLIP 分數對語義細節不敏感、可被 game | [2211.09800 (InstructPix2Pix)](https://arxiv.org/abs/2211.09800) |

## 怎麼誠實讀分數

- **controllability 與 fidelity 幾乎都分開量**：Cosmos-Transfer1 把 Edge F1/Blur SSIM（服從度）與 DOVER Quality Score（保真度）並列卻不連成曲線——讀者得自己腦補 Pareto。看到單一 leaderboard 數字時要問「另一軸在哪」。
- **CFG scale 是隱藏旋鈕**：同一模型把 guidance ω 調高，服從度與 FID 同時變化。不報 ω（或不報整條掃描曲線）的對比，等於只在 Pareto 上挑一個對自己有利的點。
- **單模態強 ≠ 多模態強**：多 condition 同時加會互相干擾（multi-condition interference）。在單一 condition 上刷高分，不保證多條件混合時仍服從——這層幾乎沒有標準 benchmark。
- **條件還原 metric 會自我循環**：用生成結果重抽 condition 來打分，detector 的偏誤會同時污染 GT 與預測；Edge F1 絕對值偏低（如 0.28）多半是 detector 噪聲，不是「不可控」。

## 現況與缺口

最大缺口：**沒有公認的「controllability-fidelity Pareto」benchmark**。CFG 那條 IS/FID 取捨曲線（[2207.12598](https://arxiv.org/abs/2207.12598)）是唯一被廣泛接受的取捨量法，但只覆蓋 text/class condition，不含軌跡、edge、depth、segmentation 等空間/結構 condition。空間 condition 這邊（Cosmos-Transfer1、ControlNet 系）服從度與品質各報各的，沒人沿控制強度系統掃出整條前緣。其次，**多 conditioning 干擾**（multi-condition interference）少有系統量測——多 condition 混合下「該服從誰、犧牲誰」基本靠人眼。第三，**物理保真**幾乎缺席：上述 metric 多在量「視覺對齊」與「分佈距離」，沒一個檢查服從 condition 後輸出是否仍守物理（見 evaluation-physics）。

> 註：本頁引用的 Cosmos-Transfer1 數值（Edge F1=0.28、Blur SSIM=0.96、Depth si-RMSE=0.49、Mask mIoU=0.68、Quality Score≈8.54 vs 5.48–6.51）取自 [2503.14492](https://arxiv.org/abs/2503.14492) Table 1；DragNUWA ObjMC≈326.5 為轉述值，`UNVERIFIED`（未核對原表，僅供量級參考）。其餘均為 arXiv source-grounded。

## 連回

- [benchmarks/ overview](../overview.md)
- [foundations/evaluation-physics](../../foundations/evaluation-physics/overview.md)
- [crossing: controllability-vs-fidelity](../../crossing/controllability-vs-fidelity/overview.md)
- [crossing: text-action-trajectory-spectrum](../../crossing/text-action-trajectory-spectrum/overview.md)
- [foundations/physics-conditioning](../../foundations/physics-conditioning/overview.md)

## 參考

- Ho & Salimans, *Classifier-Free Diffusion Guidance* — [2207.12598](https://arxiv.org/abs/2207.12598)
- Zhang, Rao, Agrawala, *Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)* — [2302.05543](https://arxiv.org/abs/2302.05543)
- Huang et al., *VBench: Comprehensive Benchmark Suite for Video Generative Models* — [2311.17982](https://arxiv.org/abs/2311.17982)
- *Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control* — [2503.14492](https://arxiv.org/abs/2503.14492)
- *Cosmos-Drive-Dreams: Scalable Synthetic Driving Data Generation with World Foundation Models* — [2506.09042](https://arxiv.org/abs/2506.09042)
- Wang et al., *MotionCtrl: A Unified and Flexible Motion Controller for Video Generation* — [2306.14435](https://arxiv.org/abs/2306.14435)
- Yin et al., *DragNUWA* — [2308.08089](https://arxiv.org/abs/2308.08089)
- Brooks et al., *InstructPix2Pix: Learning to Follow Image Editing Instructions* — [2211.09800](https://arxiv.org/abs/2211.09800)
