# Video Physics Benchmarks

> 評測「生成影片是否物理可信」的 benchmark 目錄：守恆/運動/常識三類考點、各自的 metric、以及各自怎麼被 game。

## 這類在測什麼

這一格不評「影片美不美」，只評「影片裡的物理對不對」。具體拆成三種可信度：
- **守恆類**（質量/動量/能量是否守恆，物體是否憑空出現消失）—— 最難，現有模型最常翻車。
- **運動類**（拋體軌跡、碰撞、流體、光影反射是否符合直覺力學）。
- **常識類**（text-to-video 是否生成出 prompt 描述的真實世界互動，如「斜面上的彈珠會滾下來」）。

關鍵教訓：這些考點和「視覺保真度」幾乎正交 —— 一段看起來極真實的影片，物理可以完全是錯的（見下方 Physics-IQ 的 r=-0.46 不顯著結果）。

## Benchmark 全景

| Benchmark | 測什麼 | 主要 metric | 已知缺陷 / 可被 game | 出處 |
|---|---|---|---|---|
| **Physics-IQ** | 真實物理的 switch-frame 預測（給條件幀，預測後 5 秒；396 影片 / 66 場景，含力學/流體/光學/熱學/磁性） | Spatial IoU + Spatiotemporal IoU + Weighted spatial IoU + MSE，聚合成 Physics-IQ 分（相對物理變異歸一） | **視覺真實度與物理理解不相關**（Pearson r=-0.46, p=.247，**不顯著**）；最佳模型（VideoPoet multiframe）僅達物理變異 baseline 的 24.1%；MSE 偏好「不動」的保守預測 | arXiv 2501.09038 |
| **VideoPhy** | text-to-video 物理常識（solid-solid / solid-fluid / fluid-fluid 互動） | 人評 joint score（語意遵循 ∧ 物理常識）；自動評估器 **VideoCon-Physics** | 依賴人評為金標準；最佳 CogVideoX-5B 僅 39.6% joint；auto-evaluator 與人評有 gap，跨新模型外推不保證 | arXiv 2406.03520 |
| **VideoPhy-2** | action-centric 物理常識（200 actions），重守恆律（質量/動量） | 人評 semantic + physical + rule grounding；自動評估器 **VideoPhy-AutoEval** | **hard subset 最佳模型 joint 僅 22%**；模型在守恆律上普遍失敗；hard 子集分數低=天花板效應，難分辨前段班 | arXiv 2503.06800（ICLR 2026） |
| **PhyGenBench / PhyGenEval** | physics-aware T2V 物理常識（160 prompts / 27 物理定律 / 4 domain） | **PhyGenEval** 三層階層評估：關鍵物理現象偵測 → 物理順序驗證 → 整體自然度，各層用 VLM + GPT-4o 生成的物理問題 | 評估鏈依賴 VLM/LLM judge，judge 本身物理能力是上限（見 PhysBench）；prompt 集偏「教科書式」單定律，組合場景覆蓋弱 | arXiv 2410.05363（ICML 2025） |
| **VBench** | 通用 video gen 品質，拆成 16 disentangled 維度 | 各維度專屬 metric，附人評偏好校準 | 16 維大多是 **perceptual quality**；物理只能從 motion smoothness / dynamic degree 間接推測，沒有真正的物理維度 | arXiv 2311.17982（CVPR 2024） |
| **VBench-2.0** | intrinsic faithfulness，新增 `Physics` 與 `Commonsense` 兩個 top-level 維度 | Physics 子維（State Change / Mechanics / Material / Multi-View Consistency）+ Commonsense 子維（Motion Rationality / Instance Preservation 等） | 仍以 VLM/規則打分，子維分數可被「保守、低動態」影片刷高；單維高分不保證跨維一致 | arXiv 2503.21755（Mar 2025） |
| **PhysBench** | **VLM 對物理世界的「理解/感知」**（非純 video-gen；10,002 video-image-text / 4 domain × 19 sub-class） | 多選/判斷準確率，8 capability dimension | 評的是 **evaluator 自己懂不懂物理**，不是生成器 —— 它決定上面那些 VLM-as-judge 鏈的可信度上限；75 個 VLM 普遍弱於常識推理 | arXiv 2501.16411（ICLR 2025） |
| **Morpheus** | video gen 的物理推理，用**真實物理實驗**對照（80 真實影片，守恆律導向） | physics-informed metric（把影片映射到共同物理表徵，用 PINN + VLM 比對守恆量） | 用真實實驗當地面真值，覆蓋場景有限（80）；映射到物理表徵本身有誤差；「美觀但違物理」仍是普遍結論 | arXiv 2504.02918 |

> 標 `UNVERIFIED`：T2VPhysBench / WorldModelBench-physics / Cosmos-Reason-as-evaluator 在本次檢索中未取得可確認的 arXiv ID + 指標細節，故**不列入主表**，待補。PhyWorldBench（arXiv 2507.13428）僅見標題、未細查指標，亦標 `UNVERIFIED`。

## 怎麼誠實讀分數

- **高分代表什麼**：在「該 benchmark 的特定物理切片 + 該 metric」下，生成影片比同類更不違物理。不代表跨守恆軸都對。
- **高分不代表什麼**：不代表視覺真實度高 —— Physics-IQ 直接量到兩者**不相關**（r=-0.46, p=.247，統計上不顯著）。反向也成立：影片越像真的，物理越對是沒有根據的推論。
- **human-eval 偏 perceptual**：VideoPhy / VideoPhy-2 的金標準是人評，人類標註者也容易被「看起來順」誤導；joint score 把語意與物理綁在一起，單看總分會掩蓋「語意對但物理錯」的常見失敗模式。
- **VLM-as-judge 的天花板**：PhyGenEval / VBench-2.0 用 VLM 打物理分，而 PhysBench 顯示 VLM 自己物理就弱 —— judge 的物理能力是整條評估鏈的上限。
- **Goodhart 風險**：MSE 類 metric 獎勵「保守、低動態」的預測（不動就不會錯）；hard subset 分數普遍貼地板（VideoPhy-2 22%）會壓縮前段班的分辨力。當 benchmark 變成優化目標，這些縫隙就是被 game 的入口。

## 現況與缺口

SOTA 大致狀況：text-to-video 物理常識最佳 joint 分仍在 22%（VideoPhy-2 hard）到 39.6%（VideoPhy）區間，Physics-IQ 最佳模型只摸到物理變異 baseline 的 24.1% —— 即「離真懂物理還很遠」。

最大缺口：**沒有單一 benchmark 同時覆蓋 5 條守恆軸**（質量/動量/能量/角動量/電荷）。現有 benchmark 各自挑一塊（Physics-IQ 重感知預測、VideoPhy 系列重常識、Morpheus 重守恆但只 80 場景、PhyGenBench 重單定律），守恆律的系統性覆蓋仍是空白；跨 benchmark 分數也不可直接相加比較。

## 連回

- [benchmarks/ 索引](../overview.md) —— 四大 benchmark 分區的入口。
- [evaluation-physics 方法論](../../foundations/evaluation-physics/overview.md) —— 物理評估的底層方法論。
- [VBench / VBench-2.0 / PhysBench 細部](../../foundations/evaluation-physics/vbench-physics.md) —— eval suite landscape 深挖。
- [PhysBench 拆解](../../foundations/evaluation-physics/physbench.md) —— evaluator 物理理解的上限討論。
- [守恆違反地圖](../../crossing/conservation-violation-atlas/overview.md) —— 「5 守恆軸覆蓋缺口」對應的 crossing 楔子。
- [5 軸 ontology](../../cheat-sheet/ontology.md) —— eval benchmark 在 output/injection 軸為 N/A 的定位。

## 參考

- Physics-IQ — Do generative video models understand physical principles? — arXiv 2501.09038
- VideoPhy — Evaluating Physical Commonsense for Video Generation — arXiv 2406.03520
- VideoPhy-2 — A Challenging Action-Centric Physical Commonsense Evaluation in Video Generation — arXiv 2503.06800
- PhyGenBench / PhyGenEval — Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation — arXiv 2410.05363
- VBench — Comprehensive Benchmark Suite for Video Generative Models — arXiv 2311.17982
- VBench-2.0 — Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness — arXiv 2503.21755
- PhysBench — Benchmarking and Enhancing Vision-Language Models for Physical World Understanding — arXiv 2501.16411
- Morpheus — Benchmarking Physical Reasoning of Video Generative Models with Real Physical Experiments — arXiv 2504.02918
- PhyWorldBench — A Comprehensive Evaluation of Physical Realism in Text-to-Video Models — arXiv 2507.13428（`UNVERIFIED`，指標未細查）
