<!-- ontology-5axis output=action-seq injection=data-only control=text|image-init temporal=streaming domain=robotics -->

# Physical Intelligence π0 / π0.5 — Anchor Use-Case Dissection

> Black et al., **π0: A Vision-Language-Action Flow Model for General Robot Control**, arXiv [2410.24164](https://arxiv.org/abs/2410.24164) (Oct 2024)
> Physical Intelligence + 35 authors, **π0.5: a Vision-Language-Action Model with Open-World Generalization**, arXiv [2504.16054](https://arxiv.org/abs/2504.16054) (Apr 2025)
> Code/weights: [`github.com/Physical-Intelligence/openpi`](https://github.com/Physical-Intelligence/openpi) (released 2024-09; π0.5 weights 2025-Q2)
>
> 為什麼進 robotics-data-gen anchor 名單：PI 是本 handbook 整條 robotics 線的**下游終點客戶**。Cosmos-Predict / Sora robot variants / RoboCasa / V-JEPA-2-AC 都把 "生成出來的 video / latent / sim demo" 喂給某個 generalist VLA 做 pre-train——而 2024-25 該位置的事實標準就是 π0。任何 robotics 合成資料路線的 ROI 評估，最終必須落到「PI policy 在真機 success rate 提升多少」這個唯一 ground truth。

---

## 1. TL;DR

**PI 的 bet**：跨 embodiment 真實遙操資料 + VLM backbone + flow-matching action head → 一個 50Hz 連續控制的 generalist policy。π0 (2024-10) 在 7 種 robot 平台、超過 10,000 小時遙操資料上訓練，從 PaliGemma 3B VLM 初始化，用 flow matching 出 50Hz action chunk，把 OpenVLA-style autoregressive token VLA 在 dexterous task（疊衣、整理桌面、組裝紙盒）上甩開一個量級。π0.5 (2025-04) 在 π0 之上加 **異質 co-training**——多機器人 demo + web data + 高層 semantic prediction + object detection——第一次把端到端 learning system 跑到「整理沒見過的家庭廚房 / 臥室」這個 open-world generalization 等級。

**為什麼這篇是 physics-gen handbook 的關鍵**：本倉所有 robotics-data-gen 路線（pixel-video / latent-WM / sim-augment）最終都要回答**「合成資料能不能讓 π0-class policy 在真機 success rate 上漲？」**——這是唯一不會自欺欺人的 metric。PI 的工程細節（action chunk 規格、normalization 規約、camera layout）反過來變成所有合成資料 generator 的硬規格約束（[openpi #872](https://github.com/Physical-Intelligence/openpi/issues/872)、[#449](https://github.com/Physical-Intelligence/openpi/issues/449)）。

---

## 2. Core mechanism

### π0：VLM backbone + flow-matching action expert

```
   image_t (3 cams) ─┐
   image_{t-1}     ─┤
   text prompt     ─┤──►  PaliGemma 3B VLM   ──► cross-attn ──┐
   robot proprio   ─┘     (frozen / lightly tuned)            │
                                                              ▼
                                          ┌───────────────────────────────┐
                                          │  Action Expert (~300M params) │
                                          │  flow matching, 10 steps      │
                                          │  → action chunk a_{t:t+H}     │
                                          │  H = 50 (1 sec @ 50Hz)        │
                                          └──────────────┬────────────────┘
                                                         ▼
                                          7-DoF × 2 arms / mobile / multi-EE
```

- **Flow matching** (vs DDPM, vs autoregressive token VLA)：直接學 velocity field $v_\theta(a, t \mid o)$，推理 10 步 ODE 解出 1 秒 action chunk；比 OpenVLA 的 7-token autoregressive 至少**省 5× 推理延遲**，且支援連續 high-frequency control。
- **Action expert** 是個獨立的 transformer 並行於 PaliGemma 後段——VLM 不負責出 action token，只出 conditioning。VLM 與 action expert 之間用 cross-attention 接（[#951 "Pi0.7 cross-attention between vlm and action expert"](https://github.com/Physical-Intelligence/openpi/issues/951) 把這層暴露出來）。
- **資料**：自家 fleet 7 種 embodiment（UR5 / bimanual Franka / Trossen / mobile arm）+ Open X-Embodiment 子集；總時長 10,000h 級。

### π0.5：異質 co-training + 高層 semantic prediction

π0.5 沿用 π0 的雙塔結構，但在訓練資料與目標函數兩端都加維度：

- **資料**：多機器人 demo + **web image-text** + **object detection labels** + **semantic subtask annotation**（"now picking the cup", "now wiping the counter"）。
- **損失**：除了原本 flow-matching action loss，再加 (1) high-level semantic subtask prediction (autoregressive text)（2）low-level action 同時學。本質上是 VLM 那一端從「只當 conditioner」升級為「同時是 high-level planner」。
- **效果**：第一次有端到端 learning system 在**從未見過的真實家庭**做 long-horizon 任務（整理廚房、收拾床鋪）。論文核心句：**"knowledge transfer is essential for effective generalization"**——這正是合成資料路線的存在理由。

---

## 3. 五軸定位 + 同軸對手

| Axis | π0 / π0.5 | 註 |
|---|---|---|
| 1. Output | `action-seq` | 50Hz action chunk（H=50）；不出像素、不出 3D |
| 2. Injection | `data-only` | 物理隱式從 10k+ hr 真實遙操 + web data 學會；無 sim、無 PDE、無 hard constraint |
| 3. Control | `text` + `image-init` | text instruction + 多相機當前幀；π0.5 加 object detection 條件，但仍歸 image |
| 4. Temporal | `streaming` | 50Hz 連續控制；action chunk 之間 receding horizon 滾動 |
| 5. Domain | `robotics` | 多 embodiment 但統一是物理機器人 |

**同軸對手（output=action-seq, domain=robotics）**

| 對手 | 五軸主要差異 | 對比要點 |
|---|---|---|
| **OpenVLA / RT-2** (2307.15818 / 2406.09246) | injection=data-only, temporal=autoregressive | autoregressive token VLA → 推理慢 5-10×，無法 50Hz；π0 用 flow matching 換到 continuous control，這條路線後續所有 VLA（GR00T, RDT, Octo）都跟進 |
| **[V-JEPA-2-AC](../../foundations/latent-world-models/v-jepa-2.md)** | output=latent-tokens, control=action+image-init | 走 latent-WM 路線，**62h** robot video 就能 zero-shot pick-place；π0 路線需 10,000h 真實遙操。兩者是 "scale data" vs "scale representation" 的世紀對賭 |
| **[Cosmos-Predict](../../foundations/foundation-physics-models/cosmos-wfm.md) + VLA stack** | output=pixel-video, injection=data-only | NVIDIA 把 Cosmos 當 "video pre-train" 餵下游 VLA（GR00T），假設「video physics prior 可以 transfer 到 action」；π0 證明「不經過 pixel 也可」，這條 stack 是否值得 = robotics-data-gen 子路線 1 的核心問號 |
| **DreamerV4 / Decart Oasis** | output=latent-tokens, temporal=latent-rollout/streaming-cache | World-model + planner 路線，但都還沒在真機 dexterous task 達到 π0 量級 |

**Cross-axis 必要說明（per Check 9b）**：`output=action-seq + injection=data-only` 在矩陣裡是 ✓ 合法格，無 §8 必解釋條款。`domain=robotics` 非 generalist，符合 Check 9c 白名單。

---

## 4. ⚡ shines / ❌ breaks

### ⚡ 真正領先的 regime

- **Dexterous long-horizon 任務**：疊衣服（90 sec+）、組裝紙盒、清桌面——OpenVLA / RT-2 在這些任務上幾乎是 0%，π0 報 60-90%。
- **跨 embodiment scale**：同一份 weights 在單臂、雙臂、移動底盤都能 fine-tune，是目前公開模型中最 "broad" 的 generalist policy。
- **學習速度**：PI 內部報「100h 新任務遙操 → 上線」級別的 fine-tune 速度；π0.5 進一步把這個門檻拉低（部分新家庭僅靠 high-level data + 少量 demo）。
- **連續高頻控制**：50Hz 是 dexterous bimanual 的 viable 下限，autoregressive VLA 達不到。

### ❌ Known failure modes

[Penn PAL Lab 第三方 in-the-wild 評估](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/) 把 failure 落地到具體百分比：

- **OOD 物件**：玻璃茶壺倒水 **0%**、玩具廚房櫃門 **0%**、咖啡機操作 **0%**——任何 PaliGemma 沒見過的 articulation / 透明物件直接崩。
- **Instruction 模糊**："Close the toilet" 0% vs "Close the white lid for the toilet" 100%。nonsense input 會默認抓 training-dominant 物件（如 marker pen）。
- **Contact-rich force control**：「對細物（手指）施力過大、對重物施力過小」——沒有 tactile feedback，只能靠視覺猜，clearance 估錯就撞碗。
- **Memory-less freeze**：架構無 cross-chunk memory，手在把手上時視覺像 idle → policy 凍結。腕部相機完全遮擋 → 0% 進度。
- **Cross-embodiment 不夠 free**：[openpi #449](https://github.com/Physical-Intelligence/openpi/issues/449) 與 [#872](https://github.com/Physical-Intelligence/openpi/issues/872) 反覆問「換到 Mobile Franka Panda 12-DoF 要改什麼」——share projection layer 還是 separate encoder 沒有 canonical 答案，[Discussion #740](https://github.com/Physical-Intelligence/openpi/discussions/740) 把 "π0 到底是不是真 cross-embodiment" 直接拿出來辯。

---

## 5. Reproduction notes

- **openpi repo**：[github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)，2024-09 release，223+ open issues（截 2026-05）。
- **GPU 預算**：fine-tune 一個下游 task：8× A100 80GB 一晚（~1B params total），LoRA 變體 ([#944](https://github.com/Physical-Intelligence/openpi/issues/944)) 可降到 2-4× A100。從頭訓 ([#436](https://github.com/Physical-Intelligence/openpi/issues/436)) 需要 PI 級別資料（10,000h+）+ 64+ A100 數週——基本不可複現。
- **典型踩坑**：
  - Action dim hardcode 為 7 或 8（Franka），改 12-DoF mobile arm 要改 `src/openpi/training/config.py`（[#872](https://github.com/Physical-Intelligence/openpi/issues/872)）。
  - π0.5 checkpoint 的 SigLIP positional embedding 在 patch grid 上 L2 norm 嚴重不均（[#947](https://github.com/Physical-Intelligence/openpi/issues/947)），影響 finetune 收斂。
  - LeRobot dataset + JointPos action 的 config 還在摸索（[#933](https://github.com/Physical-Intelligence/openpi/issues/933)）。
  - LIBERO 推理 reproduce 有暗坑（[#936](https://github.com/Physical-Intelligence/openpi/issues/936)）。

---

## 6. Cross-line synthesis

PI 是本 handbook 4 條技術路線的**下游消費端**，不是 producer：

- **Pixel-WM → π0**：NVIDIA GR00T 路線假設 Cosmos-Predict 生成的機器人 video 能當 pre-train 燃料。但 PI 自己沒走這條（π0 paper 完全不依賴合成 video），這是 robotics-data-gen 子路線 1（[overview](./overview.md)）的核心驗證題：合成 video 替代多少真實遙操小時？目前公開資料**沒有 head-to-head 證據**證明 Cosmos→π0 比同預算多收真實 demo 划算。
- **Latent-WM → π0**：[V-JEPA-2-AC](../../foundations/latent-world-models/v-jepa-2.md) 用 62h robot video + frozen encoder 做到 zero-shot pick-place，PI 用 10,000h 真實遙操做到 long-horizon dexterous——一個 representation-pilled，一個 scale-pilled。Cross-pretrain（V-JEPA encoder 接 π0 action head）是公開 backlog 沒人做過的明顯實驗。
- **Diff-sim → π0**：Genesis / MJX 還沒成為 PI 公開的訓練資料來源；[RoboCasa](../../foundations/data-engine/robocasa.md) sim demo 進入 PI 訓練池的證據也不足。PI 的 stance 近似「真實遙操 > sim」——這是合成路線最大威脅。
- **Surrogate → π0**：FNO/GraphCast 這層離 robotics 太遠，無直接接點。

**跨 handbook 引用**：
- VLA-Handbook 把 π0/π0.5 列為 VLA architecture 章的旗艦案例（action-output 端權威）。
- 本倉視角：π0 是**驗證合成資料 ROI 的 ground truth**——任何 robotics-data-gen 路線的成敗在「PI policy success rate 提升 N%」這個數字上見真章。
- Bridge：[`/bridge-to-vla/generative-data-for-vla.md`](../../bridge-to-vla/overview.md)。

---

## 7. References

**Canonical**
- π0: Black, Brown, Driess, Esmail et al., arXiv [2410.24164](https://arxiv.org/abs/2410.24164), Oct 2024
- π0.5: Physical Intelligence + 35 authors, arXiv [2504.16054](https://arxiv.org/abs/2504.16054), Apr 2025
- π0 blog: [physicalintelligence.company/blog/pi0](https://physicalintelligence.company/blog/pi0)
- π0.5 paper PDF: [pi.website/download/pi05.pdf](https://www.pi.website/download/pi05.pdf)

**Secondary / wild eval**
- [GRASP Lab "Evaluating π0 in the Wild"](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/) — 第三方 OOD failure 量化
- [Cloderic notes on π0](https://www.cloderic.com/content/2025-02-27-notes-on-pi0)
- [openpi repo issues / discussions](https://github.com/Physical-Intelligence/openpi/issues)

---

## §8 Pitfall log

> Severity 標尺：🔴 blocker · 🟠 major · 🟡 minor。

### §8.1 🔴 Cross-embodiment 不是自動的（[#449](https://github.com/Physical-Intelligence/openpi/issues/449), [#872](https://github.com/Physical-Intelligence/openpi/issues/872), [Discussion #740](https://github.com/Physical-Intelligence/openpi/discussions/740))

Action dim 在 `src/openpi/training/config.py` 是 7/8（Franka）hardcode。換 12-DoF mobile arm 要自己改 config + normalization stats，並且 shared projection vs separate encoder 沒 canonical 答案。**Workaround**：先 LoRA fine-tune 在新 embodiment 的 200+ demo 上；不要期待 zero-shot。社區共識：「π0 是 multi-embodiment trained，不等於 cross-embodiment generalizing」。

### §8.2 🔴 OOD 物件直接崩到 0%（[Penn PAL eval](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/))

玻璃茶壺 / 玩具櫃 / 咖啡機 success rate **0%**。**根因**：PaliGemma 沒看過該物件 + 沒有 LLM-style commonsense fallback。**Workaround**：先用 detection prompt（π0.5 引入 object detection 條件）強制 grounding；OOD task 不要直接上 π0，先收 100+ demo fine-tune。**對 physics-gen handbook 的啟示**：合成 video 補 OOD 物件覆蓋率是有明確 ROI 的方向（如果 transfer 真的成立）。

### §8.3 🟠 Prompt 措辭極度敏感（[Penn PAL eval](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/))

"Close the toilet" → 0%；"Close the white lid for the toilet" → 100%。Nonsense input 退化到 training-dominant 物件。**Workaround**：production deployment 必須加一層 LLM rewriter normalize 指令；不要把使用者原話直接喂 policy。

### §8.4 🟠 Memory-less freeze + 腕部遮擋崩潰（[Penn PAL eval](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/))

架構無 cross-chunk memory，手放在把手上時視覺像 idle scene → policy 凍結。腕部相機完全遮擋 → 0% 進度，無 active search。**Workaround**：上層加 task-state FSM 強制推進；外部 retry / nudge 邏輯；硬體上保證 wrist cam 不被衣物完全擋住。

### §8.5 🟠 π0.5 SigLIP positional embedding 失衡（[#947](https://github.com/Physical-Intelligence/openpi/issues/947))

官方 π0.5 checkpoint 載入後 SigLIP 學到的 posemb 在 patch grid 上 L2 norm 有極端不均。Downstream finetune 在某些 spatial region 收斂異常。**Workaround**：先 audit posemb norm；考慮重新初始化或 LoRA 只在 action expert 而不碰 vision tower。

### §8.6 🟡 從頭訓不可複現 / LIBERO 推理暗坑（[#436](https://github.com/Physical-Intelligence/openpi/issues/436), [#936](https://github.com/Physical-Intelligence/openpi/issues/936))

從頭訓需要 PI 級別 10,000h+ 資料 + 64+ A100 數週——基本上只能用發布的 checkpoint。LIBERO benchmark 上推理結果與論文對不齊，目前 issue 未閉。**Workaround**：用 LoRA fine-tune ([#944](https://github.com/Physical-Intelligence/openpi/issues/944)) 路線；reproduce LIBERO 數字前對齊 action chunk 規格 + normalization stats。

### §8.7 🟡 Force / contact-rich 任務無 tactile（[Penn PAL eval](https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/))

純視覺輸入無法估計握力強度，捏細物太用力 / 抓重物太鬆。**Workaround**：硬體加 tactile sensor 並在 fine-tune 加 force 通道（[Force Prompting](../../foundations/physics-conditioning/force-prompting.md) 路線的下游應用點）；或限制部署任務避開 force-critical 場景。

---

> Cross-axis descriptive notes（per ontology v2 9-Descriptive）：本條為 `injection=data-only`，與 `temporal=streaming` 完全相容；本條 `control=text|image-init` 在 `domain=robotics` 範圍內合法（不需額外解釋條款）。π0/π0.5 不在 generalist 白名單（Check 9c），明確標為 robotics。
