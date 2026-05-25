<!-- ontology-5axis output=pixel-video injection=sim-in-loop-infer|aux-loss control=image-init|force|param temporal=clip-parallel domain=rigid -->

# PhysGen — Rigid-Body Physics-Grounded Image-to-Video Generation

> Liu, Ren, Gupta, Wang. ECCV 2024. arXiv [2409.18964](https://arxiv.org/abs/2409.18964). Code: [stevenlsw/physgen](https://github.com/stevenlsw/physgen).

---

## 1. One-paragraph TL;DR

PhysGen 是 2024 年第一個把**硬 sim**（2D rigid-body solver, PyMunk）跟**學習式 video diffusion**（SEINE / SVD-class）端到端串起來的 image-to-video 系統。Prior art 兩端是斷的：純 data-driven I2V（Stable Video Diffusion / Sora-class）對「我給這個物體一個 30N 的水平力」這種顯式力學輸入沒有 controllability —— 模型只能從文字或起始幀「猜」要發生什麼；而 graphics 的 rigid simulator 可以精確算碰撞但渲染出來不像照片。PhysGen 的取捨是**把 simulator 放在中間做控制信號生成器**：image → 自動切分剛體 + 估參 → PyMunk 跑 2D dynamics → diffusion 把 simulated motion lift 回 photoreal video。它對 handbook ontology 上 Axis 2 (injection) 的價值在於**它同時是 `sim-in-loop` 與 `aux-loss`** —— sim 提供 trajectory ground truth，diffusion 端再以這條 trajectory 為 conditioning，這跟下游 Force Prompting / PhysGen3D 一系列「explicit force as control」工作的祖先就是它。

## 2. Core mechanism

三段式 pipeline。每段都是「foundation-model perception + 古典物理求解 + 條件式生成」的代表組合。

```
                    ┌──────────────────────────────────────────┐
   single image ──► │ Stage 1: Perception                      │
   (+ optional      │   - Grounded-SAM  → 物體 mask             │
    text label)     │   - GeoWizard     → depth / normal       │
                    │   - Intrinsic decomp → albedo / shading  │
                    │   - LLM/prior     → mass, friction, μ    │
                    └─────────┬────────────────────────────────┘
                              │  rigid bodies + params
                              ▼
   user input ────► ┌──────────────────────────────────────────┐
   (force F,        │ Stage 2: 2D rigid-body simulation        │
    torque τ,       │   PyMunk (Chipmunk 2D) solver            │
    init velocity)  │   t=0…T, dt~1/60s                        │
                    │   outputs: per-frame SE(2) pose / mask    │
                    └─────────┬────────────────────────────────┘
                              │  simulated motion trajectory
                              ▼
   inpainted bg ──► ┌──────────────────────────────────────────┐
   (Inpaint-        │ Stage 3: Diffusion render + refine       │
    Anything)       │   SEINE-style I2V with motion conditioning│
                    │   ControlNet-like flow / mask guidance    │
                    │   → photoreal video clip                  │
                    └──────────────────────────────────────────┘
                              │
                              ▼
                       16-frame physically grounded clip
```

關鍵 design 決定：
- **Sim 與 render 解耦**：simulator 不可微，但因為它只負責產生 conditioning，gradient 不需要穿過它 —— 這跟 Genesis-style 全 differentiable pipeline 的 cost 取捨完全不同。
- **2D 而非 3D**：PyMunk 是 Chipmunk 的 Python binding，平面剛體 only。作者刻意選 2D 是因為單張 image 估 3D 形狀+質量+摩擦是 ill-posed，2D 限定可以在 side-view / top-down 視角下做到 robust。代價在 §4 / §8。
- **Force 是顯式 input**：用戶在 image 上點一個方向 + magnitude，simulator 把它注入 → 跟 Force Prompting (NeurIPS 2025) 的 control 介面血緣相同，但 PhysGen 把 force → motion 的部分交給 explicit sim，Force Prompting 則完全靠 ControlNet 隱式學。

## 3. 五軸定位 + 同軸對手

| 軸 | 值 | 註 |
|---|---|---|
| Output | `pixel-video` | 16 frame RGB clip |
| Injection | `sim-in-loop` + `aux-loss` | PyMunk 在 inference loop；diffusion 端用 motion mask 做 conditioning |
| Control | `image-init` + `force` + `param` | force/torque + 可調 mass/friction |
| Temporal | `clip-parallel` | 一次 denoise 整段 clip，非 autoregressive |
| Domain | `rigid` | 2D 剛體，明確排除 soft/fluid |

同軸對手（output=pixel-video）：

- **Sora / Veo** — 純 `data-only`，無 force conditioning，long-horizon 物理崩潰是 known mode。詳見 [Sora dissection](../video-world-models/sora.md)。PhysGen 用 sim 補了 Sora 缺的「explicit force → trajectory」控制。
- **Force Prompting** (NeurIPS 2025, [force-prompting.md](./force-prompting.md)) — 同樣 force-conditioned I2V，但**沒有 explicit simulator**：直接讓 CogVideoX-5B + ControlNet 從 15k Blender-synthetic videos 學 force→motion mapping。PhysGen 是「sim 顯式 + render 學習」；Force Prompting 是「sim 完全藏進權重」。後者 generalize 更廣（不限剛體形狀），但 debug 完全黑盒。
- **PhysDiff** ([physdiff.md](./physdiff.md)) — guidance-gradient motion-only diffusion（人體動作 + 物理懲罰），不生像素；跟 PhysGen 在 injection 軸上同為「物理 + diffusion」但 output 不同（motion vs pixel）。
- **PhysDreamer** (Zhang et al., **ECCV 2024**, arXiv [2404.13026](https://arxiv.org/abs/2404.13026)) — 同期不同團隊（MIT/Stanford）的 3D 變體：給定靜態 3DGS，估材料 stiffness，用 MPM 在 3D 跑 elastic dynamics，再用 video model 蒸出 dynamics prior。PhysDreamer 強在 elastic / soft，PhysGen 強在 rigid-body contact；兩者互補，不直接競爭。

> 註：user prompt 把 PhysDreamer 標為 CVPR 2024，**實際是 ECCV 2024**；標題也不是 "Physical Property Estimation from Static 3D Models"，而是 "Physics-Based Interaction with 3D Objects via Video Generation"。此處以 arxiv / ECCV 官方為準。

## 4. ⚡ Where it shines / ❌ Where it breaks

⚡ Shines
- **Explicit, debug-able force conditioning**：你給 (Fx, Fy, τ)，PyMunk 直接吃；錯了能單獨打開 sim 看，不像 ControlNet-only 方法那樣只能 ablation。
- **Determinism inside the sim layer**：相同 seed + 相同 force → 相同 trajectory（diffusion 端仍隨機）。這對 evaluation 跟 prediction-vs-truth 對比是稀缺優勢。
- **Side-view / top-down rigid scenes 表現最好**：撞球、推箱、拋射、單擺、骨牌 —— PyMunk 設計初衷就是這類 2D game physics，accuracy 對 perception 而言「夠用」。
- **GPU 預算友善**：sim 完全 CPU，只有 diffusion stage 上 GPU；對比 PhysDreamer / PhysGen3D 需要 3D 表徵 + MPM/material 訓練便宜很多。

❌ Breaks
- **2D-only 是硬限制**：相機平移、繞物體軌道、out-of-plane 旋轉全部會破。
- **Rigid-only**：cloth、liquid、granular、軟橡皮全失敗 —— 而且不是「報錯」，是 silently 給出物理錯但看起來 plausible 的視覺，這在下游使用最危險。
- **Perception 是 cascading single-point-of-failure**：Grounded-SAM mask 切錯 / GeoWizard depth 錯 → simulator 給錯 trajectory → diffusion 強行 render → 用戶看不出但物理已崩。
- **Physical parameter 估計弱**：mass / μ 是用 prior 或粗略 heuristics 給的，沒有真實校準；同樣形狀的塑膠盒 vs 鐵盒 simulator 用的是同一組參數。
- **Clip 長度短**：典型 16 幀（~2 秒），長 horizon 跨 clip 接續會有 visual 跳變。
- **OOD 物體形狀**：不規則剛體被 PyMunk 近似成 convex polygon set，contact 點位置會錯，導致彈跳角度肉眼可辨偏差。

## 5. Reproduction notes

最小可跑 setup（驗證自 [stevenlsw/physgen](https://github.com/stevenlsw/physgen) README）：

```bash
git clone --recurse-submodules https://github.com/stevenlsw/physgen.git
conda create -n physgen python=3.9
conda activate physgen
pip install -r requirements.txt
# 額外需下載：SEINE checkpoint, Grounded-SAM weights, GeoWizard weights
```

- **GPU 預算**：repo 沒列明確 VRAM 數字 `[TBD: verify VRAM minimum from issues]`。經驗法則 SEINE + Grounded-SAM 同時載入需 ~24GB（單張 A5000/3090 級）。Sim 階段純 CPU。
- **依賴鏈**：Pymunk (sim) + Grounded-Segment-Anything (mask) + GeoWizard (depth/normal) + Inpaint-Anything (背景補) + SEINE (I2V diffusion)。任何一個 stage 換模型都會破整條 pipeline 的 conditioning 接口。
- **典型踩坑**：
  1. submodule clone 漏 `--recurse-submodules` → Grounded-SAM 缺權重。
  2. SEINE 權重需從另一 repo 手動下載，README 引用的 link 偶爾變動 `[TBD: confirm current SEINE checkpoint URL]`。
  3. 自帶 demo 只在 side-view / top-down image 上 robust；正面視角 demo 視覺 OK 但 sim 完全錯（contact 點被投影壓扁）。

## 6. Cross-line synthesis

PhysGen 在 handbook 五條技術路線上的接口：

- **vs `foundation-physics-models/`（Cosmos-WFM, [cosmos-wfm.md](../foundation-physics-models/cosmos-wfm.md)）**：Cosmos 走 `data-only` + `text` conditioning，強在 scale 與 generalization；PhysGen 走 `sim-in-loop` + `force`，強在 controllability。兩者可 compose：Cosmos 提供 photoreal prior，PhysGen 的 sim trajectory 作為 ControlNet-style guidance 注入。這是 2026 出現的「foundation video model + per-scene physics adapter」的早期原型。
- **vs `diffusion-physics/` PINN ([pinn.md](./pinn.md))**：PINN 走純 `aux-loss`（PDE residual 加到 loss）；PhysGen 的 `aux-loss` 是 sim 給的 motion mask 跟 diffusion 輸出之間的 L2/perceptual，是**間接 PDE constraint** —— 力學律已經被 PyMunk 解完，diffusion 只負責 visual match。所以 PhysGen 在 injection 軸上是 **hybrid (sim-in-loop ∩ aux-loss)** 而非單純 PINN。
- **vs `crossing/controllability-vs-fidelity/`**（[controllability-vs-fidelity](../../crossing/controllability-vs-fidelity/)）：PhysGen 是該 trade-off 圖上的一個明確 anchor —— controllability 高（force 顯式可調）但 fidelity 受 SEINE-class 渲染上限，且 domain 窄（rigid 2D）。Sora 在另一端：fidelity 高、controllability 弱。Force Prompting 嘗試 Pareto 中段。
- **vs `differentiable-simulators/`**：PhysGen 故意**不**用 diff-sim —— PyMunk 不可微，但因 gradient 不需穿過去（sim 只給 conditioning），整體訓練/推理成本遠低於 Genesis-train。代價是失去「用 diffusion loss 反推物理參數」的能力。

## 7. References

主要文獻
- **Liu, Ren, Gupta, Wang.** *PhysGen: Rigid-Body Physics-Grounded Image-to-Video Generation.* ECCV 2024. arXiv [2409.18964](https://arxiv.org/abs/2409.18964). [ECCV poster page](https://eccv.ecva.net/virtual/2024/poster/1012). [Springer ch.](https://link.springer.com/chapter/10.1007/978-3-031-73007-8_21).
- Project page: [stevenlsw.github.io/physgen](https://stevenlsw.github.io/physgen/) · Code: [github.com/stevenlsw/physgen](https://github.com/stevenlsw/physgen).

血緣 / 後續
- **Zhang et al.** *PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation.* ECCV 2024. arXiv [2404.13026](https://arxiv.org/abs/2404.13026). [project page](https://physdreamer.github.io/). — 3D + elastic 變體。
- **Chen et al.** *PhysGen3D: Crafting a Miniature Interactive World from a Single Image.* CVPR 2025. arXiv [2503.20746](https://arxiv.org/abs/2503.20746). [project page](https://by-luckk.github.io/PhysGen3D/). — 同團隊 / 同 family 的 3D 後繼。
- **Gillman et al.** *Force Prompting.* NeurIPS 2025. arXiv [2505.19386](https://arxiv.org/abs/2505.19386). [force-prompting.github.io](https://force-prompting.github.io/). — 取消顯式 sim，純 ControlNet 學 force conditioning。

工具
- **PyMunk** — Chipmunk 2D physics 的 Python binding. [pymunk.org](http://www.pymunk.org/).
- **SEINE** — Image-to-video diffusion backbone used by PhysGen.
- **Grounded-Segment-Anything** / **GeoWizard** / **Inpaint-Anything** — perception stack.

## 8. §8 Pitfall log

PhysGen 在實作 / 部署時的已知失效 + workaround。Severity: 🔴 silent-bad (看不出但物理錯) · 🟠 noisy-bad (視覺也壞) · 🟡 cost/setup。

### §8.1 🟠 PyMunk 2D-only — 視角錯就崩
PyMunk 是 Chipmunk 的 2D wrapper，所有物體被視為 SE(2) rigid body。任何含 out-of-plane motion（如球從遠處往相機飛）的 prompt 都會被 sim 強行投影成平面動作，輸出視覺與物理矛盾。
**Workaround**：強制限定 side-view / top-down 輸入；對 3D 場景改用 PhysGen3D 或 PhysDreamer。`[TBD: verify GitHub issue # tracking this]`

### §8.2 🔴 Rigid body decomposition cascade failure
Grounded-SAM mask 邊緣鋸齒 / 切到陰影 → PyMunk 把錯誤輪廓當 collider → contact 點全錯 → diffusion 仍然 render 出「貌似合理」的視覺。User study 中這類錯誤幾乎無法被肉眼察覺，但精確 force prediction 任務上會穩定偏差。
**Workaround**：手動 review mask；對關鍵 demo 用 SAM-HQ 替換 base SAM。

### §8.3 🟡 ControlNet-style conditioning weight 敏感
Stage 3 的 motion mask → diffusion 的 conditioning strength 對 photorealism vs sim-faithfulness 是直接 trade-off：強 → 視覺像 sim 截圖；弱 → 視覺漂亮但動作偏離 sim。沒有 principled 調法。
**Workaround**：per-domain 校 weight；論文 default 通常偏 sim-faithful。

### §8.4 🔴 Physical parameter estimation 弱
Mass / friction / restitution 是用 prior / heuristic 給的，沒有從 image 反推或從多幀觀察校準。同樣的紙箱跟同樣的鐵箱會被當成同一個物體。對「同形狀不同材質」demo 完全失靈。
**Workaround**：作為 user-overridable param 暴露給使用者；下游 PhysGen3D 在這軸有改進。

### §8.5 🔴 Non-rigid scenes silently fail
布料、水、煙、頭髮、繩子在 PyMunk 裡沒有對應 primitive，會被 forced 成 rigid polygon。Pipeline 不會報錯，diffusion 會 render 一個「rigid 化的布料」—— 視覺常常 plausible，但完全違背原本物理。
**Workaround**：對 input image 加 domain classifier，非 rigid 直接拒絕，引導至 PhysDreamer (elastic) 或專用 cloth/fluid 線。

### §8.6 🟠 Clip 長度 ~16 frames 撐不住 long-horizon
SEINE backbone 限制；超過 clip 長度需 stitch，而 sim 跨 clip 重新初始化容易丟動量。
**Workaround**：用 auto-regressive stitching + carry-over velocity；或改用 latent-rollout 後繼方法（PhysGen3D 部分緩解）。

### §8.7 🟡 Repo dependency hell
SEINE / Grounded-SAM / GeoWizard / Inpaint-Anything 各自 pin 的 torch / diffusers 版本不一致；conda env 解出來常衝突。
**Workaround**：用 README 指定的 python 3.9 + 鎖 requirements；新版 diffusers 不要升級。

### §8.8 🟠 Convex decomposition 對不規則 collider 失真
凹型物體（U 型件、椅子）被 PyMunk 拆成 convex polygons，contact 點偏離真實表面，反彈角度肉眼可辨偏差 5–15°。
**Workaround**：對關鍵物體手動指定 polygon vertices；或限制 demo 用凸形物體。

---

> See also: [overview.md](./overview.md) (本 zone 6 種 injection 對比) · [Sora](../video-world-models/sora.md) · [Force Prompting](./force-prompting.md) · [PhysDiff](./physdiff.md) · [Cosmos WFM](../foundation-physics-models/cosmos-wfm.md) · [PINN](./pinn.md) · [controllability-vs-fidelity](../../crossing/controllability-vs-fidelity/).
