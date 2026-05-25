<!-- ontology-5axis output=action-seq|mesh injection=score-conditioned|sim-in-loop control=text|contact temporal=joint-rollout|autoregressive domain=robotics|rigid -->

# PhysDiff — Physics-Guided Human Motion Diffusion Model

> Yuan, Song, Iqbal, Vahdat, Kautz. **ICCV 2023 (Oral)**. arXiv [2212.02500](https://arxiv.org/abs/2212.02500). Project: [nvlabs.github.io/PhysDiff](https://nvlabs.github.io/PhysDiff/).

## 1. TL;DR

純資料驅動的 human-motion diffusion（MDM 一脈）在文生動作 / 動作生成上分數很高，但生成的骨架普遍 **floating（懸空）/ foot sliding（滑步）/ ground penetration（穿地）**，原因是 denoising 過程裡完全沒有任何接觸或重力訊號 — MDM 用 foot-contact loss 是 soft penalty，無法 hard-enforce。PhysDiff 提出一個極簡的補丁：**在 denoising 迴圈裡夾一個 physics simulator**，每隔幾步把當前的 motion sample 餵給一個已預訓練好的 humanoid imitator（UHC / Isaac Gym + PPO），讓 simulator rollout 一段「真的能站著走的軌跡」，再把這個物理可行的版本注回 diffusion chain 當下一步起點。

這就是 ontology Axis 2 「`score-conditioned`」 + 「`sim-in-loop`」雙軸的 **canonical 樣板**：物理規律不是 architecture-level 進去（像 HNN）、也不是 training-time loss（像 PINN），而是 **inference 時的迭代投影**，所以可以 **plug 在任何 pretrained motion diffusion model 上**而不重訓 score network。對 video-physics / Sora 的 foot-floating 問題，這條路是目前最直接可借鑒的範式。

實測在 HumanML3D + MDM denoiser 上：ground penetration 11.29mm → 0.998mm、floating 18.88mm → 2.60mm、foot sliding 1.41mm → 0.51mm，整體 Phys-Err 降 86%（作者報數）。

## 2. Core mechanism

PhysDiff 不是一個新的 score network，而是一個 **projection operator** 插在 reverse diffusion 的某些 timestep 上。設 reverse process 為 $x_T \to x_{T-1} \to \cdots \to x_0$（$x$ 是動作序列、SMPL 參數），令 $\hat{x}_{t-1} = \text{denoise}(x_t, t)$ 為 score-step 出來的 candidate，PhysDiff 加上：

$$
x_{t-1} = \text{PhysProj}(\hat{x}_{t-1}) = \text{UHC-rollout}(\hat{x}_{t-1})
$$

`PhysProj` 把 $\hat{x}_{t-1}$ 餵給 Universal Humanoid Controller（[UHC](https://github.com/ZhengyiLuo/UHC)，後續論文升級到 Isaac Gym + PPO 訓練的 imitator），simulator 內 PD controller + residual force 跑 N 步 rollout 並 imitate 該參考動作 — 跑得出來的軌跡就是物理可行的；跑不出來（contact violation、IK 失敗）就回退或截斷。輸出 $x_{t-1}$ 再走下一輪 denoising。

關鍵設計細節（作者實測最佳）：**"End 4, Space 1"** — 只在 denoising **末段 4 個 timestep** 應用 projection、間隔 1 步。早期高 noise 階段做 projection 反而傷品質（because the simulator can't imitate noise-level garbage），這是 score-conditioned 路線跟 PINN 路線的本質區別：**晚做、少做、做精準**。

```
text prompt c
     │
     ▼
  x_T (noise)                                              ┌────────────────────┐
     │                                                     │ Physics Simulator  │
     ▼                                                     │ (Isaac Gym + UHC)  │
  ┌─────────────┐         ┌────────┐                       │ PD ctrl + res-force│
  │ denoise(t) │ ──x̂─►  │ in     │  if t ∈ {T-3..T} ──►  │ PPO-trained imitator│
  │ (MDM score) │   ▲    │ proj   │                       └─────────┬──────────┘
  └─────────────┘   │    │ window │                                 │
        ▲           │    └────────┘                                 ▼
        │           │       else: x̂ passes through            x'  (physical)
        │           │                                                │
        │           └────────────────────────────────────────────────┘
        │
        x_{t-1} ◄────────────────────────────────────────── feed back

  Loop t = T → 0;  score-conditioned every step, sim-in-loop only last 4.
```

注意 PhysProj 是 **non-differentiable**（simulator 是 black box），所以這條路不能用來訓 score network，只能在 sampling 時用 — 跟 Genesis / MJX 的「可微 sim 反傳 gradient 訓 score」是兩條完全不同的路線（見 §6）。

## 3. 五軸定位 + 同軸對手

| Axis | PhysDiff |
|---|---|
| Output | `action-seq` (SMPL pose seq) + `mesh` (rigged human) |
| Injection | **`score-conditioned` + `sim-in-loop`** — 雙標記，本倉這對組合的範例 |
| Control | `text` (HumanML3D)、`contact` (隱式 via simulator) |
| Temporal | `joint-rollout`（一次 clip）但 projection 內含 `autoregressive` simulator step |
| Domain | `robotics`（humanoid）/ `rigid`（剛體連桿） |

**同軸對手**：

- **PINN-style aux loss**（→ [`./pinn.md`](./pinn.md)）— 訓練時加 contact / momentum loss。優勢：inference 0 額外 cost；劣勢：soft constraint、weight tuning 噩夢、超出訓練 distribution 立刻失效。PhysDiff 把「規律檢查」從 train-time 搬到 inference-time，**用 sim 的 ground truth 取代 hand-crafted PDE loss**。
- **Hamiltonian / Lagrangian NN**（→ [`./hamiltonian-lagrangian-nn.md`](./hamiltonian-lagrangian-nn.md)）— 架構天生保守。優勢：hard guarantee；劣勢：只能處理 closed 動力系統，無法表達 contact discontinuity（人走路的本質就是不斷打開/關閉接觸對）。PhysDiff 用 simulator 代理整個複雜接觸模型，**換掉一整類 PDE 寫不出來的場景**。
- **PhysGen**（→ [`./physgen.md`](./physgen.md)）— rigid-body pipeline，static image → physics-grounded video。同樣是 sim-in-loop，但 PhysGen 是 **pipeline-style**（perception → sim → render），不是 **iterative-loop**。PhysDiff 把 sim 嵌進 denoising 迴圈，是更深的耦合。
- **Force Prompting**（→ [`./force-prompting.md`](./force-prompting.md)）— 把力當 conditioning 餵 video diffusion。Injection 比 PhysDiff 弱（`implicit-from-data` + `force` control），完全不接 simulator。
- **MDM (baseline, [Tevet 2022](https://arxiv.org/abs/2209.14916))** — PhysDiff 的 denoiser 就是 MDM。MDM 用 foot-contact loss + sample-prediction（預測 $x_0$ 而非 noise），這讓 PhysDiff 的後處理 projection 變得可行（你要 project 一個 motion，不能 project 一團 noise）。PhysDiff vs MDM 的 ablation 是這條 line 最乾淨的對照組。

## 4. ⚡ shines / ❌ breaks

⚡ **真正領先的 regime**：

- **零訓練成本擴充任何 pretrained motion diffusion model** — 把 MDM、MotionDiffuse、MLD 都當成 black-box denoiser，加個 projection 就能拿到 >78% physical plausibility 提升。production-friendly 程度極高。
- **可量化 artifact 解決** — 不是「看起來更好」這種主觀指標，是 ground penetration / floating / foot sliding 三個 mm-level 數值同時降一個量級。Reproducible by definition。
- **Sample-prediction MDM 是天然搭檔** — MDM 在每個 step 輸出的是 $\hat{x}_0$（可理解為當前對最終動作的估計），這正好是 simulator 可以 imitate 的對象；若是 noise-prediction 路線（DDPM 原版），projection 需要先 denoise 才能投影，多一層 cost。

❌ **Known failure modes**：

- **必須有可用的 humanoid simulator + pretrained imitator** — UHC 只能處理 SMPL-skeleton humanoid，要換成手部精細動作（MANO）/ 動物 / 變形體，imitator 要重訓。**這直接決定了 PhysDiff 不能套到 weather / fluid / soft-body** — 沒有對應的 imitator 概念。
- **Manifold drift / OOD motion** — projection 會把 sample 拉去 simulator manifold，但 simulator manifold ≠ data manifold。Yuan et al. 自己 ablation 顯示 projection 太頻繁（每 step 都做）反而傷 FID — 因為 sample 被拉出 score network 的高密度區、score guidance 接不回來。**"End 4, Space 1" 是 fragile tuning**，換 denoiser 要重調。
- **Compute cost 一次膨脹 ~3-5×** — 每個 projection step 內含 simulator rollout 數十 sim-step + PPO policy forward；inference latency 從 MDM 的 ~秒級拉到分鐘級。對 real-time 應用基本不可行。
- **Imitator failure 沒 graceful fallback** — UHC 跑不出來時的處理（截斷 / 回退 / 強制 IK）是黑盒；作者 paper 也沒詳述失敗率 — 對 highly dynamic / 非 SMPL-distribution 動作（武術、雜技、舞蹈）特別容易爆。
- **Sim accuracy 是品質上限** — Isaac Gym 的接觸模型有 ~5mm 等級的穿插（rigid-rigid contact 數值穩定性問題），所以 PhysDiff 報的「ground penetration 0.998mm」這個下限其實是 sim 自己的 floor，不是物理真值。
- **沒有外部物件互動** — 原版 PhysDiff 只有 humanoid 跟 ground plane，[PhysHOI (Wang 2023)](https://arxiv.org/abs/2312.04393) 才把 object interaction 補上（contact graph 顯式建模），且 PhysHOI 走的是 pure RL imitation 路線，不是 diffusion + projection。

## 5. Reproduction notes

**官方代碼狀態**：截至 2026-05，project page [nvlabs.github.io/PhysDiff](https://nvlabs.github.io/PhysDiff/) 沒有公開官方 code repository。`[TBD: verify NVlabs/PhysDiff GitHub release status]`。社群目前主要參考：

- **MDM 部分**：直接用 [GuyTevet/motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model)（HumanML3D / KIT / HumanAct12 預訓練 checkpoint 齊全）。
- **UHC simulator 部分**：[ZhengyiLuo/UHC](https://github.com/ZhengyiLuo/UHC)（MuJoCo 實現，論文後續升級到 Isaac Gym）；後續 [PHC (Luo 2023, ICCV)](https://arxiv.org/abs/2305.06456) 的 Perpetual Humanoid Control 是更穩的 imitator，建議現在重現以 PHC 替代原版 UHC。
- **接口部分**：要自己寫 `denoise_step → SMPL → UHC reference → rollout → SMPL back → next denoise_step` 的 glue code，這是踩坑重災區。

**最小 GPU 預算**：1× A6000 / 4090 跑 inference 可行；Isaac Gym 對 GPU memory 比較吃（~12GB for 一個 humanoid + ground），加上 MDM 自己 ~2GB，~16GB sweet spot。**訓練不需要**（只插 inference）。

**典型踩坑**：

1. UHC 期待 25 fps（後升 30 fps）motion input，MDM 輸出 20 fps，要做時間重採樣（簡單線性插值常導致 IK 抖動）。
2. SMPL pose-axis-angle ↔ rotation-matrix 表示在 MDM / UHC 之間轉換時，root orientation 容易翻轉 180° — 軸定義差異，第一次調試會 debug 半天。
3. Isaac Gym 在 headless docker 裡需要特殊 setup（X11 / `libGL` 依賴），CI 不友好；本地開 GUI debug 比 server-only 快很多。
4. Projection schedule "End 4, Space 1" 沒法直接搬到 50-step DDIM / 1000-step DDPM — 是 timestep ratio，不是 absolute step；要按 reverse process 長度比例 rescale。

## 6. Cross-line synthesis

**PhysDiff vs 可微 sim 路線**（[Genesis](../differentiable-simulators/genesis.md) / [MuJoCo MJX](../differentiable-simulators/mujoco-mjx.md)）：兩條路線都「用 simulator」，但用法剛好相反：

- PhysDiff：**non-differentiable sim** 當 inference-time projector，**不**反傳 gradient，所以 simulator 可以是任意黑盒（PhysX / Isaac / Bullet）。優點：implementation 簡單；缺點：score network 學不到「物理偏好」。
- Genesis-train / MJX-train：**differentiable sim** 在 training 時餵 gradient 進 score network，讓 model 內化物理 prior。優點：inference 0 額外 cost；缺點：可微 sim 對 contact 數值極脆弱，目前 only works for smooth dynamics（fluid > rigid > granular）。

**結論**：兩條路線在 contact-rich 場景目前 **互補不替代**。PhysDiff 是 contact-discontinuity 的暴力解，可微 sim 是 long-term 優雅解但還不成熟。

**對 video generation（Sora-line）的延伸 — 開放問題**：Sora 的 foot-physics / object permanence 也是 score-conditioned model + 違反守恆律的 case。直接搬 PhysDiff 思路的困難：

1. Video diffusion 的 latent 不是 SMPL — 沒有對應的「humanoid imitator」可投影。
2. Video diffusion 的 denoising step 數遠多於 motion（如 100+），projection 算力指數爆炸。
3. 需要 **video → mesh → sim → mesh → video** 的閉環，每一段都是當前 SOTA 邊界。

短期可行方向：先做 **partial projection**（只投影 character body region 而非整幅 video，類似 inpainting），對應 zone [`crossing/conservation-violation-atlas/`](../../crossing/conservation-violation-atlas/) 列出的 foot-contact / ground-penetration sub-violation 表 — 那邊是這篇方法**最值得照搬的具體目標清單**。

**與 PhysGen / Force Prompting 的差異**詳見 §3；簡言之 PhysDiff 是「inference-loop」、PhysGen 是「pipeline」、Force Prompting 是「conditional input」，三者的耦合深度遞減。

## 7. References

**Canonical**：
- Yuan, Song, Iqbal, Vahdat, Kautz. *PhysDiff: Physics-Guided Human Motion Diffusion Model*. **ICCV 2023 (Oral)**. arXiv [2212.02500](https://arxiv.org/abs/2212.02500). Project page: [nvlabs.github.io/PhysDiff](https://nvlabs.github.io/PhysDiff/).

**直接依賴**：
- Tevet et al. *Human Motion Diffusion Model (MDM)*. **ICLR 2023**. arXiv [2209.14916](https://arxiv.org/abs/2209.14916). Code: [GuyTevet/motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model).
- Luo et al. *Universal Humanoid Control (UHC)*, supporting Kinpoly (NeurIPS 2021) / EmbodiedPose (NeurIPS 2022). Code: [ZhengyiLuo/UHC](https://github.com/ZhengyiLuo/UHC). MuJoCo-based; later升級為 Isaac Gym 在 PhysDiff 採用。

**後續 / 互補**：
- Luo et al. *Perpetual Humanoid Control (PHC)*. **ICCV 2023**. [openaccess paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Luo_Perpetual_Humanoid_Control_for_Real-time_Simulated_Avatars_ICCV_2023_paper.pdf) — UHC 升級版，更穩的 imitator，建議重現時替代。
- Wang et al. *PhysHOI: Physics-Based Imitation of Dynamic Human-Object Interaction*. arXiv [2312.04393](https://arxiv.org/abs/2312.04393). 補上 object interaction，但走 pure RL imitation 路線（非 diffusion）。
- Luo et al. *Universal Humanoid Motion Representations for Physics-Based Control*. arXiv [2310.04582](https://arxiv.org/abs/2310.04582).
- `[TBD: verify "PhysSampleSpace" — 2024/25 candidate paper of this name not located via web search; may be a misremembered title. Closest 2025 hits: "Physics-Informed Diffusion Models" arXiv 2403.14404; "Physics-informed diffusion models in spectral space" (NeurIPS 2025)]`.

## 8. §8 Pitfall log

1. **UHC embedded-sim 的能力上限**：UHC 是 SMPL-only、ground-plane only。任何手部精細動作（MANO）/ 物件互動 / 不平地形 = 直接超出 imitator distribution、projection 退化為「拒收」。對 dance / sports / cooking 之類 motion，外掛 PhysHOI 或更新 imitator。**Severity: high**, workaround: 換 PHC 或 PhysHOI。
2. **Manifold drift from projection**：projection schedule 過密（每 step 都投）會把 sample 拉出 score network 高密度區，FID 上升。作者 "End 4, Space 1" 是針對 MDM + 1000-step diffusion 調的，**換 DDIM / classifier-free guidance scale 都要重調**。Severity: medium, workaround: grid search projection schedule（每組訓練 / inference setting 都要重來，30+ GPU-hour）。
3. **OOD motion → 非物理 fallback**：當 text prompt 描述了 UHC 沒見過的動作（如「moonwalk」、「contact juggling」），simulator imitate 失敗，PhysDiff 退化回 raw MDM 輸出（甚至更糟，因為 projection 中途打斷了 score chain）。**Severity: high**, workaround: 加 imitator-confidence gate，confidence 太低就 skip projection（paper 沒做）。
4. **Computational cost per denoising step**：projection step ≈ 30-50 sim steps × Isaac Gym contact solve，相當於 MDM denoise step 的 ~10-20× FLOPs。整體 inference 從 ~2s 拉到 ~30s/clip。**Severity: medium (research)**, **high (production)**, workaround: 只在最後 1-2 step 投影（品質有損但速度可接受）；或 distill PhysDiff 結果回去訓練純 score network（作者未做，是 open direction）。
5. **Sim accuracy → motion accuracy ceiling**：Isaac Gym rigid-contact solver 本身有 mm-level 穿插容忍，PhysDiff 報的「ground penetration 0.998mm」是 sim floor 而非物理零。要更精，需要 MuJoCo MPR 或 Bullet hard-contact，但這兩個 GPU acceleration 差。Severity: low（多數應用夠用），workaround: 換 simulator 看 contact 模型實現。
6. **Object interaction 完全缺席**：PhysDiff 原版假設 humanoid 與 ground 是唯一互動，桌上拿杯、開門、踢球這類場景無法處理。[PhysHOI](https://arxiv.org/abs/2312.04393) 引入 contact graph 是第一個解法，但需重訓 imitator + diffusion model 都要新 data pipeline。**Severity: high (任何 HOI 應用)**, workaround: PhysHOI 路線 + 等待 PhysDiff-HOI hybrid 出現。
7. **Code 未官方開源**：截至 2026-05 NVlabs 沒有 release official PhysDiff repo，社群重現需自行串 MDM + UHC + projection glue，這層 engineering 不是論文裡的「24 行 ablation 表」可以代替的 — 重現難度被低估。Severity: medium, workaround: 用 PHC 替代 UHC + 自寫 projection wrapper（約 200-400 LOC）。

---

**Cross-refs**: [`./overview.md`](./overview.md) (zone overview) · [`../../crossing/conservation-violation-atlas/`](../../crossing/conservation-violation-atlas/) (foot/ground violation specification) · [`../differentiable-simulators/genesis.md`](../differentiable-simulators/genesis.md) / [`../differentiable-simulators/mujoco-mjx.md`](../differentiable-simulators/mujoco-mjx.md) (可微 sim 對照組).
