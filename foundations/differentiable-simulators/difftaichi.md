<!-- ontology-5axis output=N/A injection=sim-in-loop-train control=action|trajectory|force|param temporal=streaming domain=fluid|rigid|soft -->

# DiffTaichi

## 1. One-paragraph TL;DR

DiffTaichi (Hu et al., arXiv [1910.00935](https://arxiv.org/abs/1910.00935), ICLR 2020) 是「可微物理 sim 領域」的學術錨點 paper — 它把 Taichi DSL（同團隊 SIGGRAPH Asia 2019 的稀疏資料結構語言）擴成「自帶 reverse-mode AD」的 imperative GPU 語言，並在一篇論文裡放出 **10 個 differentiable simulators**（MPM-elastic 2D/3D、liquid、wave、water renderer、rigid-body、mass-spring、billiards、+2 video-only）作為「同一 DSL 可以撐多種物理 + 反向傳播」的證明。對本 handbook 重要的不是它 2020 年最快（已被超越），而是 **它是 Genesis 的直接父代**：同一作者群（Yuanming Hu, MIT CSAIL → Taichi Graphics）用 Taichi runtime 拼出 Genesis 的多 solver 棧，差別是 DiffTaichi 是「PDE 友善的單檔 demo 集合」，Genesis 是「rigid+soft+fluid 同場耦合的 robotics 平台」。讀完 DiffTaichi 才能解釋 Genesis 為什麼選 Taichi 不選 JAX/CUDA、為什麼 MPM 的 differentiability 比 rigid 早 GA、以及為什麼這條路線的 contact-rich gradient 問題從 2020 拖到 2026 仍是公開瓶頸。

## 2. Core mechanism

DiffTaichi 的核心是兩個設計選擇：**megakernel + 兩階自動微分**。

```
            ┌──────────── User-written Taichi kernel ────────────┐
            │  @ti.kernel def advect(...):                       │
            │      for i,j in grid:                              │
            │          v_new[i,j] = ...   ← 多 stage 融進一個 megakernel │
            └─────────────┬──────────────────────────────────────┘
                          │
                          ▼  Source Code Transformation (SCT)
            kernel-內：每個 megakernel 自動產生對偶 gradient kernel
                          │  (preserves parallelism + arithmetic intensity)
                          ▼
            kernel-間：lightweight tape 紀錄 kernel 呼叫順序
                          │
                          ▼  Reverse replay
              end-to-end ∂L/∂params  ← 直接接到 PyTorch optimizer
```

關鍵點：
- **Megakernel**：使用者自然把多階段（如 P2G → grid update → G2P 的 MPM 三步）融進一個 kernel，AD 在 kernel 內走 SCT；不是傳統 op-graph autodiff（避免每個小 op 都建 graph 的 overhead）。
- **Two-scale AD**：kernel-內走 source transform（保留 SIMT 並行 + arithmetic intensity），kernel-間走 tape（記錄呼叫序）。這是和 PyTorch / TensorFlow 純 tape AD、JAX 純 trace-based AD 的本質差異。
- **Imperative + JIT**：寫起來像 Python for-loop，編譯後是 CUDA/Metal/CPU SIMT kernel。
- **論文實測**（paper §5）：手寫 elastic sim DiffTaichi 版 **比 hand-tuned CUDA 短 4.2× 但速度相當**，**比 TensorFlow 實作快 188×**；neural-network controller 在 10 個 sim 上「typically optimized within tens of iterations」。

## 3. 五軸定位 + 同軸對手

| 軸 | DiffTaichi | [Genesis](./genesis.md) | [MuJoCo MJX](./mujoco-mjx.md) | [NVIDIA Warp](./nvidia-warp.md) | [Aerial Gym](./aerial-gym.md) |
|---|---|---|---|---|---|
| Output | N/A（自身非生成模型） | N/A | N/A | N/A | N/A |
| Injection | `sim-in-loop-train`（PDE-first，原生可微） | sim-in-loop-train（rigid `.grad` 未 GA） | sim-in-loop-train（JAX-native） | sim-in-loop-train（CUDA Python） | sim-in-loop-train（**非可微**，issue #58） |
| Control | `action`+`trajectory`+`force`+`param`（PDE-first；contact 較弱） | 最完整（+contact） | action+trajectory+force+contact | action+force+contact | param+action（drone-specific） |
| Temporal | `streaming` | streaming | streaming | streaming | streaming |
| Domain | `fluid`+`rigid`+`soft`（MPM-heavy） | robotics+rigid+soft+fluid+granular | robotics+rigid | rigid+soft+fluid | drone（rigid 子集） |

**同軸對手分群**：
- **「PDE/MPM 友善的學術 DSL」**：**DiffTaichi（本篇）**、Brax-MPM extensions（少）。DiffTaichi 是這格的事實標準。
- **「JAX 派」**：MJX、Brax。同樣 trace-based AD，但 JAX 不擅長動態控制流；MPM 等不規則 stencil 寫起來很彆扭，這也是 Genesis 選 Taichi 不選 JAX 的關鍵理由之一。
- **「CUDA Python 派」**：NVIDIA Warp。和 DiffTaichi 設計理念最近（imperative + GPU + AD），但 vendor-locked CUDA，跨 backend 弱。
- **「成熟 robotics platform」**：MJX、Isaac Sim。contact 模型遠成熟，但 PDE / MPM / 流體 是另外一個世界。

DiffTaichi 的真正獨佔位置 = **「一個 paper 證明同一 imperative DSL 可以同時撐 MPM / SPH / mass-spring / rigid / height-field water 並全部反傳」** — 它是後續一整條路線（Genesis、ChainQueen、PlasticineLab）的方法論模板，不是要拿來「打贏 MJX」。

## 4. ⚡ shines / ❌ breaks

### ⚡ 真正領先的 regime

- **MPM / SPH / wave 等 PDE / particle-grid hybrid**：megakernel + SCT 對「不規則 stencil + 大量 small ops」極友善；同樣的 sim 在 PyTorch / TensorFlow 寫法是噩夢（paper 量到 188× 落差）。
- **教學 / 學術 baseline**：10 個 example sim 是「single-file, readable Python」，每個 < 300 LOC；學術圈想跑一個 differentiable MPM 控制 task，DiffTaichi 是最低 entry。
- **Diff-physics 路線的論文模板**：ChainQueen、PlasticineLab、ThinShellLab、DiffTactile 多數沿用 DiffTaichi 的 megakernel + tape 結構；想理解整條 line 必須從這篇切入。
- **跨 backend**：Taichi 的 CUDA / Metal / CPU 後端都跑（與 Warp 的 CUDA-only 形成對照）— 對 Apple Silicon 用戶幾乎是唯一可微 sim 選項。

### ❌ Known failure modes

- **Contact-rich gradient 噪聲未解**：rigid_body.py 的可微接觸是 soft penalty（spring + damper），實際 contact discontinuity 處 gradient 仍噪。billiards.py issue #14 已揭：不同 initial condition 下 loss landscape 多峰、optimization 容易發散。
- **單作者主導 + 維護斷層**：repo 由 Yuanming Hu 個人維護（2019-2021 集中提交），DiffTaichi 框架本身已 **merge 進 Taichi 主 repo**（README：「The DiffTaichi differentiable programming framework is now officially part of Taichi」），這個 example repo 自 2022 起活躍度斷崖；2023-2025 的 issue 多數零回應或長尾。
- **Genesis 把人力吸走**：同一作者群 2024 中後期重心轉到 Genesis；DiffTaichi 變成「歷史 reference」而非 active platform — 想用 diff-sim 撐 production robotics，正解是看 Genesis（含其 §8 速度爭議）或 MJX-JAX，不是 DiffTaichi。
- **API 與 Taichi 主線版本漂移**：Taichi 自 v1.0 → v1.7 多次破壞性變更；DiffTaichi example 在新版 Taichi 下會壞（issue #65 diffmpm.py crash, taichi-dev/taichi issue #8377 「Fix Diff Taichi error with liquid.py」皆此類）。pin 到特定 Taichi 版本是現實 workaround。
- **沒 robotics scene format**：DiffTaichi 是 single-file demo 集合，沒有 URDF / MJCF / USD parser；想接 Franka / Allegro / 真機 robot 必須自己 reimplement scene loading（Genesis 之所以存在的核心驅動之一）。
- **CPU/GPU performance 不對稱**：issue #57 揭 diffmpm.py 在某些 grid size 下 CPU 比 GPU 快（kernel launch overhead 主導）— 教學 demo 預設值不適合直接 scale。
- **NaN 累積**：mass_spring.py issue #61「loss 重複跑後變 nan」— gradient explosion 在長 rollout 沒 clip / detach 機制是常見坑。

## 5. Reproduction notes

```bash
# 經典單檔 demo
git clone https://github.com/taichi-dev/difftaichi
cd difftaichi/examples
pip install taichi==1.5.0     # 注意：pin 版本，1.7+ 部分 example 壞掉
python diffmpm.py             # 2D MPM elastic — 最 canonical 入門
python rigid_body.py          # 2D rigid-body locomotion controller
python liquid.py              # 3D MPM liquid（可能需 taichi 1.5.x，見 issue #8377）
```

- **GPU 預算**：單 RTX 3060 / Apple M-series 都能跑全部 example。10 simulator 沒有任何一個需要多卡。
- **License**：MIT（依 Taichi 生態）；example repo 自身未顯示獨立 LICENSE 檔（[TBD: verify 直接讀 LICENSE 確認]）。
- **典型踩坑**：
  - 不要用最新 Taichi（pin 到 README 指定版本，常是 v1.4-1.5 區間）
  - `ti.kernel` 內的 Python 控制流被 SCT 重寫；想 detach 變數要看 Taichi 主倉 discussion #8573 的 workaround
  - `field(s) are not placed` 錯誤（issue #46）= Taichi field lifetime / 宣告順序問題，不是 DiffTaichi 本身 bug
  - GPU 比 CPU 慢時：grid 太小 / kernel 太短，把 batch 或 grid 加大

## 6. Cross-line synthesis

DiffTaichi 與本倉其他 4 條 generation 路線的接點：

1. **neural surrogate × DiffTaichi**：DiffTaichi 的 MPM/SPH/wave rollout 是 [FNO](../neural-surrogates/fno.md) / MeshGraphNet 類 surrogate 的 ground-truth 工廠 — 這也是 ChainQueen / PlasticineLab 論文線的標配。Genesis 把這條 pipeline 「平台化」，但方法論上仍是 DiffTaichi 的延伸。
2. **diff-sim × video WM**：DiffTaichi 沒自帶 photo-real renderer，但 water_renderer.py 已展示「sim → CNN-渲染 → 反傳到物理參數」的 sketch；這是 [PhysGen](../physics-conditioning/physgen.md) / Force Prompting 類 sim-in-loop video conditioning 的雛形。
3. **diff-sim × VLA**：DiffTaichi 自身不適合 VLA（無 robotics scene format、contact 弱）；但它證明「Python imperative + reverse AD + GPU」可行，這條 axis 後來被 MJX-JAX、Brax、Genesis 拿來做 VLA fine-tuning。
4. **與 v2 ontology 的對應**：DiffTaichi 是 `injection=sim-in-loop-train` 的純粹版（Genesis 同 tag 但加 hard-constraint flavor）。Cross-axis Check 9b 對 `output=N/A` 的 sim 不適用；§8 不需解釋「pixel-video + hard-constraint」例外。

**真正獨佔的 composition**：當研究問題是 **「PDE / MPM 為主、需要對物理參數（楊氏模量、黏度、摩擦）做端到端梯度下降」** 且不需要 robotics scene + contact-rich 操作 → DiffTaichi 仍是最輕量、最讀得懂的選項。換到 contact-rich manipulation 或 cross-material 工業場景 → 升級到 Genesis 或 MJX。

## 7. References

**Primary**：
- arXiv: [1910.00935](https://arxiv.org/abs/1910.00935) v3 (Feb 2020), Hu, Anderson, Li, Sun, Carr, Ragan-Kelley, Durand
- 作者主機構：MIT CSAIL（Hu, Anderson, Ragan-Kelley, Durand）+ Adobe Research（Carr）+ NVIDIA / Adobe（Sun, Li）。Yuanming Hu 後來成立 Taichi Graphics。
- ICLR 2020 OpenReview: https://openreview.net/forum?id=B1eB5xSFvr — 公開 review；presentation type 在 OpenReview poster 頁未顯標 spotlight/poster，社區常引為「ICLR 2020 spotlight」但 [TBD: verify 官方 program]
- 同團隊前作：Taichi (SIGGRAPH Asia 2019) — 稀疏資料結構 DSL，DiffTaichi 是其 AD 擴展

**Code / 二手**：
- GitHub: [taichi-dev/difftaichi](https://github.com/taichi-dev/difftaichi) — 10 example sims，merge 進 Taichi 主倉後實質凍結
- ar5iv full-text mirror: https://ar5iv.labs.arxiv.org/html/1910.00935
- Yuanming Hu publication page: https://yuanming.taichi.graphics/publication/2020-difftaichi/
- 後續 line work: ChainQueen (ICRA 2019, 早於 DiffTaichi)、PlasticineLab (ICLR 2021)、ThinShellLab (ICLR 2024)
- 直系繼承者: [Genesis](./genesis.md) — 同 Taichi runtime，把 10 個 single-file sim 擴成統一 robotics 平台

## 8. §8 Pitfall log

| # | Issue / 來源 | 原文摘錄 / 數據 | Severity | Workaround |
|---|---|---|---|---|
| 8.1 | [difftaichi#14 "Unexpectedly bad results for billiards.py variant"](https://github.com/taichi-dev/difftaichi/issues/14) | billiards.py 在不同初始條件下 optimization 發散；contact-rich 場景 loss landscape 多峰 | **High** | 改用 soft-contact / penalty model；接受 diff-sim 在 contact 處 gradient noisy 是固有限制 |
| 8.2 | [difftaichi#61 "Mass spring loss is nan after repeatedly running"](https://github.com/taichi-dev/difftaichi/issues/61) | mass_spring.py 多次 run 後 loss = NaN — gradient explosion / state 未 reset | **High**（教學 demo 自身就壞，影響第一印象） | 加 gradient clip；每 epoch 重置 sim state；pin Taichi 版本 |
| 8.3 | [difftaichi#65 "Diffmpm.py crashes during the forward simulation"](https://github.com/taichi-dev/difftaichi/issues/65) + [taichi#8377 "Fix Diff Taichi error with liquid.py"](https://github.com/taichi-dev/taichi/issues/8377) | Taichi 主線 v1.7 與 DiffTaichi example 不相容；多個 example 在新版 Taichi 下 crash | **High**（直接卡住新人 reproduce） | pin `taichi==1.5.0`（或 README 指定版本）；不要用 latest |
| 8.4 | [difftaichi#57 "why using GPU is much slower than CPU in this diffmpm.py example?"](https://github.com/taichi-dev/difftaichi/issues/57) | 預設 grid size 下 CPU 反而比 GPU 快 — kernel launch overhead 主導 | **Medium** | 加大 grid / batch；對小規模 demo 改 backend=cpu |
| 8.5 | [difftaichi#46 "how to fix field(s) are not placed error"](https://github.com/taichi-dev/difftaichi/issues/46) + [taichi discussion #8573 "Difftaichi, ignore a few derivative expansion"](https://github.com/taichi-dev/taichi/discussions/8573) | Taichi field lifetime / 宣告順序與 AD tape 紀錄交互；複雜 sim 容易撞 | **Medium** | 嚴格按 `ti.init` → field 宣告 → kernel 定義 → tape 順序；對不需 grad 的 field 顯式 detach |
| 8.6 | 維護斷層 — 單作者 + Genesis 吸走人力 | repo 主提交集中在 2019-2021；2023 後 issue 多數零回應；README 註記框架已 merge 進 Taichi 主倉 | **Medium**（不是 bug 而是定位） | 把 DiffTaichi 當「歷史 + 教學」reference；production / 新研究 改用 Genesis 或 MJX |
| 8.7 | 無 robotics scene format | 無 URDF / MJCF / USD 載入；single-file demo 與 robot 平台無接口 | **High**（對 VLA / robotics 用戶） | 接 robotics 走 MJX / Isaac / Genesis；DiffTaichi 留給 PDE 任務 |
| 8.8 | Tape memory 在長 rollout 爆炸 | Reverse-mode AD 必須 checkpoint 所有 intermediate state；長 rollout（>1000 步）memory 線性增長 | **Medium** | 手動 checkpoint / gradient truncation（BPTT-style）；Taichi 主倉 discussion #8573 提到部分 derivative 可手動 ignore |
| 8.9 | Spotlight/Poster 不明 | OpenReview poster 頁未顯標 presentation type；社區廣泛標為 spotlight 但無官方 confirm | **Low**（引用問題） | 引用時寫「ICLR 2020」即可，避免「spotlight」未驗證表述 |

---

**[TBD: verify]**：
- [TBD: verify DiffTaichi 在 ICLR 2020 是否官方 spotlight（OpenReview 頁未顯；社區引用常標 spotlight）]
- [TBD: verify DiffTaichi example repo 的 LICENSE 檔內容（README 未顯示，需開 repo 直查）]
- [TBD: verify 論文裡「10 simulators」清單的後 2 個（video-only，README 未列檔名）— 可能是 smoke / cloth render demo]
