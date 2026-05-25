<!-- ontology-5axis output=N/A injection=sim-in-loop|hard-PDE control=action|trajectory|force|contact|physical-param temporal=streaming domain=robotics|rigid|soft|fluid -->

# Genesis (Genesis-Embodied-AI)

## 1. One-paragraph TL;DR

Genesis 是 2024-12-19 開源的「universal differentiable simulator」，由 20+ 研究實驗室合作兩年的成果（CMU 牽頭、Apache 2.0 授權），目標是把 rigid / articulated / MPM(soft, granular) / SPH(liquid) / FEM / PBD(cloth, thin-shell) / Stable-Fluid 全部塞進「同一個 Taichi-backed Python 引擎」，並承諾「世界最快」（單卡 4090 跑 Franka 43M FPS / 430,000× realtime）。**Genesis 自己不是生成模型** — 它在本 handbook 出現的理由是：它是 video world-model / VLA 訓練的 **sim-in-loop oracle + 數據工廠**（cross-material 是其他 sim 給不出來的）。但 release 後速度宣稱被 Stone Tao（ManiSkill author）公開反駁，benchmark 被指 inflate 約 **150×**（見 §8），需要區分 marketing claim 與實測 regime。

## 2. Core mechanism

Genesis 的引擎核心是 **Taichi-kernel-on-GPU 的多 solver 統一框架**，所有 solver 共用一套 spatial-hash + broad-phase + coupling layer：

```
                ┌────────────── Genesis Scene ──────────────┐
                │                                            │
   text/URDF ─▶ │ ┌─Rigid Solver─┐  ┌─MPM Solver─┐           │
                │ │ (articulated)│  │ (soft/snow/│           │
                │ │  contact LCP │  │  granular) │           │
                │ └──────┬───────┘  └─────┬──────┘           │
                │        │ rigid-MPM/SPH/PBD coupling        │
                │ ┌─SPH ─┴┐ ┌─PBD ─┐ ┌─FEM ─┐ ┌─Stable-Fluid─┐
                │ │ liquid│ │ cloth│ │ defor│ │   smoke/gas  │
                │ └───────┘ └──────┘ └──────┘ └──────────────┘
                │                                            │
                │   ⬇ autodiff (only MPM + Tool today)       │
                └────────────────────────────────────────────┘
                            │
                    ⬇ photo-real renderer (LuisaRender / OptiX)
                            │
                    ⬇ generative data engine (text→scene)
```

關鍵設計選擇：
- **Python 100%**：Taichi DSL 把 kernel 翻成 CUDA/Metal/CPU，沒有 C++ binding 維護負擔
- **可微範圍有限**：v0.3/0.4 series 公開承認「currently differentiable = MPM solver + Tool Solver；rigid 的 differentiability still being added」
- **Coupling-first 設計**：rigid-MPM / rigid-SPH / rigid-PBD 三條 coupling 是 USP，這是 MJX / Brax / Warp 還缺的

## 3. 五軸定位 + 同軸對手

| 軸 | Genesis | MuJoCo MJX | NVIDIA Warp | Brax | DiffTaichi | Isaac Sim/Lab |
|---|---|---|---|---|---|---|
| Output | N/A (state/contact/render→RGB) | N/A | N/A | N/A | N/A | N/A + Omniverse RGB |
| Injection | sim-in-loop（+ 部分 hard-PDE on MPM） | sim-in-loop | sim-in-loop | sim-in-loop | sim-in-loop（PDE-first） | sim-in-loop |
| Control | action+trajectory+force+contact+**param**（最完整） | action+trajectory+force+contact | action+force+contact | action+trajectory | param+force（PDE-first） | action+trajectory+force |
| Temporal | streaming | streaming | streaming | streaming | streaming | streaming |
| Domain | robotics+rigid+**soft+fluid+granular** | robotics+rigid | rigid+soft+fluid | rigid（locomotion） | soft+fluid | robotics+rigid（+ Omniverse soft via SDK） |

**同軸對手分群**：
- **「成熟 + 窄 domain」陣營**：MuJoCo MJX（contact 模型最成熟、VLA 評測事實標準）、Brax（JAX-native, locomotion RL 標配）
- **「廣 domain + 商用」陣營**：NVIDIA Warp + Isaac Lab（Cosmos pipeline 接得最好，但封閉度高）
- **「廣 domain + 開源 + 新銳」**：**Genesis（這篇）**、DiffTaichi（學術 baseline，沒 robotics 整合）

Genesis 的真正獨佔位置 = **「rigid + soft + fluid + granular 同場耦合 + 開源 + Python」** — 其他 sim 任何一個都缺至少一條。

## 4. ⚡ shines / ❌ breaks

### ⚡ 真正領先的 regime

- **Cross-material coupling**：rigid arm × soft dough × granular sand 同場演化，這在 MJX / Brax / Warp 公開版本要自己拼。對 deformable manipulation / liquid pouring / soft-body VLA 數據工廠是現成 oracle。
- **單一 Python API**：沒有 C++ binding hell；學生兩天能 prototype。
- **多 backend**：CUDA / AMD / Apple Metal 都跑 — Taichi runtime 換 target 不改 code。
- **Photo-realistic renderer + 生成式資料引擎**：自帶 LuisaRender + 「text → 4D scene」管道（雖然這條 pipeline 細節未完全 open，仍是 design 賣點）。
- **Sim-in-loop oracle for video WM**：因為它能丟出 ground-truth contact + force + material state，pixel WM 的 physics-fidelity loss 有 reference。

### ❌ Known failure modes

- **Speed claims contested**：原 43M FPS / 430,000× 數字被多方獨立 reproduce 失敗。Stone Tao 報告（見 §8）顯示，啟用 self-collision + multi-substep + 連續 action 後，數字掉到 0.29M FPS（**~150× 落差**），在 contact-rich manipulation 比 ManiSkill/SAPIEN **慢 3–10×**。Genesis team issue #181 之後出了修正 benchmark，承認部分設定有問題但保留部分 high-end 數字。
- **Differentiability 表面承諾 > 實際覆蓋**：截至 0.3.x → 0.4.x，只有 MPM + Tool Solver 是 fully differentiable；rigid-body 的 gradient「coming soon」拖了一年（這對「用 Genesis 訓 VLA / diff-MPC」是硬限制 — 跟 MJX-JAX 的 differentiability 比仍落後）。
- **Contact discontinuity 經典問題沒解**：所有 hard-contact sim 通病 — 接觸 / 分離瞬間 gradient 噪聲大；Genesis 沒提出新 contact model（不像 arXiv 2506.14186 「soft gradient for hard contact」這類專門方案）。
- **Constraint solver 調參敏感**：早期 demo video 出現 cube 從 gripper 穿過 / 漂浮，作者承認是「poorly tuned constraint solver configurations」 — 對非作者用戶意味陡峭的調參曲線。
- **Camera + rendering 同時開時 throughput 崩**：Stone Tao 量到 render-on 時從 430,000× 跌到 ~10× realtime，遠輸 Isaac Lab / ManiSkill 同任務的 ~1,000×。對 video-WM 訓練（需要 RGB 大量輸出）是直接瓶頸。
- **與 MJX 缺乏 head-to-head**：官方 benchmark 主比 Isaac Gym（已 EOL）+ MJX；MJX-Warp 出來後社群普遍認為 MJX-Warp 在 rigid 場景已追上或反超（見 google-deepmind/mujoco discussion #2303）。

## 5. Reproduction notes

**Repo / install**（2026-05 時點驗證）：

```bash
# 主倉
git clone https://github.com/Genesis-Embodied-AI/Genesis
cd Genesis
pip install -e .          # or: pip install genesis-world

# 跑第一個 demo（Franka + soft cube）
python examples/rigid/franka_arm.py
```

- License: **Apache 2.0**（已查 LICENSE 檔）。
- Python 100%；最低 GPU 預算：單 RTX 3060 12GB 可跑單環境 rigid + MPM；多環境 parallel 建議 4090 / A100。
- 典型踩坑：
  - `gs.init(backend=gs.cuda)` vs `gs.metal` vs `gs.cpu`：backend 切換時 numerical 不完全 bit-exact
  - MPM resolution 預設偏粗；soft demo 看起來「沒物理感」常是 grid_size 沒調
  - **不要直接信 README benchmark 數字** — 自己跑 `examples/speed_benchmark.py` 並開 self-collision + 多 substep
  - rigid-body 的 `.grad` 還沒上 → 想做 diff-MPC 的工作流目前要降級到 MPM-only 場景
- 文檔：https://genesis-world.readthedocs.io/

## 6. Cross-line synthesis

Genesis 與其他 4 條 generation 路線的接點：

1. **pixel-video WM × Genesis**：用 Genesis 跑 cross-material rollout，photoreal renderer 出 RGB → 餵 Cosmos-Predict / Sora-class WM 當「物理正確」訓練資料。USP 在「soft/granular/liquid」這些 Isaac+Omniverse 也吃力的 domain。
2. **latent-WM × Genesis**：把 Genesis state 當 ground-truth latent，訓 DreamerV4-style latent rollout 的 reconstruction target，逼 latent 內化 contact / deformation 動力學。
3. **neural surrogate × Genesis**：用 Genesis 的 MPM/SPH ground truth 訓 FNO / MeshGraphNet 類 surrogate；之後用 surrogate 取代 inner-loop sim 在 RL/diff-MPC 中加速。**這是 DiffTaichi 路線的自然延伸，Genesis 給了更廣 domain 的 oracle。**
4. **VLA × Genesis (sim-in-loop)**：rigid arm + soft / liquid 操作的 VLA fine-tuning 數據工廠。但 Stone Tao 警告 contact-rich manipulation 速度其實不如 ManiSkill/SAPIEN — 純 rigid manipulation 還是 MJX/ManiSkill 較穩。

**真正獨佔的 composition**：當 generation 任務 *必須* 同時涵蓋 rigid + soft + fluid（例如：機器人攪粥、軟體機器人游泳、granular pile manipulation），Genesis 是目前唯一一個 open-source、單 API、可微（部分）的 oracle。其他組合用 MJX + Warp + 自己拼 coupling 是 viable 但工程成本高。

## 7. References

**Primary**：
- GitHub: https://github.com/Genesis-Embodied-AI/Genesis （main repo, Apache 2.0）
- Project page: https://genesis-embodied-ai.github.io/
- Docs: https://genesis-world.readthedocs.io/
- 沒有正式 paper（截至 2026-05；release 走 GitHub + project page；個別子系統如 ThinShellLab、DiffTactile 有 ICLR 2024 paper）

**Critique / 二手實測**：
- Stone Tao（ManiSkill author）"How fast is the new hyped Genesis simulator?"：https://stoneztao.substack.com/p/the-new-hyped-genesis-simulator-is — **最關鍵的反駁來源**
- Speed benchmark repo（Stone Tao）：https://github.com/zhouxian/genesis-speed-benchmark
- MuJoCo discussion #2303 「Genesis claims 10-80× MJX, true?」：https://github.com/google-deepmind/mujoco/discussions/2303
- Silicon Valley Robotics Center 2026 RL sim comparison：https://www.roboticscenter.ai/rl-environments/best-2026

**Marketing-side reference（要批判性看）**：
- The Decoder「430,000× faster than reality」：https://the-decoder.com/genesis-speeds-up-ai-robot-training-with-simulations-430000x-faster-than-reality/
- MarkTechPost release writeup：https://www.marktechpost.com/2024/12/19/meet-genesis-an-open-source-physics-ai-engine-redefining-robotics-with-ultra-fast-simulations-and-generative-4d-worlds/

## 8. §8 Pitfall log

| # | Issue / 來源 | 原文摘錄 / 數據 | Severity | Workaround |
|---|---|---|---|---|
| 8.1 | [Issue #181](https://github.com/Genesis-Embodied-AI/Genesis/issues/181) 「Simulation Speed Benchmark likely has significant issues leading to overstated numbers」by StoneT2000 | (a) benchmark 用「fastest physics setting」而其他 tutorial 不用；(b) 「one action followed by 999 steps of no actions」→ rigid solver early-exit；(c) self-collision 預設關閉 | **Critical**（影響選型決策） | 自己重跑：開 self-collision + multi-substep + 連續 random action。Genesis team 後續放修正 benchmark，**保留 43M FPS（self-collision-only）但承認 random-action 場景掉到 27M FPS**。 |
| 8.2 | Stone Tao 二手實測（substack） | 「publicly reported numbers are off by **150×**」「3–10× slower than ManiSkill/SAPIEN on collision-rich manipulation」「rendering on: drops to ~10× realtime vs Isaac Lab/ManiSkill ~1,000×」 | **High** | 用 MJX/ManiSkill 做純 rigid manipulation；Genesis 留給 cross-material 任務 |
| 8.3 | Differentiability 覆蓋不完整 | 官方 docs（0.4.x）：「MPM solver and Tool Solver currently differentiable, differentiability for other solvers being added soon (starting with rigid-body)」— **rigid `.grad` 從 2024-12 拖到 2026-05 仍未完全 GA** | **High**（對 diff-MPC / sim-in-loop training 是硬限制） | rigid diff 任務改用 MJX-JAX（differentiable）或 Brax；Genesis 只在 MPM/soft 任務用其 autodiff |
| 8.4 | Constraint solver 調參敏感（demo video glitch） | 作者公開承認「cubes glitching stems from poorly tuned constraint solver configurations」 | **Medium** | 跟 Genesis Discord / GitHub Discussions 對齊「known good」config；不要直接信 example 預設值 |
| 8.5 | Camera + render throughput | Stone Tao：「render on → 430,000× → ~10× realtime」 | **High**（直接卡 video-WM 數據生成） | 拆 pipeline：state-only fast rollout 在 Genesis；視覺 render 走 Isaac/Omniverse 或 offline batch |
| 8.6 | 缺正式 paper / 第三方 peer review | release 走 GitHub + 部落格；社群是主要驗證渠道 | **Medium**（學術引用 / 工程 baseline 兩用都不穩固） | 引用時標 commit SHA + 自跑 benchmark；不要引「世界最快」的 marketing 表述 |
| 8.7 | Contact discontinuity 老問題未解 | Genesis 沒提出新 contact model（vs arXiv 2506.14186 等專門方案）；hard-contact gradient 仍 noisy | **Medium**（對 contact-rich diff-MPC） | 改 MPM 軟接觸近似；或外掛 implicit-diff contact scheme |
| 8.8 | 與 MJX-Warp 的相對位置 | [mujoco discussion #2303](https://github.com/google-deepmind/mujoco/discussions/2303) 社群共識：rigid-only 場景 MJX-Warp 已追上或反超 Genesis | **Medium**（選型 default 仍應是 MJX） | rigid VLA 評測選 MJX；Genesis 只在「rigid+soft+fluid 同場」時是 first choice |

---

**[TBD: verify]**：
- [TBD: verify Genesis 對 AMD ROCm backend 在 2026-05 時點的成熟度 — Taichi runtime 宣稱支援但社群實測案例少]
- [TBD: verify Genesis 「generative data engine (text→4D scene)」具體實作開源程度 — release 時 demo video 多，code 是否完整放出未確認]
- [TBD: verify Genesis 0.4.x 是否已 ship rigid-body differentiability（官方 wording 一年來反覆「coming soon」）]
