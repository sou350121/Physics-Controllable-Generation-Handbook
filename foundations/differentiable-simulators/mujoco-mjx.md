<!-- ontology-5axis output=N/A injection=sim-in-loop|hard-PDE control=action|trajectory|force|contact|physical-param temporal=streaming domain=robotics|rigid -->

# MuJoCo MJX

## 1. One-paragraph TL;DR

MJX (MuJoCo XLA) is Google DeepMind 把 MuJoCo physics engine 原樣移植到 JAX/XLA 的版本，2023 年 10 月隨 MuJoCo 3.0 主線發布，用 `pip install mujoco-mjx` 取得。它存在的理由不是「新模擬器」 — 而是把 MuJoCo 那套被引用最久的 contact-rich rigid-body dynamics 拉上 GPU/TPU，讓 thousand-env scale 的 RL / VLA 評估在單卡上幾分鐘內跑完，順便提供 JAX-native 的可微梯度。它填的 prior gap 是：原版 MuJoCo CPU-only 跑 8192 env 是天荒地老，Brax 雖然 JAX-native 但 contact 模型過於簡化、sim2real gap 大；MJX 試圖兼顧「MuJoCo 級 contact 真實度」與「Brax 級 GPU 並行」。代價：JIT compile 緩慢、contact 處理仍受限於 JAX 靜態 shape 假設、梯度在 contact 邊界有 known noise。

## 2. Core mechanism

三件核心事情：

1. **Algorithm-level mirror**：MJX 不是「另一個物理 sim」，而是 `mjData` / `mjModel` 的 JAX 重寫，演算法（Newton/CG/PGS solver、soft contact constraint）與 C++ MuJoCo 完全對齊，模型 XML 共用。
2. **Static-shape vectorisation**：所有 tensor shape 在 JIT trace 時固定 → `vmap` 出去就是 thousand-env batch。代價：contact 處理時間隨 *possible* contact 數而非 *active* contact 數線性擴張。
3. **Differentiability via XLA autodiff**：penalty-based soft contact model 對 `q, qd, ctrl` 自然可微；對 model parameter（mass, friction）則部分可微（見 §8）。

```
mjModel(XML) ──┐
               ├─► mjx.put_model ─► JAX pytree
mjData ────────┘                       │
                                       ▼
                              jax.vmap(mjx.step)  ── 8192 envs
                                       │
                                       ▼
                  jax.grad(loss(rollout))  ─► ∂loss/∂(ctrl, q, θ)
```

要點：`mjx.step` 是 pure function，沒有 C++ 那種 in-place mutation，但相對地 trace overhead 很高，第一次 compile 通常 1–3 分鐘（MuJoCo Playground 技術報告數字）。

## 3. 五軸定位 + 同軸對手

| 軸 | MJX 值 | 註 |
|---|---|---|
| Output | `N/A` | 不是生成模型；產出 ground-truth state/contact/render 給生成模型用 |
| Injection | `sim-in-loop` (主) / `hard-PDE` (邊界) | 嚴格滿足 Newton-Euler；contact 用 penalty soft model |
| Control | `action|trajectory|force|contact|physical-param` | ctrl input、external wrench、contact mask 都可作 input |
| Temporal | `streaming` | 一步一步 forward；可微 backward through time |
| Domain | `robotics|rigid` | 偏 articulated rigid；soft body 透過 flex 但不算主路 |

**同軸對手對比**：

| 對手 | 相對於 MJX 的差異 | 何時選對手 |
|---|---|---|
| **Brax** | 純 JAX，contact 模型更簡（spring-damper），更快但 sim2real 落差大；Brax envs 已被官方建議遷移到 MuJoCo Playground | 純 locomotion + 不在意 contact 細節 |
| **Genesis** | 統一 rigid+soft+fluid，Taichi backend，自稱 10-80× 快於 MJX（社群實測有爭議，見 issue #2303） | mixed-material 場景；不在意官方 RL stack |
| **NVIDIA Warp** | CUDA-first，MJX-Warp 變體解決了 MJX-JAX 的 contact 擴展瓶頸，**但 Warp 版目前不支援 differentiability** | NVIDIA 卡 + 純 forward sim + mesh collision |
| **Isaac Sim (Isaac Lab)** | 工業強度、Omniverse 渲染、與 Cosmos 接得最深；但不可微、學術門檻高 | sim-to-real with photoreal pixels |

定位一句話：**MJX 是 contact-rich rigid manipulation + JAX-native RL 的預設選擇**；要 differentiable + Nvidia-only 性能上限走 MJX-Warp（forward）或 DiffMJX 補丁（backward）。

## 4. ⚡ Where it shines / ❌ Where it breaks

### ⚡ Shines

- **Contact 真實度**：30-DoF REEM-C humanoid 在 RTX 4090 上跑 8192 parallel envs × 200M PPO steps 只要 56 分鐘（~60k env-steps/s），這在 Brax 上 contact 細節會崩。
- **MuJoCo Playground (RSS 2025 outstanding demo)**：六款真機（Berkeley Humanoid、Unitree Go1/G1、LEAP hand、Franka）在 8 週內 zero-shot sim-to-real，純 MJX-trained policy + Madrona batch renderer，視覺策略也能 single-GPU 端到端。
- **VLA evaluation oracle**：因 XML 模型庫跟 DM Control / dm_robotics 同源，做 manipulation benchmark 共識度高（OpenVLA、Octo、π0 都報 MuJoCo-based eval）。

### ❌ Breaks

- **JIT compile slowness**：Playground tasks 1–3 分鐘編譯；改一個 `nconmax` 就重編。Dev iteration 痛點。
- **Contact gradient noise**：penalty-based contact 在 stiffness 拉高（要逼近 hard contact）時 autograd 出來的梯度錯誤 — `arXiv:2506.14186 (DiffMJX)` 量化過：標準 MJX 在 contact-rich 任務上 gradient 與 finite-difference 對不上，需 adaptive timestep + contact-from-distance estimator 補救。
- **靜態 shape 詛咒**：contact 計算成本 ∝ `nconmax`（possible contacts）而非 `ncon`（active），1000 個 box 場景就算只有 10 個接觸也付 1M pair-check 的錢。MJX-Warp 部分緩解，但放棄 differentiability。
- **梯度為零的隱蔽 bug**：Issue #1344 — `dof_frictionloss` 的梯度恆為零（dry friction 沒接到 autodiff graph）；做 system identification 會踩。
- **PD control 行為與 C++ MuJoCo 不一致**：Issue #1545 報導 quadruped 站姿 PD gains 在 MJX 跟 MuJoCo 同 XML 但 contact force 抖動；`impratio` tuning 是黑藝術。
- **Soft body / cloth / fluid**：MJX 不是設計給這些；統一仿真要轉 Genesis 或 Warp。

## 5. Reproduction notes

最小可跑 setup：

```bash
pip install mujoco-mjx jax[cuda12]  # JAX-native path
# or 完整 RL stack：
pip install mujoco_playground brax
```

```python
import jax, mujoco
from mujoco import mjx

m = mujoco.MjModel.from_xml_path("humanoid.xml")
mx = mjx.put_model(m)            # 上 device
dx = mjx.make_data(mx)
step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
batch_dx = jax.tree.map(lambda x: jax.numpy.broadcast_to(x, (8192,) + x.shape), dx)
batch_dx = step(mx, batch_dx)    # 8192 envs forward 1 dt
```

GPU 預算（實測 anchor，非官方保證）：
- RTX 4090 × 1：humanoid 8192 env 60k steps/s；27-DoF Franka pick 約 150k steps/s
- A100 × 1：MuJoCo Playground 全 benchmark 用此跑
- TPU v4：可 push 到 1M+ steps/s 但需要 model 簡化（contact ↓）

**典型踩坑**：
1. 第一次 `jit` 等 90 秒以為當機 — 把 progress bar 關了；用 `chex.set_n_cpu_devices` 開多卡前先 warm up 一個。
2. `nconmax` 設太低跑一段 NaN，設太高慢成狗；以 `max(ncon)` × 1.5 做起點調。
3. CPU-fallback 比 C++ MuJoCo 慢 20–50× — MJX **不是 CPU 友善版**，永遠 GPU/TPU。
4. `mjx.put_data` 是 host→device 拷貝，每 step 呼叫會殺掉所有並行收益。
5. 要 differentiate 過 contact：先用 `solver=Newton`、`impratio≈1`、`solref/solimp` 軟化；硬 contact 直接放棄 grad，改 finite-difference 或 BPTT 短軌跡。

## 6. Cross-line synthesis

MJX 對另外 4 條技術路線怎麼接：

| 路線 | 接法 |
|---|---|
| **pixel-WM** (Sora/Cosmos-Predict/Genie-2) | MJX + Madrona batch renderer 產出 paired (state, RGB, depth, action) → 用作 video WM finetuning 資料；NVIDIA Cosmos 走 Isaac/Omniverse 是同一思路的對手實作 |
| **latent-WM** (DreamerV4 / V-JEPA-2) | MJX 作 sim-in-loop oracle：Dreamer 訓練時讓 latent rollout 跟 MJX rollout 算 KL/MSE；可微梯度允許直接 BPTT 進 latent encoder（但要小心 contact noise） |
| **diff-sim 路線本身** | MJX 是 baseline；研究方向是修它 — DiffMJX 補 contact gradient，Genesis 換 Taichi backend，Warp 換 CUDA。所有 diff-sim 論文都對著 MJX 跑 head-to-head |
| **neural surrogate** (GraphCast / MeshGraphNet 風) | MJX 產出大量 trajectory 當 supervised data 訓 surrogate → 推理時用 surrogate 取代 MJX 拉速度。剛體 manipulation surrogate（如 `arXiv` 2024 ContactNets-style）這條路線在重啟 |

最關鍵 composition：**MJX (state oracle) × pixel-WM (perception) × VLA (action policy)** — Playground 已示範前兩件，VLA 端的整合是 2026 的開放戰場。

## 7. References

Canonical：
1. Todorov, Erez, Tassa, *MuJoCo: A physics engine for model-based control*, IROS 2012 — MuJoCo 原始 paper（contact model 出處）
2. MuJoCo 3.0 release notes + MJX discussion #1101, 2023-10 — MJX 上線公告
3. `mujoco.readthedocs.io/en/stable/mjx.html` — 官方 MJX 文檔（最新版 caveat 都在這）

二手實測 / 對比：
4. Zakka et al., *MuJoCo Playground*, arXiv:2502.08844 / RSS 2025 — 六款真機 sim-to-real，benchmark 最完整
5. *Hard Contacts with Soft Gradients (DiffMJX)*, arXiv:2506.14186 — contact gradient 問題的量化與補丁
6. `github.com/google/brax/discussions/409` — Brax↔MJX 整合官方說明（Brax envs 已不主力維護）
7. `github.com/google-deepmind/mujoco/discussions/2303` — Genesis 自稱 10–80× 快於 MJX 的社群辯論
8. Silicon Valley Robotics Center, *Best Robot Simulators for RL 2026* — 第三方 2026 對比

## 8. §8 Pitfall log

| # | Issue / Source | 摘要 | Severity | Workaround |
|---|---|---|---|---|
| 8.1 | `mujoco#1344` ([MJX] Gradients w.r.t. `dof_frictionloss` always zero, closed) | dry friction 參數沒接到 autograd graph，做 sysID 看似收斂其實沒在學 friction | High（沉默失敗） | 改用 `dof_damping` 近似 dry friction；或在 loss 端加 friction-aware regulariser |
| 8.2 | `mujoco#1545` (Contact problems in MJX, closed) | 同 XML 在 MuJoCo C++ 站得住、MJX 抖動；`impratio=100` 不穩，`impratio` 低又滑 | High（C++↔MJX 數值不一致） | 重 tune `solref`/`solimp`/`impratio`；接受 MJX-tuned 跟 C++ tuned 不是同一組參數 |
| 8.3 | arXiv:2506.14186 §3 | hard contact 區域 autodiff 梯度與 finite-difference 不一致；contact 剛分離時梯度消失 | High（差分控制器發散） | DiffMJX patch (adaptive timestep + contact-from-distance)；或在 trajectory 級短窗 BPTT |
| 8.4 | `mjx.rst` 官方 docs | JIT compile 1–3 分鐘；改 `nconmax` / 改 XML / 改 batch shape 都要重編 | Medium（dev velocity） | 固定 schema；用 `jax.persistent_compilation_cache` 跨 run 留 cache |
| 8.5 | `mjx.rst` Limitations 段 + Playground 報告 §4 | contact cost ∝ `nconmax`（possible）而非 `ncon`（active）；多物件場景線性炸 | Medium（cost model 反直覺） | 切換 MJX-Warp（forward only，無 grad）；或砍 `nconmax` 然後容忍 NaN-recover |
| 8.6 | `discussions#2812` (MJX GPU vs CPU) | 小模型 / 少 env 時 GPU 反而比 CPU MuJoCo 慢 | Low（誤用） | < 64 envs 留在 C++ MuJoCo；MJX 是 thousand-env 工具 |
| 8.7 | MJX-Warp doc | MJX-Warp 不支援 differentiability | Medium（路徑分叉） | forward-only 任務（RL training）用 Warp；diff loss / sysID 留 JAX |
| 8.8 | Genesis 對比辯論 #2303 | 10–80× 數字爭議，benchmark 條件不對等 | Low（行銷話術） | 自己跑 representative scene 跨 sim 對齊，再決定 |
| 8.9 | `[TBD: verify]` — 各家 Apple Silicon Metal backend 對 MJX 的覆蓋度 | Apple Silicon 路徑官方說支援，但社群實測 perf 與 CUDA 差距未知 | Low | 大規模 RL 不要押 Apple Silicon；dev 用可以 |

---

*寫作日期：2026-05-25。MJX 仍在快速演進 — `nconmax` 行為、Warp backend 覆蓋度、DiffMJX 是否上 upstream 都建議 6 個月內回查官方 changelog。*
