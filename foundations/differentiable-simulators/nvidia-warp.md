<!-- ontology-5axis output=N/A injection=sim-in-loop-train control=action|trajectory|force|param temporal=streaming domain=robotics|rigid|soft|fluid -->

# NVIDIA Warp

## 1. One-paragraph TL;DR

NVIDIA Warp 是 NVIDIA 自 2022 年起放出、Apache-2.0 開源（[github.com/NVIDIA/warp](https://github.com/NVIDIA/warp)，2026-05 為 v1.13.0、6.7k star）的「Python-first GPU simulation framework」 — 它的賭注是：**讓研究員用純 Python 寫 kernel，然後把 Python AST JIT-compile 成原生 CUDA**，同時自動生成 adjoint code 提供可微梯度。對本 handbook 而言，Warp 的核心地位不在「它是不是最好的 sim」，而在 **它是 NVIDIA 整條 Physical-AI stack 的黏合層**：上接 [Isaac Sim / Isaac Lab](https://developer.nvidia.com/isaac/sim) 的 robotics 場景，下接 [Cosmos WFM](../foundation-physics-models/cosmos-wfm.md) 的 video-generation 資料管道，並且在 2025-03 變成 [Newton](https://github.com/newton-physics/newton)（NVIDIA × Google DeepMind × Disney Research 三方合作、後捐給 Linux Foundation 的 open robotics physics engine）的底層 GPU runtime。換句話說：要訓 Cosmos / GR00T 等 NVIDIA 系 video-WM / VLA，你寫的「sim 端」幾乎一定會碰到 Warp（直接或經由 Newton / MuJoCo-Warp 間接）。Warp 自己 `output=N/A`（不是生成模型），但它是「state → physically-correct rollout → 餵 WM 訓練」的單一最強 NVIDIA-native 路徑。

## 2. Core mechanism

Warp 的編譯路徑跟 Genesis（Taichi-backed）和 MJX（JAX/XLA-backed）都不同 — 它是直接 Python AST → C++/CUDA source → nvcc/clang → PTX → 載回 Python process 當 dynamic library，並用 `warp.Tape` 紀錄 kernel launch 供反向重放。

```
   @wp.kernel def step(...) ─┐
        │ (Python AST)       │
        ▼                    │  warp.Tape (forward 記錄)
   AST walker / codegen      │      │
        │                    │      │ replay backward
        ▼                    │      ▼
   C++/CUDA source (.cu)     │  adjoint kernel (auto-generated)
        │                    │      │
        ▼                    │      ▼
   nvcc / clang → PTX        │  ∂loss/∂(inputs)
        │                    │      │
        ▼                    │      ▼
   dlopen back into Python ──┘  → torch.Tensor / jax.Array (zero-copy)
        │
        ▼
   wp.launch(step, dim=N, ...)  ← 跟 PyTorch/JAX 用 __cuda_array_interface__ 互通
```

四個關鍵設計選擇：

- **Kernel-based, not tensor-based**：每個 thread 可獨立 `if`/`break`/`return`，不需 mask trick — 對 contact / sparse particle / mesh query 這類「條件密集」的物理 kernel 是天然 fit（[NVIDIA blog 2022](https://developer.nvidia.com/blog/creating-differentiable-graphics-and-physics-simulation-in-python-with-nvidia-warp/)）。
- **Auto-adjoint code generation**：Warp 在生成 forward kernel 時同步生成 adjoint kernel（reverse-mode AD），forward + backward 大約是 1 + 1 pass，記憶體不需要 stash 所有 intermediate（[NVIDIA blog 2026-03](https://developer.nvidia.com/blog/build-accelerated-differentiable-computational-physics-code-for-ai-with-nvidia-warp/)）。
- **Zero-copy interop with PyTorch / JAX / Paddle**：`wp.to_torch(arr)` / `wp.from_torch(t)` 直接共用 device buffer（`__cuda_array_interface__`），沒有 host round-trip。這是它能塞進 RL / WM training loop 的關鍵。
- **Primitive library**：built-in mesh queries（closest-point / ray-cast / BVH）、hash grid、SDF、quaternion、spatial transform、稀疏 FEM — 不像 Taichi 只給 DSL，Warp 自帶 robotics / graphics 的「物理 prelude」。

Sample kernels（從官方 examples）：

- **Particle integrator**（semi-implicit Euler）：每個 thread 更新一顆 particle 的 `x`, `v`，包含 gravity + force accumulation
- **Contact detection**：對每個 particle launch 一個 thread，內部呼叫 `wp.mesh_query_point` 找最近三角形，回 closest point + normal
- **Smoke / fluid (Stable-Fluid-style)**：advect / project / diffuse 三個 kernel chain，全 GPU 上跑

## 3. 五軸定位 + 同軸對手

| 軸 | NVIDIA Warp | [Genesis](./genesis.md) | [MuJoCo MJX](./mujoco-mjx.md) | DiffTaichi | PyTorch CUDA ext. |
|---|---|---|---|---|---|
| Output | N/A | N/A | N/A | N/A | N/A |
| Injection | sim-in-loop（+ Newton/MuJoCo-Warp 為 RL backbone） | sim-in-loop | sim-in-loop | sim-in-loop（PDE-first） | 不適用（generic kernel） |
| Control | action+force+param（透過 Newton 加 trajectory + contact） | action+trajectory+force+contact+param | action+trajectory+force+contact | param+force | N/A |
| Temporal | streaming | streaming | streaming | streaming | N/A |
| Domain | rigid+soft+fluid（透過 Newton + 自帶 FEM/SPH） | rigid+soft+fluid+granular | rigid only | soft+fluid | generalist |

**同軸對手分群**：

- **「Python-on-GPU kernel JIT」**：Warp（CUDA codegen, NVIDIA-only） vs DiffTaichi（Taichi runtime, multi-backend） vs Genesis（Taichi DSL 上層 + 物理 solver 庫）
- **「Tensor-based GPU sim」**：MJX（JAX/XLA） vs Brax（JAX） — 跟 Warp 路線根本不同
- **「商用低階 CUDA」**：PyTorch CUDA extensions / Triton — 自由度更高但要自己寫 C++ binding 跟手刻 adjoint

**Warp vs PyTorch CUDA extension model（這篇 USP 核心對比）**：

| 維度 | PyTorch CUDA ext. | NVIDIA Warp |
|---|---|---|
| 寫 kernel 的語言 | C++ / CUDA + pybind11 binding | 純 Python（`@wp.kernel` 裝飾器） |
| 編譯時機 | build-time（`setup.py build_ext`） | runtime JIT（first launch 編譯，後續快取） |
| 反向梯度 | 手寫 `backward()` 或外接 autograd Function | 自動生成 adjoint kernel |
| Iteration speed | 改 kernel → rebuild 數十秒~分鐘 | 改 kernel → next launch 秒級 recompile |
| 自由度 | 任意 CUDA feature（warp-level primitive、cuBLAS 整合等） | 受 Warp DSL 約束（不能任意呼叫 cuBLAS） |
| 對 ML 框架的整合 | PyTorch 一等公民 | PyTorch / JAX / Paddle 三邊 zero-copy |
| Robotics primitives | 自己寫 | built-in（mesh / SDF / hash grid / FEM） |

Warp 的真正獨佔位置 = **「Python ergonomics × CUDA performance × auto-adjoint × NVIDIA ecosystem hook」**。哪一條單獨拿出來都有對手，但「四條同時拿」目前只有 Warp。

## 4. ⚡ shines / ❌ breaks

### ⚡ 真正領先的 regime

- **「我想改一個 contact kernel 然後 30 秒後 RL 訓練 loop 就跑」** — vs PyTorch CUDA ext. 的 rebuild 週期，研究員 iteration 速度差一個量級。
- **跟 Cosmos / Isaac Lab 接的「合法路徑」**：[Isaac Sim Replicator 的 Cosmos Synthetic Data Generation](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_cosmos.html) 管道、[Newton](https://github.com/newton-physics/newton) RL backbone、[GR00T N1](https://nvidianews.nvidia.com/news/nvidia-isaac-gr00t-n1-open-humanoid-robot-foundation-model-simulation-frameworks) 訓練 stack 全部以 Warp 為 GPU runtime。要做 NVIDIA-aligned Physical-AI 工程，繞不開。
- **MuJoCo-Warp 性能**：DeepMind 與 NVIDIA 合作的 MuJoCo-Warp port 對 locomotion 達 252× vs JAX、manipulation 達 475× vs JAX（[NVIDIA blog 2026-03](https://developer.nvidia.com/blog/build-accelerated-differentiable-computational-physics-code-for-ai-with-nvidia-warp/)）。整體 Newton stack 對 humanoid sim ~70× / in-hand manipulation ~100× 提速。
- **CFD / FEM 級數值 workload**：Autodesk XLB 跑 lattice-Boltzmann fluid 在 A100 上比 JAX 快 ~8×、記憶體少 2.5–3×。
- **真 auto-adjoint**：相比 Genesis 「只有 MPM + Tool solver 可微」、MJX「contact 邊界梯度噪聲」，Warp 對使用者寫的任意 kernel 自動生 adjoint（限制見 §8）。

### ❌ Known failure modes

- **GPU-only 路徑窄**：CPU backend 存在但 essentially 是 debugging fallback；非 NVIDIA GPU（AMD ROCm / Apple Metal）**不支援**。如果你的 lab 一半機器是 Mac / AMD，Genesis（Taichi backend）反而更通用。
- **Auto-diff 不是全自動 oracle**：control-flow heavy kernel 的 adjoint 仍可能漏 dependency（見 issue #1470「Extend autograd array access checks past tape capture」、#1451「Support gradients for array-rooted composite augmented assignments」），需要使用者讀 generated adjoint code debug。
- **Contact discontinuity 經典問題沒解**：Warp 自帶的 collision primitive 是 hard-contact，gradient 在 contact 邊界仍 noisy — 這是 diff-sim 共通通病，Warp 沒有提出新 contact model（vs 例如 「soft gradient for hard contact」這類專門方案）。要做 contact-rich diff-MPC 仍要靠 Newton / MuJoCo-Warp 的 soft-constraint 重新 wrap。
- **DSL 限制**：Warp kernel 不能任意呼叫 cuBLAS / cuDNN / 既有 .cu 檔案 — 嚴格說是「Python 子集 → CUDA 子集」。要混既有 CUDA codebase 仍得走 PyTorch CUDA ext. + Warp 雙線並 zero-copy 銜接。
- **Parallel kernel compilation 偶發 race**（issue #1474）：大規模 launch 多 kernel 時 codegen 可能撞 race；workaround 是預編譯 + cache warm-up。
- **「Python 不全部支援」**：kernel return type annotation 會被靜默忽略（issue #1471） — 對嚴格 typing 工作流是不安寧的 sharp edge。

## 5. Reproduction notes

**Repo / install**（2026-05 時點驗證）：

```bash
# 主倉
git clone https://github.com/NVIDIA/warp
cd warp

# 或直接 pip
pip install warp-lang            # core
pip install "warp-lang[examples]"  # + USD / extra examples

# 第一個 kernel
python examples/core/example_sim_cartpole.py
```

- License: **Apache 2.0**；Python 3.10+；CUDA-capable GPU（Maxwell or newer, GeForce GTX 9xx+）；driver R545+。
- 文檔：https://nvidia.github.io/warp/
- 典型踩坑：
  - 第一次 `wp.launch` 慢 → JIT compile + cache miss；warm-up 完之後才量 throughput
  - `warp.Tape()` context 內所有 launch 才會被紀錄供 backward；之外的 launch 不會出梯度
  - PyTorch interop 要用 `wp.to_torch(arr, requires_grad=True)`，否則 PyTorch 不會把 Warp 梯度接進 autograd graph
  - 想在 Isaac Lab 內用：Isaac Lab 自帶 Warp 版本 pin；自己 pip upgrade Warp 會破環境 — 跟 Isaac Lab release notes 對齊
  - 改 kernel 後 host process 不重啟也行（cache 自動 invalidate），但 cache dir 偶爾要手動清（`~/.cache/warp`）

## 6. Cross-line synthesis

Warp 在本 handbook 的價值是「NVIDIA 系 generation 訓練資料管道的物理層」 — 它跟其他 4 條 generation 路線都有具體接點：

1. **pixel-video WM × Warp**：核心管線是 [Isaac Sim Replicator → Warp 物理 rollout → Omniverse 渲染 RGB → Cosmos Predict / Cosmos-WFM 訓練資料](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_cosmos.html)。跟 [Cosmos-WFM](../foundation-physics-models/cosmos-wfm.md) 的接法是「Cosmos 用 Warp-rolled-out 物理 ground-truth 視訊做 fine-tune / eval」。這條跟 Genesis 競合：Genesis 也能跑 cross-material rollout，但 NVIDIA 內部管線 default 是 Warp + Omniverse。
2. **latent-WM × Warp**：用 Warp 跑物理 oracle 出 state trajectory，當 [DreamerV4](../latent-world-models/dreamer-v4.md) latent reconstruction target。**zero-copy `wp.to_torch`** 讓這條接得最乾淨 — 不用每步序列化 numpy。
3. **neural surrogate × Warp**：FEM / CFD 級 ground truth（XLB / Warp 自帶 FEM）餵 [FNO](../neural-surrogates/fno.md) / MeshGraphNet surrogate，之後 surrogate 取代 inner-loop Warp 在 diff-MPC 加速。Warp 對 FEM mesh 有 first-class primitive，這條比 MJX / Brax 自然。
4. **VLA × Warp**：[Newton](https://github.com/newton-physics/newton)（Warp-based） + [Isaac Lab](https://developer.nvidia.com/isaac/sim) + GR00T N1 是 NVIDIA 系 humanoid VLA 訓練的官方 stack。Warp 自己不直接訓 VLA — 它是 Isaac Lab / Newton 跑 thousand-env RL rollout 的底層 runtime。

**獨佔的 composition** = 「要做 NVIDIA-native 的 Cosmos / GR00T / Newton 整鏈 fine-tune」時，Warp 是唯一合法路徑。其他組合（自己拼 Genesis + Cosmos）技術上可行但失去 NVIDIA 的 reference workflow 支撐。

## 7. References

**Primary**：
- GitHub: https://github.com/NVIDIA/warp （main repo, Apache 2.0, v1.13.0 @ 2026-05-04, 6.7k★）
- Docs: https://nvidia.github.io/warp/
- Project page: https://developer.nvidia.com/warp-python
- NVIDIA blog 2022 「Creating Differentiable Graphics and Physics Simulation in Python with NVIDIA Warp」by Miles Macklin（Warp 創始 PM/director）: https://developer.nvidia.com/blog/creating-differentiable-graphics-and-physics-simulation-in-python-with-nvidia-warp/
- NVIDIA blog 2026-03 「Build Accelerated, Differentiable Computational Physics Code for AI with NVIDIA Warp」: https://developer.nvidia.com/blog/build-accelerated-differentiable-computational-physics-code-for-ai-with-nvidia-warp/

**Ecosystem 接點**：
- Newton physics engine（Warp-based, Linux Foundation 2025-09 捐贈）: https://github.com/newton-physics/newton ; announcement: https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/
- Cosmos Synthetic Data Generation in Isaac Sim Replicator: https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_cosmos.html
- GTC 2025 session S72709「MuJoCo-Warp and Newton」: https://www.nvidia.com/en-us/on-demand/session/gtc25-s72709/
- Isaac Sim / Isaac Lab: https://developer.nvidia.com/isaac/sim

**沒有正式 arxiv paper**（截至 2026-05 時點）：Warp 走 NVIDIA blog + GTC talk + GitHub release path；學術引用通常引 Macklin 2022 blog + GitHub commit SHA。

## 8. §8 Pitfall log

| # | Issue / 來源 | 原文摘錄 / 數據 | Severity | Workaround |
|---|---|---|---|---|
| 8.1 | [Issue #1474](https://github.com/NVIDIA/warp/issues/1474) 「Parallel kernel compilation can fail due to warp codegen race condition」 | 多 kernel 同時首次 launch 時 codegen race | **High**（CI / multi-process launch 場景會偶發爆） | 先預編譯（`wp.load_module(...)` warm-up）或序列化首次 launch |
| 8.2 | [Issue #1470](https://github.com/NVIDIA/warp/issues/1470) 「Extend autograd array access checks past tape capture」 | tape capture 結束後 array 仍被改寫但沒檢查 → backward 拿到錯梯度 | **Critical**（**差異化 diff-sim 用戶必看** — 靜默錯誤） | 嚴格保持 forward arrays 在 backward 結束前 immutable；不要在 tape 外做 in-place update |
| 8.3 | [Issue #1451](https://github.com/NVIDIA/warp/issues/1451) 「Support gradients for array-rooted composite augmented assignments」 | `arr[i] += expr` 的某些寫法不出 gradient | **High**（容易寫出但靜默漏梯度） | 改寫成顯式 `tmp = arr[i]; arr[i] = tmp + expr` 並 inspect 生成的 adjoint code |
| 8.4 | [Issue #1471](https://github.com/NVIDIA/warp/issues/1471) 「Kernel return annotations are silently ignored」 | `def foo(...) -> wp.float32:` 的 return type 被 ignore | **Medium**（typing 工作流不一致） | 暫時用 docstring 補；等 Warp 修 |
| 8.5 | [Issue #1469](https://github.com/NVIDIA/warp/issues/1469) 「capture_while() fails during multi-stream CUDA graph capture」 | multi-stream CUDA graph + `capture_while` 衝突 | **Medium**（高 throughput RL stack 才會踩） | 改用 single-stream graph capture，或拆 step 不用 `capture_while` |
| 8.6 | Contact discontinuity 老問題未解 | Warp 對 mesh / SDF collision 是 hard-contact；接觸 / 分離 instant gradient noisy | **High**（diff-MPC contact-rich 任務） | 用 Newton soft-constraint wrap；或 MPM 軟接觸近似；或 implicit-diff contact scheme |
| 8.7 | GPU-only 真實成本 | CPU backend 是 debug fallback；AMD ROCm / Apple Metal 不支援；H100 / A100 與 RTX 4090 上 numerical 不完全 bit-exact | **Medium**（混異質 fleet 的 lab 跨機器 reproduction 痛） | 跨機器 reproduction 要 pin GPU model + driver；混跑 Genesis（Taichi backend）做 sanity check |
| 8.8 | DSL 子集限制 | Warp kernel 不能呼叫 cuBLAS / cuDNN / 既有 .cu | **Medium**（混既有 CUDA codebase 工程成本） | 雙線：純物理 kernel 走 Warp；GEMM / conv 級走 PyTorch + zero-copy `wp.to_torch` |
| 8.9 | Newton stack 仍在 alpha | Newton（Warp-based, 2025-09 LF 捐贈）API 仍在 churn；MuJoCo-Warp 與 Newton 路線分工不清 | **Medium**（早期採用者風險） | 鎖 Newton commit + Warp commit；不要在 production stack 直接吃 main |

---

**[TBD: verify]**：
- [TBD: verify Warp v1.13.0 對 H200 / Blackwell GPU 的 tuning 狀態 — release notes 提了 Hopper 路徑但 Blackwell 細節未確認]
- [TBD: verify Cosmos data pipeline 內 Warp 與 Omniverse PhysX 的分工切點 — 部分文檔模糊（PhysX 給 rigid + Warp 給 soft/fluid？還是 Newton 取代 PhysX？）2025-09 之後 Newton 路線是否完全替換 PhysX 在 robotics RL 角色仍需追蹤]
- [TBD: verify Warp 對「pure-PyTorch 工作流」的相對成本 — 沒看到公開的 Warp kernel vs torch.compile(custom-CUDA) 在同一 contact kernel 的 head-to-head benchmark]
