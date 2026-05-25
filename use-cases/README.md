# Use Cases — 7 個下游

| Use case | 主軸 | 重要 zone |
|---|---|---|
| [robotics-data-gen](robotics-data-gen/overview.md) | 用生成 video / latent / sim 替代真實 demo | video-WM · latent-WM · diff-sim · data-engine |
| [autonomous-driving-sim](autonomous-driving-sim/overview.md) | Closed-loop driving WM | video-WM · long-horizon · controllability |
| [aerial-sim](aerial-sim/overview.md) ★ | 無人機 closed-loop sim + 合成 aerial footage | diff-sim · long-horizon · data-engine |
| [embodied-policy-rollout](embodied-policy-rollout/overview.md) | WM-as-policy / MPC-on-WM | latent-WM · long-horizon · evaluation |
| [scientific-discovery](scientific-discovery/overview.md) | Neural surrogate 替代 PDE solver | neural-surrogates · material-and-dynamics |
| [media-and-content](media-and-content/overview.md) | 影片 / 廣告 / 電影 | video-WM · diffusion-physics · controllability |
| [digital-twin](digital-twin/overview.md) | 工廠 / 手術 / 工業 | diff-sim · 3d-aware · data-engine |

## 為什麼是 use-cases 不是 embodiments

不像 Spatial-Handbook 按 embodiment（aerial / driving / manipulation / marine）切，本倉按「**生成模型給誰用**」切 ——
因為物理可控生成是 **upstream pipeline**，下游可以是不同 embodiment / 不同行業。

## 與 sister handbooks 的對應

- robotics-data-gen / embodied-policy-rollout ↔ VLA-Handbook
- autonomous-driving-sim ↔ Spatial-Handbook driving embodiment
- **aerial-sim ↔ Spatial-Handbook `embodiments/aerial/` ★**（spatial 最深 embodiment，本倉提供生成端視覺資料來源）
- digital-twin / scientific-discovery 是本倉獨有的下游
