# Onboarding — Physics-Controllable Generation Handbook

讀者 30 分鐘搞清楚這個 handbook 怎麼用。

## 你是誰？挑一條路徑

### A. 我做 VLA / robot policy，想知道 generative 路線能不能省掉真實 demo
1. 先讀 [`crossing/sim-vs-gen-data/overview.md`](crossing/sim-vs-gen-data/overview.md) — 何時 generative data 贏 sim data
2. 再讀 [`use-cases/robotics-data-gen/overview.md`](use-cases/robotics-data-gen/overview.md)
3. 對比 [`bridge-to-vla/overview.md`](bridge-to-vla/overview.md) 三篇邊界 essay

### B. 我做自駕 world-model，想看 closed-loop 怎麼撐住
1. [`foundations/long-horizon-rollout/overview.md`](foundations/long-horizon-rollout/overview.md) — drift / error accumulation
2. [`use-cases/autonomous-driving-sim/overview.md`](use-cases/autonomous-driving-sim/overview.md)
3. [`companies/`](companies/overview.md) 看 Wayve GAIA / Tesla / Cosmos-Drive

### C. 我做影片生成，想理解「物理感」從哪來
1. [`foundations/video-world-models/overview.md`](foundations/video-world-models/overview.md)
2. [`foundations/physics-conditioning/overview.md`](foundations/physics-conditioning/overview.md)
3. [`crossing/conservation-violation-atlas/overview.md`](crossing/conservation-violation-atlas/overview.md) — 誰在哪些守恆律上崩

### D. 我做科學模擬 / PDE，想看 neural surrogate 能不能替代 solver
1. [`foundations/neural-surrogates/overview.md`](foundations/neural-surrogates/overview.md)
2. [`foundations/material-and-dynamics/overview.md`](foundations/material-and-dynamics/overview.md)
3. [`use-cases/scientific-discovery/overview.md`](use-cases/scientific-discovery/overview.md)

### E. 我想理解整個 landscape
1. [`cheat-sheet/ontology.md`](cheat-sheet/ontology.md) — 5-axis 切法
2. [`cheat-sheet/functional_map.md`](cheat-sheet/functional_map.md) — 一張圖找方法
3. [`cheat-sheet/timeline.md`](cheat-sheet/timeline.md) — 三年內路線演化

## Cheat-sheet 區的閱讀順序

```
ontology.md           ─┐
functional_map.md     ─┼─→ controllability_input_matrix.md ─→ physics_violation_atlas.md
timeline.md           ─┘                                       (拉到 crossing/conservation-violation-atlas)
```

## 想貢獻 dissection？

讀 [`AGENTS.md`](AGENTS.md) 的 8 段模板 + [`CONTRIBUTING.md`](CONTRIBUTING.md)。

新增 dissection 需要：
- HTML comment header `<!-- ontology-5axis ... -->`
- 同軸對手 2-3 個
- §8.x Pitfall log（GitHub-validated）

## 為什麼是「三冊一體」

| 冊 | 視角 | Status |
|---|---|---|
| [VLA-Handbook](https://github.com/sou350121/VLA-Handbook) | Action 端 | ✅ active |
| [Spatial-Intelligence-Handbook](https://github.com/sou350121/Spatial-Intelligence-Handbook) | Perception 端 | ✅ Mintlify live |
| **本倉** Physics-Controllable-Generation-Handbook | Generation 端 | 🟡 skeleton (2026-05-25) |

三個視角覆蓋 embodied AI 的完整 perception → world-model → action 閉環。
