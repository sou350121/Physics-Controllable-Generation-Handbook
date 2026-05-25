# Physics-Controllable Generation Handbook

> **照见 / Pulsar 系列・第三冊**
> Sister to [VLA-Handbook](https://github.com/sou350121/VLA-Handbook) and [Spatial-Intelligence-Handbook](https://github.com/sou350121/Spatial-Intelligence-Handbook).
>
> 跨 video / latent / 3D / simulator / surrogate 五條技術路線的 **physics-as-conditioning 世界模型** 手冊。

## 為什麼這個 handbook 存在

當 Sora、Veo、Cosmos、Genie、V-JEPA、Genesis、GraphCast 同時冒出來，它們表面看是不同領域：影片生成、機器人模擬器、駕駛 world-model、神經 PDE solver。但底層問題是同一個 —

> **如何讓生成模型尊重（甚至顯式建模）物理規律，並接受可控輸入（文字 / 動作 / 軌跡 / 力 / 接觸）？**

這條 axis 上的設計選擇決定了：
- VLA pre-training 能不能用合成資料替代真實 demo
- 自駕能不能 closed-loop 跑 long-horizon
- 科學模擬能不能脫離 PDE solver
- 影片生成的「物理感」要靠資料還是靠 inductive bias

這個 handbook 把這條 axis 上的 **5 條技術路線 × 6 種下游應用 × 5 個 cross-cutting wedge** 系統化，並標註誰違反守恆律、誰能 closed-loop、誰能拿 force 當 conditioning。

## 結構

| 路徑 | 內容 |
|---|---|
| [`foundations/`](foundations/overview.md) | 13 個技術 zone：video / latent world-model · diffusion-physics · diff-sim · neural surrogate · controllability · evaluation · 3D-aware gen · long-horizon · material · data-engine · foundation-physics-models |
| [`use-cases/`](use-cases/overview.md) | 6 個下游：robotics data-gen · driving sim · embodied policy rollout · scientific discovery · media · digital twin |
| [`crossing/`](crossing/overview.md) | 5 個 USP wedge：pixel-vs-latent · sim-vs-gen data · controllability-vs-fidelity · conservation violation atlas · text-action-trajectory spectrum |
| [`cheat-sheet/`](cheat-sheet/overview.md) | 5-axis ontology · functional map · timeline · controllability matrix |
| [`deployment/`](deployment/overview.md) | calibration · compute · failure modes · inference cost · safety |
| [`benchmarks/`](benchmarks/overview.md) | VBench-Physics / PhysBench / PDEBench / 自製對齊評測 |
| [`companies/`](companies/overview.md) | NVIDIA Cosmos · World Labs · OpenAI Sora · Google Genie/Veo · Meta V-JEPA · Wayve · Decart · PI · Genesis |
| [`bridge-to-vla/`](bridge-to-vla/overview.md) | 給 VLA pre-training 的接口 |
| [`bridge-to-spatial/`](bridge-to-spatial/overview.md) | 與 Spatial-Intelligence-Handbook 的接口 |
| [`reports/physics-gen-daily/`](reports/physics-gen-daily/README.md) | Pulsar 自動抓 arxiv → qwen 評級 → 每日落地（Phase 2） |

## 5-axis Ontology v1

每篇 dissection 頂部會有 `<!-- ontology-5axis ... -->` HTML comment，五軸是：

1. **Output space** — pixel-video / latent / 3D-scene / mesh / particle / field / action-seq
2. **Physics injection** — implicit-from-data / constraint-loss / sim-in-loop / energy-based / score-conditioned / hard-PDE
3. **Controllability input** — text / action / trajectory / force / contact / image-prompt / 3D-prompt / multi
4. **Temporal paradigm** — single-frame / autoregressive / joint-rollout / latent-rollout / hierarchical
5. **Domain coupling** — generalist / robotics / driving / fluid / rigid / bio / weather

詳見 [cheat-sheet/ontology.md](cheat-sheet/ontology.md)。

## Status

- [x] Skeleton + ontology v1 (2026-05-25)
- [ ] 12 anchor dissections (Cosmos / Sora / Genie-2 / V-JEPA-2 / DreamerV4 / Genesis / MuJoCo-MJX / GraphCast / FNO / Wayve GAIA / PhysGen / VBench-Physics)
- [ ] Mintlify Hobby deploy
- [ ] Pulsar Phase 1 (daily arxiv → reports/physics-gen-daily/)
- [ ] 30+ dissection target

## License

Apache 2.0 — same as [VLA-Handbook](https://github.com/sou350121/VLA-Handbook).
