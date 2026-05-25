# Bridge: Generative Data for VLA

> 🚧 **Stub** — waiting on anchor dissections. See [overview.md](overview.md) for design rationale.

## 主題

生成資料能否替代真實 demo 訓 VLA — 兩端契約、Pareto、何時 fail。

## 預計內容

- **何時生成資料贏**：long-tail 場景、長 horizon、多 embodiment 泛化
- **何時真實 demo 贏**：精細接觸 (force/contact) 任務、sim2real gap 大的場景
- **三條合成路線**：純 video（Sora-class）/ action-conditioned WM（Genie-2 / V-JEPA-2）/ sim-augmented（Genesis + DR）
- **兩端契約**：data schema / 座標系 / action token 是否 grounded / 是否帶 force

## 引用 anchor

待寫完後引用：
- [`/foundations/foundation-physics-models/cosmos-wfm.md`](../foundations/foundation-physics-models/cosmos-wfm.md)
- [`/foundations/latent-world-models/v-jepa-2.md`](../foundations/latent-world-models/v-jepa-2.md)
- [`/crossing/sim-vs-gen-data/overview.md`](../crossing/sim-vs-gen-data/overview.md)
- VLA-Handbook（sister repo）對應 dissection

## TODO

- [ ] PI data engine 公開資料整理
- [ ] Cosmos pipeline 中 sim / gen / real 比例實測
- [ ] RoboCasa vs 真實 demo 的 ablation 整理
