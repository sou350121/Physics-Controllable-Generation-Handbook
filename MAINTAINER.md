# Maintainer Notes

## 倉位置

- GitHub: `sou350121/Physics-Controllable-Generation-Handbook`（待建）
- Local checkout: `/home/claudeuser/Physics-Controllable-Generation-Handbook/`
- Sister handbooks: VLA-Handbook · Spatial-Intelligence-Handbook
- Mintlify (Phase 2): `<TBD>.mintlify.app`

## Pulsar integration (Phase 2)

- Daily arxiv 抓 → qwen3.5-plus 評級 → 寫 `reports/physics-gen-daily/`
- GitHub Actions workflow: `.github/workflows/pulsar-physics-gen-daily.yml`（待加）
- Schedule: weekday 00:30 UTC (沿用 spatial 模式，避免同分鐘搶 DashScope quota — 建議改成 00:40 UTC)
- Secret: `DASHSCOPE_API_KEY`（從 spatial 復用 value `sk-3cb6841934bd4df987d2a4fe8dac5839`）

## 評級 keyword pool (待調)

候選 keyword（compose 自 spatial pulsar `_config.py`）：
- World model: world model, neural simulator, video world model, action-conditioned video
- Physics: differentiable physics, physics-informed, PINN, neural PDE, neural surrogate
- Controllability: ControlNet, classifier guidance, force-conditioned, trajectory-conditioned
- Models: Cosmos, Sora, Veo, Genie, V-JEPA, Dreamer, GAIA, PhysGen, MeshGraphNet, FNO

## 與 sister repos 的協調

- 新 dissection 若主題已在 sister repo 出現，§6 cross-line synthesis **必須** cross-ref，不要重寫
- bridge-to-vla / bridge-to-spatial 維護由本倉負責；對方倉若也有 bridge dir，注意對齊
- Cosmos 系列：spatial 已有 `companies/nvidia_cosmos.md`，本倉聚焦其 Cosmos-Predict / Cosmos-Reason / Cosmos-Drive 物理可控生成部分，差異化在「物理控制 axis」

## Release

- 第一個 milestone：12 anchor dissections + Mintlify deploy
- 第二個：Pulsar Phase 1 daily
- 第三個：30+ dissections + §8 pitfall coverage > 80%
