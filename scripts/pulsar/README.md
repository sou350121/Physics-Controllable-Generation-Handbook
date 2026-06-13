# Pulsar — Physics-Gen Daily Pipeline

Phase 1 standalone：arxiv → qwen3.5-plus evaluate → write `reports/physics-gen-daily/` → auto commit.
姊妹於 [Spatial Pulsar](https://github.com/sou350121/Spatial-Intelligence-Handbook/tree/main/scripts/pulsar)，同形態、physics-gen 領域調校。**不接 Telegram，整合走 git**。

## 檔案

```
_config.py        集中 config（arxiv feeds / keyword pool / rating prompt / 5-axis tags）
collect.py        arxiv RSS → keyword-A 過濾 ∩ ¬reject-C → 60d dedup → stdout JSON
rate.py           qwen3.5-plus 評 ⚡/🔧/📖/❌ + 一句話 reason + 5-axis tags（drop ❌）
post.py           markdown 落地 reports/physics-gen-daily/YYYY-MM-DD.md（90d prune；TG graceful skip）
run_daily.py      編排 collect → rate → post（單 cron entry）
cron_runner.sh    self-hosted cron wrapper（備用；archive-first 再 informational audit）
state/            seen_arxiv_ids.json dedup cache（gitignored）
```

## 領域調校（vs Spatial）

- **arxiv categories**：cs.CV / cs.LG / cs.AI / cs.GR / cs.RO / **physics.flu-dyn** / **cond-mat.soft**
- **keyword pool**：world model / diffusion-physics / differentiable-sim / neural-surrogate(PDE/CFD/weather) /
  physics-conditioning / controllability / 3D-aware gen / 長程 rollout / 生成式機器人數據
- **rating prompt**：scope 到 physics-controllable generation；tags 取 5 軸（output/injection/control/temporal/domain）
- **output 路徑**：`reports/physics-gen-daily/`

## 部署（live）

GitHub Actions [`.github/workflows/pulsar-physics-gen-daily.yml`](../../.github/workflows/pulsar-physics-gen-daily.yml)：
- schedule：`40 0 * * 1-5`（weekday 00:40 UTC ≈ 08:40 CN，錯開 Spatial 的 00:30）
- secret：`DASHSCOPE_API_KEY`
- 設計：先 commit 日檔（scoped `reports/physics-gen-daily/`）→ 再跑 audit（informational, `continue-on-error`），
  日檔不被無關審計阻斷（取自 Spatial 復活教訓）。

## 本地測試

```bash
export DASHSCOPE_API_KEY=sk-...
python3 scripts/pulsar/run_daily.py            # 完整跑（weekday；週末 arxiv 空）
PHYSGEN_DRY_RUN=1 python3 scripts/pulsar/run_daily.py   # 評級但跳 TG
PHYSGEN_DATE=2026-06-15 python3 scripts/pulsar/collect.py   # 指定日期（繞 weekend gate）
```
