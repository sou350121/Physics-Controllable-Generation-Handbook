# Pulsar — Physics-Gen Daily Pipeline

Phase 1 standalone：arxiv → qwen evaluate → write `reports/physics-gen-daily/` → auto commit.

## 待移植檔（從 Spatial Pulsar）

```
_config.py        集中 config
collect.py        arxiv RSS → keyword filter
rate.py           qwen3.5-plus 評 ⚡/🔧/📖/❌
post.py           markdown 落地（不接 TG）
run_daily.py      編排 collect → rate → post
cron_runner.sh    self-hosted cron wrapper（備用）
```

## 移植要點

- 改 keyword pool（physics-gen 專屬，見 `docs/pulsar-integration.md`）
- 改 output 路徑：`reports/physics-gen-daily/`
- 改 arxiv categories：cs.LG, cs.CV, cs.GR, cs.RO, **physics.flu-dyn**, **cond-mat.soft**
- TG 保持 graceful skip（如 spatial 設計）

## 部署 (Phase 2)

GitHub Actions `.github/workflows/pulsar-physics-gen-daily.yml`：
- schedule: `40 0 * * 1-5`（weekday 00:40 UTC ≈ 08:40 CN，錯開 spatial 的 00:30）
- secrets: `DASHSCOPE_API_KEY`
