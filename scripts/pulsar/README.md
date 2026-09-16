# Pulsar — Physics-Gen Daily Pipeline

Phase 1 standalone：arxiv → qwen3.5-plus evaluate → write `reports/physics-gen-daily/` → auto commit.
姊妹於 [Spatial Pulsar](https://github.com/sou350121/Spatial-Intelligence-Handbook/tree/main/scripts/pulsar)，同形態、physics-gen 領域調校。**不接 Telegram，整合走 git**。

## 檔案

```
_config.py        集中 config（arxiv feeds / keyword pool / rating prompt / 5-axis tags）
collect.py        arxiv RSS → 跨 feed 去重 → keyword-A 過濾 ∩ ¬reject-C → 90d dedup → stdout JSON
rate.py           qwen3.5-plus 評 ⚡/🔧/📖/❌ + 一句話 reason + 5-axis tags（drop ❌）
post.py           markdown 落地 reports/physics-gen-daily/YYYY-MM-DD.md（90d prune；TG graceful skip）
run_daily.py      編排 collect → rate → post（單 cron entry）
cron_runner.sh    self-hosted cron wrapper（備用；archive-first 再 informational audit）
state/            seen_arxiv_ids.json dedup cache（**tracked**，見下）
test_pipeline.py  collect 階段去重的回歸閘（audit.yml 會跑）
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

## 去重（2026-09-16 修）

兩個獨立的洞，兩個都是「綠燈 + 看起來合理的報告」，所以誰都看不見：

1. **跨 feed 沒去重**。arxiv 會把同一篇 cross-list 到多個分類，這條管線讀 **7 個 feed**，
   原始聯集就把同一篇算了好幾次。實測 44 份已提交報表：**43 天有重複，共 359 行**；
   2026-09-14 那份 11 條目其實只有 7 篇，`2609.12441` 在 cs.CV / cs.RO / cs.LG 各出現一次，
   **各自送去評級、各自計費**。姊妹 Spatial repo 七月就修了（98587c2），這邊沒跟上。

2. **dedup cache 被 gitignore，而 runner 是無狀態的**。`state/` 原本整個不進 git —— Phase 1
   跑在有持久磁碟的 `cron_runner.sh` 上，那是對的。改跑 GitHub Actions 後每次
   `actions/checkout` 都是全新工作區，**被 ignore 的 cache 就等於不存在的 cache**：
   `load_seen()` 每天回 `{}`，dedup **一次都沒生效過**。實測 `reports/physics-gen-daily/`：
   273 次同一 arXiv id 跨日重複刊出，其中 272 次（99.6%）落在本該擋掉的窗口內。

**不變式**：`DEDUP_WINDOW_DAYS >= REPORT_RETENTION_DAYS`（原本 60 vs 90，記憶比檔案短，
同一篇就會在兩份還看得到的報告裡各出現一次）。由 `_config.py` 在 import 時直接 raise，
並由 `test_pipeline.py` 守住。cache 現在跟著 commit —— workflow 的 `git add` 排在
「沒新內容就不 commit」判斷**之後**，以免每天多出空 commit。
