# Reports · Pulsar-pipeline 自動產出

Pulsar 每個 weekday 自動掃 arxiv、用 qwen3.5-plus 評級、把當天 physics-controllable-generation
相關論文寫成一份 markdown，append 進 `physics-gen-daily/`。**不接 Telegram，整合走 git**（同 VLA / Spatial
姊妹倉決定）；commit 後 Mintlify 7s rebuild 即線上可讀。

| Subdirectory | Cadence | Format |
|---|---|---|
| `physics-gen-daily/` | 每 weekday (00:40 UTC ≈ 08:40 CN) | 當天 arxiv cs.CV / cs.LG / cs.GR / cs.RO / cs.AI / physics.flu-dyn / cond-mat.soft 過濾後 LLM 評級 (⚡/🔧/📖) |
| `weekly/` | 每週五 (02:30 UTC ≈ 10:30 CN) | 前瞻偵察 — 本週主軸 / 意外信號 / 五軸熱度 / 可證偽觀察清單（彙整本週日報的 ⚡/🔧） |

## 機制

```
arxiv RSS → keyword-A 寬鬆過濾 ∩ ¬keyword-C reject → 60d dedup
          → qwen3.5-plus 評 ⚡/🔧/📖/❌（5 軸 tag：output/injection/control/temporal/domain）
          → 寫 physics-gen-daily/YYYY-MM-DD.md（90d 自動 prune）
          → git commit（scoped）→ push → Mintlify rebuild
```

管線程式：[`scripts/pulsar/`](../scripts/pulsar/README.md)。部署：GitHub Actions
`.github/workflows/pulsar-physics-gen-daily.yml`（weekday 00:40 UTC，錯開 Spatial 的 00:30）。

> 這份 archive 是**逐日歸檔內容**，不進 Mintlify 側欄逐頁列表（避免數百筆日期條目），由本 landing page +
> RSS 串接。
