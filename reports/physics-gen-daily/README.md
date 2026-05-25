# Physics-Gen Daily Reports

Pulsar Phase 1 自動產出。每個 weekday 一個 `YYYY-MM-DD.md`，內容：

- arxiv (cs.LG/CV/GR/RO + physics.flu-dyn + cond-mat.soft) 當日新論文 keyword-filtered
- 每篇 qwen3.5-plus 評 ⚡/🔧/📖/❌ + 一句話 takeaway
- 90 天 retention（超過自動清）

第一筆預計：Phase 2 部署後（待 scripts/pulsar/ 移植完成）。

## 評級語義

- ⚡ **影響重大**（新 SOTA / 範式 / Cosmos-tier release）
- 🔧 **工程實作有料**（reproducible，改進明確）
- 📖 **值得知道**（背景知識 / 綜述）
- ❌ **不收**（無物理 / 無 controllability）

## Format

```markdown
# Physics-Gen Daily — YYYY-MM-DD

## ⚡
- [Title](arxiv-url) — 一句話 takeaway

## 🔧
...

## 📖
...
```
