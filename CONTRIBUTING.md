# Contributing

## 提 PR 前

1. 讀 [`AGENTS.md`](AGENTS.md) 的 8 段模板。
2. 新 dissection 放 `foundations/<zone>/<method-slug>.md`，slug 用 lowercase-kebab。
3. 頂部加 `<!-- ontology-5axis output=... injection=... control=... temporal=... domain=... -->`。
4. §8 Pitfall log 需附 GitHub issue / talk timestamp / 二手實測來源，不接受純臆測。

## 哪些不接受

- 純 paper summary（沒對比、沒失效）
- 沒五軸 header
- 沒 §8 pitfall（除非真的找不到任何已知問題，請在 §8 註記「無已知失效記錄」）
- 引入 figure 但沒給 attribution
- 把 sister repo 已涵蓋的主題重複寫（請在該 dissection §6 cross-line synthesis 加一個 cross-ref，不要另開）

## Audit gates

`scripts/handbook_audit.py` 會檢：
- 頂部 5-axis header 存在且 5 軸都有值
- 同 zone 內無重複 method slug
- README ↔ overview 同步（每個子目錄要有 overview.md）
- 內鏈 root-relative（`/foundations/...` 不要寫成 `../foundations/...`）

CI 不過不能 merge。
