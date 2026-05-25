# Scripts

Tooling for handbook maintenance + Pulsar Phase 1 daily pipeline.

| Script | 角色 | Status |
|---|---|---|
| `handbook_audit.py` | 13 audit checks（ontology header, internal links, etc.） | 🟡 待移植自 spatial |
| `gen_mintlify_nav.py` | 從目錄結構生成 docs.json navigation | 🟡 待移植 |
| `sync_readme_from_overview.py` | 子目錄 overview.md → README.md mirror | 🟡 待移植 |
| `inject_ontology_headers.py` | 在新 dissection 自動補 5-axis HTML comment header | 🟡 待移植 |
| `query_ontology.py` | 查 ontology 五軸的全文索引 | 🟡 待移植 |
| `pulsar/` | Daily arxiv pipeline | 🟡 待從 spatial clone |

## 移植 plan

1. 從 `/home/claudeuser/Spatial-Intelligence-Handbook/scripts/` 複製
2. 改 `_config.py`：RSS sources / keyword pool / output dir
3. 改 `gen_mintlify_nav.py`：本倉 nav 結構
4. 改 `handbook_audit.py`：5-axis values 對應本倉 ontology v1
