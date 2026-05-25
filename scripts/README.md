# Scripts

Tooling for handbook maintenance + Pulsar Phase 2 daily pipeline.

| Script | 角色 | Status |
|---|---|---|
| **[`ontology_kmeans_audit.py`](./ontology_kmeans_audit.py)** | **Hybrid data-driven + LLM-driven ontology sanity check** — embed dissection headers + TL;DR via DashScope text-embedding-v3 → per-axis k-means + intra/inter cluster cosine sim → flag potential mislabels | ✅ **ready (user-triggered)** |
| `handbook_audit.py` | 14 audit checks（ontology header, internal links, etc.） | 🟡 待移植自 spatial |
| `gen_mintlify_nav.py` | 從目錄結構生成 docs.json navigation | 🟡 待移植 |
| `sync_readme_from_overview.py` | 子目錄 overview.md → README.md mirror | 🟡 待移植 |
| `inject_ontology_headers.py` | 在新 dissection 自動補 5-axis HTML comment header | 🟡 待移植 |
| `query_ontology.py` | 查 ontology 五軸的全文索引 | 🟡 待移植 |
| `pulsar/` | Daily arxiv pipeline | 🟡 待從 spatial clone |

## `ontology_kmeans_audit.py` — Quick start

### What it does
Cross-validates the 5-axis Ontology v2.0 LLM labels against embedding clusters.
Catches LLM tagging drift, taxonomy gaps, and outliers WITHOUT making any
automatic changes.

**Why hybrid?**
- LLM labels are expressive but drift across agents/prompts.
- K-means on embeddings is deterministic but blind to design intent.
- Hybrid: use k-means to flag candidates; LLM/human decides.

### Run

```bash
export DASHSCOPE_API_KEY=sk-...           # required
pip install --user numpy                  # optional (speeds up; falls back to pure-python)
python3 scripts/ontology_kmeans_audit.py
```

Output: `/tmp/ontology-kmeans-audit-YYYY-MM-DD.md`

### Options

```
--cap N          process only first N dissections (debug)
--axis output    only audit one axis (output / injection / control / temporal / domain)
--out PATH       output report path
```

### What's in the report

For each of the 5 axes:
- **Values used** in current 19 dissections
- **Non-canonical** flag (any value not in v2 spec)
- **Intra-cluster cosine sim** per label (higher = embedding agrees with LLM grouping)
- **Mislabel flags**: dissections closer to another label's cluster centroid than their own
- **K-means cluster purity** (k = #unique labels) — how cleanly embedding clusters match LLM labels

### How to interpret

| Signal | What it might mean |
|---|---|
| Many mislabel flags on Axis 2 (injection) | LLM is tagging similar-content papers with different injection labels — taxonomy is fuzzy |
| Low intra-cluster sim on a value | That value covers a semantically wide range — consider splitting |
| K-means purity < 0.6 | Embedding doesn't see the LLM's boundary — boundary is fuzzy OR embedding misses signal |
| Non-canonical values | Audit script Check 9 will catch these; fix dissection header |

### Limitations

- N=19 is small for k-means; treat output as **directional**, not definitive.
- Embedding sees TL;DR prose; if TL;DR doesn't mention the axis explicitly, embedding may not see the difference.
- Axis 5 (domain) is least actionable (many dissections have 1-2 domain values).
- This is **sanity-check / candidate generation**, not an autolabeler.

### Cost estimate

- 19 dissections × 1 embedding call ≈ 19 calls (~$0.0005 total at DashScope rates)
- ~30 seconds wall-clock including retries
