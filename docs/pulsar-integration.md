# Pulsar Integration

`reports/physics-gen-daily/` 是 Pulsar Phase 1 自動產出區。

## 路徑

- **Pipeline scripts**: `scripts/pulsar/` （與 Spatial Pulsar 同套架構）
- **Output**: `reports/physics-gen-daily/YYYY-MM-DD.md`
- **GitHub Actions**: `.github/workflows/pulsar-physics-gen-daily.yml`（Phase 2 加）
- **Schedule**: 建議 weekday 00:40 UTC（避開 spatial 的 00:30 同分鐘 DashScope quota 搶）

## Pipeline (clone 自 Spatial Phase 1)

| 檔 | 角色 |
|---|---|
| `_config.py` | RSS / qwen / keyword 集中 config |
| `collect.py` | 4-5 arxiv RSS (cs.LG/CV/GR/RO + physics.flu-dyn) → keyword filter → dedup |
| `rate.py` | qwen3.5-plus 評 ⚡/🔧/📖/❌ |
| `post.py` | markdown 落地；不接 TG（純 git） |
| `run_daily.py` | 一次 orchestrate |

## 環境變數

- `DASHSCOPE_API_KEY` — 必需，從 spatial 共用 `sk-3cb6841934bd4df987d2a4fe8dac5839`
- `PHYS_DRY_RUN=1` — test 模式
- `PHYS_DATE=YYYY-MM-DD` — backfill

## Keyword pool（v0.1 候選）

複用 spatial pool + 加 physics-gen 專屬：
- world model, neural simulator, video world model, action-conditioned video
- differentiable physics, PINN, neural PDE, neural surrogate, FNO, MeshGraphNet
- ControlNet, classifier guidance, force-conditioned, trajectory-conditioned
- Cosmos, Sora, Veo, Genie, V-JEPA, Dreamer, GAIA, PhysGen, ContactGen, ForceGen

## Phase 1 → Phase 2

- Phase 1: standalone daily（本倉 GitHub Actions）
- Phase 2: 整合 Pulsar 主倉 `memory/domains.json` 註冊 `physics_gen` domain，與 vla / ai_app 並列
- Phase 3: weekly summary / cross-domain insight 加入

## 不接 TG

跟 Spatial 同決定：handbook 整合走 git，Mintlify 7s rebuild = 同等即時性。
