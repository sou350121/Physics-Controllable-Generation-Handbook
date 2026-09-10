"""Physics-Controllable-Generation Handbook — Pulsar pipeline config (Phase 1).

Standalone version: runs anywhere with Python 3.9+ + the env vars below.
Sister pipeline to Spatial-Intelligence-Handbook's scripts/pulsar/ — same shape,
physics-gen domain tuning (arxiv categories, keywords, rating prompt, 5-axis tags).

Env vars required:
    DEEPSEEK_API_KEY    — DeepSeek deepseek-flash (primary rater)

Env vars optional:
    DASHSCOPE_API_KEY   — Aliyun qwen (fallback rater; run degrades loudly
                          without it rather than emitting placeholder sheets)
    PHYSGEN_DEEPSEEK_MODEL — override the DeepSeek model id
    TELEGRAM_BOT_TOKEN  — enable TG push (skipped gracefully if absent)
    TELEGRAM_CHAT_ID    — TG target chat ID
    PHYSGEN_DRY_RUN=1   — collect + rate only, skip writes (dev/test)
    PHYSGEN_DATE        — override "today" in YYYY-MM-DD (for backfill)

Default workflow: handbook integration via git (commit reports/physics-gen-daily/).
TG is optional opt-in (same decision as VLA / Spatial: git integration only).
"""
from __future__ import annotations
import os
from pathlib import Path

# ---- Paths ----------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STATE_DIR = REPO_ROOT / "scripts" / "pulsar" / "state"
REPORTS_DIR = REPO_ROOT / "reports" / "physics-gen-daily"
WEEKLY_DIR = REPO_ROOT / "reports" / "weekly"

# ---- LLM ------------------------------------------------------------
# Primary: DeepSeek. Fallback: DashScope qwen. Transport lives in _llm.py —
# see that module's docstring for the measurements behind these constants.
#
# DeepSeek became primary on 2026-09-10 after the DASHSCOPE_API_KEY Actions
# secret was revoked and every CI rating call started returning HTTP 401 while
# the workflow still reported success (reports/ 08-31..09-04 are placeholders).
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# `deepseek-flash` is the real id; deepseek-v4-flash / deepseek-chat /
# deepseek-reasoner are aliases onto it. deepseek-v4.1-* does not exist.
DEEPSEEK_MODEL = os.environ.get("PHYSGEN_DEEPSEEK_MODEL", "deepseek-flash")
# 65536 is load-bearing, not arbitrary: DeepSeek scales how much it reasons to
# the max_tokens it is given, and reasoning is charged against the same budget.
# At 1024 it spent the entire budget thinking and returned EMPTY content with
# finish_reason="length" (measured on this repo's single-paper prompt).
DEEPSEEK_MAX_TOKENS = 65536

# DashScope OpenAI-compatible endpoint (CodingPlan Pro since 2026-04)
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-plus"  # qwen3.5-plus alias on DashScope; cheaper than max
LLM_TIMEOUT = 180  # seconds; bumped for thinking mode
LLM_RETRY = 3
LLM_RETRY_BACKOFF = 5  # base seconds; (attempt+1) * backoff

# ---- Telegram -------------------------------------------------------
# Kept for parity with sister pipelines but NOT used by default (git integration only).
TG_API = "https://api.telegram.org/bot{token}/sendMessage"
TG_PARSE_MODE = "HTML"
TG_MAX_LEN = 4096  # Telegram message limit

# ---- RSS sources ----------------------------------------------------
# arxiv categories most relevant to physics-controllable *generation* / world models.
# Differs from Spatial (which is perception-side): we add cs.GR (graphics/sim/render),
# physics.flu-dyn (CFD surrogates) and cond-mat.soft (MPM / particle / soft-matter).
#
# HOST: must be rss.arxiv.org, NOT export.arxiv.org.
# Measured 2026-09-10: http://export.arxiv.org/rss/<cat> 301-redirects to https and
# then returns HTTP 200 with a *well-formed but item-less* channel — zero <item>
# elements for all 7 categories, while https://rss.arxiv.org/rss/<cat> returns
# 34–860. The dead feed even self-identifies: its <link> and <atom:link href> both
# point at rss.arxiv.org. Because it is a 200 with valid XML, fetch_rss raised
# nothing and the pipeline logged the cheerful "Fetched: 0 papers across 7 feeds"
# on 09-08 and 09-09 while exiting 0. Do not "simplify" this back to export.*.
ARXIV_RSS_HOST = "https://rss.arxiv.org/rss"
ARXIV_FEEDS = {
    "cs.CV": f"{ARXIV_RSS_HOST}/cs.CV",            # video gen / diffusion / 3D
    "cs.LG": f"{ARXIV_RSS_HOST}/cs.LG",            # world models / neural surrogates
    "cs.AI": f"{ARXIV_RSS_HOST}/cs.AI",            # foundation models
    "cs.GR": f"{ARXIV_RSS_HOST}/cs.GR",            # graphics / simulation / rendering
    "cs.RO": f"{ARXIV_RSS_HOST}/cs.RO",            # robotics-data-gen / sim-to-real
    "physics.flu-dyn": f"{ARXIV_RSS_HOST}/physics.flu-dyn",  # fluid surrogates
    "cond-mat.soft": f"{ARXIV_RSS_HOST}/cond-mat.soft",      # MPM / particle / soft matter
}

# Category sort priority (lower = surfaced first). Shared by rate.py + post.py.
CAT_PRIORITY = {
    "cs.CV": 0,
    "cs.GR": 1,
    "cs.LG": 2,
    "cs.RO": 3,
    "cs.AI": 4,
    "physics.flu-dyn": 5,
    "cond-mat.soft": 6,
}

# Skip arxiv on weekends (no new papers; same as VLA / Spatial)
SKIP_WEEKENDS = True

# ---- Filter keywords (Layer A — broad inclusion) --------------------
# A paper passes Layer A if title or abstract contains any of these.
# Tuned for physics-controllable generation (per ontology v2 5 axes).
KEYWORDS_A = [
    # Generative world models / video
    "world model", "world simulator", "video generation", "video diffusion",
    "video prediction", "video world model", "generative world model",
    "neural simulator", "learned simulator", "interactive world",
    # Diffusion / score / flow
    "diffusion model", "latent diffusion", "score-based", "flow matching",
    "consistency model", "rectified flow",
    # Physics injection (this handbook's USP axis)
    "physics-based", "physics-informed", "physically plausible", "physical consistency",
    "physical commonsense", "PINN", "conservation law", "Hamiltonian neural",
    "Lagrangian neural", "physics constraint", "physically grounded",
    # Differentiable / classical simulation
    "differentiable simulation", "differentiable physics", "physics simulation",
    "material point method", "MPM", "smoothed particle", "SPH", "fluid simulation",
    "cloth simulation", "rigid body", "soft body", "contact dynamics", "physics engine",
    # Neural surrogates / PDE / scientific
    "neural surrogate", "neural operator", "Fourier neural operator", "neural PDE",
    "operator learning", "weather forecast", "climate model", "turbulence",
    "computational fluid", "graph network simulator",
    # Controllability / conditioning
    "controllable generation", "controllable video", "action-conditioned",
    "trajectory-conditioned", "force prompt", "camera control", "motion control",
    # 3D-aware generation
    "3D generation", "3D-aware", "scene generation", "Gaussian splatting", "4D generation",
    # Embodied data / policy from video
    "robot data generation", "sim-to-real", "video pretraining", "world model policy",
    "embodied world model", "long-horizon rollout", "autoregressive video",
]

# ---- Filter keywords (Layer B — boost / promotion signals) ---------
# Papers with these in title get rated higher priority (⚡/🔧 vs 📖).
KEYWORDS_B_BOOST = [
    "real-time", "interactive", "streaming", "online",
    "open-source", "open weight", "open-world",
    "benchmark", "VBench", "physics benchmark",
    "foundation model", "state-of-the-art", "SOTA",
    "robot", "driving", "embodied", "controllable",
]

# ---- Filter keywords (Layer C — reject / noise) ---------------------
# Papers with these in title are rejected outright (irrelevant to physics-gen).
KEYWORDS_C_REJECT = [
    "medical imaging", "medical image", "tumor", "MRI", "X-ray", "histopathology",
    "speech recognition", "ASR", "audio classification", "music generation",
    "sentiment analysis", "text classification", "named entity", "machine translation",
    "recommendation system", "click-through", "advertising",
    "federated learning", "blockchain", "cryptograph",
    "drug discovery", "protein folding",  # bio-only, not physics-gen world models
]

# ---- Rating prompt (Layer D — LLM) ----------------------------------
# 4-tier rating per VLA / Spatial convention.
RATING_PROMPT_SYSTEM = """你是 Physics-Controllable-Generation Handbook 的論文評級助手。

本 handbook 關注「physics-as-conditioning 世界模型 / 可控生成」：video / latent / 3D / simulator /
surrogate 五條路線，核心問題是「物理規律如何進入生成模型、如何被條件控制」。

每篇 paper 用 4 級之一評：
- ⚡ load-bearing breakthrough（範式信號 / 新能力 / 高引用潛力）
- 🔧 engineering value（複現 / SOTA / production-ready，但無範式創新）
- 📖 reference（survey / 教學 / 重要 baseline）
- ❌ reject（離題 / 弱 novelty / 已被取代 / 純 NLP-LLM 無生成或世界模型成分）

評級基於：
1. 是否屬於 physics-controllable generation 範疇（video/world model / diffusion-physics /
   differentiable-sim / neural-surrogate(PDE/CFD/weather) / physics-conditioning(PINN/守恆) /
   controllability / 3D-aware gen / 長程 rollout / 生成式機器人數據）
2. 是否觸及本倉 USP 軸「physics injection」（data-only / aux-loss / sim-in-loop / guidance-gradient /
   architecture-bias / hard-constraint）— 觸及越深越高分
3. 是否有 reproducibility（GitHub repo / 數據集 / 訓練 recipe）
4. 是否解 known limitation（守恆違反 / 長程漂移 / 可控性-保真度 trade-off）或開新方向

回答格式（嚴格 JSON）：
{
  "rating": "⚡" | "🔧" | "📖" | "❌",
  "reason": "一句話評理由（中文）",
  "tags": ["axis 標籤", ...]
}
tags 從 5 軸詞彙選 1-3 個（output / injection / control / temporal / domain），例如
["output:pixel-video", "injection:guidance-gradient", "domain:fluid"]。
"""

# ---- Daily report format --------------------------------------------
REPORT_TOP_N = 5  # top N ⚡/🔧 picks to push + write to markdown
REPORT_RETENTION_DAYS = 90  # auto-clean reports/physics-gen-daily/ older than N days

# ---- Weekly synthesis -----------------------------------------------
WEEKLY_TITLE = "Physics-Gen Weekly"   # report H1 / banner label
WEEKLY_LOOKBACK_DAYS = 7        # how many days of dailies to aggregate
WEEKLY_RETENTION_WEEKS = 26     # auto-clean reports/weekly/ older than N weeks
# Forward-looking (前瞻偵察) per VLA convention — weekly = scout, not retrospective.
WEEKLY_PROMPT_SYSTEM = """你是 Physics-Controllable-Generation Handbook 的週度前瞻偵察員。

輸入是本週每日 arxiv 評級裡的 ⚡/🔧 論文（physics-controllable generation 領域：video/world-model /
diffusion-physics / differentiable-sim / neural-surrogate / physics-conditioning / controllability /
3D-aware gen / 長程 rollout / 生成式機器人數據）。

週報不是日報的索引，是**前瞻判斷**。寫成 markdown，4 節：
1. **## 本週主軸** — 2-3 個反覆出現的主題/技術方向（每個 1-2 句，點名代表論文）。
2. **## 意外信號** — 1-3 個出乎意料或反共識的點（沒有就寫「本週無明顯意外」）。
3. **## 五軸熱度** — 這週論文在 5 軸（output/injection/control/temporal/domain）上偏向哪裡；
   特別關注 USP 軸 injection（data-only→hard-constraint 的物理注入強度光譜）這週移動到哪。
4. **## 可證偽觀察清單** — 2-4 條**下週可檢驗**的具體預測或待觀察項（要可證偽，不要空話）。

只依據輸入論文，不要編造不存在的論文或數字。語氣精煉、判斷性強，避免堆砌。
"""

# ---- Memory / dedup -------------------------------------------------
DEDUP_FILE = STATE_DIR / "seen_arxiv_ids.json"
DEDUP_WINDOW_DAYS = 60  # don't re-rate papers seen in last 60 days


# ---- Env vars -------------------------------------------------------
def get_env(name: str, required: bool = True) -> str:
    """Return env var or raise if required."""
    v = os.environ.get(name, "")
    if required and not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def is_dry_run() -> bool:
    return os.environ.get("PHYSGEN_DRY_RUN", "") == "1"


def today_str() -> str:
    import datetime
    return os.environ.get("PHYSGEN_DATE", datetime.date.today().isoformat())
