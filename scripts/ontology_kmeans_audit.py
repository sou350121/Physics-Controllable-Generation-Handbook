#!/usr/bin/env python3
"""
Ontology K-Means Audit (Hybrid: Data-driven + LLM-driven)
=========================================================

Cross-validates the 5-axis Ontology v2.0 LLM labels against embedding
clusters. Catches LLM tagging drift, taxonomy gaps, and outliers WITHOUT
making any automatic changes.

Why hybrid?
  - LLM labels are expressive but drift across agents/prompts.
  - K-means on embeddings is deterministic but blind to design intent.
  - Hybrid: use k-means to flag candidates; LLM/human decides.

Usage
-----
    export DASHSCOPE_API_KEY=sk-...
    pip install --user numpy   # optional but recommended
    python3 scripts/ontology_kmeans_audit.py [--cap N] [--axis output|injection|...]

Output
------
    /tmp/ontology-kmeans-audit-YYYY-MM-DD.md

What it does
------------
  1. Walks foundations/**/*.md + use-cases/aerial-sim/*.md, parses
     <!-- ontology-5axis ... --> headers.
  2. Extracts §1 TL;DR (first H2 section content) from each.
  3. Embeds (header + TL;DR) via DashScope text-embedding-v3.
  4. For each of 5 axes:
       a. Group dissections by their v2 axis value (multi-value → primary).
       b. Compute intra-cluster mean cosine sim (same-label dissections).
       c. Compute inter-cluster mean cosine sim (different-label pairs).
       d. Per dissection: cosine sim to own-cluster centroid vs nearest other.
       e. Flag if nearest_other_sim > own_cluster_sim → potential mislabel.
  5. Lloyd's k-means (k = unique label count) on full embeddings → cluster
     purity per axis. Reports any label whose cluster assignment dominantly
     belongs to another label.

What it does NOT do
-------------------
  - Doesn't modify any file.
  - Doesn't decide who's right (LLM vs k-means).
  - Doesn't run in CI by default (user-triggered).

Limitations
-----------
  - With only 19 dissections, k-means is underpowered (high variance).
    Treat output as **directional**, not definitive.
  - Axis 5 (domain) often has 1-2 values per dissection — clustering
    those is noisy. Axis 2 (injection) is the most actionable.
"""

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

HANDBOOK_ROOT = Path(__file__).resolve().parent.parent
DISSECTION_GLOBS = [
    "foundations/**/*.md",
    "use-cases/**/*.md",
]
EXCLUDE_NAMES = {"overview.md", "README.md"}

DASHSCOPE_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
)
EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024
MAX_CHARS = 2000  # truncate input per call

# v2 canonical axis values (mirror cheat-sheet/ontology.md)
CANONICAL_VALUES = {
    "output": {"pixel-video", "latent-tokens", "3d-explicit", "3d-implicit",
               "particle", "field", "action-seq", "motion", "N/A"},
    "injection": {"data-only", "aux-loss", "sim-in-loop-train",
                  "sim-in-loop-infer", "guidance-gradient",
                  "architecture-bias-soft", "hard-constraint", "N/A"},
    "control": {"text", "action", "trajectory", "force", "contact",
                "image-init", "3d-init", "param", "camera", "layout", "N/A"},
    "temporal": {"single-frame", "streaming", "autoregressive", "clip-parallel",
                 "latent-rollout", "streaming-cache", "hierarchical"},
    "domain": {"generalist", "robotics", "driving", "fluid", "rigid", "soft",
               "granular", "bio", "weather", "astro", "N/A"},
}


# ─── Step 1: parse headers + TL;DR ────────────────────────────────────────────

HEADER_RE = re.compile(
    r"<!--\s*ontology-5axis\s+"
    r"output=([^\s]+)\s+"
    r"injection=([^\s]+)\s+"
    r"control=([^\s]+)\s+"
    r"temporal=([^\s]+)\s+"
    r"domain=([^\s-]+)\s*-->",
    re.IGNORECASE,
)

def parse_dissection(path):
    # type: (Path) -> Optional[Dict[str, Any]]
    """Parse one dissection: return dict or None if header missing."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = HEADER_RE.search(text)
    if not m:
        return None
    axes = {
        "output": m.group(1).split("|"),
        "injection": m.group(2).split("|"),
        "control": m.group(3).split("|"),
        "temporal": m.group(4).split("|"),
        "domain": m.group(5).split("|"),
    }
    # TL;DR = content up to second H2 (after the first "## ..." headline)
    tldr_match = re.search(
        r"##\s+[^\n]+\n+(.+?)(?=\n##\s)", text, re.DOTALL,
    )
    tldr = tldr_match.group(1).strip() if tldr_match else ""
    return {
        "path": str(path.relative_to(HANDBOOK_ROOT)),
        "axes": axes,
        "tldr": tldr[:MAX_CHARS],
        "header_raw": m.group(0),
    }

def discover_dissections(root=HANDBOOK_ROOT):
    # type: (Path) -> List[Dict[str, Any]]
    seen = set()
    out = []
    for pattern in DISSECTION_GLOBS:
        for p in root.glob(pattern):
            if p.name in EXCLUDE_NAMES or p in seen:
                continue
            seen.add(p)
            d = parse_dissection(p)
            if d is not None:
                out.append(d)
    return sorted(out, key=lambda d: d["path"])


# ─── Step 2: embedding via DashScope ──────────────────────────────────────────

def embed_one(text, api_key):
    # type: (str, str) -> List[float]
    """Single embedding call. Retries 3× on transient errors."""
    payload = {
        "model": EMBED_MODEL,
        "input": {"texts": [text[:MAX_CHARS]]},
        "parameters": {"text_type": "document"},
    }
    body = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                DASHSCOPE_ENDPOINT,
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                emb = data["output"]["embeddings"][0]["embedding"]
                if len(emb) != EMBED_DIM:
                    raise RuntimeError(
                        f"unexpected embedding dim {len(emb)} (want {EMBED_DIM})"
                    )
                return emb
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, RuntimeError) as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"embedding failed after 3 retries: {last_err}")

def embed_all(dissections, api_key):
    # type: (List[Dict[str, Any]], str) -> List[List[float]]
    """Embed (header + TL;DR) for each dissection."""
    embeds = []
    for i, d in enumerate(dissections, 1):
        text = f"{d['header_raw']}\n\n{d['tldr']}"
        print(f"[{i}/{len(dissections)}] embedding {d['path']}", file=sys.stderr)
        embeds.append(embed_one(text, api_key))
    return embeds


# ─── Step 3: cosine sim + k-means (pure python fallback) ──────────────────────

def cosine_sim(a, b):
    # type: (List[float], List[float]) -> float
    if HAS_NUMPY:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if (na and nb) else 0.0

def centroid(vectors):
    if not vectors:
        return None
    if HAS_NUMPY:
        return np.mean(np.asarray(vectors, dtype=np.float64), axis=0).tolist()
    n = len(vectors)
    return [sum(v[i] for v in vectors) / n for i in range(len(vectors[0]))]

def kmeans_lloyd(vectors, k, max_iter=50, seed=0):
    """Minimal Lloyd's k-means. Returns (assignments, centroids)."""
    if HAS_NUMPY:
        rng = np.random.default_rng(seed)
        X = np.asarray(vectors, dtype=np.float64)
        n = X.shape[0]
        if n < k:
            return list(range(n)), X.tolist()
        init_idx = rng.choice(n, size=k, replace=False)
        centroids = X[init_idx].copy()
        for _ in range(max_iter):
            # cosine distance: 1 - cosine_sim
            norms = np.linalg.norm(centroids, axis=1)
            cn = norms == 0
            norms[cn] = 1
            cnorm = centroids / norms[:, None]
            xnorm = X / np.maximum(np.linalg.norm(X, axis=1)[:, None], 1e-12)
            sims = xnorm @ cnorm.T
            assign = sims.argmax(axis=1)
            new_centroids = []
            for ci in range(k):
                pts = X[assign == ci]
                new_centroids.append(
                    pts.mean(axis=0) if len(pts) else centroids[ci]
                )
            new_centroids = np.array(new_centroids)
            if np.allclose(new_centroids, centroids, atol=1e-6):
                break
            centroids = new_centroids
        return assign.tolist(), centroids.tolist()
    # pure-python fallback
    import random
    rng = random.Random(seed)
    n = len(vectors)
    if n < k:
        return list(range(n)), list(vectors)
    centroids = [vectors[i] for i in rng.sample(range(n), k)]
    for _ in range(max_iter):
        assign = []
        for v in vectors:
            best_ci, best_sim = 0, -2.0
            for ci, c in enumerate(centroids):
                s = cosine_sim(v, c)
                if s > best_sim:
                    best_ci, best_sim = ci, s
            assign.append(best_ci)
        new_centroids = []
        for ci in range(k):
            members = [vectors[i] for i, a in enumerate(assign) if a == ci]
            new_centroids.append(centroid(members) if members else centroids[ci])
        # convergence
        delta = max(
            abs(a - b) for c1, c2 in zip(centroids, new_centroids) for a, b in zip(c1, c2)
        )
        centroids = new_centroids
        if delta < 1e-6:
            break
    return assign, centroids


# ─── Step 4: per-axis audit ───────────────────────────────────────────────────

def primary_value(values):
    # type: (List[str]) -> str
    """Pick the first non-N/A value as the primary label."""
    for v in values:
        if v and v.upper() != "N/A":
            return v
    return "N/A"

def per_axis_audit(axis, dissections, embeds):
    # type: (str, List[Dict[str, Any]], List[List[float]]) -> Dict[str, Any]
    """For one axis: compute intra/inter sim + per-dissection nearest-cluster check."""
    labels = [primary_value(d["axes"][axis]) for d in dissections]
    unique_labels = sorted(set(labels))

    # canonical value sanity
    canonical = CANONICAL_VALUES.get(axis, set())
    non_canonical = [v for v in unique_labels if v not in canonical]

    # group by label
    groups = {l: [] for l in unique_labels}
    for i, lab in enumerate(labels):
        groups[lab].append(i)

    # centroid per label
    centroids = {l: centroid([embeds[i] for i in idxs]) for l, idxs in groups.items()}

    # intra-cluster avg cosine sim (members ↔ own centroid)
    intra_avg = {}
    for l, idxs in groups.items():
        if len(idxs) < 2:
            intra_avg[l] = None
            continue
        sims = [cosine_sim(embeds[i], centroids[l]) for i in idxs]
        intra_avg[l] = sum(sims) / len(sims)

    # per-dissection: am I closer to my own centroid or another's?
    mislabel_flags = []
    for i, d in enumerate(dissections):
        own = labels[i]
        own_sim = cosine_sim(embeds[i], centroids[own])
        nearest_other = None
        nearest_other_sim = -2.0
        for l, c in centroids.items():
            if l == own:
                continue
            s = cosine_sim(embeds[i], c)
            if s > nearest_other_sim:
                nearest_other, nearest_other_sim = l, s
        if nearest_other_sim > own_sim:
            mislabel_flags.append({
                "path": d["path"],
                "own_label": own,
                "own_sim": round(own_sim, 4),
                "nearer_label": nearest_other,
                "nearer_sim": round(nearest_other_sim, 4),
                "gap": round(nearest_other_sim - own_sim, 4),
            })

    # k-means with k = # unique labels (>=2)
    kmeans_result = None
    if len(unique_labels) >= 2 and len(dissections) >= 2 * len(unique_labels):
        k = len(unique_labels)
        assign, _ = kmeans_lloyd(embeds, k)
        # cluster purity: dominant label fraction per cluster
        purity = []
        for ci in range(k):
            members = [labels[i] for i, a in enumerate(assign) if a == ci]
            if not members:
                continue
            dominant = max(set(members), key=members.count)
            frac = members.count(dominant) / len(members)
            purity.append({"cluster": ci, "size": len(members),
                           "dominant_label": dominant, "purity": round(frac, 3)})
        kmeans_result = {"k": k, "purity": purity}

    return {
        "axis": axis,
        "n_unique_labels": len(unique_labels),
        "labels": unique_labels,
        "non_canonical": non_canonical,
        "intra_avg_sim": {l: round(v, 4) if v else None for l, v in intra_avg.items()},
        "mislabel_flags": mislabel_flags,
        "kmeans": kmeans_result,
    }


# ─── Step 5: markdown report ──────────────────────────────────────────────────

def format_report(audits, dissections):
    # type: (List[Dict[str, Any]], List[Dict[str, Any]]) -> str
    today = datetime.now().strftime("%Y-%m-%d")
    out = []
    out.append(f"# Ontology K-Means Audit — {today}\n")
    out.append(
        f"> Hybrid (LLM-curated + k-means) sanity check on **{len(dissections)} "
        f"dissections × 5 axes**. **No files modified.** Flags potential mislabels "
        f"and non-canonical values for human review.\n"
    )
    out.append(
        "> Method: embed `(5-axis header + §1 TL;DR)` via DashScope "
        "text-embedding-v3 → per-axis centroid distance + Lloyd k-means.\n"
    )

    out.append("## Summary\n")
    out.append("| Axis | N labels | Non-canonical | Mislabel flags | K-means purity (avg) |")
    out.append("|---|---|---|---|---|")
    for a in audits:
        purity = "—"
        if a["kmeans"]:
            ps = [c["purity"] for c in a["kmeans"]["purity"]]
            purity = f"{sum(ps)/len(ps):.2f}" if ps else "—"
        nc = ", ".join(a["non_canonical"]) if a["non_canonical"] else "0"
        out.append(
            f"| {a['axis']} | {a['n_unique_labels']} | {nc} | "
            f"{len(a['mislabel_flags'])} | {purity} |"
        )
    out.append("")

    for a in audits:
        out.append(f"## Axis: `{a['axis']}`\n")
        out.append(f"**Values used**: {', '.join('`'+l+'`' for l in a['labels'])}\n")
        if a["non_canonical"]:
            out.append(
                f"⚠️ **Non-canonical values** (not in v2 spec): "
                f"{', '.join('`'+v+'`' for v in a['non_canonical'])}\n"
            )
        out.append("**Intra-cluster similarity** (higher = labels embedding-consistent):\n")
        for l, v in a["intra_avg_sim"].items():
            out.append(f"  - `{l}`: {v if v is not None else 'n=1, skip'}")
        if a["mislabel_flags"]:
            out.append(
                "\n**Potential mislabels** (dissection closer to another cluster's centroid):\n"
            )
            out.append("| Path | Own label | Own sim | Nearer label | Nearer sim | Gap |")
            out.append("|---|---|---|---|---|---|")
            for f in a["mislabel_flags"]:
                out.append(
                    f"| `{f['path']}` | `{f['own_label']}` | {f['own_sim']} "
                    f"| `{f['nearer_label']}` | {f['nearer_sim']} | **+{f['gap']}** |"
                )
        else:
            out.append("\n✅ No mislabel flags on this axis.\n")
        if a["kmeans"]:
            out.append(f"\n**K-means (k={a['kmeans']['k']}) cluster purity**:\n")
            for c in a["kmeans"]["purity"]:
                out.append(
                    f"  - cluster {c['cluster']} (n={c['size']}): "
                    f"dominant `{c['dominant_label']}` @ purity {c['purity']}"
                )
        out.append("")

    out.append("---")
    out.append("## How to act on this report\n")
    out.append(
        "1. **Mislabel flags** → review each: is the LLM label wrong, or is "
        "the embedding model confused (e.g., shared vocabulary)?\n"
    )
    out.append(
        "2. **Low intra-cluster sim** → the LLM is using one label for "
        "semantically diverse dissections; consider splitting the value.\n"
    )
    out.append(
        "3. **K-means cluster purity < 0.6** → the embedding doesn't see "
        "the same boundary as the LLM; either the boundary is fuzzy or the "
        "embedding misses some signal (try larger context).\n"
    )
    out.append(
        "4. **Non-canonical values** → audit script `handbook_audit.py "
        "Check 9` should catch these; if not, fix dissection header.\n"
    )
    out.append("")
    out.append(
        "## Limitations\n"
        f"- N={len(dissections)} is small; k-means variance is high.\n"
        "- Embedding signal is dominated by the TL;DR prose; if the TL;DR "
        "doesn't mention the axis explicitly, the embedding may not see "
        "the difference.\n"
        "- This is a **directional sanity check**, not a labeler.\n"
    )
    return "\n".join(out)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cap", type=int, default=None,
                   help="cap number of dissections processed (debug)")
    p.add_argument("--axis", choices=list(CANONICAL_VALUES),
                   default=None, help="only audit one axis")
    p.add_argument("--out", default=None,
                   help="output report path (default /tmp/ontology-kmeans-audit-<date>.md)")
    args = p.parse_args(argv)

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    dissections = discover_dissections()
    if args.cap:
        dissections = dissections[: args.cap]
    if not dissections:
        print("ERROR: no dissections found", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(dissections)} dissections.", file=sys.stderr)

    print(f"Embedding via DashScope {EMBED_MODEL}...", file=sys.stderr)
    embeds = embed_all(dissections, api_key)
    print("Embedding complete.", file=sys.stderr)

    axes = [args.axis] if args.axis else list(CANONICAL_VALUES)
    audits = [per_axis_audit(a, dissections, embeds) for a in axes]

    report = format_report(audits, dissections)
    out_path = (
        Path(args.out) if args.out
        else Path(f"/tmp/ontology-kmeans-audit-{datetime.now().strftime('%Y-%m-%d')}.md")
    )
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
