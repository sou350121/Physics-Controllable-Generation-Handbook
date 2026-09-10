#!/usr/bin/env python3
"""LLM rate papers ⚡/🔧/📖/❌ — DeepSeek deepseek-flash primary, qwen fallback.

Usage:
    python3 scripts/pulsar/collect.py | python3 scripts/pulsar/rate.py
    # or:
    python3 scripts/pulsar/rate.py --in papers.json --out rated.json

Reads JSON list from stdin, writes JSON list with added .rating / .reason / .tags
to stdout. Skips ❌ papers from output (configurable).

Requires: DEEPSEEK_API_KEY (primary). DASHSCOPE_API_KEY enables the fallback.
Pure stdlib + urllib for HTTP (no openai SDK to keep deps light).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _llm
from _config import RATING_PROMPT_SYSTEM, CAT_PRIORITY

# The 4 tiers the prompt is allowed to return. Anything else is a failure, not
# a verdict — see rate_one().
VALID_RATINGS = {"⚡", "🔧", "📖", "❌"}

# If more than this share of papers fail to rate, the sheet is not worth
# publishing. 2026-08-31..09-04 each shipped "⚡0 · 🔧0 · 📖80" with every line
# reading "(rating error: HTTP 401)" and the workflow still went green, because
# per-paper failures were swallowed into 📖 and main() returned 0 regardless.
MAX_FAILURE_RATE = 0.30


def call_qwen(messages: list[dict], api_key: str) -> str:
    """Rate-path LLM call. DeepSeek primary, qwen fallback. Return assistant text.

    Name and signature kept for back-compat with existing callers; the transport
    (both providers, retries, salvage, circuit breaker) now lives in _llm.chat.
    `api_key` is accepted but no longer used — _llm resolves each provider's own
    key, since one call can be served by either. Raises if BOTH providers fail.

    Deliberately no response_format: sending {"type":"json_object"} to DeepSeek
    makes it emit a literal {"type": "json_object"} line before the payload.
    See _llm.py for the measurements.
    """
    return _llm.chat(messages, temperature=0.1, expect_json=True,
                     validate=_is_usable_verdict)


def _is_usable_verdict(parsed) -> bool:
    """A salvaged payload only counts if it carries one of the 4 real tiers.

    Without this a well-formed but content-free 200 (e.g. the leaked
    {"type": "json_object"} marker) would satisfy salvage_json, never reach the
    qwen fallback, and land as a per-paper error instead of a retry.
    """
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else None
    return isinstance(parsed, dict) and parsed.get("rating") in VALID_RATINGS


def rate_one(paper: dict, api_key: str) -> dict:
    """Rate a single paper. Returns paper dict + .rating / .reason / .tags."""
    user_msg = (
        f"Title: {paper['title']}\n"
        f"Category: {paper.get('category', 'n/a')}\n"
        f"Abstract: {paper['abstract'][:1500]}"  # cap to avoid token bloat
    )
    if paper.get("boost"):
        user_msg += "\n(Title contains production/aerial/benchmark signal — boost priority.)"

    messages = [
        {"role": "system", "content": RATING_PROMPT_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    raw = call_qwen(messages, api_key)

    # salvage_json handles the prose-prefix / code-fence / malformed-element
    # shapes DeepSeek produces at HTTP 200. _llm.chat already rejected anything
    # it could not salvage, so a None here means the contract changed.
    result = _llm.salvage_json(raw)
    if isinstance(result, list) and result:
        result = result[0]  # model wrapped the single object in an array
    if not isinstance(result, dict):
        raise ValueError(f"no JSON object recoverable from response for {paper['id']}")

    rating = result.get("rating", "")
    if rating not in VALID_RATINGS:
        # An out-of-vocabulary rating used to be silently coerced to 📖, which is
        # indistinguishable from a real 📖 verdict. Treat it as a failure.
        raise ValueError(f"invalid rating {rating!r} for {paper['id']}")

    paper["rating"] = rating
    paper["reason"] = result.get("reason", "")
    paper["tags"] = result.get("tags", [])
    # Record who actually answered — the header used to hardcode "qwen3.5-plus"
    # even on the days when no qwen call had succeeded at all.
    paper["rated_by"] = _llm.provider_banner()
    return paper


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", help="input JSON file (default: stdin)")
    ap.add_argument("--out", dest="outfile", help="output JSON file (default: stdout)")
    ap.add_argument("--keep-rejects", action="store_true", help="keep ❌ rated papers in output")
    ap.add_argument("--max", type=int, default=80, help="cap papers to rate (avoid LLM cost blow up)")
    args = ap.parse_args()

    if args.infile:
        papers = json.loads(Path(args.infile).read_text())
    else:
        papers = json.loads(sys.stdin.read())

    if not papers:
        print("  (no papers to rate)", file=sys.stderr)
        out = []
    else:
        # Boost papers first, then by category priority
        papers.sort(key=lambda p: (not p.get("boost"), CAT_PRIORITY.get(p.get("category"), 9)))
        papers = papers[:args.max]

        _llm.preflight()
        rated = []
        failures = 0
        for i, p in enumerate(papers):
            print(f"  Rating {i+1}/{len(papers)}: {p['id']} ({p.get('category')})", file=sys.stderr)
            try:
                rated.append(rate_one(p, ""))
            except Exception as e:
                # Per-paper resilience is deliberate — one unlucky abstract must
                # not lose the other 98. What was NOT deliberate is letting a
                # 100%-failure run look like a successful one; see the gate below.
                print(f"  ERROR rating {p['id']}: {e}", file=sys.stderr)
                failures += 1
                p["rating"] = "📖"
                p["reason"] = f"(rating error: {e})"
                p["tags"] = []
                rated.append(p)

        rate_of_failure = failures / len(papers)
        if rate_of_failure > MAX_FAILURE_RATE:
            print(
                f"\nFAIL: {failures}/{len(papers)} papers ({rate_of_failure:.0%}) could not be "
                f"rated — above the {MAX_FAILURE_RATE:.0%} threshold. Refusing to emit a "
                f"placeholder sheet; the report for today is intentionally NOT written so "
                f"the workflow sentinel fires. Check the provider errors above.",
                file=sys.stderr,
            )
            return 1
        if failures:
            print(f"  {failures}/{len(papers)} papers failed to rate (within threshold)",
                  file=sys.stderr)

        if not args.keep_rejects:
            out = [p for p in rated if p.get("rating") != "❌"]
            print(f"  Kept {len(out)}/{len(rated)} (dropped ❌)", file=sys.stderr)
        else:
            out = rated

    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.outfile:
        Path(args.outfile).write_text(text)
    else:
        print(text)

    # Summary
    if out:
        from collections import Counter
        ratings = Counter(p["rating"] for p in out)
        print(f"  Rating distribution: {dict(ratings)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
