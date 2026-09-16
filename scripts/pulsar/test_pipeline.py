#!/usr/bin/env python3.11
"""Regression check for the collect-stage dedup. No network, no LLM, no pytest.

    python3 scripts/pulsar/test_pipeline.py

Mirrors the sibling Spatial repo's scripts/pulsar/test_gates.py. Every case here
is a measured incident from the 2026-09-16 audit, not a hypothetical:

  * The cross-feed dedup did not exist in this repo. arxiv cross-lists one paper
    under several categories and this pipeline reads seven feeds, so the raw
    union listed the same paper up to three times in one sheet — 359 duplicate
    lines across 43 of the 44 committed reports, each duplicate separately rated
    and separately billed.
  * The 60-day dedup cache was gitignored while the runner is stateless, so
    load_seen() returned {} on every CI run and the dedup never fired at all:
    273 cross-day re-emissions, 272 inside the window.
  * DEDUP_WINDOW_DAYS (60) was shorter than REPORT_RETENTION_DAYS (90) — two
    constants that had to agree with nothing enforcing the ordering.
"""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _config  # noqa: E402
import collect  # noqa: E402


class WindowInvariants(unittest.TestCase):
    """Pairs of constants that must agree, with something enforcing it."""

    def test_dedup_memory_outlives_the_published_archive(self):
        """A paper must stay in the dedup memory for at least as long as the
        report that published it stays readable, or the archive ends up holding
        two live copies of it. 60 vs 90 until 2026-09-16."""
        self.assertGreaterEqual(_config.DEDUP_WINDOW_DAYS, _config.REPORT_RETENTION_DAYS)

    def test_weekly_lookback_covers_its_own_cadence(self):
        """run_weekly aggregates WEEKLY_LOOKBACK_DAYS of dailies and runs every
        Friday; a lookback shorter than the gap would silently skip days."""
        self.assertGreaterEqual(_config.WEEKLY_LOOKBACK_DAYS, 7)


class DedupCacheIsDurable(unittest.TestCase):
    def test_cache_is_tracked_by_git(self):
        """The runner is stateless (actions/checkout). An ignored cache file is
        an absent cache file: load_seen() returns {} and the dedup never fires.
        This is the whole 2026-09-16 finding in one assertion."""
        rel = _config.DEDUP_FILE.relative_to(_config.REPO_ROOT)
        out = subprocess.run(["git", "check-ignore", "-q", str(rel)],
                             cwd=_config.REPO_ROOT, capture_output=True)
        self.assertNotEqual(out.returncode, 0,
                            f"{rel} is gitignored — the CI runner will start every "
                            "day with an empty cache")

    def test_cache_is_populated_and_parses(self):
        seen = collect.load_seen()
        self.assertGreater(len(seen), 0, "committed dedup cache is empty")
        self.assertTrue(all(isinstance(v, str) and len(v) == 10 for v in seen.values()))

    def test_workflow_commits_the_cache_after_the_no_change_check(self):
        """Order matters: staged before the check, every quiet day becomes an
        empty commit; never staged at all, the cache never leaves the runner."""
        wf = (_config.REPO_ROOT / ".github" / "workflows"
              / "pulsar-physics-gen-daily.yml").read_text(encoding="utf-8")
        self.assertIn("git add scripts/pulsar/state/seen_arxiv_ids.json", wf)
        self.assertLess(wf.index("No new physics-gen-daily report"),
                        wf.index("git add scripts/pulsar/state/seen_arxiv_ids.json"))

    def test_load_seen_survives_a_corrupt_cache(self):
        real = _config.DEDUP_FILE
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "seen.json"
            p.write_text("{not json")
            collect.DEDUP_FILE = p
            try:
                self.assertEqual(collect.load_seen(), {})
            finally:
                collect.DEDUP_FILE = real


class CrossFeedDedup(unittest.TestCase):
    """arxiv cross-lists; seven feeds means up to seven copies of one paper."""

    def test_cross_listed_paper_is_counted_once(self):
        papers = [
            {"id": "2609.12441", "title": "IMPLY", "abstract": "physics video", "category": "cs.CV"},
            {"id": "2609.12441", "title": "IMPLY", "abstract": "physics video", "category": "cs.RO"},
            {"id": "2609.12441", "title": "IMPLY", "abstract": "physics video", "category": "cs.LG"},
            {"id": "2609.13146", "title": "SNAP3D", "abstract": "physics 3d", "category": "cs.CV"},
        ]
        out, seen_today = [], set()
        for p in papers:                       # the block under test, in isolation
            if p["id"] in seen_today:
                continue
            seen_today.add(p["id"])
            out.append(p)
        self.assertEqual([p["id"] for p in out], ["2609.12441", "2609.13146"])

    def test_collect_contains_the_cross_feed_dedup(self):
        """Guard the wiring: the block must be in collect_today(), before the
        seen-cache dedup, not just reimplemented in this test."""
        src = Path(collect.__file__).read_text(encoding="utf-8")
        self.assertIn("Cross-feed dedup", src)
        self.assertLess(src.index("seen_today"), src.index('p["id"] not in seen'))


class NoDuplicateLinesInCommittedReports(unittest.TestCase):
    def test_the_most_recent_report_lists_each_paper_once(self):
        """2026-09-14 shipped 11 entries for 7 papers. New sheets must not."""
        import re
        import datetime
        dated = []
        for f in _config.REPORTS_DIR.glob("*.md"):      # README.md etc. are not sheets
            try:
                dated.append((datetime.date.fromisoformat(f.stem), f))
            except ValueError:
                continue
        if not dated:
            self.skipTest("no dated reports yet")
        day, latest = max(dated)
        # Only enforce on sheets written after the fix; the historical archive is
        # append-only and is deliberately left as the record of what happened.
        if day <= datetime.date(2026, 9, 16):
            self.skipTest(f"{latest.stem} predates the cross-feed dedup fix")
        ids = re.findall(r"\((\d{4}\.\d{4,5})\s*·", latest.read_text(encoding="utf-8"))
        self.assertEqual(len(ids), len(set(ids)), f"{latest.name} lists a paper twice")


if __name__ == "__main__":
    unittest.main(verbosity=2)
