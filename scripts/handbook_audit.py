#!/usr/bin/env python3
"""
Handbook Audit — Physics-Controllable-Generation-Handbook 自动质量门槛 lint。

Sister to Spatial-Intelligence-Handbook's scripts/handbook_audit.py — same shape,
physics-gen-tuned checks (5-axis ontology = output/injection/control/temporal/domain;
no spatial-specific HKUST/atlas/zone-count assertions).

8 个 check：
  1. Broken Links        — 所有 .md 内相对路径 markdown link 必须 resolve
  2. Ontology 5-axis     — 每篇 dissection (foundations/ + use-cases/ 内容档) 顶部必含
                           <!-- ontology-5axis ... -->，且 5 轴 (output/injection/control/
                           temporal/domain) 全有 (ontology.md Check 9 承诺的落地)
  3. Mintlify Nav        — 每个 repo 内 .md 必须出现在 docs.json nav；nav 引用的 page 必须存在
                           (dated report archives + AUDIT_* + README 豁免；.mdx extensionless)
  4. No empty groups     — docs.json 不得含空 group (Mintlify 侧栏空 entry)
  5. README/overview sync — 每个 sub-folder README.md == overview.md (Mintlify drops README)
  6. UNVERIFIED 纪律     — UNVERIFIED 标记不能是孤行/纯标题 (必须同行带具体 claim)
  7. Stale TODO          — 顶层文件里的 (TBD)/(待补)/(待写) 列出 (INFO，不 fail)
  8. Dissection anchors  — dissection 应含 X-Ray / TL;DR / Napkin / Timeline 锚点 (WARN，不 fail)

Exit code: 0 = 全 PASS（WARN/INFO 不阻塞）, 1 = 至少一项 FAIL。
只用 stdlib。Python 3.9+。
"""
from __future__ import annotations

import json
import re
import sys
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import Iterable

CheckResult = namedtuple("CheckResult", ["name", "status", "summary", "details"])
# status ∈ {"PASS", "WARN", "FAIL", "INFO"}

# ----------------------------------------------------------------------
# 常量 / 规则配置
# ----------------------------------------------------------------------

# 与 gen_mintlify_nav.py 保持一致：README + AUDIT_* 不进 nav。
def is_excluded_stem(stem: str) -> bool:
    return stem == "README" or stem.startswith("AUDIT_")


# 5-axis ontology v2 (physics-gen)。每篇 dissection 必须声明 5 轴坐标。
ONTOLOGY_AXES = ["output", "injection", "control", "temporal", "domain"]
ONTOLOGY_HEADER_MARKER = "<!-- ontology-5axis"
ONTOLOGY_HEADER_END = "-->"
# dissection 来源目录（含 ontology header 的内容档所在处）
DISSECTION_DIRS = ["foundations", "use-cases"]

# 相对链接 + fenced code stripper
RELATIVE_LINK_RE = re.compile(r"\]\((\.{1,2}/[^)\s#]+)(#[^)]*)?\)")
FENCE_RE = re.compile(r"^```")

# UNVERIFIED 纪律：孤行 / 纯标题 / 引言行 (无 claim) 形式
UNVERIFIED_SUSPICIOUS_RE = re.compile(r"^\s*(?:[#>*\-]+\s*)*`?UNVERIFIED`?\s*$")

# Stale TODO
STALE_TODO_RE = re.compile(r"\((?:TBD|待補|待补|待写|待寫)\)")

# Dissection 模板锚点 (AGENTS.md v2 template) — WARN-only completeness signal
DISSECTION_ANCHORS = {
    "X-Ray": [r"X[- ]?[Rr]ay"],
    "TL;DR": [r"TL;?DR"],
    "Napkin": [r"Napkin", r"📌"],
    "Timeline": [r"研究全景", r"[Tt]imeline", r"📍"],
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def iter_markdown_files(repo_root: Path) -> Iterable[Path]:
    """遍历仓库所有 .md，排除 .git / node_modules / 隐藏目录。"""
    for md in repo_root.rglob("*.md"):
        parts = md.relative_to(repo_root).parts
        if any(p.startswith(".") and p != "." for p in parts):
            continue
        if "node_modules" in parts:
            continue
        yield md


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def find_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def strip_fenced_code_blocks(text: str) -> str:
    """剥掉 ``` 包裹的 fenced code block，避免把模板里的示例链接当真链接。"""
    out: list[str] = []
    inside = False
    for line in text.splitlines():
        if FENCE_RE.match(line.lstrip()):
            inside = not inside
            out.append("")
            continue
        out.append("" if inside else line)
    return "\n".join(out)


def collect_dissections(repo_root: Path) -> list[Path]:
    """dissection = DISSECTION_DIRS 下非 overview/README 的内容 .md。"""
    out: list[Path] = []
    for d in DISSECTION_DIRS:
        base = repo_root / d
        if not base.exists():
            continue
        for md in base.rglob("*.md"):
            if ".git" in md.parts:
                continue
            if md.name in ("overview.md", "README.md"):
                continue
            out.append(md)
    return sorted(out)


# ----------------------------------------------------------------------
# Check 1 — Broken Links
# ----------------------------------------------------------------------


def check_1_broken_links(repo_root: Path) -> CheckResult:
    broken: list[str] = []
    total = 0
    for md in iter_markdown_files(repo_root):
        text = strip_fenced_code_blocks(read_text(md))
        for m in RELATIVE_LINK_RE.finditer(text):
            total += 1
            rel_path = m.group(1)
            try:
                target = (md.parent / rel_path).resolve()
                target.relative_to(repo_root.resolve())
            except (ValueError, OSError):
                broken.append(f"{md.relative_to(repo_root)} → {rel_path}（解析失败）")
                continue
            if not target.exists():
                broken.append(f"{md.relative_to(repo_root)} → {rel_path}")
    ok = total - len(broken)
    if broken:
        details = [f"  断链：{b}" for b in broken[:50]]
        if len(broken) > 50:
            details.append(f"  …还有 {len(broken) - 50} 条未列出")
        return CheckResult("Broken Links", "FAIL",
                           f"{ok}/{total} relative links resolve，{len(broken)} 条断链", details)
    return CheckResult("Broken Links", "PASS", f"{total}/{total} relative links resolve", [])


# ----------------------------------------------------------------------
# Check 2 — Ontology 5-axis header
# ----------------------------------------------------------------------


def check_2_ontology_5axis(repo_root: Path) -> CheckResult:
    dissections = collect_dissections(repo_root)
    missing_header: list[str] = []
    missing_axes: list[str] = []
    for d in dissections:
        text = read_text(d)
        head = text[:600]
        if ONTOLOGY_HEADER_MARKER not in head:
            missing_header.append(str(d.relative_to(repo_root)))
            continue
        start = text.find(ONTOLOGY_HEADER_MARKER)
        end_idx = text.find(ONTOLOGY_HEADER_END, start)
        if end_idx < 0:
            missing_header.append(str(d.relative_to(repo_root)) + " (未闭合 -->)")
            continue
        block = text[start:end_idx + len(ONTOLOGY_HEADER_END)]
        # axes use `key=value` form (multi-value via |, N/A allowed)
        missing = [ax for ax in ONTOLOGY_AXES if f"{ax}=" not in block]
        if missing:
            missing_axes.append(f"  {d.relative_to(repo_root)} 缺轴：{', '.join(missing)}")

    details: list[str] = []
    if missing_header:
        details.append(f"  {len(missing_header)} dissection(s) 缺 ontology-5axis header:")
        details += [f"    - {p}" for p in missing_header[:10]]
        if len(missing_header) > 10:
            details.append(f"    ... and {len(missing_header) - 10} more")
    details += missing_axes[:10]

    if missing_header or missing_axes:
        return CheckResult("Ontology 5-axis", "FAIL",
                           f"{len(missing_header)} 缺 header + {len(missing_axes)} 缺轴 (共 {len(dissections)} dissection)",
                           details)
    return CheckResult("Ontology 5-axis", "PASS",
                       f"all {len(dissections)} dissections 带完整 5-axis header "
                       f"({'/'.join(ONTOLOGY_AXES)})", [])


# ----------------------------------------------------------------------
# Check 3 — Mintlify Nav coverage
# ----------------------------------------------------------------------


def check_3_mintlify_nav(repo_root: Path) -> CheckResult:
    docs_json = repo_root / "docs.json"
    if not docs_json.exists():
        return CheckResult("Mintlify Nav", "INFO",
                           "docs.json 不存在；跳过（如未设置 Mintlify 部署可忽略）", [])
    try:
        cfg = json.loads(docs_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return CheckResult("Mintlify Nav", "FAIL", f"docs.json 解析失败：{e}", [])

    nav_pages: set[str] = set()

    def walk(node):
        if isinstance(node, str):
            nav_pages.add(node)
        elif isinstance(node, dict):
            for k in ("pages", "groups", "tabs"):
                if k in node:
                    for child in node[k]:
                        walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(cfg.get("navigation", {}))

    # Dated report archives (reports/<sub>/*.md) are archive content, not sidebar pages.
    def is_report_archive(rel: Path) -> bool:
        return rel.parts[:1] == ("reports",) and len(rel.parts) >= 3

    repo_md: set[str] = set()    # must appear in nav
    nav_known: set[str] = set()  # legitimately maps to a real file
    for md in repo_root.rglob("*.md"):
        if ".git" in md.parts:
            continue
        if is_excluded_stem(md.stem):
            continue
        rel = md.relative_to(repo_root)
        pid = str(rel)
        nav_known.add(pid)
        if not is_report_archive(rel):
            repo_md.add(pid)
    # .mdx served extensionless
    for mdx in repo_root.rglob("*.mdx"):
        if ".git" in mdx.parts:
            continue
        nav_known.add(str(mdx.relative_to(repo_root).with_suffix("")))

    missing_in_nav = sorted(repo_md - nav_pages)
    missing_files = sorted(nav_pages - nav_known)

    details: list[str] = []
    if missing_in_nav:
        details.append(f"  {len(missing_in_nav)} md not in docs.json nav:")
        details += [f"    - {p}" for p in missing_in_nav[:10]]
        if len(missing_in_nav) > 10:
            details.append(f"    ... and {len(missing_in_nav) - 10} more")
    if missing_files:
        details.append(f"  {len(missing_files)} nav entries with no matching file:")
        details += [f"    - {p}" for p in missing_files[:10]]
        if len(missing_files) > 10:
            details.append(f"    ... and {len(missing_files) - 10} more")

    if missing_in_nav or missing_files:
        return CheckResult("Mintlify Nav", "FAIL",
                           f"{len(missing_in_nav)} orphan md + {len(missing_files)} dangling nav "
                           f"entries — run `python3 scripts/gen_mintlify_nav.py > docs.json`", details)
    return CheckResult("Mintlify Nav", "PASS",
                       f"all {len(repo_md)} md present in docs.json nav (no orphans, no dangling)", [])


# ----------------------------------------------------------------------
# Check 4 — No empty nav groups
# ----------------------------------------------------------------------


def check_4_no_empty_groups(repo_root: Path) -> CheckResult:
    docs_json = repo_root / "docs.json"
    if not docs_json.exists():
        return CheckResult("No empty groups", "INFO", "docs.json 不存在；跳过", [])
    try:
        cfg = json.loads(docs_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return CheckResult("No empty groups", "FAIL", f"docs.json 解析失败：{e}", [])

    empties: list[str] = []

    def walk(node, path=""):
        if isinstance(node, dict):
            if "pages" in node and not node["pages"]:
                empties.append(f"  empty group: {path or node.get('group', '?')}")
            for k in ("pages", "groups", "tabs"):
                for child in node.get(k, []):
                    label = child.get("group") or child.get("tab") if isinstance(child, dict) else None
                    walk(child, f"{path}/{label}" if label else path)
        elif isinstance(node, list):
            for child in node:
                walk(child, path)

    walk(cfg.get("navigation", {}))

    if empties:
        return CheckResult("No empty groups", "FAIL",
                           f"docs.json 含 {len(empties)} 个空 group",
                           empties[:10] + [
                               "  修复: regen python3 scripts/gen_mintlify_nav.py > docs.json"])
    return CheckResult("No empty groups", "PASS", "docs.json 所有 group 都非空", [])


# ----------------------------------------------------------------------
# Check 5 — README/overview sync
# ----------------------------------------------------------------------


def check_5_readme_overview_sync(repo_root: Path) -> CheckResult:
    mismatched: list[str] = []
    total = 0
    for ov in repo_root.rglob("overview.md"):
        if ".git" in ov.parts:
            continue
        total += 1
        readme = ov.parent / "README.md"
        if not readme.exists():
            mismatched.append(f"  {ov.parent.relative_to(repo_root)}/README.md 缺失 (overview 有，README 无)")
            continue
        if read_text(readme) != read_text(ov):
            mismatched.append(f"  {ov.parent.relative_to(repo_root)}/ README.md != overview.md")
    if mismatched:
        return CheckResult("README/overview Sync", "FAIL",
                           f"{len(mismatched)}/{total} sub-folder README 与 overview 不同步 — "
                           f"run `python3 scripts/sync_readme_from_overview.py`", mismatched[:15])
    return CheckResult("README/overview Sync", "PASS",
                       f"all {total} sub-folders README.md == overview.md", [])


# ----------------------------------------------------------------------
# Check 6 — UNVERIFIED discipline
# ----------------------------------------------------------------------


def check_6_unverified_discipline(repo_root: Path) -> CheckResult:
    offenders: list[str] = []
    for md in iter_markdown_files(repo_root):
        for i, line in enumerate(read_text(md).splitlines(), start=1):
            if "UNVERIFIED" in line and UNVERIFIED_SUSPICIOUS_RE.match(line):
                offenders.append(f"  {md.relative_to(repo_root)}:{i}  «{line.strip()[:50]}»")
    if offenders:
        return CheckResult("UNVERIFIED Discipline", "FAIL",
                           f"{len(offenders)} 处 UNVERIFIED 是孤行/纯标题 (必须同行带具体 claim)",
                           offenders[:15])
    return CheckResult("UNVERIFIED Discipline", "PASS", "所有 UNVERIFIED 都附具体 claim", [])


# ----------------------------------------------------------------------
# Check 7 — Stale TODO (INFO)
# ----------------------------------------------------------------------


def check_7_stale_todo(repo_root: Path) -> CheckResult:
    hits: list[str] = []
    for name in ("README.md", "ONBOARDING.md", "AGENTS.md", "MAINTAINER.md", "CONTRIBUTING.md"):
        p = repo_root / name
        if not p.exists():
            continue
        for i, line in enumerate(read_text(p).splitlines(), start=1):
            if STALE_TODO_RE.search(line):
                hits.append(f"  {name}:{i}  «{line.strip()[:60]}»")
    return CheckResult("Stale TODO", "INFO",
                       f"{len(hits)} 处 (TBD)/(待补)/(待写) — roadmap 正常" if hits
                       else "0 处 (TBD)/(待补)/(待写) — roadmap 正常", hits[:15])


# ----------------------------------------------------------------------
# Check 8 — Dissection template anchors (WARN)
# ----------------------------------------------------------------------


def check_8_dissection_anchors(repo_root: Path) -> CheckResult:
    incomplete: list[str] = []
    dissections = collect_dissections(repo_root)
    for d in dissections:
        text = read_text(d)
        missing = [name for name, pats in DISSECTION_ANCHORS.items() if not find_any(pats, text)]
        if missing:
            incomplete.append(f"  {d.relative_to(repo_root)} 缺锚点：{', '.join(missing)}")
    if incomplete:
        return CheckResult("Dissection Anchors", "WARN",
                           f"{len(incomplete)}/{len(dissections)} dissection 缺核心锚点 "
                           f"(X-Ray/TL;DR/Napkin/Timeline) — 升 v1 时补", incomplete[:15])
    return CheckResult("Dissection Anchors", "PASS",
                       f"all {len(dissections)} dissections 带核心锚点", [])


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


CHECKS = [
    check_1_broken_links,
    check_2_ontology_5axis,
    check_3_mintlify_nav,
    check_4_no_empty_groups,
    check_5_readme_overview_sync,
    check_6_unverified_discipline,
    check_7_stale_todo,
    check_8_dissection_anchors,
]


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"=== Physics-Gen Handbook Audit ({today}) ===\n")

    results: list[CheckResult] = []
    for i, fn in enumerate(CHECKS, start=1):
        try:
            res = fn(repo_root)
        except Exception as e:
            res = CheckResult(fn.__name__, "FAIL", f"check raised exception: {e!r}", [])
        results.append(res)
        print(f"[CHECK {i} / {res.name}] {res.status}  {res.summary}")
        for line in res.details:
            print(line)
        print()

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "INFO": 0}
    for r in results:
        counts[r.status] += 1

    print(f"Total: {counts['PASS']} PASS / {counts['WARN']} WARN / "
          f"{counts['FAIL']} FAIL / {counts['INFO']} INFO")
    exit_code = 1 if counts["FAIL"] > 0 else 0
    print(f"Exit code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
