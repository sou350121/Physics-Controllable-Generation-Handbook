# AGENTS.md — 給 AI agents (Pulsar / Claude / Codex) 的編輯指南

姊妹倉：[VLA-Handbook](https://github.com/sou350121/VLA-Handbook) · [Spatial-Intelligence-Handbook](https://github.com/sou350121/Spatial-Intelligence-Handbook)。三倉共用同一套寫作協定。

## 寫作對象 (Reader Persona)

不是學生、不是論文評審。是一位 **已經實作過 VLA / world-model / diff-sim 一條鏈** 的工程師或研究員，他想知道：

- 這個方法跟我已知的另一個方法在五軸上差在哪？
- 哪些 paper claim 在真實系統會崩？什麼工況崩？
- 我要怎麼 compose 它跟另一條路線（pixel-WM × diff-sim、neural surrogate × VLA）？

寫作不需要解釋 diffusion / NeRF / contact dynamics 是什麼 — 直接進入「對比 / 取捨 / 失效模式」。

## Dissection 寫作模板 — v2（spatial-style，2026-05-26 升級）

對齊 sister Spatial-Intelligence-Handbook 風格。每篇 dissection（`foundations/<zone>/<paper-or-method>.md`）：

```
<!-- ontology-5axis output=... injection=... control=... temporal=... domain=... -->

# <Method Name> 解構（<English>）

> **發布時間**：YYYY-MM · arXiv [NNNN.NNNNN](url)
> **論文**：*Full Title*
> **作者**：A, B, C, ... (affiliation 簡記)
> **核心定位**：1-2 句說這篇方法在 v2 ontology 上落在哪、解了什麼 prior gap。

**Status:** v0.5 — 解構基於 paper 摘要 + GitHub issues + 二手分析。完整 benchmark 數字 / param 細節 待維護者升 v1。
**TL;DR:** 用 ≤4 句寫完整故事：① 它做什麼新事 ② 它的核心 trick ③ 為什麼這對我們的 v2 axis ★ ④ 一個關鍵實證數字（提升 % / 違反 metric / GPU 預算）。

**X-Ray.** 一段分析性論斷（150-300 字）。**不是 summary** ——
- 把這篇放回 v2 ontology / 5 axis Pareto 上
- 標出它解了哪些**舊的工程坑**
- 預測它打不開的 envelope（什麼會超出論文範圍）
- 對 Physics-Gen handbook 讀者的意義（為什麼這該寫 anchor）

## 📍 研究全景時間線

```ascii
   YYYY        YYYY              YYYY              YYYY-MM            YYYY?
   prior ────► prior2 ──────► closest ────────► YOU ARE HERE ──► future
   ... position the method in lineage ...
```

★ = 主要新點。**仍未解：xxx**（留給下一代）。

---

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作（or vs 同軸對手）

| 維度 | 前作 | 本方法 |
|---|---|---|
| ... | ... | ... |

### 1.2 ⚡ Eureka Moment

> **核心 trick 一句話** —— 用一句話講明白他們做對了什麼，配 1-2 句直覺。

### 1.3 信息流（架構圖）

```ascii
side-by-side ASCII diagram showing prior vs this method, OR layered architecture
```

---

## §2 · 數學層

### 📌 Napkin Formula

```
   核心公式（1-3 行最重要的 equation）

   Cost: O(...)  vs prior: O(...)
```

**直覺**：用 2-3 句講為什麼這個 formula 是 trick 所在。

### 2.x Loss / 訓練細節

如果有 multi-task loss / aux loss / guidance gradient 等，在這列出。

---

## §3 · 數據層 / 訓練 scale

訓練資料規模 + 怎麼來的 + scale 增長 vs 同類方法。

---

## §4 · 代碼層

| 項 | 狀態 |
|---|---|
| Repo | github.com/... |
| Checkpoint | HF link / 大小 |
| License | OpenRAIL++ / Apache-2.0 / community-only |
| Inference GPU | ... |
| Streaming | ✅ / ❌ |
| Metric scale | ✅ / ❌ |

---

## §5 · 評測 / Benchmark

| Benchmark | Metric | 前 SOTA | 本方法 | Δ |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

解讀 1-2 段 — **哪部分 Δ 是真的 capability，哪部分是 data leakage / benchmark Goodhart**。

---

## §6 · Issues & Limitations

### 6.1 論文自述 limitations
列 paper 自己標的 3-5 條。

### 6.2 Hidden Assumptions
我們從架構推斷的 4-6 條隱含假設（讀者該知道的）。

### 6.x GitHub-validated 失敗模式（atlas 聯動）

| 失敗 / 問題 | GitHub evidence | 嚴重度 |
|---|---|---|
| ... | [issue #N](url): 摘錄 | 🔴 / 🟠 / 🟡 |

**Maintainer 響應度**：N open / M closed (YYYY-MM-DD)。

---

## §7 · 比較 & 面試 Tip

| 同軸對手 | Axis 2 (injection) | Streaming | Metric? | Open? | Status |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... |

> **🎤 Interview Tip.** 「<典型問題>」正確答：「...」錯答：「...」（為什麼錯）

### 7.1 Falsifiable predictions

1. ✅ **VERIFIED <日期>**：... (附鏈)
2. **YYYY-MM 前**：...
3. **YYYY-MM 前不會發生**：...

---

## §8 · For the Reader（按 persona 分流）

- **VLA / robot policy 工程師** —— ...
- **自駕 closed-loop 工程師** —— ...
- **影片生成工程師** —— ...
- **神經 PDE / surrogate 研究者** —— ...
- **物理 conditioning 研究者** —— ...
- **Research 學生** —— ...

不是每篇都要全部 6 persona，但對應該方法切實的至少 3 條。

---

## References

- **<Method>** — 作者 et al. *Venue Year* · [arXiv:NNNN.NNNNN](url) · [code](url)
- **前作 / lineage** — ...
- **第三方分析** — blog / talk / 二手 reproduction ...

---

## Boundary

- 完整 X 解構（YYY 細節）→ [`./other.md`](./other.md)
- 跨 zone wedge → [`crossing/...md`](../../crossing/.../overview.md)
- 與 sister handbook 接口 → [`bridge-to-*/...md`](../../bridge-to-vla/...md)
- 與 5 axis 全景 → [`cheat-sheet/ontology.md`](../../cheat-sheet/ontology.md)

---

## ✍️ 維護者註（v0.5 → v1 升級清單）

本 v0.5 基於 X / Y / Z。下次升 v1 時補：

1. ⏳ <未驗證項目 1>
2. ⏳ <未驗證項目 2>
...

---

[← Back to <zone>](./overview.md)

Sources:
- [<canonical paper>](url)
- [<github>](url)
- [<secondary>](url)
```

### 不寫的東西（拒絕 anti-pattern）

- 不寫「§1. TL;DR」這種雙重編號（用 `# Method 解構` + 開頭 X-Ray 段；正文用 `## §1 · 架構`）
- 不寫 paper abstract 摘要 —— 我們要 **comparison + failure + composition**
- 不寫"我覺得很 cool" —— 對比落到 v2 axis / failure mode / cost
- TBD / UNVERIFIED 是 **設計特性不是恥辱** —— 留給維護者升 v1，比假裝確定強

## 不寫的東西

- 不寫「我覺得很 cool」之類主觀讚美。對比一定要落到五軸 / failure mode / cost。
- 不寫沒實作過的範式（純 hand-wavy abstract paper）— 留給 reports/ daily 抓進來再決定。
- 不複製 paper abstract — handbook 的價值在 **comparison + failure + composition**，不在 summary。

## 與三個 sister handbook 的分工

| Handbook | 視角 |
|---|---|
| VLA-Handbook | Action 一端 — 給定觀察輸出動作 |
| Spatial-Intelligence-Handbook | Perception 一端 — 從感測還原 3D / pose |
| **本倉** | Generation 一端 — 從文字/動作生成可控物理觀察 |

跨倉引用走 `bridge-to-*/` 目錄。

## Pulsar pipeline 接口

`reports/physics-gen-daily/` 是 Pulsar Phase 1 自動產出區。
- daily：arxiv cs.LG/cs.CV/cs.GR/cs.RO + arxiv physics.flu-dyn / cond-mat.soft → keyword filter → qwen3.5-plus 評 ⚡/🔧/📖/❌
- 不接 TG，git push only（Mintlify rebuild 7s）
- 詳見 [`docs/pulsar-integration.md`](docs/pulsar-integration.md)
