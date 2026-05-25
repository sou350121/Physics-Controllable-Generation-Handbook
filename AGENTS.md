# AGENTS.md — 給 AI agents (Pulsar / Claude / Codex) 的編輯指南

姊妹倉：[VLA-Handbook](https://github.com/sou350121/VLA-Handbook) · [Spatial-Intelligence-Handbook](https://github.com/sou350121/Spatial-Intelligence-Handbook)。三倉共用同一套寫作協定。

## 寫作對象 (Reader Persona)

不是學生、不是論文評審。是一位 **已經實作過 VLA / world-model / diff-sim 一條鏈** 的工程師或研究員，他想知道：

- 這個方法跟我已知的另一個方法在五軸上差在哪？
- 哪些 paper claim 在真實系統會崩？什麼工況崩？
- 我要怎麼 compose 它跟另一條路線（pixel-WM × diff-sim、neural surrogate × VLA）？

寫作不需要解釋 diffusion / NeRF / contact dynamics 是什麼 — 直接進入「對比 / 取捨 / 失效模式」。

## Dissection 寫作模板 (8 段)

每篇 dissection（`foundations/<zone>/<paper-or-method>.md`）：

```
<!-- ontology-5axis output=... injection=... control=... temporal=... domain=... -->

# <Method Name>

## 1. One-paragraph TL;DR
為什麼這個方法存在，解決哪個 prior gap。

## 2. Core mechanism
數學/架構/loss 的核心 — 不超過半頁，配 1 個 ASCII/markdown 圖。

## 3. 五軸定位 + 同軸對手
參照 ontology v1 標出五軸值；列出同軸 2-3 個競爭方法。

## 4. Where it shines / where it breaks
- ⚡ 真正領先的 regime
- ❌ Known failure modes（最好引 GitHub issue / 作者 talk / 二手實測）

## 5. Reproduction notes
最小可跑 setup、GPU 預算、典型踩坑。

## 6. Cross-line synthesis
與其他 4 條技術路線（pixel-WM / latent-WM / diff-sim / surrogate）怎麼接。

## 7. References
canonical paper + 3-5 二手實測。

## 8. §8.x Pitfall log
GitHub-validated 已知問題（issue #、原文摘錄、severity、workaround）。
```

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
