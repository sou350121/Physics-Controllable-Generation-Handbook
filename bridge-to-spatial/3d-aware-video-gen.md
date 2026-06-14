# Bridge: 3D-Aware Video Generation — Generative 3DGS vs Reconstructive 3DGS

> **本倉 (Physics-Gen) = generation 端** · **Spatial-Handbook = perception 端**
> **共同表徵**：3D Gaussian Splatting (3DGS) · **相反方向**：synthesize（從文字/單圖外推）vs reconstruct（從多張真實照片擬合）

**Status:** v1 — opinionated draft. Grounded in [`foundations/3d-aware-generation/generative-gaussian-splatting.md`](../foundations/3d-aware-generation/generative-gaussian-splatting.md) (GGS anchor) and Spatial's reconstruction line. Cross-repo benchmark deltas marked `UNVERIFIED` where not directly grounded.

**TL;DR:** 兩冊收的是**同一個表徵、相反的資訊流向**。Spatial 的 3DGS / VGGT 線是 *reconstructive*：N 張真實照片（或 sensor）→ 擬合出一個忠於現實的 splat field，資訊只能**保存**不能**捏造**。Physics-Gen 的 generative 3DGS（[GGS](../foundations/3d-aware-generation/generative-gaussian-splatting.md)、World Labs Marble）是 *synthesizing*：文字 / 單圖 → 外推出一個 splat field，**遮擋面必須 hallucinate**。同表徵讓兩端可以無縫對接（一個 `.splat` 既可被 reconstruct 出來、也可被 generate 出來，downstream renderer 不在乎來源）；但 **metric scale 與 coordinate frame 的保證等級天差地遠**——這就是 contract 的核心，也是最容易爆的 seam。

---

## 1 · 為什麼是「同表徵、相反方向」

3DGS（Kerbl et al. 2023）是一個顯式 3D 表徵：場景 = 一堆帶 `(μ, Σ, α, SH)` 的 Gaussian primitive，可被可微分 rasterizer 渲染到任意視角。它本身**不規定資訊從哪來**。兩冊各取一條方向：

- **Spatial（reconstructive）**：輸入是**已存在的觀測**（多視角照片、RGB-D、video stream）。3DGS-original / Mip-Splatting / GS-SLAM 是 per-scene optimization；VGGT / MapAnything / VGGT-Ω 是 feed-forward foundation model——單次 transformer 前向出 3D tensor。資訊量 = 觀測量；模型不該發明沒看到的東西。**忠實度是賣點**。
- **Physics-Gen（generative）**：輸入是**不完整 spec**（文字、單張圖、camera trajectory）。GGS 把 3DGS 當 video-diffusion 的中間 bottleneck（LDM U-Net → epipolar transformer → splat decoder → render）；資訊量 < 場景量，差額靠 video-diffusion prior **補完 / 編造**。**多視角一致性是賣點**（同一面牆兩個 view 是同一批 splat，結構性保證不漂）。

一句話：**reconstruction 把「我看到的」固化成 3D；generation 把「我想要的」外推成 3D。** 表徵相同，所以可以接；來源相反，所以保證不同。

---

## 2 · 兩端契約（interface table）

| 契約欄位 | Spatial 端（reconstructive 3DGS / VGGT）提供 | Physics-Gen 端（generative 3DGS）提供 | seam 在哪 |
|---|---|---|---|
| **表徵 schema** | `(μ, Σ, α, SH)` per Gaussian，或 feature-3DGS | 同 schema（GGS 用 feature-3DGS，splat 數遠少於百萬量級 vanilla） | ✅ 對齊：downstream renderer 不在乎來源 |
| **Coordinate frame** | SLAM / VGGT 可給 world frame（含 pose graph）；單前向 model 多給 camera-relative | GGS 是 **pose-conditional**（Plücker camera embedding 必填），輸出在指定 camera 軌跡的相對 frame | ⚠ generated 場景缺 absolute world anchor；要對齊真實場景需額外 registration |
| **Metric scale** | VGGT 線 **un-metric**；MapAnything 是少數**原生 metric**（factored repr `D×R×T×s`）`UNVERIFIED` | GGS 訓練在 RealEstate10K / ScanNet++，**scale 不保證 metric**（YouTube 房地產無真實尺度） | 🔴 **最大 seam**：generated 場景幾乎都是 scale-ambiguous，不能直接量距離 |
| **遮擋面 / 未觀測區** | 留空 / 標 uncertainty（沒看到就沒有） | **hallucinate**（diffusion prior 填補），可能與物理不符 | ⚠ 兩端對「未知區」哲學相反：留白 vs 編造 |
| **動態 / 物理** | 多為 static（4DGS 是 dynamic 子分支）；無 force / contact | GGS 同病：`injection=data-only`，static rigid，無 dynamic / 無守恆 | 🔴 兩端都不碰物理；要 dynamic 得各自外接 |
| **可審計性** | 可回溯到輸入照片（有 ground truth） | 無 ground truth（生成的，無「正確答案」可比對） | ⚠ generated 場景的「正確性」無法驗證，只能驗一致性 |

---

## 3 · 核心問題：一個 generated 場景何時 metrically usable downstream？

這是 contract 的 payoff 問題。Downstream（VLA training scene、driving sim、機器人 navigation）通常需要**真實尺度 + 穩定 world frame**。Reconstructive 3DGS 從真實觀測來，scale 與 frame 至少**可恢復**；generative 3DGS 預設**兩者都缺**。判斷一個 generated splat field 能不能往下游餵：

1. **它是否被 metric anchor 約束過？** GGS 純從 text / 單圖外推 → scale-free，**不可直接量距離**。若 generation 是 image-init 且那張圖有已知 intrinsics / 已知物件尺寸，可後驗 rescale；否則只能當「視覺 plausible 但幾何不可信」用。
2. **它的 coordinate frame 有沒有 registration 路徑？** Pose-conditional generation 給的是相對 camera 軌跡的 frame。要塞進一個既有的 world（例如真實機器人工作空間），必須跑一次 registration（ICP / feature matching）把 generated splat 對齊到 reconstructed anchor。
3. **未觀測面 hallucination 會不會誤導 downstream？** 機器人會去「桌子背面」抓東西，但桌背是 diffusion 編的——這裡 generated 場景的 hallucinated 幾何**可能直接導致 policy 學歪**。

**判準（honest version）**：generated 3DGS 對「視覺增廣 / domain randomization / 新視角 rendering」**夠用**；對「需要真實尺度與可信幾何的 closed-loop 控制 / metric navigation」**不夠用，除非先 register 到一個 reconstructed metric anchor**。這正是兩冊互補的點：**Spatial 出 metric anchor scene，Physics-Gen 出 corner-case multiplication，兩者對齊後一起餵 sim**（見 GGS anchor §6 cross-line synthesis）。

---

## 4 · 互補 pipeline（兩端怎麼接）

```
Spatial (reconstruct)              Physics-Gen (generate)
──────────────────────             ──────────────────────
真實機器人攝影機                    文字 / 單圖 / camera traj
   │ VGGT / 3DGS-SLAM                  │ GGS / Marble
   ▼                                   ▼
metric anchor scene  ◀──register──  scale-free corner-case scene
   │  (真實尺度 + world frame)         │  (視覺多樣但幾何待校)
   └──────────────┬────────────────────┘
                  ▼
        Isaac Sim / Genesis  →  scene multiplication  →  VLA training
```

- **Spatial → Physics-Gen**：reconstructed anchor scene 當 generation 的 image-init / 幾何約束，把 generative 的 scale 釘到真實。
- **Physics-Gen → Spatial**：generated corner-case（罕見光照、罕見佈局）擴充 reconstructed 場景庫，補 long-tail——這是 reconstruction 拿不到的（你不能去「重建」一個不存在的場景）。

---

## 5 · 開放 seam（未解）

- **🔴 Metric scale handshake 沒有標準**：目前沒有一個約定俗成的 metadata flag，讓 generation 端宣告「我這個 splat field 是 / 不是 metric」、reconstruction 端宣告「我這個 anchor 的 scale source 是什麼」。Downstream 只能憑經驗假設。**這是兩冊最該共同定義的 schema 欄位**。
- **⚠ Hallucinated 幾何的 uncertainty 標註**：generated 場景的遮擋面沒有 confidence map。Reconstructive 線（VGGT-Ω 等）開始輸出 per-pixel uncertainty；generative 線基本沒有。理想 contract：generated splat 每個 primitive 帶「observed vs hallucinated」標記。
- **⚠ Dynamic 兩端都缺**：GGS / Marble static，Spatial 主線也 static（4DGS 是子分支）。要 dynamic 場景兩端都得外接（pixel-WM 補 agent、diff-sim 補 contact）——見 [`bridge-to-spatial/nerf-3dgs-meet-world-model.md`](./nerf-3dgs-meet-world-model.md)。
- **可審計性不對稱**：reconstruction 可對照 ground truth，generation 不能。長期看，「generated 場景的物理合理性如何驗證」是 open research，不是工程坑。

---

## Boundary

- Generative 3DGS 的單篇 method 解構（GGS / World Labs Marble）→ [`foundations/3d-aware-generation/`](../foundations/3d-aware-generation/)
- 顯式 3D 表徵當 world model（GS-as-WM）→ [`bridge-to-spatial/nerf-3dgs-meet-world-model.md`](./nerf-3dgs-meet-world-model.md)
- Reconstructive 3DGS / feed-forward 3D 的完整解構 → Spatial-Handbook [`foundations/3dgs-family/`](https://github.com/sou350121/Spatial-Intelligence-Handbook/tree/main/foundations/3dgs-family) · [`foundations/feed-forward-3d/`](https://github.com/sou350121/Spatial-Intelligence-Handbook/tree/main/foundations/feed-forward-3d)（VGGT 線）
- Spatial 端 world-model 視角 → Spatial-Handbook [`foundations/world-model/`](https://github.com/sou350121/Spatial-Intelligence-Handbook/tree/main/foundations/world-model)

## References

- Generative Gaussian Splatting — Schwarz, Müller, Kontschieder. **arXiv 2503.13272**, ICCV 2025. https://katjaschwarz.github.io/ggs/
- 3D Gaussian Splatting (root anchor，兩冊共用) — Kerbl et al. **SIGGRAPH 2023**, ACM TOG 42(4).
- VGGT (feed-forward 3D，reconstruction 線 anchor) — **CVPR 2025** `UNVERIFIED canonical link`. 詳見 Spatial-Handbook dissection.
- MapAnything (原生 metric feed-forward) — Spatial-Handbook [`foundations/feed-forward-3d/mapanything_dissection.md`](https://github.com/sou350121/Spatial-Intelligence-Handbook/tree/main/foundations/feed-forward-3d/mapanything_dissection.md) `UNVERIFIED scale claim`.
