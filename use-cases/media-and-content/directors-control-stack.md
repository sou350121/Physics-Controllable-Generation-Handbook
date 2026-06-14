<!-- ontology-5axis output=pixel-video injection=data-only control=image-init|camera|trajectory temporal=clip-parallel domain=generalist -->

# 導演的控制堆疊 —— 以及為什麼停在 previz 不是 final frame 解構

> **解構對象**：不是某一個模型，而是 2024–2026 累積出來的「**導演控制堆疊**」（reference / first-last-frame / camera / in-context-edit / structure）—— 加上它在商業 production 裡的**真實落點**（廣告 production-real vs 高端電影 previz-only）。
> **錨定來源**：可控生成 survey（arXiv [2507.16869](https://arxiv.org/html/2507.16869v3)，v3）+ CameraCtrl（arXiv [2404.02101](https://arxiv.org/abs/2404.02101)）+ Runway Aleph（[research post](https://runwayml.com/research/introducing-runway-aleph)，2025-07-25）+ Coca-Cola 2025 假日廣告（[Marketing Dive](https://www.marketingdive.com/news/what-coca-cola-learned-generative-ai/741709/)）+ ILM「Star Wars: Field Guide」（[Decrypt](https://decrypt.co/320224/ilm-makes-star-wars-field-guide-short-film-using-generative-ai)，2025）+ match moving（[Wikipedia](https://en.wikipedia.org/wiki/Match_moving)）。版本宣稱已標日期；快速演進領域，**讀者請以發布日為準**。
> **為什麼進名單**：本倉前面拆的是**引擎**（Sora / AnimateDiff），但 creative-tech / VFX practitioner 每天面對的不是引擎、是**控制面**——「我能把哪幾根桿子交給生成式？哪幾根還得用傳統 pipeline？」這篇把那組桿子列成一張表，並回答 VFX 圈最痛的一句：**為什麼 gen-video 卡在 previz 進不了 final frame**。答案不在「畫質不夠」，在 axis-3 控制的**類型**——導演要的是**學習式/相對**控制（已成熟），VFX 整合要的是 **metric 控制**（還沒有）。

---

## §1 · TL;DR（學習式/相對控制已成熟；VFX 要的 metric 控制還沒有）

**一句話**：2025 的 gen-video 把「**導演級的相對控制**」基本解決了——你能指定角色（reference）、指定首尾幀（first-last-frame）、推鏡頭（camera）、對已有鏡頭做物件級增刪（in-context edit）、餵結構圖（depth/pose/sketch）。但這些控制全部是 **learned / relative**（從資料統計學來的、相對於生成內容自身的座標系）。

**而 VFX final-frame 整合要的是另一種控制**：把 CG 以**正確的 position / scale / orientation / motion 相對於真實拍攝 plate** 插進去——這需要逐幀三角化出**校準過的 metric 相機解 + 點雲**（[match moving](https://en.wikipedia.org/wiki/Match_moving)）。生成式相機控制（CameraCtrl 等）給的是學習式軌跡操控，**不是對齊真實 plate 的 metric 解** → 所以生成的像素**不能直接拿去和真實素材合成交付**。

> ★**核心契約**：媒體裡**外觀是產品、物理只需可信不需正確**——這正是本倉 `injection=data-only` 在 media vertical 容忍度最高的原因。但**一旦生成像素要與真實素材合成**，幾何與相機就從「可信近似」升級成 **metric 量測約束**，data-only 路線當場破功。導演要的控制（reference / keyframe / camera / in-context-edit）**已成熟**；VFX 要的 metric 控制**還沒有** → 結論：**停在 previz / augmentation，不是 final frame**。

**三條 falsifiable 判斷**（standalone 短內容 / previz 已綠燈；final-frame 合成仍紅燈）：
1. 學習式控制堆疊在「**不需與真實素材像素級合成**」的場景已 production-ready（社群短片、廣告、previz、storyboard）。
2. final-frame VFX 整合的瓶頸**不是解析度/時長/畫質**，是 axis-3 控制缺 metric 相機解——這是**幾何問題不是視覺問題**。
3. 在 metric 相機/深度求解被生成式原生產出（而非事後 match-move 補）之前，gen-video 在高端 pipeline 的角色是 **previz / augmentation，不是 delivery**。

---

## §2 · 控制堆疊（導演手上的五根桿子）

可控生成 survey（arXiv [2507.16869](https://arxiv.org/html/2507.16869v3)）把條件分成 **7 類**：**Structure**（pose / depth / sketch / bbox）· **Identity** · **Image**（reference）· **Temporal**（flow / trajectory / camera / motion）· **Audio** · **Other** · **Universal**——這與本倉 5 軸的 `control=` 列舉**直接呼應**（structure→layout/pose、image→`image-init`、temporal→`camera`/`trajectory`）。下面把 survey 的分類**翻譯成導演實際在用的五根操作桿**：

| 桿子 | 代表能力（版本/日期） | survey 類別 | 本倉 axis-3 | 控制性質 | production 槓桿點 |
|---|---|---|---|---|---|
| **① reference / identity** | Runway **Gen-4**：single reference image →「無限角色一致」，官方定位 VFX「seamlessly sit beside live action」；Sora 2 **cameo**（最佳 2–4s、≤2 角色） | Identity / Image | `image-init` | **learned / relative**（identity 嵌入，非 3D rig） | 跨鏡頭**角色一致性**——廣告/系列短片的命脈 |
| **② first-last-frame（關鍵幀）** | Kling **FLF2V**；Pika **Pikaframes**（1–10s）；Veo **3.1** image-to-image transition | Temporal（motion 端點） | `image-init`×2 | **learned**（端點給定、中間插值由統計補） | **最 production-friendly 的桿**——控制「in 與 out」，接得上 storyboard |
| **③ camera（Plücker）** | **CameraCtrl**（arXiv [2404.02101](https://arxiv.org/abs/2404.02101)）：plug-and-play 相機**位姿**控制模組，疊在 video diffusion 上、其他不動；用 **Plücker embedding**→相機 encoder→注入 **temporal attention**。MotionCtrl 統一相機+物件；CameraCtrl II / ReCamMaster 擴到動態場景 | Temporal（camera） | `camera` | ★**相對/學習式、非 metric 相機解** | 推軌/環繞/手持感——但**不是**可對齊真實 plate 的校準相機 |
| **④ in-context edit（新槓桿）** | Runway **Aleph**（付費 2025-07-25、API 2025-08-01）：對**輸入片**做物件 add/remove/transform、生不同相機角度、改風格/光 | Other / Universal | `image-init`（video-in） | **learned**（video-to-video 編輯） | **最接近 VFX-pipeline-native**——把生成式插進「已有素材」而非從零生成 |
| **⑤ structure（ControlNet-for-video）** | Wan **VACE**：統一 T2V / ref2v / v2v / inpaint / outpaint，接 depth / pose / sketch / flow / layout / bbox（確切 channel 清單 **UNVERIFIED** primary） | Structure | `layout` / pose / depth | **learned**（per-frame 結構條件複製到 clip） | 把分鏡的**構圖/動作**約束住——AnimateDiff/ControlNet-video 血統的延續 |

**讀法**：①②⑤ 是「**生成內容的內部控制**」（角色/端點/構圖）；③④ 是「**朝向 VFX pipeline 的兩根桿**」——③ 想控相機、④ 想編輯已有素材。但**③ 的相機是學習式的、④ 的編輯不輸出 metric 幾何**——這正是 §4 那道牆的兩個入口。

> **與引擎章的分工**：本篇**不重拆** Sora / AnimateDiff 引擎本身（連 [sora.md](../../foundations/video-world-models/sora.md) · [animatediff.md](../../foundations/controllability-mechanisms/animatediff.md)）。這裡只談**控制面**——同一個 `data-only` 引擎，導演透過上面五根桿子去**操作**它。

---

## §3 · 商業現實（廣告 production-real；高端電影 previz-only）

兩個 2025 的標誌性案例，劃出 gen-video 商業落點的**上下界**：

| 維度 | 廣告線：**Coca-Cola 2025 假日廣告** | 高端電影線：**ILM「Star Wars: Field Guide」** |
|---|---|---|
| 工具 | Runway / Kling / Luma **Dream Machine** | ILM 內部 text-to-video 生物測試 |
| 規模/產出 | **~10,000 幀 / ~5,000 片段**，**~2 個月** vs 傳統 ~1 年 | **2 分鐘**生成短片，**ILM 首個** text-to-video 生物測試（2025） |
| 落點 | **production-real**（真的當成品投放） | **previz-only**（生物/概念測試，非交付鏡頭） |
| 自評/外評 | **公眾批評連兩年**——"AI can't smile" 的 **uncanny**（人臉微表情翻車） | ILM 自評：「快速 sketch 的 **promise** + prompt 工作流的 **limitation** … **not a finished product** … early-stage」 |
| 教訓 | 省時間是真的；但**人臉/情感**這類觀眾最敏感的 surface 仍是 uncanny 雷區 | 生成式在 ILM 的角色是**加速 ideation**，不是產出最終像素 |

**收斂結論**：
- 廣告敢上 production-real，是因為**很多鏡頭不需要與真實素材像素級合成**（純生成片段拼接即可），且觀眾對廣告的「合成感」容忍度較高——但**人臉**例外（Coca-Cola 兩年都栽在這）。
- 電影守在 previz-only，是因為高端 VFX 的命脈是**和實拍 plate 無縫合成**——而那需要 §4 的 metric 相機解，**生成式現在給不出**。
- 注：**Sora 2**（OpenAI 已宣布下線：app **2026-04-26**、API **2026-09-24**）在此只當「**一個世代證明了什麼可能**」的能力標記——**不是前向 production 目標**。它證明了 cameo/一致性的上限，但工程依賴不該押在一個即將退役的 API 上。

---

## §4 · 為什麼停在 previz：match-move 的 metric-geometry 牆

這是全篇的**核心機制**。VFX 把 CG 整合進實拍，靠的是 **match moving**（[Wikipedia](https://en.wikipedia.org/wiki/Match_moving)）：

> **match moving 要幹的事**：把 CG 元素以**正確的 position / scale / orientation / motion**，**相對於被拍攝物**插入鏡頭——為此必須**逐幀三角化**出相機的 **position / rotation / lens（焦距/畸變）**，解出一個**校準過的 metric 相機解 + 場景點雲**。有了這個解，CG 渲染相機才能和實拍相機**逐幀對齊**，合成才不穿幫。

```ascii
   傳統 VFX final-frame pipeline                生成式相機控制（CameraCtrl 等）
   ─────────────────────────────              ─────────────────────────────
   實拍 plate (真實相機)                         text / ref / 首尾幀
        │                                            │
        ▼  逐幀特徵追蹤 + 三角化                       ▼  Plücker embedding（相機射線參數化）
   ┌──────────────────────┐                     ┌──────────────────────┐
   │ METRIC 相機解        │                      │ 相機 encoder         │
   │ position/rotation/   │  ◀── 校準、可量測     │ → temporal attention │  ◀── 學習式、相對
   │ lens + 點雲(公尺尺度) │                      │   注入 video diffusion│
   └──────────────────────┘                     └──────────────────────┘
        │                                            │
        ▼  CG 渲染相機逐幀對齊真實相機                  ▼  生出「看起來像那樣推鏡」的像素
   ★ 與真實 plate 像素級合成 ✅                    ★ 無校準解、無法對齊真實 plate ❌
        │                                            │
        ▼                                            ▼
   FINAL FRAME（可交付）                          PREVIZ / standalone（不可與實拍合成交付）
```

**牆在哪**：CameraCtrl 用 **Plücker embedding** 把相機射線參數化餵進 temporal attention——它能讓模型**生出「像是那個機位/運動」的像素**，但它**不解出一個校準的 metric 相機**，更不會輸出與某段**真實 plate 對齊**的相機外參+點雲。於是：

- **standalone / previz**：只要不和真實素材合成，"相對相機運動可信" 就夠了 → ✅ 已成熟。
- **final-frame 合成**：必須 metric-aligned 到真實 plate → ❌ 生成式給的是學習近似，**不是量測**。

**這是幾何問題，不是畫質問題**。哪怕生成畫質再上一個量級，只要相機控制停在「學習式/相對」、不輸出校準 metric 解，gen-video 就跨不過 final-frame 的合成門檻。④ in-context edit（Aleph）是最接近的嘗試——它**對已有素材操作**，理論上更靠近 pipeline；但它生「不同相機角度」時同樣**不導出 metric 相機解**，仍是學習式重繪。

---

## §5 · 五軸定位（learned/relative control vs metric control）

| Axis | 值 | 理由 |
|---|---|---|
| **Output** | `pixel-video` | 五根桿子最終都 decode 成 RGB 幀；無 3D/explicit 幾何輸出 |
| **Injection** | `data-only` | 控制堆疊全靠資料統計——reference 嵌入、首尾幀插值、Plücker→attention、結構條件，**無物理 loss / 無 sim / 無 hard-constraint**（media vertical 物理容忍度最高的直接後果） |
| **Control** | `image-init \| camera \| trajectory` | image-init=reference + 首尾幀 + video-in(Aleph)；camera=CameraCtrl 位姿；trajectory=MotionCtrl 物件路徑；structure(layout/pose/depth) 經 VACE 但本篇聚焦三主桿 |
| **Temporal** | `clip-parallel` | 整段 clip 一次 denoise（FLF2V / Aleph / CameraCtrl 皆 clip 窗口內），跨 clip 銜接仍難 |
| **Domain** | `generalist` | 廣告/電影/社群通用內容，無固定場域（⚠️ audit Check 9c 對 use-case 篇的 generalist 取態見 §8.7） |

★**本篇的五軸洞察**：以上五軸**全落在 `data-only` / `learned` 半邊**。VFX final-frame 要的是另一種東西——**校準 metric 量測**——它在五軸的 `control` 裡**沒有對應值**（`camera` 是學習式位姿、不是 metric 相機解）。**這個「缺一根 metric 控制軸」正是 previz↔final-frame 的分界線**，也是本倉「外觀是產品、物理只需可信」契約**唯一失效**的地方：當生成像素要與真實 plate 合成，幾何/相機從近似升級成**量測約束**。

---

## §6 · 跨路線綜合

- **連 [appearance-dynamics-decoupling.md](./appearance-dynamics-decoupling.md)**：那篇講「外觀 vs 動態解耦」——本篇是它的**控制面對偶**。導演的五根桿子里，①reference 控**外觀身份**、②③④⑤ 控**運鏡/動態/構圖**。media 之所以能 `data-only` 走天下，正因為**外觀是產品、動態只需可信**；而 final-frame 的牆，恰恰是**動態（相機運動）需要從「可信」升級到「metric 正確」**時才豎起來——和解耦篇「動態正確性何時變成硬約束」的論點同源。
- **連 [overview.md](./overview.md)**：overview 列的 wishlist（ControlNet-video 在電影工程的整合）——本篇給出**為什麼整合卡在 previz** 的根因（metric 牆），是那條 wishlist 的「下半場」。
- **連引擎章**：[sora.md](../../foundations/video-world-models/sora.md)（`data-only` reference point，含 Sora-2 cameo 一致性上限）· [animatediff.md](../../foundations/controllability-mechanisms/animatediff.md)（"freeze base + 學 temporal sidecar"，⑤structure 桿的血統）。**本篇不重拆引擎**，只談架在引擎上的控制面。
- **連 [ontology.md](../../cheat-sheet/ontology.md)**：survey 的 7 類條件 ↔ 本倉 5 軸 `control=` 列舉的對照（structure→layout/pose、image→image-init、temporal→camera/trajectory），是 ontology「多模態 controllability 是 2026 主戰場」論點的 media-vertical 實例。

---

## §7 · 參考

- 可控生成 survey（7 類條件分類）：arXiv [2507.16869](https://arxiv.org/html/2507.16869v3)（v3）
- **CameraCtrl**（Plücker embedding → 相機 encoder → temporal attention）：arXiv [2404.02101](https://arxiv.org/abs/2404.02101)；MotionCtrl / CameraCtrl II / ReCamMaster 為同線擴展
- Runway **Aleph**（in-context video edit，付費 2025-07-25、API 2025-08-01）：[runwayml.com/research/introducing-runway-aleph](https://runwayml.com/research/introducing-runway-aleph)
- Runway **Gen-4**（single-reference 角色一致、VFX 定位）·Sora 2 **cameo**·Kling **FLF2V**·Pika **Pikaframes**·Veo **3.1**·Wan **VACE**（產品/官方頁面，能力宣稱按發布日，**part UNVERIFIED**）
- **Coca-Cola 2025 假日廣告**（~10,000 幀 / ~5,000 片段 / ~2 個月；公眾批評）：[marketingdive.com](https://www.marketingdive.com/news/what-coca-cola-learned-generative-ai/741709/)
- **ILM「Star Wars: Field Guide」**（2 分鐘、首個 text-to-video 生物測試、previz-only 自評，2025）：[decrypt.co/320224](https://decrypt.co/320224/ilm-makes-star-wars-field-guide-short-film-using-generative-ai)
- **match moving**（metric 相機解 + 點雲、逐幀三角化）：[en.wikipedia.org/wiki/Match_moving](https://en.wikipedia.org/wiki/Match_moving)

---

## §8 踩坑日誌

> 來源：(a) 官方/產品自承 limitation、(b) 公開案例可觀察破綻、(c) 概念解讀陷阱。Severity：H=阻止用於該用途 / M=工程可繞 / L=cosmetic。**所有「角色一致/相機可控」能力宣稱皆產品方自述，未經獨立驗證者標 UNVERIFIED。版本宣稱按發布日，快速演進、易過期。**

| # | 來源 | 摘錄 / 觀察 | Severity | Workaround |
|---|---|---|---|---|
| §8.1 | **match move 的 metric 牆**（核心） | 生成式相機控制（CameraCtrl/Plücker）是**學習式/相對**，**不輸出校準 metric 相機解**，無法對齊真實 plate → 不能直接交付 final-frame 合成 | **H** | final-frame 仍走傳統 match-move + CG 合成；gen-video 限 previz / standalone / augmentation |
| §8.2 | Coca-Cola 2025（公眾批評，連兩年） | **"AI can't smile"**——人臉微表情 **uncanny**，是觀眾最敏感 surface | **H** | 含人臉特寫的鏡頭審慎用生成式；情感戲回實拍；生成式留給環境/B-roll/概念 |
| §8.3 | ILM 自評 | 「**not a finished product … early-stage**」——high-end pipeline 只把它當 ideation 加速 | **H** | 定位為 previz/storyboard 工具；不承諾交付鏡頭；產出進 review 而非 master |
| §8.4 | **Sora 2 下線**（app 2026-04-26 / API 2026-09-24） | 當「能力標記」可以，當**前向 production 依賴**會踩空 | **H** | 工程依賴分散到 shipping 的引擎（Veo / Kling / Wan / Runway）；別綁單一退役 API |
| §8.5 | **Wan VACE channel 清單** | depth/pose/sketch/flow/layout/bbox 的**確切支援 channel 未經 primary 源確認** | M（**UNVERIFIED**） | 採用前以官方 model card / repo 逐項驗證所需 conditioning channel 是否真支援 |
| §8.6 | **能力宣稱 = 產品方自述** | Gen-4「無限角色一致」、Sora 2 cameo「最佳 2–4s/≤2 角色」、Kling FLF2V 等**皆官方宣稱**（**UNVERIFIED** by independent eval） | M（**UNVERIFIED**） | 上 production 前自行壓測一致性/時長極限；不照單全收 marketing 數字 |
| §8.7 | **audit Check 9c**（generalist 白名單） | ontology 規定 `domain=generalist` 僅 Sora/Veo/Cosmos 可標；本篇為 **use-case 控制面綜述**（橫跨多 generalist 引擎），非單一模型 anchor → 故沿用 header 指定值並在此聲明 | L | 視作 use-case 級 generalist（聚合多 foundation 引擎的控制面），非模型級宣稱；若 audit 嚴格化可改標 `media` |
| §8.8 | **「相對相機可控」≠「metric 對齊」**（最常見誤判） | demo 裡推軌/環繞很順，被誤以為「能對齊真實鏡頭」——兩者是**學習近似 vs 量測**兩回事 | M | 評估 final-frame 適配時只認「**能否導出校準相機外參+點雲對齊 plate**」這個二元軸，不看運鏡是否流暢 |
| §8.9 | first-last-frame 中段不可控 | FLF2V/Pikaframes 只給**端點**，中間運動由統計插值——複雜中段動作可能違背意圖 | M | 把長動作切成多個短 FLF 段、密集打關鍵幀；複雜物理中段別指望端點控制 |
| §8.10 | in-context edit（Aleph）仍不導出幾何 | 「生不同相機角度」是**學習式重繪**、非 metric 視角合成——靠近 pipeline 但**未過 metric 牆** | M | 當高階 v2v 編輯/概念探索用；不當作可合成 final plate 的多視角幾何源 |
