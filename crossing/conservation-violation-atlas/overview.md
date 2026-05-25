# Conservation Violation Atlas

> 各方法在質量 / 動量 / 能量 / 接觸無穿透 / 因果一致性上的違反程度地圖。

> v0.1 框架，待 12 anchor dissection 完成後填實。

## 五類守恆律

| 律 | 評測方式 | 哪些方法最容易違反 |
|---|---|---|
| Mass conservation | 流體 / 軟體模擬中總質量是否守恆 | Sora-style pixel WM（fluid 場景） |
| Momentum conservation | 剛體碰撞前後動量總和 | 純 implicit video WM |
| Energy conservation | 擺 / 振盪 / 落體系統 | 多數 video WM |
| Non-penetration (contact) | 兩個剛體不可互相穿透 | 多數 video WM、部分 3D 生成 |
| Causal consistency | 物體不無中生有 / 消失、不違反因果 | 短 clip 內好，long-horizon 崩 |

## 矩陣（v0.1，預期值）

|              | Mass | Momentum | Energy | Contact | Causality |
|--------------|------|----------|--------|---------|-----------|
| Sora         | ❌   | ❌       | ❌    | ❌      | 🟡（短 ok） |
| Veo          | ❌   | ❌       | ❌    | ❌      | 🟡 |
| Cosmos-Predict | ❌ | ❌      | ❌    | 🟡      | 🟡 |
| Cosmos-Reason+Predict | 🟡 | 🟡 | 🟡   | 🟡      | ✅ |
| V-JEPA-2     | N/A  | N/A      | N/A    | N/A     | 🟡 (latent) |
| DreamerV4    | N/A  | N/A      | N/A    | N/A     | 🟡 (latent) |
| Genesis (sim)| ✅   | ✅       | ✅    | ✅      | ✅ |
| GraphCast    | ✅   | ✅       | ✅    | N/A    | ✅ |
| FNO          | 🟡   | 🟡       | 🟡    | N/A    | ✅ |
| PhysDiff     | 🟡   | 🟡       | 🟡    | 🟡     | 🟡 |
| ContactGen   | N/A  | 🟡       | N/A    | ✅      | ✅ |

> ✅=satisfies, 🟡=partial / domain-locked, ❌=routinely violates, N/A=not applicable to output space。

## 為什麼這 atlas 是 USP

沒有公開 benchmark 系統地測這 5 個 axis。本 handbook 目標是每篇 dissection 的 §8 pitfall log 都標出該方法在此 atlas 上的位置 —— 累積成全景。

## 評測 implementation

待補（指向 `foundations/evaluation-physics/` 的具體 metrics）。

## Open question

- 「Causal consistency」是否該獨立為第 6 條，或屬於 momentum/energy 的衍生？
- Multi-frame 守恆 vs frame-pair 守恆要分開計？
