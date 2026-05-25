# Controllability Input Matrix

> v0.1 placeholder — 主流方法 × 9 種 controllability input。✅=原生支援 / 🟡=可改造 / ❌=不支援。

| 方法 | text | action | trajectory | force | contact | image | 3D | param | multi |
|---|---|---|---|---|---|---|---|---|---|
| Sora | ✅ | ❌ | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 |
| Veo | ✅ | ❌ | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 |
| Cosmos-Predict | ✅ | 🟡 | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 |
| Cosmos-Drive | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | 🟡 | ❌ | ✅ |
| Genie-2 | 🟡 | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 |
| V-JEPA-2 | 🟡 | ✅ | ❌ | ❌ | ❌ | 🟡 | ❌ | ❌ | 🟡 |
| DreamerV4 | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Wayve GAIA | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ |
| Decart (interactive WM) | 🟡 | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 |
| World Labs gen-3D | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | 🟡 |
| Genesis | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| MuJoCo MJX | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| GraphCast | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ |
| FNO | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ |
| PhysGen | ✅ | ❌ | 🟡 | 🟡 | 🟡 | ✅ | ❌ | 🟡 | 🟡 |
| ForceGen | ❌ | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | 🟡 |
| ContactGen | ❌ | ❌ | ❌ | 🟡 | ✅ | 🟡 | 🟡 | ❌ | 🟡 |

## 觀察

- **影片端**（Sora/Veo/Cosmos-Predict）目前在 text+image 兩根柱子上飽和，trajectory/force 進度慢
- **遊戲端**（Genie-2 / Decart）action 是原生第一公民，但物理 implicit-from-data，沒 force/contact
- **機器人端**（Cosmos-Drive / Wayve GAIA）trajectory 開始成熟，force/contact 仍稀缺
- **Sim 端**（Genesis / MJX）幾乎全綠但生成能力 = 0（不是生成模型）
- **Surrogate 端**（GraphCast / FNO）只吃 param，但 fidelity 在自家領域最高

## Gap / 空格

幾乎沒有方法在 **text + action + force + contact** 同時 ✅。這是 robotics-data-gen 真正稀缺的接口。

## TODO

- [ ] 加入「multi=多選時的組合方式」（concat / cross-attn / per-stage）
- [ ] 加入 cost annotation（GPU + latency）
