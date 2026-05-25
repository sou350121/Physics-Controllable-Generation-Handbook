# Controllability Mechanisms

> Ontology Axis 3 的工程實現 —— 9 種 conditioning input 怎麼接進 video / latent / 3D 生成模型。

## 9 種輸入的工程接法

| Input | 接法選項 | 代表 |
|---|---|---|
| `text` | cross-attn / prefix tuning / classifier-free guidance | Sora · Veo |
| `action` | latent action token / concat / cross-attn | Genie-2 · DreamerV4 |
| `trajectory` | trajectory embedding / spatial-temporal cond | Cosmos-Drive · GAIA-traj |
| `force` | force vector → embedding / contact field cond | ForceGen · ContactGen-force |
| `contact` | contact graph / mask field | ContactNets · CC-Diff |
| `image-init` | image token concat / cross-attn | SVD · Cosmos-img2vid |
| `3d-init` | NeRF/3DGS feature lift → cond | World Labs |
| `param` | scalar embedding / hypernetwork | NeuralPhysics · ParamControlNet |
| `camera` | camera pose / extrinsics conditioning（v2 新增） | Cosmos · GAIA-2 · Generative-GS |
| `layout` | scene-graph / BEV / road-graph（v2 新增） | GAIA-2 · Cosmos-Drive |

> v2 移除 `multi` 字面 keyword —— 多值直接用 `|` 分隔，例如 `control=text|action|trajectory|camera`。

## ControlNet-for-physics

ControlNet 在 image diffusion 取得成功 → 同套思路向 video / physics 延伸：
- ControlNet-video (depth / pose / sketch)
- ControlNet-physics (contact map / force field / occupancy)

## Dissection wishlist (3 篇)

- [ ] Genie-2 latent action 設計（與 latent WM zone 互鏈）
- [ ] Cosmos-Drive trajectory conditioning 工程
- [ ] ForceGen / ContactGen 對比

## §8 共通 pitfall

- Multi-conditioning 互相干擾 —— text 蓋住 trajectory 的常見現象
- 訓練時見的 conditioning 分布範化失敗
- Conditioning strength 與 fidelity 的 Pareto（指向 crossing/controllability-vs-fidelity）
