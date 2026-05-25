# Differentiable Simulators

> 可微 simulator 不是「生成模型」 —— 但它是生成模型的 oracle、loss 來源、訓練資料工廠。本倉收的角度是：**diff-sim 怎麼餵生成模型**。

## 5-axis defaults

- `output=N/A`（自己不是生成模型；但生產 ground-truth video/state/contact）
- `injection=sim-in-loop`
- `control=action|trajectory|force|contact|param`（最完整一條）
- `temporal=streaming`
- `domain=robotics|rigid|soft|fluid`

## Anchor methods

| Sim | 重點 |
|---|---|
| MuJoCo MJX | JAX 移植，GPU 並行；contact 模型成熟 |
| Genesis | 全新可微 sim，rigid+soft+fluid 統一 |
| Warp | NVIDIA 線；CUDA-first，與 Cosmos 接 |
| Brax | JAX-first，rigid for RL |
| DiffTaichi | 學術 baseline，PDE 友善 |
| NVIDIA Isaac Sim | 工業強度，與 Omniverse / Cosmos 接 |
| FleX | NVIDIA position-based dynamics（granular/cloth） |

## 與生成模型接的 3 種模式

1. **作為訓練資料源** —— Cosmos 從 Isaac/Omniverse pull video
2. **作為訓練 loss** —— 在 sim 中 rollout 比對 NN-WM 結果
3. **作為推理 oracle** —— Cosmos-Reason 用 sim 評生成 video 的物理性

## Dissection wishlist (4-5 篇)

- [ ] MJX vs Genesis vs Warp 對比（成熟度 / 場景 / GPU 效率）
- [ ] Genesis 統一 rigid+soft+fluid 的設計取捨
- [ ] Sim-in-loop training 對 video WM 的實測收益
- [ ] NVIDIA Omniverse → Cosmos 資料管道
- [ ] DiffTaichi → 神經 PDE 的學術橋

## §8 共通 pitfall

- Contact discontinuity 仍是 differentiability 弱點 —— 梯度估計噪聲
- Sim2real gap：sim-in-loop 訓出來的 WM 在真實場景物理感反而下降
- GPU 並行 contact 解算的擴展性瓶頸（thousand-env scale 才有 RL signal）
