# Companies

| 公司 | 主軸 | 與本倉的相關性 |
|---|---|---|
| NVIDIA Cosmos | World Foundation Model 全棧（Predict + Reason） | 旗艦 — 多 dissection |
| OpenAI | Sora pixel video gen | implicit-physics 旗艦 |
| Google DeepMind | Veo, Genie, GraphCast, GenCast | 跨 video+latent+surrogate 三線 |
| Meta FAIR | V-JEPA / V-JEPA-2 | LeCun latent 派旗艦 |
| World Labs | 3D scene gen | 3D-aware 旗艦 |
| Wayve | GAIA driving WM | driving sim 旗艦 |
| Decart | Real-time interactive WM | latent rolling WM |
| Physical Intelligence (PI) | VLA + data engine | robotics-data-gen 接口 |
| Genesis Embodied | 統一可微 sim | diff-sim 新興 |
| Runway / Pika / Kling / Hunyuan / Wan | 影片商業線 | media use-case |
| ECMWF / DeepMind weather | Surrogate prod | scientific 旗艦 |
| **Skydio** | aerial autonomy 教科書（VIO + on-board NN） | aerial-sim 旗艦 |
| **DJI / Autel** | 消費級 + 工業級 drone sensor stack + 內部 sim | aerial-sim 工業參考 |
| **UZH RPG** | Champion-Level Drone Racing (Nature 2023) — sim-to-real 範式 | aerial-sim 學術旗艦 |
| **NTNU ARL** | Aerial Gym 開源 — GPU 並行 drone sim | diff-sim aerial |

## Dissection wishlist

優先 anchor（Phase 1 6 篇）：
- [ ] NVIDIA Cosmos（含子線 Predict / Reason / Drive / Robotics）
- [ ] OpenAI Sora
- [ ] Google Veo (1/2/3) + Genie 1/2
- [ ] Meta V-JEPA / V-JEPA-2
- [ ] Wayve GAIA-1/2
- [ ] World Labs 路線

第二批（Phase 2）：
- [ ] Decart interactive WM
- [ ] Genesis embodied / Physical Intelligence
- [ ] DeepMind GraphCast / GenCast / Pangu
- [ ] Runway / Kling / Hunyuan / Wan SOTA 對比
- [ ] **Skydio / DJI / UZH RPG**（aerial 三巨頭，搭配 aerial-sim use-case）

## 與 Spatial-Handbook companies 的重疊

- Spatial-Handbook 已收 nvidia_cosmos / wayve_world_model / world_labs / physical_intelligence
- 本倉用「生成端」視角重寫：強調 Cosmos 的 Predict（生成）線、World Labs 的 3D gen 而非 reconstruction
- 引用時 §6 cross-line synthesis 必須交叉鏈接 sister repo
