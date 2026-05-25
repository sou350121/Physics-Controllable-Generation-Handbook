# Text-Action-Trajectory Spectrum

> 9 種 controllability input 的光譜對比，誰能組合誰，缺口在哪。

## Spectrum

```
[ 抽象 / 高層意圖 ]                                                [ 具體 / 物理量 ]
        text → image-prompt → 3d-prompt → action → trajectory → contact → force → physical-param
       (Sora)    (SVD)        (WorldLabs)  (Genie)  (Cosmos-Drive) (ContactGen) (ForceGen) (NeuralPhysics)
```

從左到右：抽象度下降，物理量化度上升，**互相之間有 composability 階梯**。

## Composability 階梯

- Text + image：成熟（90% 主流 video gen）
- Text + trajectory：成熟（Cosmos-Drive）
- Text + action：開始成熟（Genie-2）
- Action + trajectory：自然搭配
- Trajectory + force：罕見，是 robotics 真正需要的接口
- Action + contact + force：sim 原生，gen 還沒
- All-of-above：理論可組，沒人做出來

## Empirical gap

幾乎沒有方法在 robotics 場景同時接受 **text + action + force + contact** —— 這是 robotics-data-gen use-case 的真正稀缺點。

## 與 cheat-sheet/controllability_input_matrix.md 的差異

- 矩陣是 method × input 的存在性
- 本 wedge 是 input 之間的 **composability** 與 **缺口**

## Open question

- Force 與 contact 能不能透過 text + action 隱式蘊含？（PhysGen 嘗試）
- 是不是有「universal conditioning interface」可以同時吃所有 9 種？
