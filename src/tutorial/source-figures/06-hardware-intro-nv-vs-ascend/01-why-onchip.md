# 图 1 · 为什么需要片上存储

正文位置：第 1.1、1.2 节。

```text
                        延迟（时钟周期，量级示意）
   离计算单元很远  │  HBM / GM    ~~~~~~~~~~~~~~~  数百周期
                  │      │
                  │  L2 cache     ~~~~~~~~~        （量级低于 HBM，具体值未确认）
                  │      │
                  │  shared / UB  ~~~~~            （量级再低，具体值未确认）
                  │      │
   离计算单元很近  │  寄存器       ~                几个周期（算术指令）

   容量：越靠近计算单元越小
   带宽：越靠近计算单元越高
   可编程性：shared / UB / L0 可显式分配；寄存器由编译器分配
```

## 这张图想说什么

* 访存延迟比算术延迟大两个数量级（CUDA Best Practices Guide：未缓存访存“hundreds of
  clock cycles”，算术指令约 4 周期，compute capability 7.0）。所以优化方向是
  **让数据先靠近计算单元、并在片上被复用**。
* 越靠近计算单元的存储容量越小，于是只能分块（tile）；分块之后想重叠搬运与计算，
  就需要多缓冲。而多缓冲又要和片上容量、同步一起算。

## 事实来源

* 延迟对比：`../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md`。
* 层级顺序：`../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md`。
* 昇腾侧各级存储的角色见 `02-ascend-hierarchy.md`。
* 图中标注“具体值未确认”的地方，本章没有找到可跨型号引用的公开延迟数字，**不填估计值**。

## 许可

自绘，随本书正文。
