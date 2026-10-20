# 图 2 · NVIDIA 侧存储层级（以 H100 / Hopper 为例）

正文位置：第 3.2 节。

```text
  一个 SM（Streaming Multiprocessor）
  ┌───────────────────────────────────────────────────────────────┐
  │  寄存器文件（256 KB / SM，64K × 32-bit）                       │
  │      ▲ 最近、最快、最小；编译器分配，线程私有                  │
  │      │                                                        │
  │  shared memory / L1（合并 256 KB / SM）                        │
  │      ▲ 可编程片上 SRAM；shared 可配置到 228 KB                 │
  │      │ 单 block 最多 227 KB（CUDA 保留 1 KB）                  │
  │      │                                                        │
  │  （Blackwell：另有 Tensor Memory，累加器专用，见 04 图）        │
  └───────────────────────────────────────────────────────────────┘
              │  所有 SM 共享
              ▼
  ┌───────────────────────────────────────────────────────────────┐
  │  L2 cache：50 MB（H100）/ 126 MB（GB200）                      │
  └───────────────────────────────────────────────────────────────┘
              │
              ▼
  ┌───────────────────────────────────────────────────────────────┐
  │  HBM / 显存：80 GB HBM3，> 3 TB/s（H100 SXM5）                 │
  │              B200 支持 HBM3/HBM3e，容量最高 180 GB             │
  └───────────────────────────────────────────────────────────────┘
```

## 关键约束（都带出处）

* H100：每 SM 256 KB 寄存器文件；合并 L1/shared 256 KB；shared carveout 可选值为
  `0, 8, 16, 32, 64, 100, 132, 164, 196, 228` KB；单 block 最多 227 KB；静态 shared
  仍限制 48 KB，超过需显式 opt-in。
* H100：L2 50 MB；HBM3 80 GB、带宽 > 3 TB/s（厂商博客标注初步规格）。
* B200（cc10.0）：shared 上限与 Hopper 相同（228 KB/SM、227 KB/block）；GB200 的 L2
  为 126 MB；B200 支持 HBM3/HBM3e，容量最高 180 GB。

## 事实来源

* `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md`
  （H100 层级、容量、带宽；Table 3）。
* `../tilelang-corpus/05-nvidia-backend/raw/nvidia-hopper-tuning-guide.md`
  （carveout 列表、每 block 上限、1 KB 保留）。
* `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blackwell-tuning-guide.md`
  （B200/GB200 数字）。

均为厂商公开资料，**非实测**。

## 许可

自绘，随本书正文。
