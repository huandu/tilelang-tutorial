# 图 3 · NVIDIA 执行层次与 SIMT

正文位置：第 3.1 节。

```text
  Grid（一次 kernel launch 的全部分块）
   │
   ├── Thread Block / CTA（最多 1024 线程、每 SM 最多 32 个）
   │    │
   │    ├── Warp（32 个线程；SIMT 的最小调度单位）
   │    │    ├── Thread 0   ─┐
   │    │    ├── Thread 1    │  同一 warp 执行同一条指令，
   │    │    ├── ...         │  但每个线程可以访问不同地址
   │    │    └── Thread 31  ─┘
   │    │
   │    └── Warp ...
   │
   └── Thread Block ...

  （Hopper/Blackwell 新增一层）
  Thread Block Cluster（portable 最大 8；opt-in 最大 16）
   └── 多个 block 保证同时调度，可通过 DSM 直接读写彼此的 shared memory
```

## 关键数字

H100（cc9.0）公开规格：每 warp 32 线程；每 SM 最多 64 warp / 2048 线程 / 32 block；
每 block 最多 1024 线程；每 cluster 最多 16 block。portable cluster 最大 8，H100/B200
通过 opt-in 可到 16。

## 这张图想说什么

* **SIMT = 每个线程独立寻址 + warp 级合并发射。** 收益是灵活（分支、离散访存），
  代价是 warp 内分支发散会串行执行，且访存需要合并才能高效。
* **层次越高，共享越强、同步越粗。** `thread → warp → block → cluster`，每一层都有
  自己的「可以互相看到什么」和「怎么对齐」。

## 事实来源

* 执行层次与规格：`../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md`
  （Hopper 博客；「Compute capability: V100 vs. A100 vs. H100」表）。
* cluster / DSM / 上限：`../tilelang-corpus/05-nvidia-backend/raw/nvidia-hopper-tuning-guide.md`
  与 `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blackwell-tuning-guide.md`。
* SIMT 的访存合并：`../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md`。

## 许可

自绘，随本书正文。
