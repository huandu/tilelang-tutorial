# 示例 07：昇腾的 pipe 模型与 tile 下界

## 目的

NVIDIA 侧的设计靠「流量 × 波数」的启发式（`../03-carver-cost-model/`），
昇腾侧不是：它把每条语句变成任务，按**物理 pipe** 逐个收费，再把结果交给 Z3 排。
常量表在

```
_vendor/tilelang-v0.1.15/src/ascend/transform/estimate_latency.h:27-158
```

规则在 `src/ascend/transform/estimate_latency.cc`。这个脚本抽出其中**决定 tile 尺寸**
的两半——Cube 的吞吐代价与 GM→L1 的 MTE2 代价——并解出它们相等时的 tile。

关键公式（`estimate_latency.cc:695-707` 与 `:920`）：

```
Cube 的 ops  = 2 * M * N * align(K, 16)        # 乘加已经算两次
Cube 的 II   = ceil(ops / cube_throughput_<dtype>)
MTE2 的 II   = max( ceil(bytes / aic_mte2_to_l1_bandwidth),
                    descriptors * mte2_gm_to_l1_min_ii )
```

两式相除，`K` 与两个因子 2 都消掉，得到一条纯形状判据。

## 运行前提

* Python 3.9+，只用标准库。
* **不需要 GPU / NPU / TileLang / PyTorch。**
* 所有常量都是**照抄源码**的，脚本头部逐条列了出来。

## 运行命令

```bash
python3 ascend_pipe_model.py
python3 ascend_pipe_model.py --dtype fp16
python3 ascend_pipe_model.py --sweep
python3 ascend_pipe_model.py --tile 512 256 128
```

## 期望输出

正确性判据：退出码为 0。实际输出（节选）：

```text
   M=N     cube II     mte2 II   cube/mte2  binding
    64          16         128        0.12  MTE2
   128          64         128        0.50  MTE2
   192         144         128        1.12  Cube
   256         256         164        1.56  Cube

Closed form: the Cube is the bottleneck iff
    max( 2K(M+N)/mte2_bw , descriptors*mte2_min_ii ) <= 2MNK/cube_throughput
which for large tiles (the issue floor is 128 cycles)
reduces to  M*N / (M + N) >= cube_throughput / mte2_bw = 81.92,
    i.e. for a square tile M = N >= 163.8

tile M= 256 N= 224 K= 128 bf16  Cube II    1792  MTE2 II    1229  MTE1 II    256
     -> binding Cube  needs >= 3 buffers  (latency/period = 1.004)
      the shape in tilelang/ascend/README.md's GEMM+ReLU example
tile M= 128 N= 128 K=  32 bf16  Cube II     128  MTE2 II     164  MTE1 II     32
     -> binding MTE2 GM->L1  needs >= 4 buffers  (latency/period = 2.159)
      a typical NVIDIA-side tile, for contrast: MTE2-bound here
```

三点值得注意：

1. **官方的 256×224×128 例子恰好落在 Cube 侧**，脚本没有任何针对它的调参；
   这是模型自洽的一个旁证（不是证明）。
2. **直方图上的翻转点在 M=N≈164**，与闭式解一致；128 那边是 MTE2 瓶颈，
   而且此时 MTE2 的 II 是被「描述符最小发射间隔」128 cycles 顶住的，不是带宽。
3. `latency/period = 1.004` 那一行说明**不要把模型当判决**：Cube 的 8 cycle
   completion overhead 恰好把第一章 11 节的 stage 公式从 2 顶到 3，
   而官方例子用的是 `num_stages=2`。

## 性能口径

**不测量任何性能。** 全部单位是 cycle，没有换算成时间——因为换算需要时钟，
而素材里没有一条可引用的公开时钟数字。这也正是这套模型的优点：它不需要时钟。

## 作者实际做到的验证级别

**CPU 逻辑验证。** 在 macOS (Apple Silicon)、Python 3.11 上运行过默认、`--sweep`、
`--dtype fp16/fp32` 与多组 `--tile`；输出与本文件一致。没有在昇腾 NPU 上运行过
任何内容，也没有跑过 `EstimateLatency` 这个 pass 本身。

## 已知局限

* 这是**教学用的抽版本**。真实的 `EstimateAscend` 还要算 Vector 的 ops/cycle、
  `ScaleBytesByLanes` 的 lanes 缩放、每个物理路径的完成延迟、以及基于 pipe mask 的
  II 合成（`estimate_latency.cc:750-970`）。本脚本只保留 Cube、MTE1、MTE2 三条。
* 「needs >= N buffers」用的是第 11 章的 `1 + ceil(latency/period)`,它在这套模型里
  没有经过验证。真实的多缓冲份数是 `PrepareMultiBuffer` /
  `MaterializeMultiBuffer` 和 Z3 求解的结果（第 23 章）。
* 常量是按芯片世代手工标定的（第 23 章 5.1 节点明了这一点）；换芯片要重标。
* `descriptors=2`（A、B 各一个）是一个假设，真实描述符数由 codegen 决定。
