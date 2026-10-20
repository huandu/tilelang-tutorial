# 图 2：roofline 的正确读法

**用途**：正文第 2.2–2.4 节。

## 2.1 一张图里有几条 roof，而不是一个点

```text
                        log(FLOP/s)
                             ^
   peak(tensor_core.fp16) ---|================================  <- roof A
                             |          ^
   peak(cuda_core.fp32) -----|----------|=====================  <- roof B
                             |          |
                             |          |   compute-bound 区
                             |        /'|
                             |      /'  |     <- 斜率 = 带宽
                             |    /'    |
                             |  /'  memory-bound 区
                             |/'________|____________________> log(FLOP/Byte)
                             0        BP_B   BP_A

   BP_A = peak(tensor_core.fp16) / bandwidth
   BP_B = peak(cuda_core.fp32)   / bandwidth
```

同一个算子只有**一个**真实的位置（它的 AI），但有**若干条**可能的 roof。
把 AI 落在哪条 roof 下面，是设计者要声明的判断，不是算子自带的属性。

用 H200 标定值（`TileOPs/.../h200.yaml`）算出的四个平衡点：

| roof | peak | 平衡点 |
| --- | --- | --- |
| `tensor_core.fp8` | 1380.75 TFLOP/s | 306.67 FLOP/Byte |
| `tensor_core.bf16` | 759.94 TFLOP/s | 168.78 FLOP/Byte |
| `tensor_core.fp16` | 729.26 TFLOP/s | 161.97 FLOP/Byte |
| `cuda_core.fp32` | 57.27 TFLOP/s | 12.72 FLOP/Byte |

带宽 4800 GB/s × 0.9380 = 4502.4 GB/s（`code/01`）。

## 2.2 roof 之下还有一层：模型没有抓手的区域

```text
   SOL time
      ^
      |                                        ___________
      |                                   ____/
      |                              ____/
      |                         ____/
      |____________________/___|
      |<-- 没有抓手 --------->|<-- 模型开始有意义 -->
      0                    latency floor

   当 sol_time 与实测时间都低于延迟下限时，启动开销占主导，
   模型对这一行的读数「没有抓手」（TileOPs roofline.md:40-41）。
   同一段代码在 prefill shape 上有意义，在 decode shape 上没有。
```

## 2.3 八种误读各自的偏移量

```text
误读                                        在案例算子上把答案带偏
---------------------------------------------------------------
M1 用 datasheet 峰值                     compute time 低估 1.36x
M2 用张量核 roof 给逐元素算子定价        平衡点 12.72 -> 161.97 (12.7x)
M3 按输入 dtype 声明 roof                同一算子三个上限 306/162/12.7
M4 用一个 stage 的 AI 代表融合后的算子   AI 2.79 被读成 1.40 (2.0x)
M5 超越函数按 1 FLOP 记账                softmax 行 0.8k -> 1.7k cycles (2.08x)
M6 只数 payload 不数查表                 63.57 MB -> 67.76 MB（真实设计 4.65x）
M7 在模型没有抓手的尺度上读数字          一个 decode step 的 SOL 是 0.003 us
M8 把 >105% 的效率当好性能               那是公式错误，不是性能
---------------------------------------------------------------
```

## 来源

* 机器参数（标定后）：R2（`TileOPs/.../h200.yaml:15-50`）；
  datasheet 峰值对照：R10（`nvidia-blog-hopper-architecture-in-depth.md:135-143`）。
* 「没有抓手」的定义与 80% / 105% 分档：R1（`roofline.md:40-41`、`:156-163`）。
* 八种误读的数字：`code/02-roofline-misuse/roofline_misuse.py` 的实际输出。
