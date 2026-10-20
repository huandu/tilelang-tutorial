# 第 02 章 代码

本章读者已经会写 PyTorch，但还没有性能建模经验，而且写作环境**没有 GPU / NPU**。
所以这里的代码只有一个使命：**把“一个算子要读多少字节、算多少次”变成可运行、可复算的账本。**

两个示例都只用 Python 标准库，在任意机器上离线可跑，输出就是正文里的表格。

```
code/
├─ README.md                       本文件
├─ 01-op-traffic-model/
│  ├─ README.md
│  └─ op_traffic.py                单个算子的访存量/运算量/算术强度/时间下界
└─ 02-step-budget/
   ├─ README.md
   └─ step_budget.py               一次训练迭代 / 一次推理请求的分解账本
```

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-op-traffic-model/op_traffic.py` | Python 3.9+（只用标准库） | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实跑过默认参数、`--machine h100-spec`、`--op` 过滤与多个形状；输出见该目录 README |
| `02-step-budget/step_budget.py` | Python 3.9+（只用标准库） | **CPU 逻辑验证**：同上，实跑过默认参数与放大后的模型参数 |

两个示例相互独立，可以单独复制出去运行；都不依赖其他章节的文件。

## 这两个示例不做什么

* **不测量性能。** 它们算的是“理论下界”，不是预测。真实 kernel 只会比它慢。
* **不替代 profiler。** 想知道真实时间，必须在有卡的机器上用 Nsight Compute（NV）或
  msProf（昇腾）测，见第 12 章。
* **不做 autotune。** 参数选择与搜索在第 11 章。

## 在真实机器上怎么接着往下做

本章 4.3 节讲了 `torch.compile` 能覆盖到哪一步。要把它变成观察，需要的步骤是：

1. 用 `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1` 跑一遍模型，拿到带名字的 kernel 列表
   （名字形如 `triton_poi_fused_<aten算子>_<序号>`）；
2. 用 `TORCH_COMPILE_DEBUG=1` 或 `torch.compiler.load_cache_artifacts` 打开 Inductor
   生成的 Triton 源码，看它把哪些算子融合进了同一个 kernel；
3. 用 Nsight Compute 对热点 kernel 取 achieved FLOPs 与 DRAM 字节数，和本章脚本算出的
   下界对比，得到“roofline 利用率”。

这三步的完整流程在第 12 章；本章只解释为什么它会给出“够用 / 不够用”的结论。

## 统一说明

* 机器参数一律在代码里注明出处（见各 `README.md` 与 `../references.md`）；
  本章没有任何一条性能数字来自本机实测。
* 版本一律为 TileLang `v0.1.15`（见 `../../VERSIONS.md`）；本书不使用“最新版”这类表述。
