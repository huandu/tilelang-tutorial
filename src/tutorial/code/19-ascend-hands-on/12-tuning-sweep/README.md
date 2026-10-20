# 示例 12：配置扫描驱动（纯标准库）

## 目的

本章 §8.6。在**上机器之前**把「测什么、比什么、怎么记」定下来：

1. 生成 GEMM / 向量两类配置；
2. 用 Atlas A2 的片上容量做**资源账预筛**，把明显放不下的配置先剔掉（省机器时间）；
3. 打印一张「实验记录表」模板，字段与 M 树 SKILL §12.2 要求的测量条件对齐
   （设备、CANN、torch_npu、哪棵树、pass config、正确性、延迟、瓶颈、原始样本）。

## 运行前提

* Python 3.9+，仅标准库。**不需要 NPU。**

## 运行命令

```bash
python3 tune_sweep.py                     # GEMM 配置预筛
python3 tune_sweep.py --mode vec --tile 4096 8192 --stages 1 2 3
```

## 期望输出（示意）

```
== 片上容量（Atlas A2，字节）==
    l1:   524288 B (512 KB)
   l0a:    65536 B (64 KB)
   ...
== GEMM 配置扫描（按资源账预筛）==
  BM= 128 BN= 256 BK=  64 stages=2 float16->float32 | l1=... l0c=128KB | OK
  BM= 256 BN= 256 BK=  64 stages=3 float16->float32 | ... | OVER: l0c: need ... > cap ...
  ...
预筛剔除：N 个配置放不下（这些不用上机器）。
== 实验记录表模板（照着填，A/B 才可比）==
  config = '<BM/BN/BK/stages>'
  ...
```

## 作者实际做到的验证级别

**CPU 逻辑验证**：纯标准库，在写作机上实际运行过。

## 已知局限

* 容量数字是 **Atlas A2** 的（L1 512KB / L0A、L0B 各 64KB / L0C 128KB / UB 192KB），
  来自 CATLASS 的 `atlasA2_hardware_info.md`。**Ascend 950 的容量可能不同**，
  请用 `GetCoreMemSize` 之类的平台接口核实后再套用。
* 资源账只估「用户 buffer」的占用，**不含 layout padding、多缓冲的额外维度与 SDK 预留**；
  真正的容量判定以编译/启动是否失败为准。
* 它只做规划，**不做测量**。测量请用 `../08-msprof-bench/`。
