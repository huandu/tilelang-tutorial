# 示例 6：把两个 TileLang kernel 装进昇腾 ACLGraph

## 目的

正文「启动开销与执行模型」一节说明：一批小算子按 eager 逐条下发时，host 侧
（Python → C++、Tiling、申请 workspace、Launch）的耗时可能盖过 device 上的执行时间，
这就是 host-bound；把下发序列捕获成一张图、之后只做图回放，是绕开它的标准做法。
在昇腾上这个机制叫 ACLGraph，用 `torch.npu.NPUGraph` 暴露。

本示例按官方示例
`_vendor/tilelang-ascend-v0.1.2.000/examples/aclgraph/rms_rope_aclgraph.py`
的形态写了一遍最小版本：定义 RMS Norm 算子 → 捕获 → 回放 → 对答案。

## 运行前提

* 昇腾 950（或兼容）设备一台；
* CANN 已安装并 `source set_env.sh`；
* `torch` + `torch_npu` 可用，`torch.npu.is_available()` 为真；
* 昇腾版 `tilelang`（`tilelang-ascend`，锁定版本 `v0.1.2.000-release`）。

## 运行命令

```bash
cd chapters/26-runtime-jit-abi/code/06-ascend-aclgraph
python3 aclgraph_capture.py
```

## 期望输出

```text
replayed shape: (1024, 512) dtype: torch.float16
matches reference: True
```

## 正确性判据

* 捕获阶段不报错（捕获期间算子只被记录、不执行）；
* `graph.replay()` 之后 `sink` 与 `out + out` 在 `atol=1e-2, rtol=1e-2` 内相等；
* 想进一步确认图模式真的生效：按上游 README 的做法打开 host 日志
  （`ASCEND_GLOBAL_LOG_LEVEL=1` 等），在日志里搜 `capture model`，
  `0` = 未启用、`1` = 已启用。

## 已知限制（来自官方 README）

* 捕获期间的操作**不会立即执行**，只在 `replay()` 时执行；
* 回放会把同一段计算重复执行，所以输入张量的地址必须在捕获期间固定下来——
  想换输入要配合静态输入缓冲（上游示例用的是原地算子 + 复用同一块显存）；
* 官方 README 建议把 `TL_ASCEND_AUTO_SYNC` / `TL_ASCEND_MEMORY_PLANNING` /
  `TL_ASCEND_AUTO_CV_SYNC` / `TL_ASCEND_AUTO_CV_COMBINE` 一起打开，
  让同步与内存规划在编译期定好，避免图内出现 host 侧分支。

## 作者实际做到的验证级别

**仅静态核对**：`python3 -m py_compile` 通过；kernel 主体与官方示例
`examples/aclgraph/rms_rope_aclgraph.py:21-59` 逐行比对（只删了第二个算子和
形状簿记）；`torch.npu.NPUGraph()` / `torch.npu.graph(...)` 与官方示例
`:271-272` 一致。本机无昇腾 NPU，**未运行**。
