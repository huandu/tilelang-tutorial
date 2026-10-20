# 示例 09：ACLGraph 入图（A 树）

## 目的

本章 §9.2。用最小的 `RMSNorm` 演示 ACLGraph 的 **capture + replay**：

```python
g = torch.npu.NPUGraph()
with torch.npu.graph(g):
    out = tilelang_rms_norm(q, eps)   # 只记录，不执行
g.replay()                            # 真正执行；可反复调用
```

## 运行前提

* Ascend NPU + CANN + `torch_npu` + TileLang-Ascend（A 树，需 `source set_env.sh`）；
* `hidden_size` 能被 32 整除（`block_M=32`）。

## 运行命令

```bash
python3 rms_rope_aclgraph.py --b 2 --s 512 --hidden 2048
```

想确认入图是否真的生效，先打开 Host 日志：

```bash
export ASCEND_HOST_LOG_FILE_NUM=1000
export ASCEND_PROCESS_LOG_PATH=logs
export ASCEND_GLOBAL_LOG_LEVEL=1
python3 rms_rope_aclgraph.py
```

## 期望输出

```
eager correctness passed
aclgraph replay correctness passed
```

并在 `logs/` 里能搜到 `capture model`（值为 1 表示启用）。

**本机没有 NPU，以上输出未产生。**

## 作者实际做到的验证级别

**仅静态核对**：`py_compile` 通过；`torch.npu.NPUGraph`、`torch.npu.graph`、
`T.alloc_shared`、`T.tile.mul/add/div/sqrt`、`T.reduce_sum` 的用法对照
A 树 `examples/aclgraph/{README.md,rms_rope_aclgraph.py}` 与
`docs/TileLang-Ascend Programming Guide.md:2564-2587` 逐条核过。

## 已知局限

* 官方示例里 RMSNorm 与 RoPE 是**两个算子**链在一起；这里只保留 RMSNorm，
  以免引入 RoPE 的 mask / sin-cos 细节。要复现完整示例请直接用官方文件。
* ACLGraph 只降低**下发开销**，不做图编译/图优化；如果 kernel 本身是设备侧瓶颈，
  入图不会变快。
* 原示例的 RoPE 是 in-place 算子（不设 `out_idx`）；本示例的 RMSNorm 有输出，
  因此设了 `out_idx=[-1]`。
