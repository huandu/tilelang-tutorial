# pass_order：静态核对昇腾 AutoSchedule 的 pass 顺序

## 目的

正文第 4.1 节说「AutoSchedule 是 13 个 pass 的链」。这个脚本把这句话变成可执行的检查：
从 `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py` 的
`if auto_schedule_enabled:` 块里按顺序抽出 pass 调用，再去
`src/ascend/transform/` 找每个 pass 的 FFI 注册点。

它不做编译、不导入 `tvm`/`tilelang`、不连 NPU。

## 运行前提

* Python 3.9+，标准库。
* 需要锁定版本的源码树。默认会在脚本的上级目录里找 `_vendor/tilelang-v0.1.15`；
  找不到时用 `--repo` 指定。

## 运行命令与预期输出

```bash
python3 ascend_pass_order.py
```

预期（本机实测，节选）：

```text
   1. NormalizeControlFlowForSchedule src/ascend/transform/normalize_control_flow_for_schedule.cc:240
   ...
   6. AutoSchedule                   src/ascend/transform/auto_schedule.cc:1400
   ...
  13. RestoreWhileLoops              src/ascend/transform/restore_while_loops.cc:117

ok: 顺序一致，且每个 pass 都在 src/ascend/transform 里有 FFI 注册
```

退出码 0 表示顺序一致且 13 个 pass 都注册过；非 0 会逐条打印差异。

## 验证级别

`在 CPU 上验证过逻辑`——脚本在本机实跑并返回 0，
`python3 -m py_compile ascend_pass_order.py` 通过。**未在 NPU 上验证。**
