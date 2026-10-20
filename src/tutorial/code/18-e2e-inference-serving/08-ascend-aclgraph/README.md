# 示例 08：昇腾侧的 torch_npu 与 ACLGraph 入图

## 目的

把“TileLang 算子如何在昇腾上被反复执行而不被 host 侧下发拖死”讲清楚：

1. TileLang 的昇腾 kernel 怎么与 `torch_npu` 的算子落在同一条 stream 上；
2. ACLGraph 的 capture / replay 两阶段模型；
3. 入图对形状的要求，以及“为什么 vLLM-Ascend 只对 decode 入图”。

`aclgraph_decode_block.py` 里是 RMSNorm + 一步 decode 的最小骨架，
结构对照 corpus 里的
`07-tilelang-ascend-adapter/raw/tilelang-ascend/examples/aclgraph/rms_rope_aclgraph.py`。

## 版本归属（重要）

本文件用的是**适配器仓库** `_vendor/tilelang-ascend-v0.1.2.000` 的语言面：

* `T.Kernel(m_num, is_npu=True) as (cid, vid)`
  ——`tilelang/language/kernel.py:220,247`；
* `tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC` 等四个开关
  ——`tilelang/transform/pass_config.py:38-50`；
* `T.tile.mul/add/div/sqrt`——`tilelang/language/ascend_tile.py:1026,1048,1059,1241`
  经 `tilelang/language/__init__.py:86` 暴露成 `T.tile`；
* `T.reduce_sum(buffer, out, dim, clear)`——`tilelang/language/reduce.py:79`。

**主干 v0.1.15 的昇腾方言不是这套 API。** 在
`_vendor/tilelang-v0.1.15/tilelang/ascend/language/` 里 kernel 写成
`with T.Kernel(N_CORES) as core_id`（没有 `is_npu` 参数），auto-schedule 由
pass context 开关控制（`tilelang/ascend/pipeline.py` 里的 `allow_autoschedule`），
而且 `TL_ASCEND_AUTO_SYNC` 这类键在 v0.1.15 里**不存在**
（在 `tilelang/transform/pass_config.py` 里检索 `ASCEND` 无结果）。
参考 v0.1.15 的写法请看 `examples/ascend/example_rmsnorm.py`。

## 运行前提

* 昇腾 NPU（文档与示例针对 Ascend 950；适配器示例在 910 系列上也可运行）
  + CANN（含 `bisheng`、`ld.lld`、Ascend runtime）+ PyTorch + `torch_npu`；
  `torch.npu.is_available()` 必须为 `True`。
* TileLang-Ascend **适配器** `_vendor/tilelang-ascend-v0.1.2.000`。
* **本机没有 NPU、没有 CANN、没有 torch_npu**，因此本文件从未运行。

## 运行命令

```bash
# 0) 先确认设备可见
python3 -c "import torch, torch_npu; print(torch.npu.is_available())"

# 1) 跑本示例
python3 aclgraph_decode_block.py

# 2) 想看 ACLGraph 是否真的生效：打开 host 日志，搜 capture model
export ASCEND_HOST_LOG_FILE_NUM=1000
export ASCEND_PROCESS_LOG_PATH=logs
export ASCEND_GLOBAL_LOG_LEVEL=1
python3 aclgraph_decode_block.py
grep -r "capture model" logs | head

# 3) 用 msProf 看上板性能（不改变代码）
msprof op --kernel-name="main_rms" python3 aclgraph_decode_block.py
```

## 期望输出

**正确性判据**（本节未实际执行）：

* `torch.testing.assert_close(out, reference(q, eps), rtol=1e-2, atol=1e-2)`
  通过，最后打印 `Kernel Output Match!`（与 corpus 里那份 ACLGraph 示例
  的收尾断言一致）；
* 再跑一遍不带 `torch.npu.graph` 的 eager 版本，两次结果一致——
  这是“入图没有改变语义”的判据。

**性能口径**：

* 用 `msprof op` 或 `msprof op simulator` 采（见适配器编程指南第 6 章），
  或在 TileLang 里用 `do_bench(..., backend="msprof")`
  （`tilelang/profiler/bench.py:39,260-263`，它会用一个昇腾专用的
  cache-flush kernel 把 L2 冲掉）；
* 要区分三种口径：eager 的 kernel 时间、graph replay 的 kernel 时间、
  以及端到端（含 host 下发）时间。ACLGraph 的收益主要在第三种。
* 必须一起报的变量：NPU 型号、CANN 版本、`torch_npu` 版本、
  batch / heads / hidden、是否开图、是否开 `TL_ASCEND_AUTO_CV_COMBINE`。

## 作者实际做到的验证级别

**仅静态核对。** `python3 -m py_compile aclgraph_decode_block.py` 通过；
上面列的每个 API 名都在 `_vendor/tilelang-ascend-v0.1.2.000/` 里定位到了
定义行。**没有在 NPU 上编译或运行过，也没有任何实测数字。**

## 已知局限

* 只覆盖“capture 一次 + replay 多次”这一条路径，没有覆盖动态形状、
  多流、以及 vLLM-Ascend 里把图路径限制在 decode 的调度层逻辑。
* corpus 里的 ACLGraph 示例使用 `T.tile.*` 做逐元素运算；主干 v0.1.15 的
  昇腾方言改用 `T.SimtVF` + `T.Parallel`，两者写法不同（见第 19 章）。
* `--soc-version` 的具体取值取决于你的 CANN 与硬件，本文件不写死。
