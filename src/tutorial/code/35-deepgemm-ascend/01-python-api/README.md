# 示例 01：DeepGEMM-Ascend 的最小调用（仅静态核对）

## 目的

正文 §3.1 的三个调用样例。把「API 兼容」落到可以复制的代码上：

* BF16 dense：`bf16_gemm_nt(a, b, d)`；
* FP8 dense：`fp8_gemm_nt((a, sf_a), (b, sf_b), d, recipe_a=..., recipe_b=...)`；
* M-grouped（MoE）：`m_grouped_fp8_gemm_nt_contiguous((a, sf_a), (b, sf_b), d, grouped_layout)`。

## 运行前提

* 昇腾 NPU（官方在 Ascend 950 系列上开发与验证，`README.md:19-31`）；
* CANN 9.20（提供 `bin/bisheng`、`bin/ld.lld`）；
* `torch_npu`、Python ≥ 3.10、支持 C++20 `<format>` 的编译器；
* 已按官方方式安装 `deep_gemm`（`./develop.sh` 或 `pip install . --no-build-isolation`）；
* 环境变量 `ASCEND_HOME_PATH` 或 `ASCEND_TOOLKIT_HOME` **在 `import deep_gemm` 之前**设好。

## 运行命令

```bash
# 先确认设备可用
python3 -c "import torch, torch_npu; print(torch.npu.is_available())"

python3 deepgemm_ascend_call.py --case bf16
python3 deepgemm_ascend_call.py --case fp8
python3 deepgemm_ascend_call.py --case mgrouped
```

## 期望输出

三类调用都应打印一行 shape 与 `OK`（程序里只做形状/无异常检查；
真正的数值对拍请用官方 `tests/`，它会调 `aclnn_*` 参考实现）。

**本机没有 NPU，以上输出未产生。**

## 作者实际做到的验证级别

**仅静态核对**：

* `python3 -m py_compile` 通过；
* 函数名与参数顺序逐个对照 corpus 快照
  `DeepGEMM-Ascend/csrc/apis/gemm.hpp:537-662` 的 `register_apis`；
* SF 的造法对照 `DeepGEMM-Ascend/tests/test_mega_moe.py:22-36`
  （`uint8` 指数 → `.view(torch.int16)`，不是数值转换）；
* **未运行、未编译任何 kernel**。

## 已知局限

* 示例不校验数值，只保证调用姿势正确；数值对拍请用官方 `tests/` + `aclnn_*`（§3.4）。
* `compiled_dims` 用默认值 `"nk"`（N、K 固化，M 动态）——
  如果你的 M 也在一个很小的集合里，可以把它加进去换更少的运行期开销。
* M-grouped 的每个专家行数必须对齐到 256；
  示例里用 `get_theoretical_mk_alignment_for_contiguous_layout()` 拿对齐值。

