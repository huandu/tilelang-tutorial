# 05 · RMSNorm + RoPE (+ FP8 量化) 的融合

## 目的

本章的 capstone：把 2–3 个算子融进一个 kernel，并说明融合带来的收益与代价。

| kernel | 融合了什么 | 归约次数 | 输出 |
| --- | --- | --- | --- |
| `rms_norm_rope_fused` | RMSNorm + 尾部 RoPE | 1 | `Y` |
| `rms_norm_rope_quant_fused` | RMSNorm + 尾部 RoPE + per-token FP8 | 2 | `Y`、`Q`、`Scale` |

账本上的收益（`../01-ledger/ledger.py` 可复算）：

```text
RMSNorm 单独 + RoPE 单独 + 量化单独 ：185.6 MB
三者融合成一个 kernel              ： 51.4 MB   （3.61×）
```

代价是**寄存器与 UB 压力**：三个算子共享同一块 fragment 和 shared，
`blk_m` 只能取 1，而 `x_f32` 一个 `(1, 4096)` 的 fp32 fragment 就是 16 KiB。
这就是「融合不是免费的」的地方。

## 运行前提

* NVIDIA GPU + CUDA + PyTorch(CUDA)
* TileLang **v0.1.15**
* 输入 fp16/bf16；`Q` 是 `float8_e4m3fn`

## 运行命令

```bash
cd 05-fusion
python3 -m py_compile rmsnorm_rope_quant.py   # 本机可做（无需 GPU）
python3 rmsnorm_rope_quant.py                # 真机：跑两组正确性
```

## 预期输出

**正确性判据**：

* `rms_norm_rope_fused` 与 `torch_reference_fused` 的第一个输出在 `rtol=atol=1e-2`
  内一致（fp16）；
* `rms_norm_rope_quant_fused` 的 `Scale` 与参考实现在 `rtol=atol=1e-2` 内一致，
  `Q` 反量化回来（`Q.float() * Scale`）与参考在 `rtol=atol=1e-1` 内一致
  ——fp8 只有 3 位尾数，比较时务必按量化误差口径。

**性能口径**：这个 kernel 的账本下界是「读 x + 读 w + 读 cos/sin + 写 y + 写 q + 写 scale」。
注意写了两份输出（`Y` 与 `Q`），所以访存比只做归一化大。要比较增益，
应该把「N 个算子各自的 bytes 之和」与「融合后一个 kernel 的 bytes」对比。

## 真机验证步骤（按这个顺序做，别跳步）

1. `python3 -m py_compile` 先过语法。
2. **先只编译，不运行**：`kernel = rms_norm_rope_fused(m, n, rope_dim)`。
   这一步最可能失败的是 layout inference——因为它要在同一个 kernel 里同时使用
   fragment 归约、shared 上的跨 lane 配对、以及第二个 fragment 归约。
   报错信息通常直接给出「src_layout / orig_dst_layout」的冲突对象。
3. 打印生成代码，确认三件事：`sq_f32` 那一步有没有被向量化；RoPE 那一步的两条
   store 是不是各自连续；fp8 的 cast 有没有被拆成标量。
4. 用小形状（`M=8, N=256, rope_dim=32`）先跑正确性，再上到目标形状。
5. 最后才谈性能，并且和 `../01-ledger/ledger.py` 算出来的下界比，而不是和另一个
   实现比。

## 已知的边界与备选方案

* **`blk_m = 1` 是默认值，不是最优值**。想加大 `blk_m` 就要么按 K 切行
  （对照 `../03-norm` 的 `rms_norm_splitk`），要么改用 per-thread partial 归约
  （TileOPs 的 RMSNorm 用 `per_thread_partial` 那条分支，见
  `../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/kernels/norm/rms_norm.py:44-52,72-84`）。
* **第二个归约的写法**：这里把一行 reshape 成 `(n_groups, group_size)` 后用
  `T.reduce_max(..., dim=2)` 降维。如果想手工写 read-modify-write，
  `loop_vectorize` 会因为「store 的索引不随最内层循环推进」而把这一处标量化的
  （`_vendor/tilelang-v0.1.15/src/transform/loop_vectorize.cc:1050-1060`）。
* **RoPE 那一步为什么要回到 shared**：NeoX 约定要求第 `p` 个和第 `p + H` 个元素
  同时可见，它们在 fragment 里可能属于不同线程，寄存器之间不能互相访问。
* **想同时输出 Y 与 Q**：如果下游只用 fp8，可以删掉 `Y` 以减少一次整行写回。
  这里保留 `Y` 是为了能直接和参考实现对比数值。

## 与上游示例的关系

* 整体结构（RMSNorm → 尾部 RoPE → 写回）对照适配器仓库
  `_vendor/tilelang-ascend-v0.1.2.000/examples/pos_embedding/rms_rope_fused.py`。
* 分组量化的写法与 dtype 处理对照
  `_vendor/tilelang-v0.1.15/examples/cast/example_per_token_cast_to_fp8.py:8-41`
  与 `_vendor/tilelang-v0.1.15/examples/ascend/example_per_token_cast_to_fp8.py:36-86`。

## 作者验证级别

**仅静态核对**：`python3 -m py_compile` 通过；用到的 `T.*` 名字经
`../07-api-audit/api_audit.py` 对照过 `_vendor/tilelang-v0.1.15/` 的语言面。
**未在 GPU 上编译或运行过**：上面的「真机验证步骤」是给读者的路线图，
不是作者已经执行过的记录。
