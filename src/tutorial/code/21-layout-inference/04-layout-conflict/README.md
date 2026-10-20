# 04 · 故意制造一次布局冲突

## 目的

对应第 21 章 §3.3 与 §7.5。对 GEMM 的 `A_shared` 用 `T.annotate_layout` 手工指定
一个与 swizzle 不兼容的布局，观察编译器走哪条出路：

* 包含升级？swizzle 合并？还是 `LOG(FATAL) << "Get different layout for ..."`
  （`_vendor/tilelang-v0.1.15/src/transform/layout_inference/layout_inference.cc:437-468`）？

脚本默认只构造 kernel；真正的冲突要**编译**才会触发，而且必须有 GPU。

## 运行前提

* NVIDIA GPU + CUDA 驱动 + 工具链 + PyTorch(CUDA) + TileLang `v0.1.15`。

## 运行命令

```bash
python3 annotate_layout_conflict.py --mode transpose   # 纯转置：多半硬冲突
python3 annotate_layout_conflict.py --mode plain       # 不 annotation：作对照
```

## 预期输出（**思路，不是实测**）

* `--mode transpose`：编译阶段报错，信息里含 `Get different layout for`，
  指向 `A_shared`。
* `--mode plain`：正常编译（与 `02`/`03` 的 GEMM 相同结构）。

**把这些当假设，不要当结论。** 实际是“通过 / 合并 / 失败”三者之一，取决于
v0.1.15 在该指令下的 `reuse_existing_shared_layout` 判定
（`_vendor/tilelang-v0.1.15/src/op/gemm.cc:254-266`）。

## 验证级别

**仅静态核对**：`python3 -m py_compile` 通过；`T.annotate_layout`
（`tilelang/language/annotations.py:45-62`）与 `T.Layout` 的签名已核对。
**本机没有 GPU，脚本未运行，“预期输出”未经证实。**
