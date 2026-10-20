# 第 27 章 代码

本章的代码分两类：

1. **可以在纯 CPU 上跑的数值实验**（`01`–`05`）。它们把正文里的数值推导
   （格式范围、量化误差、误差预算、scale 布局）变成可复算的输出。**这五个脚本的
   所有数字都来自实际运行，不是从别处抄的**；这是本章能给的最强验证级别。
2. **锁定版本的静态核对**（`06`）。它把「正文提到的源码事实」变成 44 条可跑的
   断言，覆盖 `_vendor/` 两个仓库与 corpus 里的 PTX / CATLASS 文档。

**没有任何一个示例需要 GPU 或 NPU**，也没有任何一个示例在加速器上运行过。
本章不出现「本机实测性能」——凡涉及硬件行为的结论，来源都是锁定版本的源码或
corpus 中的厂商文档，`references.md` 里逐条给出路径。

## 目录

```text
code/
├─ README.md                                  本文件
├─ 01-number-formats/
│  ├─ README.md
│  └─ number_formats.py                       枚举位模式，算 u / 范围 / E8M0 值域
├─ 02-quant-dequant/
│  ├─ README.md
│  └─ quant_dequant.py                        int8/int4/MXFP8/MXFP4 的往返误差
├─ 03-error-model/
│  ├─ README.md
│  └─ error_model.py                          误差预算 u/sqrt(K) 与累加器精度
├─ 04-vs-reference/
│  ├─ README.md
│  └─ reference_matching.py                   四种参照，其中两种会导致误判
├─ 05-mx-scale-layout/
│  ├─ README.md
│  └─ mx_scale_layout.py                      scale 打包/解包的四种典型错误
└─ 06-tilelang-audit/
   ├─ README.md
   └─ tilelang_audit.py                       44 条源码事实的可复现检查
```

## 运行前提

* Python 3.9+；`01`–`05` 需要 numpy（写作机用 3.11.7 + numpy 1.24.1），
  `06` 只用标准库。
* 不需要 GPU / NPU / PyTorch / TileLang。
* `06` 需要锁定版本的源码在 `_vendor/` 下；找不到时脚本打印 `skipped` 并以 0 退出。

## 运行命令

```bash
# 数值实验（在各自的目录下运行；互不依赖，可以单独复制出去）
python3 code/01-number-formats/number_formats.py --show-fp4
python3 code/02-quant-dequant/quant_dequant.py
python3 code/03-error-model/error_model.py
python3 code/04-vs-reference/reference_matching.py
python3 code/05-mx-scale-layout/mx_scale_layout.py

# 源码事实核对（在 code/06-tilelang-audit/ 下运行）
python3 tilelang_audit.py
```

## 各脚本的验证级别

| 目录 | 依赖 | 验证级别 | 作者实际做到的事 |
| --- | --- | --- | --- |
| `01-number-formats` | numpy | **CPU 逻辑验证** | 在写作机上跑通，退出码 0；输出见其 README |
| `02-quant-dequant` | numpy | **CPU 逻辑验证** | 同上 |
| `03-error-model` | numpy | **CPU 逻辑验证** | 同上 |
| `04-vs-reference` | numpy | **CPU 逻辑验证** | 同上 |
| `05-mx-scale-layout` | numpy | **CPU 逻辑验证** | 同上 |
| `06-tilelang-audit` | 标准库 | **CPU 逻辑验证**（对 `_vendor/` 的文本核对） | 44/44 通过；这只证明「正文引用的字符串确实在文件里」，不证明语义正确 |

## 这些脚本证明不了什么

* 它们**不验证**任何 TileLang kernel 的行为：没有编译、没有执行、没有性能。
* `03` 的误差模型是**均方根量级模型**，不是某个 kernel 的误差保证。真实误差
  还取决于分块顺序、split-K、epilogue 的 cast，这些必须在有卡机器上实测。
* `06` 是**启发式文本检查**：命中只说明那句话在锁定版本里存在，不说明我们理解对了。
