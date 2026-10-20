# 示例 4：看一个 pass 前后的变化（IR Lower Trace / Pass Diff）

## 目的

「某个 lowering 阶段之后 IR 就错了」是 TileLang 里最常见的生成类问题。
`gemm_lower_trace.py` 用同一个 GEMM 演示三种看法：

| 模式 | 入口 | 什么时候用 |
| --- | --- | --- |
| `trace-env` | `TL_LOWER_TRACE=1` | 想一次看到**整条 pipeline** 每个 pass 的前后 IR + 最终 codegen |
| `trace-api` | `lower_trace(func, passes)` | 只想固定比较指定的几个 pass |
| `pass-diff` | `pass_diff(func, passes)` | 维护老代码；新代码请改用 IR Lower Trace |

关键时序：`TL_LOWER_TRACE` 是在 **`import tilelang` 时**被读取的，所以脚本先解析参数、
设置环境变量，再导入 tilelang。

## 运行前提

* `--static-only`：任意机器。
* `trace-env` / `trace-api` / `pass-diff`：需要装了 tilelang 的环境；`--target cuda`
  只需要 CUDA codegen FFI，不需要 GPU（因为 `tilelang.lower()` 默认不做 device 编译）。

## 运行命令

```bash
python3 gemm_lower_trace.py --static-only       # 任何机器
python3 gemm_lower_trace.py --mode trace-env    # 写 HTML 报告
python3 gemm_lower_trace.py --mode trace-api
python3 gemm_lower_trace.py --mode pass-diff

# 直接指定输出目录
TL_LOWER_TRACE_DIR=/tmp/tl_trace python3 gemm_lower_trace.py --mode trace-env
```

## 期望输出

`--static-only`（写作机实际输出，退出码 0）：

```
py_compile: ok
需要与锁定版本核对的入口（逐个在 _vendor/tilelang-v0.1.15/ 下确认）：
  TL_LOWER_TRACE / TL_LOWER_TRACE_DIR    tilelang/env.py:390-392
  lower_trace()                          tilelang/tools/lower_trace/__init__.py:49
  enable() / disable()                   tilelang/tools/lower_trace/core.py:663-705
  pass_diff()                            tilelang/utils/pass_diff.py:1030
  transform.Simplify                     tilelang/transform/simplify.py:20
（本机未安装 tilelang：这里只核对语法与入口存在性，不生成任何报告。）
```

`--mode trace-env` 会生成（目录结构来自官方 lower_trace 文档）：

```
./tmp/lower_trace_dir/
└── gemm_lower_trace/                  # 由 sys.argv[0] 推导
    ├── report.html                    # 稳定符号链接 → 最近一次报告
    ├── codegen.cpp                    # 生成的源码（可编辑后重编）
    ├── codegen.cpp.original           # 基线快照
    ├── codegen.cpp.latest             # 最近一次 codegen 输出
    └── .run_records/
        └── run_<时间戳>_<pid>_<id>/
            ├── report.html
            ├── pipeline_c/
            │   ├── 00_BindTarget_before.tir
            │   ├── 00_BindTarget_after.tir
            │   └── ...
            └── codegen/
                ├── 42_codegen_before.tir
                └── 42_codegen_after.cpp
```

`--mode trace-api` 会打印每个 pass 的 `changed` 与增删行数，格式如下（**下面的数值是占位，
必须以你在自己环境里的实际输出为准**）：

```
pass=<pass 名>     changed=<True/False> +<新增行数> -<删除行数>
注意：文本变了不等于语义或性能变了——它只是一个可定位的线索。
```

**判据**：`trace-env` 生成 `report.html` 且能打开；`trace-api` / `pass-diff` 打印出每个
步骤的 `changed` 与增删行数而不抛异常。

## 注意事项

* 捕获 IR、每个 pass 后重写 HTML 都会增加编译开销与磁盘占用，**只在调试时开**。
* 编辑 `codegen.cpp` 后重跑可以把你手改的源码注回编译（PATCHED），但这一功能要求
  source-compiling 执行后端（`nvrtc` / `cython` / `cutedsl`）；默认 `tvm_ffi` 后端是
  预编译成二进制的，你的编辑只会被记录，不会被重编。
* IR Lower Trace 比较的是 `IRModule.script()` 的**文本**。文本变了不自动等于语义或性能
  变了。

## 作者实际做到的验证级别

**仅静态核对**：`py_compile` 通过，5 个入口逐个核对；**本机未安装 tilelang，
未生成过任何报告**。上面目录结构来自官方 lower_trace 文档，不是本机运行结果。
