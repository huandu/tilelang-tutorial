# 01-route-map —— 四条昇腾编译路线的静态抽取

## 目的

“昇腾后端”在锁定版本里不是一个东西，而是**四条可执行的编译路线**（M 树一条，
A 树两条，MLIR 仓库一条）。本脚本只读地扫描三棵树的源码，把每条路线的
**入口 / codegen 目标 / 生成什么 / 谁来编译** 抽成一张表，并对每条结论做
“锚点字符串是否存在”的断言。

## 运行前提

* Python 3.10+，无第三方依赖；
* `_vendor/tilelang-v0.1.15/`、`_vendor/tilelang-ascend-v0.1.2.000/` 与
  `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-mlir-ascend/` 存在。

## 运行命令

```bash
python3 route_map.py
# 也可以显式指定书根：
python3 route_map.py --book-root /path/to/tilelang-book
```

## 期望输出

```
route  tree                 entry                                  codegen target / artifact      compiler
-----  -------------------  -------------------------------------  -----------------------------  ---------------------------
M1     tilelang v0.1.15     target="ascend" | "asc"                AscendC(asc_*) CCE 源          bisheng --npu-arch=dav-3510
...
[ok]  4 routes, 0 anchor misses
```

## 验证级别

**CPU 逻辑验证**：本机（macOS / Apple Silicon，无 NPU）跑过，输出见正文 §3.3 与 §4.8–4.9。
脚本不导入 `tilelang`，也不调用任何编译器，**不构成运行验证**。
