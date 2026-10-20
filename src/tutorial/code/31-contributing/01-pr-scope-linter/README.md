# 贡献路径判定器（pr_scope_linter.py）

## 目的

一个 PR“该跑哪些测试”不由作者的心情决定，而由**它改了哪一类文件**决定。
这个脚本把改动路径映射到锁定版本仓库里实际存在的门禁，输出一张清单，
避免两种最常见的错误：

* 改了编译器却只跑了单元测试，没进 `testing/python/transform/`；
* 改了昇腾路径却按 CUDA 的套路跑测试（或反过来）。

## 运行前提

* Python 3.10+（只用到标准库）。
* 在 TileLang 仓库里执行，或者把改动路径作为参数传进来。
* 不需要 GPU / NPU，不需要安装 TileLang。

## 运行命令

```bash
# 用参数
python3 pr_scope_linter.py tilelang/language/copy_op.py src/transform/lower_tile_op.cc

# 用 git 输出（推荐）
git diff --name-only origin/main...HEAD | python3 pr_scope_linter.py -

# 自测（不带外部输入）
python3 pr_scope_linter.py --self-test
```

## 输出说明

脚本对每个改动路径给出：路径、判定类别、需要跑的门禁，以及该门禁在锁定版本
仓库里的定义位置。最后打印一份“合并后的门禁集合”。

## 期望输出（自测用例的判据）

`--self-test` 必须打印 `self-test passed` 并以退出码 0 结束。它内部断言了三条
已知映射，任一条不成立就报错退出：

1. `examples/softmax/example_softmax.py` → 算子/示例路径，需要示例测试门禁；
2. `src/transform/lower_tile_op.cc` → 编译器路径，需要 transform 测试门禁；
3. `tilelang/ascend/pipeline.py` → 昇腾路径，需要 Ascend 测试门禁与双后端说明。

## 验证级别

**在 CPU 上验证过逻辑**：`python3 -m py_compile pr_scope_linter.py` 通过；
`python3 pr_scope_linter.py --self-test` 通过。没有在真实硬件上运行过，也不需要。

## 已知限制

* 判定是**路径前缀启发式**，不解析 C++/Python 语义。一个 PR 同时改多棵树时，
  脚本会把两类门禁都列出来，这是有意为之（宁可多跑）。
* “必须双后端验证”这类昇腾规则写在昇腾仓库的 PR 模板里，脚本只能提醒
  “去核对模板”，无法替人判定。
