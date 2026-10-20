# 03-verify-pass-template — verify pass 的模板与它的边界

## 目的

本章 7.4 节说：`verify_buffer_init` 与 `verify_parallel_loop` 是最好的入门模板，
因为它们**不改 IR**——把“写 pass”的风险降到只有“遍历写错”一种。

本目录做两件事：

1. `extract_verify_skeleton.py` 把这两个上游 pass 的骨架连行号打印出来，
   让你对着填，而不用通读几百行实现；
2. `verify_local_alloc_limit.cc` 是照着这套骨架写出来的**第三个** verify pass：
   当某块 on-chip 分配的元素数超过上限时给一条带补救建议的警告。

## 文件

| 文件 | 放到哪里 | 作用 |
| --- | --- | --- |
| `extract_verify_skeleton.py` | 原地运行 | 抽取两个上游 verify pass 的骨架与行号 |
| `verify_local_alloc_limit.cc` | `src/transform/verify_local_alloc_limit.cc` | verify pass 本体 |
| `test_verify_local_alloc_limit.py` | `testing/python/transform/` | 四条边界用例 |

## 运行前提

* `extract_verify_skeleton.py`：Python 3.10+ 与标准库，**任何机器都能跑**。
* C++ 骨架与测试：需要一棵可编译的 TileLang 源码树（步骤同
  `../02-cpp-pass-template/README.md`）。测试不需要 GPU/NPU。

## 运行命令

```bash
# 不需要 tilelang
python3 extract_verify_skeleton.py
python3 extract_verify_skeleton.py --only verify_buffer_init

# 落地后（需要编译好的 TileLang）
python3 -m pytest testing/python/transform/test_verify_local_alloc_limit.py -q
```

## 预期输出（抽取脚本，本机已运行）

```text
=== verify_buffer_init ===
  header comment       : lines 1-16 (16 lines)
  total lines          : 426
  visitor overrides:
    L 196  void VisitStmt_(const BufferStoreNode *op) final {
    L 201  void VisitExpr_(const CallNode *op) final {
    L 296  void VisitStmt_(const EvaluateNode *op) final {
    L 315  void VisitStmt_(const BufferStoreNode *op) final {
    L 323  void VisitExpr_(const BufferLoadNode *op) final {
    L 330  void VisitExpr_(const CallNode *op) final {
  EmitReport           : L372
  pass factory         : L400  Pass VerifyBufferInit()
  CreatePrimFuncPass   : L416  CreatePrimFuncPass(pass_func, 0, "tl.VerifyBufferInit", {})
  FFI block            : lines 419-422
    registered name    : tl.transform.VerifyBufferInit

=== verify_parallel_loop ===
  header comment       : (none)
  total lines          : 170
  visitor overrides:
    L  46  void VisitStmt_(const ForNode *op) override {
    L  57  void VisitStmt_(const BufferStoreNode *op) override {
    L 113  void VisitStmt_(const SBlockNode *op) override {
  EmitReport           : L126
  pass factory         : L153  Pass VerifyParallelLoop()
  CreatePrimFuncPass   : L160  CreatePrimFuncPass(pass_func, 0, "tl.VerifyParallelLoop", {})
  FFI block            : lines 163-166
    registered name    : tl.transform.VerifyParallelLoop
```

（`verify_buffer_init.cc` 里有两个 `VisitStmt_(BufferStoreNode*)`、两个
`VisitExpr_(CallNode*)`：因为文件里有两个 visitor——`PotentialWriteCollector`
与 `BufferInitVerifier`。看到重名不要去重，先看清它们属于哪个 struct。）

## verify pass 的三条设计规则（来自上游的写法）

1. **只报不改**：`pass_func` 原样返回 `f`。这样它挂进任何 pipeline 都不会改变
   编译结果，出问题时责任边界清楚。
2. **聚合报告**：把发现收进 `reports_`，最后合成一条 `LOG(WARNING)`。
   `verify_buffer_init` 甚至在报告里数出 `N buffer(s)`。
3. **给补救办法**：每条发现都写清怎么改。`verify_buffer_init` 会对
   gemm 累加器专门给 `T.clear(C)` / `clear_accum=True` 的提示（`Kind::kGemmAccum`），
   而不是所有发现共用一句话。

外加一条纪律：**折不动就不报**。动态边界证明不了的东西报了就是假警报，
而假警报会让使用者把整条检查关掉——那这个 pass 就白写了。

## 测试的四条边界（`test_verify_local_alloc_limit.py`）

上界、上界加一、动态形状、不该管的 scope——这四类正是 verify pass 最容易写错的
地方，也正是本章 5.4 节要求的一定要覆盖的边界。

## 在这台机器上达到的验证级别

* `extract_verify_skeleton.py`：**CPU 逻辑验证**。作者在本机（macOS / Apple
  Silicon，无 tilelang）运行过，上面的预期输出就是它的真实输出。
* `verify_local_alloc_limit.cc` 与测试：**仅静态核对**。从未编译、从未运行。
  骨架里的每一块都能在 `_vendor/tilelang-v0.1.15/` 里找到对应行（见文件头注释）。
