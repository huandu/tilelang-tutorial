# 图 2：一个 C++ pass 的骨架解剖（自绘）

以 `src/transform/verify_buffer_init.cc` 为例，一个 pass 的骨架是**六块**。
下面每一块都标了它在文件里的行号，`code/03-verify-pass-template/extract_verify_skeleton.py`
可以把同样的清单现场打印出来。

```text
src/transform/verify_buffer_init.cc                                (426 行)

① 文件头注释（L1-16）
   "Warn when a non-global-scope buffer is read before anything writes it."
   └─ 写清三件事：pass 干什么、为什么需要、精度边界（"deliberately imprecise
      in one direction"）。这是 review 时评审人最先看的地方。

② includes（L18-37）
   "../op/*.h"（复用 tile op 的语义，不自己重写） / "span_utils.h"（行号渲染）
   <tvm/tirx/stmt_functor.h>（StmtExprVisitor/Mutator）/ <tvm/runtime/logging.h>

③ namespace + 别名（L39-44）
   namespace tvm::tl { using namespace tirx; using namespace ffi; }
   └─ "using namespace ffi" 是后面能写 refl::GlobalDef() 的原因。

④ 算法：一个或多个 StmtExprVisitor / StmtExprMutator（L46-396）
   struct PotentialWriteCollector : public StmtExprVisitor   (L189)
   struct BufferInitVerifier     : public StmtExprVisitor   (L216)
     ├─ 状态：reports_ / written_ / params_ / reported_     (L226-235)
     ├─ 辅助：SeedParams / CollectPotentialWrites / CheckRead (L242-294)
     ├─ 重载：VisitStmt_(EvaluateNode*)                     (L296)
     │        VisitStmt_(BufferStoreNode*)                  (L315)
     │        VisitExpr_(BufferLoadNode*)                   (L323)
     │        VisitExpr_(CallNode*)                         (L330)
     └─ 输出：EmitReport() 合成一条 LOG(WARNING)            (L372)

⑤ pass 工厂（L400-417）
   tvm::transform::Pass VerifyBufferInit() {
     auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) { ... };
     return CreatePrimFuncPass(pass_func, 0, "tl.VerifyBufferInit", {});
   }
   └─ 只报不改的 pass 在这里 return f；改 IR 的 pass 在这里 return 新 func。

⑥ FFI 注册（L419-422）
   TVM_FFI_STATIC_INIT_BLOCK() {
     namespace refl = reflection;
     refl::GlobalDef().def("tl.transform.VerifyBufferInit", VerifyBufferInit);
   }
   └─ 字符串必须与 ⑤ 里的 pass 名、Python 侧的 _ffi_api 调用三者一致。
```

## 两条会「静默失败」的地方

1. **重载函数名写错不报错。** `VisitStmt_(const BufferStoreNode *op)` 写成
   `VisitStmt(const BufferStoreNode *op)`，编译通过、pass 也跑，只是永远不生效。
   唯一的防线是测试里有一条**必须命中**的用例（`verify_buffer_init` 有
   `test_warns_when_accumulator_is_uninitialized`）。
2. **注册名与 Python 调用不一致只在运行时暴露。** 报错形态是
   `AttributeError: module ... has no attribute 'X'`，不会在编译期发现。

## 一个 Python pass 的骨架（对照）

`tilelang/transform/add_bufstore_wrapper.py` 只用四块：

```text
imports（L1-5）→ pass 工厂 def AddWrapperForSingleBufStore()（L8）
  → 闭包内一个 pass_fn(func, mod, ctx)（L21）
    → @tirx.functor.mutator class（或直接 ir_transform 双访客）
  → return prim_func_pass(pass_fn, opt_level=0, name="tl.AddWrapperForSingleBufStore")（L159）
```

没有第 ⑥ 块：Python pass 不需要 FFI 注册，它在 `tilelang/transform/__init__.py`
里被直接 import 进 `tilelang.transform` 命名空间。

来源：作者自绘，行号取自 `_vendor/tilelang-v0.1.15/`（2026-10-01 核对）。
