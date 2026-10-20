# 第 29 章 习题答案

每题给出**解题思路 → 参考答案 → 常见错误**。涉及的行号均可在
`_vendor/tilelang-v0.1.15/`（必要时 `_vendor/tilelang-ascend-v0.1.2.000/`）里现场
核对；核验日期 2026-10-01。

---

## E1

**思路**：回到 2.2 节的四问（需要 C++ 独有语义？改动频繁？多后端复用？编译器耗时
敏感？），逐个判断，重点是**给依据**而不是给结论。

**参考答案**：

| pass | 属于 | 依据 |
| --- | --- | --- |
| `HoistBroadcastValues` | **Python** | 文件在 `tilelang/transform/hoist_broadcast_values.py`；它只重写 `Broadcast`/`BufferStore`/`Bind`，不需要 tile op 或 layout 对象 |
| `LayoutInference` | **C++** | `src/transform/layout_inference/layout_inference.cc`；要访问 layout 对象、op 的 `InferLayout`、`BufferRegion`，这些接口只在 C++ 侧存在 |
| `AddWrapperForSingleBufStore` | **Python** | `tilelang/transform/add_bufstore_wrapper.py`；纯结构改写（把裸 store 包进 `For(PARALLEL)`），用 `ir_transform` 双访客就能表达 |
| `VFChecker` | **Python** | `tilelang/ascend/analysis/vf_checker.py:173`，`prim_func_pass` 工厂在 `:186`；被 `tilelang/ascend/pipeline.py:95` 直接调用 |
| `z3_schedule_python` | **Python 实现的回调，不是 pass** | 用 `@tvm_ffi.register_global_func("tl.transform.z3_schedule_python")`（`z3_scheduler.py:228`）注册；调用方是 C++ 的 `AutoSchedule`，它是算法库，不产生 `tvm.transform.Pass` |

「搬家会失去什么」：

* `LayoutInference` 搬去 Python → 要重新暴露 layout 对象与 `InferLayout` 接口，
  等于把 C++ 的抽象层在 Python 里重建一遍，且性能（编译器自身耗时）会退化。
* `HoistBroadcastValues` / `AddWrapperForSingleBufStore` 搬去 C++ → 每次小改都要
  重编扩展、重建 wheel，迭代成本显著上升；它们的 IR 改写规则简单，用 Python
  表达反而更清楚。
* `VFChecker` 搬去 C++ → 失去「快速迭代检查规则」这一优势，而它恰恰是容易调整
  的启发式规则。
* `z3_schedule_python` 搬去 C++ → Z3 的 Python 绑定与建模代码很难等价重写，
  而且这个位置本来就是故意留给 Python 的（第 23 章）。

**常见错误**：只答「C++ 快所以写 C++」。判断依据必须落到**接口可达性**与**迭代
成本**上，否则遇到 `VFChecker` 这种「C++ 更快但依然写在 Python」的例子就没法解释。

---

## E2

**思路**：把 2.3 节的「三处一致」逐处对照，问「哪一处坏了会报错、哪一处坏了不报错」。

**参考答案**：

1. **编译不会报错。** 运行时也不会报错：`_ffi_api.Bar()` 能找到注册名
   `"tl.transform.Bar"`（前缀 `tl.transform` 被 `init_ffi_api` 吃掉，余下 `Bar`
   成为属性），所以 Python 侧一切正常。
2. 表现正常的是**功能本身**：pass 被正确调用、IR 被正确改写、测试全绿。麻烦在于
   **它的身份变了**：`CreatePrimFuncPass(pass_func, 0, "tl.Foo", {})` 里的字符串是
   `TL_LOWER_TRACE` / `TL_PASS_PROFILE` 报告显示的名字，也是排查问题时你用来定位
   的标签。它现在显示成一个已经不存在的名字，等于在调试时给了你一条错线索。
3. 对应三处里的**第 2 处**（`CreatePrimFuncPass` 的第三个参数）。它最不容易被发现，
   因为它既不影响注册（注册用的是第 1 处的字符串），也不影响调用（调用走
   `_ffi_api` 属性），只在**报告文本**里出现。
4. 适配器仓库写的是 `tvm._ffi._init_api("tl.transform", __name__)`
   （`_ffi_api.py:8`），v0.1.15 写的是 `tvm_ffi.init_ffi_api("tl.transform", __name__)`
   （`:6`）。C++ 注册侧要从旧的 `TVM_REGISTER_GLOBAL("tl.transform.X")`
   `set_body_typed(...)` 改成 `TVM_FFI_STATIC_INIT_BLOCK()` +
   `refl::GlobalDef().def("tl.transform.X", X)`，并且要确认 `using namespace ffi;`
   在作用域内（否则 `reflection` 找不到）。

**常见错误**：以为「注册名和 Python 名字对上就行」。事实上 pass 有**三个**
可被外部看到的标识：注册名（调用用）、`CreatePrimFuncPass` 的名字（报告用）、
Python 包装函数名（用户的 API）。三者可以各自不同，这正是出错空间所在。

---

## E3

**思路**：把代码注释（`verify_buffer_init.cc:305-314` 与 `:183-188`）读透，
再用测试用例反推。

**参考答案**：

1. 之后 insert 保证「**store 先求值、再建立**」这条语义：写在右边的表达式会先被
   读一遍，所以 `x[i] = x[i] + 1` 里的 `x[i]` 是一次**真实读取**。如果先 insert，
   这次读取会被自己的写入「担保」，未初始化的 `x` 就漏报了。
2. `active_store_` 记录「当前正在访问其子表达式的 store」。`VisitExpr_(
   BufferLoadNode*)` 把它作为 `reader` 传给 `CheckRead`；对于跨线程作用域
   （shared），判定规则是「是否有**另一个节点**写过」，而这个 store 自己显然不算
   「另一个节点」。没有 `active_store_`，`s[i] = s[i] + A[i]` 这类自读自写在 shared
   作用域下会因为「有节点写 s」（就是它自己）而被放过，与非 shared 作用域行为不一致。
3. 会**失败**。该用例断言 `MESSAGE in err`；顺序反过来后，store 先把自己标成
   written，随后的自读命中 `written_.count(var) > 0`，于是不报告，断言不成立。
4. 「一个方向」指的是**宁可漏报、不误报**：注释说「a write anywhere earlier in
   execution order counts, even under a conditional the read is not guarded by」，
   也就是只要能证明「有写过」就放过，哪怕那次写在某条不会执行的分支里。这个取舍
   让假阳性足够罕见，检查才敢**默认打开**。代价是：条件性未初始化这类真问题它抓
   不到。

**常见错误**：把 `active_store_` 说成「性能优化」。它是**正确性**工具，不是为了
省遍历。

---

## E4

**思路**：`ir_transform(body, pre_visit, post_visit)` 是「前序 + 后序」双访客；
`tile_operation_depth` 是一个用「进入/离开」配平来表示嵌套层数的计数器。

**参考答案**：

1. `pre_visit` 在前序阶段 `+= 1`，`post_visit` 在后序阶段 `-= 1`，两者配平才能让
   `tile_operation_depth` 表示「当前处于第几层 tile op 循环内部」。只在 `pre_visit`
   自增的话，深度会单调上升、永不归零；第一个 tile op 循环之后，所有 store 都会
   被判定为「在 tile op 内部」而跳过改写——**pass 静默失效**。
2. `thread_binding_vars` 收集被 `AttrStmt("thread_extent")` 绑定的变量，也就是
   `threadIdx.*` 这类线程域变量。一个 store 的索引里出现线程域变量，说明它本来就是
   按线程写的；再给它套一层 `For(PARALLEL, 0, 1)` 既没有意义，也可能与后续
   `LayoutInference` 的假设冲突，因此命中即跳过。
3. 差别会出现在 **`TL_LOWER_TRACE`（IR Lower Trace）** 与 **`TL_PASS_PROFILE`**
   的报告里：带 `name="tl.AddWrapperForSingleBufStore"` 的 pass 会以这个名字出现；
   不带 name 的 `HoistBroadcastValues` 只能由工具自动推断显示名。
   排查编译问题时，「报告里认不出这个 pass」会显著拖慢定位。
4. 因为它要把**裸 store**（还没被任何循环包住的 store）包进 `For(PARALLEL)`，而
   之后的 pass 都假设 store 已经在并行循环里（`VectorizeLoop`、`LayoutInference`、
   `StorageRewrite` 等都按并行循环来组织工作）。晚一步挂，就再也找不到裸 store。

**常见错误**：认为 `nonlocal` 只是「让变量可写」。这里更关键的是**状态的生命周期**：
它必须跨越 `pre_visit` 与 `post_visit` 两次调用，所以只能存在闭包里，不能是局部
变量。

---

## E5

**思路**：E4 的 `tile_operation_depth` 是「按层数计数」，E5 要的是「按类型计数」，
用一个 `visitor` 反而比双访客更简单。直接照 `code/01-python-pass/` 的结构写。

**参考答案（关键代码）**：

```python
from tvm import tirx
from tvm.tirx import PrimFunc, PyStmtExprVisitor
from tvm.tirx.transform import prim_func_pass


@tirx.functor.visitor
class SerialLoopCounter(PyStmtExprVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def visit_for_(self, op):
        if op.kind == tirx.ForKind.SERIAL:
            self.count += 1
        self.visit_stmt(op.body)


def CountSerialLoops():
    def pass_fn(func: PrimFunc, mod, ctx) -> PrimFunc:
        counter = SerialLoopCounter()
        counter.visit_stmt(func.body)
        # 报告型 pass 原样返回 func；这里用 print 只是示意，正式实现应走 LOG
        print(f"serial loops: {counter.count}")
        return func

    return prim_func_pass(pass_fn, opt_level=0, name="tl.CountSerialLoops")
```

测试（两条）：

```python
def test_counts_serial(capfd):
    # 手搓一个含 T.serial 的 IRModule，断言 stderr 里出现 "serial loops: 1"
    ...

def test_silent_without_serial(capfd):
    # 只有 PARALLEL 的 IRModule，断言不出现该文本
    ...
```

为什么写在 Python 侧合理：它只需遍历 `For` 节点、读 `kind`，不碰 tile op、layout
或 `BufferRegion`；这类 pass 迭代频繁（阈值和判定规则会反复调整），写在 Python
可以免去每次重编扩展。

**常见错误**：

* 忘记 `self.visit_stmt(op.body)`，于是只数到最外层循环；
* 用 `print` 做正式输出——正式实现应当用日志（对照上游 verify pass 的
  `LOG(WARNING)`），否则输出无法被 `capfd` 之外的工具捕获，也无法分级过滤。

---

## E6

**思路**：先写规则，再写边界，最后才写代码。规则写不清就一定会写歪。

**参考答案**：

1. 规则一句话：*对每个 `local.fragment` 作用域的 buffer，若它的写入点集合非空，
   但在整个函数体内没有任何一次「不是同一条 store 自身」的读取，则报告。*
   精度边界：**不做别名分析、不做跨线程分析**——`fragment` 是每线程私有的，
   所以只看源码顺序；动态形状不做（折不动就不报）；被 `call_extern` 之类不透明
   call 带走的 buffer 视为「读过了」，不报（保守放过，降低误报）。
2. 五条边界（每条给「为什么」）：

   | 边界 | 为什么必须有 |
   | --- | --- |
   | 只写不读，且该 fragment 是 kernel 参数 | 参数由调用方持有，可能被后续代码读，不该报 |
   | 写入后立刻 `T.copy` 到 global | 这是**正常的写出口**，不该报 |
   | 写入在 `if` 分支里 | 条件写入仍然算写；只在「有写无读」才报 |
   | 同一个 buffer 在多条 store 写、一次读 | 只要有一次读就算读过，不该报 |
   | `local.var` / `local` 作用域（不是 fragment） | 不在管辖范围，防止扩面 |

3. C++ 骨架关键部分：

```cpp
struct SingleWriteFragmentVerifier : public StmtExprVisitor {
  struct Report { Buffer buffer; Span span; };
  std::vector<Report> reports_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> written_, read_;

  // 先 visit 再登记写：与 verify_buffer_init.cc:315-321 同一理由。
  void VisitStmt_(const BufferStoreNode *op) final {
    StmtExprVisitor::VisitStmt_(op);
    if (IsFragmentScope(op->buffer.scope())) written_.insert(op->buffer->data);
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    if (IsFragmentScope(op->buffer.scope())) read_.insert(op->buffer->data);
    StmtExprVisitor::VisitExpr_(op);
  }
  void EmitReport() const { /* 聚合，一条 LOG(WARNING)，附补救建议 */ }
};
```

**常见错误**：

* 把「写过但被 `call_extern` 带走」也报出来 → 误报率飙升，检查会被关掉；
* 在 `VisitExpr_(BufferLoadNode*)` 里连索引表达式里的 load 也当读取 → 索引 load
  很常见，几乎必然误报；
* 忘了「先 visit 再登记写」→ 自读自写被自我担保（同 E3）。

---

## E7

**思路**：用 2.5 节的判断表逐条对照，再去 `pipeline.py` 里确认实际顺序。

**参考答案**：

1. **必须放在 `LowerTileOp` 之前**，因为 `T.copy` 的 `src`/`dst` region 只在
   tile op 形式下存在；`LowerTileOp` 之后它们会被降成低层语句与内建调用。
   如果要依赖循环的源码顺序，还必须放在 `PipelinePlanning` /
   `InjectSoftwarePipeline` 之前。
2. **不冲突。** 看 CUDA pipeline 的实际顺序（`tilelang/cuda/pipeline.py`）：
   `PipelinePlanning`(`:129`) → `InjectSoftwarePipeline`(`:130`) →
   `LayoutInference`(`:134`) → `LowerTileOp`(`:142`)。也就是说
   `LayoutInference` 与 `LowerTileOp` **之间**是一个同时满足「tile op 还在」与
   「layout 已知」的窗口。把 pass 放进这个窗口即可。
   如果确实两难（例如必须同时早于 `PipelinePlanning` 又需要 layout），那说明这个
   检查不该做成「一个 pass」——拆成两个：前置收集信息并打 annotation，后置消费。
3. **只要它不是后端特有的，就挂全部 6 条 pipeline**：
   `tilelang/cuda/pipeline.py`、`tilelang/ascend/pipeline.py`、
   `tilelang/cpu/pipeline.py`、`tilelang/metal/pipeline.py`、
   `tilelang/rocm/pipeline.py`、`tilelang/webgpu/pipeline.py`。
   依据：判断它是否用到了某个后端独有的语义（如 TMA、`cthread`、Metal 的
   simdgroup）；只读 `T.copy` region 的检查不属于任何单一后端。
4. 「命令式 + 硬编码顺序」的理由：
   * **可读性**：`pipeline.py` 从上往下读就是执行顺序，读者不需要（也不能）去
     推理一套依赖解析器会怎么重排；
   * **可调试性**：`TL_LOWER_TRACE` 的报告按这个字面顺序编号，因此代码里的第 N 行
     与报告里的第 N 步可以一一对应；如果顺序由求解器决定，报告与实际阅读顺序就对
     不上了。

**常见错误**：只答「放在 LayoutInference 之后」。这回答不完整——「之后」没有上界，
而实际上它还有「不晚于 `LowerTileOp`」这个上界。

---

## E8

**思路**：把规则写成不等式再代数，别凭感觉。

**参考答案**：

1. **不算回归。** 规则是「至少有 `MIN_COUNT`(默认 2) 条 `speedup < 0.90`」。
   这里 `THRESHOLD=0.90`，低于阈值的只有 1 条（`case_A` 0.85x），`1 < 2`，
   因此不判为回归。
2. **结论变了：算回归。** 现在有 2 条（0.85x、0.88x）低于 0.90，`2 >= 2`，
   判为回归。
3. 那段注释的标定依据是：到 PR #3176 为止的 33 份机器人报告、1901 个样本中，
   真回归都表现为 14–23 条同时塌陷；7 份假警报恰好只有一条落在一个极端值上
   （最低仅 0.044x），其余全是 1.000。提「0.044x」是为了说明**单条再极端也不算
   证据**——噪声可以单条失真到离谱，但不会同时让一大批 case 一起变慢。
4. **不会。** 只在一边出现的 case 不进入比较集合（`compare_payloads` 用两边
   结果的交集）。对应开关是 `--fail-on-missing`（缺席）与
   `--fail-on-regression`（变差）两个不同的开关。
5. 本机运行 `python3 code/04-regression-judge/judge_regression.py --self-test`
   的真实输出：

   ```text
   场景 A：40 条里 1 条 0.85x
   **不判为回归**：1 条低于 0.9x，未达到 2 条的门槛。

   场景 B：40 条里 2 条 0.88x
   **回归：2/40 条低于 0.9x**
   ```

   与第 1、2 问一致。

**常见错误**：把「有 case 低于 0.90」直接等同于「回归」。这条规则的存在正是因为
那不等价。

---

## E9

**思路**：严格按图 4 的流程走，每一步先问「这一步能排除什么」。

**参考答案**：

1. 判定这一步：

   ```bash
   FAIL_ON_REGRESSION=1 ./maint/scripts/run_perf_regression.sh -k <你的用例>
   ```

   不能只看手上那一个数字，因为（a）单次测量包含噪声，规则明确要求
   `MIN_COUNT` 条同时越界；（b）你自己那个 case 掉可能是因为机器忙、时钟被
   降频，而不是代码变慢；（c）回归判定的价值在于**可复现且成规模**。本例 40 个
   case 里 6 个掉 10% 以上，已经满足门槛，属于回归。

2. 区分两类变慢：

   ```bash
   # 编译变慢？
   TILELANG_PASS_PROFILE=1 python3 <你的用例>.py
   # 产物（IR / 生成代码）变了？
   TL_LOWER_TRACE=terminal python3 <你的用例>.py
   ```

3. inclusive 涨 3 倍、self 不变 → 成本在**子 pass** 里。下一步：在
   `pass_timing` 报告里顺着 `depth` 往下看，找到 self 时间增长的那一层。常见原因
   是「把你的 pass 里的某个集合/映射从 O(1) 变成 O(N) 查找」，或「改变了 IR 的
   形状，让下游某个 pass 的搜索空间变大」。
4. 说明你的 pass 改变了 IR 的结构，而这个结构正是 `VectorizeLoop` 的工作对象——
   慢的根源很可能不是你的 pass 本身，而是它引起的**下游行为变化**。继续缩小范围：
   把 `TL_LOWER_TRACE` 的 diff 停在你的 pass 与 `VectorizeLoop` 之间，读出具体
   哪几个循环的 vector length 变小了；再用 `TL_LOWER_TRACE=1` 抓 codegen 产物
   （或 `kernel.get_kernel_source()`）确认生成的访存宽度。
5. 用**第 ④ 层 IR 基线**把它固定住：写一条断言「该循环在 `VectorizeLoop` 之后
   vector length 仍为 N」的用例（或对 before/after 做
   `assert_structural_equal`）。理由：性能回归本质上是 IR 形态回归，只有 IR 基线
   能在**不跑 benchmark** 的情况下把它挡住；benchmark 那条留给
   `run_perf_regression.sh`，负责发现没人预料到的回归。

**常见错误**：看到 inclusive 时间变长就直接怀疑自己的 pass 本身。先看 self 再
结论，是最省时间的一步。

---

## E10

**思路**：按 8.4 节的七类关注点逐条检查，重点补上作者最可能漏掉的三类：后端覆盖、
顺序依据、开关四处。

**参考答案**（每条都标出所属关注点）：

1. **后端覆盖**：为什么只挂 `tilelang/cuda/pipeline.py`？这条检查是否只依赖 CUDA
   独有语义？如果不是，`ascend` / `cpu` / `metal` / `rocm` / `webgpu` 五条
   pipeline 是不是也该各加一行？（对照 `VerifyBufferInit` 的 6 处）
2. **位置与顺序**：它挂在哪一行？该行满足「它需要的 IR 形态还在」吗（2.5 节的
   判断表）？需要给出理由，而不是「我加在了文件末尾」。
3. **开关的四处一致性**：新增的 config key 是否同时改了
   `src/op/builtin.h` 常量、`src/op/builtin.cc` 的
   `TVM_REGISTER_PASS_CONFIG_OPTION`、C++ 的 `ctx->GetConfig(...)`、
   `tilelang/transform/pass_config.py` 的 `PassConfigKey`？少一处就是
   「配置不生效但不报错」。
4. **默认开还是默认关**：默认值是什么？写进 docstring 了吗？依据是什么？
   `VerifyBufferInit` 默认开（误报罕见），`tl.disable_data_race_check` 默认关
   （易误报）——两条相反的先例说明这不是随便定的。
5. **测试的边界**：只有一个 IR 基线用例不够。至少要有 5.5 节那种反向用例
   （不被管辖的 scope、不透明的 extern、自读自写、参数 buffer、折不动的情况）。
   另外，`requires_cuda` 的端到端用例要用 `no_kernel_cache` fixture，否则第二次
   编译命中缓存、pass 根本不跑。
6. **命名与风格**：函数用 PascalCase、常量 `kPascalCase`；handle 类型不要存
   `*Node` 指针；比较用 `.same_as`；是否动了反射字段（那属于破坏性改动）。
7. **报告质量**：多个发现是否聚合成一条（对照 `verify_parallel_loop.cc:125-148`）？
   每条是否给了可操作的补救建议（对照 `verify_buffer_init.cc:384-389`）？

**常见错误**：只提「加个测试」。评审真正的价值在于指出**作者不知道自己不知道的
东西**——后端覆盖和 pass config 四处一致性都属于这一类。

---

## E11

**思路**：先在主干里找「做类似事」的 pass，再判断归属；顺序约束要靠 pass 的
docstring 与 pipeline 里的实际位置交叉验证。

**参考答案**：

1. **落在主干的 `src/ascend/transform/`。** 依据：主干里已经有 `insert_oob_padding.cc`
   （Python 包装 `AscendInsertOOBPadding`，`tilelang/ascend/transform/__init__.py:154-163`）
   专门处理「GM→L1 搬运的 OOB 尾部 + padding fill」。新的检查与它同主题，正确的
   做法是**扩展/校验这个 pass**，而不是在适配器仓库另起一套。适配器仓库是另一条
   路线（旧 FFI、另一套 pass 词汇），只有主干里确实没有对应能力时才回去。
2. 两条顺序约束的原因（都写在那个 docstring 里）：
   * **必须在 `InsertNd2Nz` 之后**：`InsertNd2Nz` 会按推断出的 NZ layout 重写
     UB→L1 的拷贝形态（拆成 scatter + post_copy）。在它之前做 OOB 判定，看到的
     还是旧形态的 copy，判据与最终 DMA 的目标不一致。
   * **必须在 `AutoSchedule` 之前**：padding 被表示成语义化的 `T.fill`，而
     `AutoSchedule` 要把它当成一个独立的 MTE2 任务来排；排在调度之后，这个 fill
     就没有任务身份，也就没有和 L1→L0 消费者之间的顺序保证。
3. 入口脚本是 `maint/scripts/run_local_ci_test_ascend.sh`；它跑
   `examples/ascend/` 与 `testing/ascend/` 两个目录。与 CUDA 入口
   （`maint/scripts/run_local_ci_test.sh`）的两点行为差异：**不加载 CUDA 调度插件**
   （`-p pytest_cuda_scheduler`）、**不自动探测 GPU**。
4. 与本章「所有结论必须标验证级别」对应的是 `AGENTS.md:63-66` 的第 1 条
   **「如实报告，禁止伪完成」**——未验证的结果不得表述为「已完成」或「已通过」。
   影响「怎么查 API」的是 `:83-87` 的第 1 条**「不要凭记忆猜 API」**：
   先查 API 速查表，再查 `examples/`，最后查源码
   （`tilelang/language/ascend_tile.py`、`tilelang/language/ascend.py`、
   `testing/python/language/`），并且 `tilelang/language/pto.py` 已废弃、禁止使用。

**常见错误**：默认「昇腾的改动都去适配器仓库」。v0.1.15 已经把昇腾后端收进主干
（`src/ascend/` + `tilelang/ascend/`），新的昇腾能力主线在主干；适配器是一条并行的
历史路线，改它之前要先确认它不是重复实现。
