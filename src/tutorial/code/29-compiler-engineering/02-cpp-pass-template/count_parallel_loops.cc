/*!
 * \file count_parallel_loops.cc
 * \brief Report every parallel loop in a PrimFunc, with its static extent.
 *
 * 这是一个**教学用的 C++ pass 模板**：它不改 IR（返回原 func），只做一次
 * 遍历并把结果打成一条聚合警告。这样写有三个好处：
 *
 *   1. 把「C++ pass 的全部零件」压到最短——visitor、pass func、注册宏，
 *      没有算法噪声；
 *   2. 它天然是幂等的，挂进任何 pipeline 都不会改变编译结果，
 *      可以安全地先用它把工具链跑通；
 *   3. 它与锁定版本里的两个 verify pass（verify_buffer_init /
 *      verify_parallel_loop）是同一个骨架，学会它再去看那两个文件
 *      就只剩算法差异了。
 *
 * 文件放置位置：``src/transform/count_parallel_loops.cc``。
 * 骨架来源：``src/transform/verify_parallel_loop.cc:19-24,28-56,151-166``
 * 与 ``src/transform/verify_buffer_init.cc:39-44,216-247,398-422``。
 *
 * 验证级别：**仅静态核对**。本机没有编译环境，本文件从未被编译过。
 * 构建与测试步骤见同目录 README.md。
 */

#include "span_utils.h"

#include <sstream>
#include <string>
#include <vector>

#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {

// 和上游 verify pass 一样：在 tvm::tl 里打开 tirx 与 ffi 两个命名空间，
// 后面写 StmtExprVisitor / CreatePrimFuncPass / reflection 都不用再加限定。
using namespace tirx;
using namespace ffi;

namespace {

/*!
 * \brief 统计所有 ``For(kind=kParallel)`` 节点，收集它们的静态 extent。
 *
 * ``StmtExprVisitor`` 是 TVM 的只读遍历基类：不改 IR，只做观察。
 * 需要改 IR 时换成 ``StmtExprMutator``（对应 Python 侧的
 * ``PyStmtExprMutator``），返回新的节点而不是原地改。
 *
 * visit 方法的重载规则：``VisitStmt_(const ForNode *op)`` 的命名必须是
 * ``VisitStmt_`` + ``<NodeType>Node``；写错名字不会报错，它只会永远不被调用，
 * 这是 C++ pass 最常见的一类静默失败。
 */
struct ParallelLoopReporter : public StmtExprVisitor {
  struct LoopInfo {
    Span span;
    /*! \brief 静态 extent；动态 extent 记为 -1。 */
    int64_t extent;
  };

  std::vector<LoopInfo> loops_;

  /*! \brief 把 extent 折成整数；折不动就返回 -1（动态边界）。 */
  static int64_t StaticExtent(const PrimExpr &extent) {
    if (const auto *imm = extent.as<IntImmNode>()) {
      return imm->value;
    }
    return -1;
  }

  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kParallel) {
      loops_.push_back({op->span, StaticExtent(op->extent)});
    }
    // 必须先记录再递归，否则嵌套并行循环的顺序会反过来。
    StmtExprVisitor::VisitStmt_(op);
  }

  /*! \brief 全部结果合成一条警告，避免刷屏。 */
  void EmitReport() const {
    if (loops_.empty()) {
      return;
    }
    std::ostringstream os;
    os << "Parallel loops: " << loops_.size() << " parallel loop(s)\n";
    for (size_t index = 0; index < loops_.size(); ++index) {
      os << "  [" << index + 1 << "] extent=";
      if (loops_[index].extent >= 0) {
        os << loops_[index].extent;
      } else {
        os << "<dynamic>";
      }
      // SpanHintSuffix 来自 span_utils.h，会把 span 渲染成 "--> file:line:col"。
      os << SpanHintSuffix({loops_[index].span}) << "\n";
    }
    LOG(WARNING) << os.str();
  }
};

} // namespace

using namespace tirx::transform;

tvm::transform::Pass CountParallelLoops() {
  // pass_func 的签名是固定的：接收 (PrimFunc, IRModule, PassContext)，
  // 返回一个 PrimFunc。返回新 func 就是改 IR，原样返回就是只观察。
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    ParallelLoopReporter reporter;
    reporter(f->body);
    reporter.EmitReport();
    return f;
  };
  // 参数依次是：pass_func、opt_level、报告里显示的名字、required passes。
  return CreatePrimFuncPass(pass_func, 0, "tl.CountParallelLoops", {});
}

// TVM 的注册块。注册名 "tl.transform.CountParallelLoops" 是**契约**：
// Python 侧的 tilelang/transform/_ffi_api.py 用
// tvm_ffi.init_ffi_api("tl.transform", __name__) 按这个前缀把函数挂成属性，
// 所以 _ffi_api.CountParallelLoops 才能被调用。名字对不上就 AttributeError。
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.CountParallelLoops", CountParallelLoops);
}

} // namespace tvm::tl
