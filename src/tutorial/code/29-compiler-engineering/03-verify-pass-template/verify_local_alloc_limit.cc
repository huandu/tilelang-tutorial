/*!
 * \file verify_local_alloc_limit.cc
 * \brief Warn when an on-chip allocation is too large to be a register array.
 *
 * 这是「verify pass」的第二个模板。它和 02 的报告型 pass 有三个刻意的不同，
 * 正好对应本章 7.4 节讲的 verify pass 三条设计规则：
 *
 *   1. **只报不改**：pass_func 永远返回原 func，检查结果走日志；
 *   2. **聚合报告**：所有发现收集进 reports_，最后合成一条警告，不刷屏；
 *   3. **给补救办法**：每条发现都带上「应该怎么改」，而不只是「你错了」。
 *
 * 骨架来源（都能在 _vendor/tilelang-v0.1.15/ 里核到行号）：
 *   * `src/transform/verify_parallel_loop.cc:28-56`（visitor + For 栈）
 *   * `src/transform/verify_parallel_loop.cc:125-148`（EmitReport 的聚合与措辞）
 *   * `src/transform/verify_buffer_init.cc:216-247`（带补救建议的 Report 结构）
 *   * `src/transform/verify_buffer_init.cc:400-422`（pass 工厂 + 注册）
 *
 * 文件放置位置：``src/transform/verify_local_alloc_limit.cc``。
 *
 * 验证级别：**仅静态核对**，本文件从未编译过。
 */

#include "span_utils.h"

#include <sstream>
#include <string>
#include <vector>

#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {

using namespace tirx;
using namespace ffi;

namespace {

/*! \brief 单个缓冲区的元素数上限；超过就认为它不该待在寄存器里。 */
constexpr int64_t kMaxLocalElements = 1 << 16;

/*! \brief 作用域白名单：这些是 on-chip 分配，其余交给别的 pass 管。 */
bool IsOnChipScope(const String &scope) {
  const std::string s = scope;
  return s == "local" || s == "local.fragment" || s == "local.var" ||
         s.rfind("local.", 0) == 0;
}

/*!
 * \brief 把形状折成静态元素数；任一维不是常量就返回 -1。
 *
 * 「折不动就不报」是 verify pass 的一条通用纪律：动态形状本机证明不了，
 * 报了也是假警报，而假警报会让所有人把这条检查关掉。
 */
int64_t StaticElementCount(const Buffer &buffer) {
  int64_t total = 1;
  for (const PrimExpr &dim : buffer->shape) {
    const auto *imm = dim.as<IntImmNode>();
    if (imm == nullptr) {
      return -1;
    }
    total *= imm->value;
  }
  return total;
}

struct LocalAllocLimitVerifier : public StmtExprVisitor {
  struct Report {
    Buffer buffer;
    int64_t elements;
    Span span;
  };

  std::vector<Report> reports_;

  void VisitStmt_(const SBlockNode *op) final {
    // TileLang 的 SBlock 用 alloc_buffers 声明本块的分配；这是
    // verify_parallel_loop.cc:113-123 读 annotations 的同一位置。
    for (const Buffer &buffer : op->alloc_buffers) {
      if (!IsOnChipScope(buffer.scope())) {
        continue;
      }
      const int64_t elements = StaticElementCount(buffer);
      if (elements >= 0 && elements > kMaxLocalElements) {
        reports_.push_back({buffer, elements, op->span});
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  /*! \brief 全部发现合成一条警告；报告里带文件行号与补救建议。 */
  void EmitReport() const {
    if (reports_.empty()) {
      return;
    }
    std::ostringstream os;
    os << "On-chip allocation too large: " << reports_.size()
       << " buffer(s) exceed " << kMaxLocalElements << " elements\n";
    for (size_t index = 0; index < reports_.size(); ++index) {
      const Report &report = reports_[index];
      os << "  [" << index + 1 << "] `" << report.buffer->name << "` ("
         << report.buffer.scope() << ") holds " << report.elements
         << " elements" << SpanHintSuffix({report.span}) << "\n"
         << "      Keep register arrays small; move this buffer to `shared` "
            "(or a smaller tile) and copy in chunks.\n";
    }
    os << "If this is intentional, disable the check with "
          "`PassConfigKey.TL_DISABLE_LOCAL_ALLOC_LIMIT_CHECK`.";
    LOG(WARNING) << os.str();
  }
};

} // namespace

using namespace tirx::transform;

tvm::transform::Pass VerifyLocalAllocLimit() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    // 有开关的检查照这个模式写：先读 config，被关掉就原地返回。
    if (ctx->GetConfig(kDisableLocalAllocLimitCheck, Bool(false)).value()->value) {
      return f;
    }
    LocalAllocLimitVerifier verifier;
    verifier(f->body);
    verifier.EmitReport();
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.VerifyLocalAllocLimit", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.VerifyLocalAllocLimit",
                        VerifyLocalAllocLimit);
}

} // namespace tvm::tl
