#!/usr/bin/env python3
"""Verify every source anchor cited by chapter 26 against the locked tree.

Why this exists
---------------
Chapter 26 makes dozens of claims of the form "this happens in
`<file>:<line>`". Prose rots; the tree does not. This script re-checks each
anchor against `_vendor/tilelang-v0.1.15/` (and a few against
`_vendor/tilelang-ascend-v0.1.2.000/`) and fails loudly if a line moved.

It never imports `tilelang`, `torch`, or `tvm`. It only reads text, so it runs
on any machine, with or without a GPU. That is the point: the anchor table in
the chapter is reproducible on hardware-less hosts.

Usage
-----
    python3 verify_runtime_anchors.py            # anchor table + citation scan
    python3 verify_runtime_anchors.py --quiet    # only failures + summary
    python3 verify_runtime_anchors.py --list     # dump the anchor table
    python3 verify_runtime_anchors.py --no-scan  # anchor table only

Two layers of checking
----------------------
1. **Anchor table** (below): a hand-maintained list of the lines the chapter
   leans on, each with the sentence it supports. This catches "line moved".
2. **Citation scan**: every `file.py:123` or `file.py:12-34` that appears
   anywhere in the chapter directory is extracted and checked against the
   locked trees (file exists, line in range, line is not blank). This catches
   typos in ranges the anchor table does not pin individually.

Exit code 0 = every anchor still matches, 1 = at least one moved.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Anchor:
    """One (file, line, expected-substring) triple, with the reason it matters."""

    id: str
    tree: str  # "main" = tilelang-v0.1.15, "ascend" = tilelang-ascend-v0.1.2.000
    relpath: str
    line: int
    needle: str
    why: str


MAIN = "main"
ASCEND = "ascend"

# ---------------------------------------------------------------------------
# The anchor table. Keep `needle` short but unique on its line.
# ---------------------------------------------------------------------------
ANCHORS: tuple[Anchor, ...] = (
    # --- 1. 装饰：@tilelang.jit ---
    Anchor("A01", MAIN, "tilelang/jit/__init__.py", 94, "def compile(", "模块级编译入口"),
    Anchor("A02", MAIN, "tilelang/jit/__init__.py", 164, "return cached(", "compile() 最终走缓存派发"),
    Anchor("A03", MAIN, "tilelang/jit/__init__.py", 270, "class JITImpl", "装饰器返回的对象"),
    Anchor("A04", MAIN, "tilelang/jit/__init__.py", 379, "def get_tir", "取 TIR 的入口"),
    Anchor("A05", MAIN, "tilelang/jit/__init__.py", 394, "def _infer_jit_mode", "lazy/eager 推断"),
    Anchor("A06", MAIN, "tilelang/jit/__init__.py", 466, "def compile(self", "JITImpl.compile"),
    Anchor("A07", MAIN, "tilelang/jit/__init__.py", 519, "def __call__(self", "JITImpl.__call__"),
    Anchor("A08", MAIN, "tilelang/jit/__init__.py", 551, "kernel = self._kernel_cache.get(key", "进程内 kernel 缓存查询"),
    Anchor("A09", MAIN, "tilelang/jit/__init__.py", 588, "def jit(", "jit 装饰器本体"),
    Anchor("A10", MAIN, "tilelang/jit/__init__.py", 640, "prim_func(func, eager_jit=True)", "装饰时就把函数包成 JITFunc"),
    Anchor("A11", MAIN, "tilelang/language/eager/builder.py", 1692, "def parse_args", "参数键 → 缓存键"),
    # --- 2. JITKernel：编译与 adapter ---
    Anchor("A12", MAIN, "tilelang/jit/kernel.py", 41, "class JITKernel", "编译产物的门面"),
    Anchor("A13", MAIN, "tilelang/jit/kernel.py", 140, "adapter = self._compile_and_create_adapter", "编译+建 adapter 一步完成"),
    Anchor("A14", MAIN, "tilelang/jit/kernel.py", 224, "def _compile_and_create_adapter", "同一个 instrumentation session"),
    Anchor("A15", MAIN, "tilelang/jit/kernel.py", 236, "prepare_tvm_ffi_callee_allocated_outputs", "编译期确定输出 ABI"),
    Anchor("A16", MAIN, "tilelang/jit/kernel.py", 288, "def _compile_artifact", "lower 的调用点"),
    Anchor("A17", MAIN, "tilelang/jit/kernel.py", 313, "from tilelang.engine.lower import lower_with_context", "导入 lower 入口"),
    Anchor("A18", MAIN, "tilelang/jit/kernel.py", 324, "def _create_adapter_from_artifact", "按 execution backend 选 adapter"),
    Anchor("A19", MAIN, "tilelang/jit/kernel.py", 342, 'if execution_backend == "tvm_ffi":', "默认 adapter"),
    Anchor("A20", MAIN, "tilelang/jit/kernel.py", 390, 'elif execution_backend == "torch":', '名字叫 torch，实际是 Metal'),
    Anchor("A21", MAIN, "tilelang/jit/kernel.py", 405, 'elif execution_backend == "cutedsl":', "CuTeDSL adapter"),
    Anchor("A22", MAIN, "tilelang/jit/kernel.py", 532, "def get_kernel_source", "看设备源码"),
    Anchor("A23", MAIN, "tilelang/jit/kernel.py", 545, "def get_host_source", "看 host 源码"),
    Anchor("A24", MAIN, "tilelang/jit/kernel.py", 151, "@classmethod", "from_database 重建路径"),
    Anchor("A25", MAIN, "tilelang/jit/abi.py", 16, 'CALLEE_ALLOCATED_OUTPUTS_ATTR = "tilelang_callee_allocated_outputs"', "ABI 决策属性"),
    Anchor("A26", MAIN, "tilelang/jit/abi.py", 39, "def prepare_tvm_ffi_callee_allocated_outputs", "给 PrimFunc 打标"),
    # --- 3. engine.lower ---
    Anchor("A27", MAIN, "tilelang/engine/lower.py", 109, "def lower_to_host_device_ir", "拆 host/device"),
    Anchor("A28", MAIN, "tilelang/engine/lower.py", 134, "PreLowerSemanticCheck", "lower 前的语义检查"),
    Anchor("A29", MAIN, "tilelang/engine/lower.py", 136, "mod = context.lower(mod)", "后端分岔点"),
    Anchor("A30", MAIN, "tilelang/engine/lower.py", 138, "host_mod = tirx.transform.Filter", "按 calling_conv 过滤"),
    Anchor("A31", MAIN, "tilelang/engine/lower.py", 144, "def lower_with_context", "带上下文编译"),
    Anchor("A32", MAIN, "tilelang/engine/lower.py", 190, "codegen_mod = device_codegen", "设备码生成"),
    Anchor("A33", MAIN, "tilelang/engine/lower.py", 216, "def lower(", "独立 lower() 入口"),
    Anchor("A34", MAIN, "tilelang/engine/param.py", 154, "class CompiledArtifact", "编译产物结构"),
    Anchor("A35", MAIN, "tilelang/engine/param.py", 169, "def dump_kernel_params", "params.json 的写法"),
    Anchor("A36", MAIN, "tilelang/engine/semantic_check.py", 21, "def PreLowerSemanticCheck", "四个检查器"),
    # --- 4. 缓存 ---
    Anchor("A37", MAIN, "tilelang/cache/__init__.py", 75, "def cached(", "唯一的缓存入口"),
    Anchor("A38", MAIN, "tilelang/cache/__init__.py", 38, "def _resolve_cache_dispatch", "按 backend module 名派发"),
    Anchor("A39", MAIN, "tilelang/cache/__init__.py", 69, "dispatch_map = _ascend_dispatch_map", "昇腾走另一张表"),
    Anchor("A40", MAIN, "tilelang/cache/kernel_cache.py", 243, "def _generate_key", "缓存键的生成"),
    Anchor("A41", MAIN, "tilelang/cache/kernel_cache.py", 269, "func_binary = func.script(show_meta=True).encode()", "key 的第一项是 TIR 文本"),
    Anchor("A42", MAIN, "tilelang/cache/kernel_cache.py", 140, "def _get_base_key", "key 里的 version/platform 部分"),
    Anchor("A43", MAIN, "tilelang/cache/kernel_cache.py", 169, "def _get_cache_namespace", "命名空间 = version/host-machine"),
    Anchor("A44", MAIN, "tilelang/cache/kernel_cache.py", 286, "def cached(", "KernelCache.cached"),
    Anchor("A45", MAIN, "tilelang/cache/kernel_cache.py", 347, "if not env.is_cache_enabled():", "关缓存的分支"),
    Anchor("A46", MAIN, "tilelang/cache/kernel_cache.py", 395, "kernel = self._load_kernel_from_disk(", "先查磁盘"),
    Anchor("A47", MAIN, "tilelang/cache/kernel_cache.py", 427, "kernel = JITKernel(", "未命中才编译"),
    Anchor("A48", MAIN, "tilelang/cache/kernel_cache.py", 552, "def _save_kernel_to_disk", "staging + 原子 rename"),
    Anchor("A49", MAIN, "tilelang/cache/kernel_cache.py", 629, "os.rename(staging_path, cache_path)", "发布是原子的"),
    Anchor("A50", MAIN, "tilelang/cache/kernel_cache.py", 641, "def _load_kernel_from_disk", "从磁盘重建 kernel"),
    Anchor("A51", MAIN, "tilelang/cache/kernel_cache.py", 807, "def _verify_cache_dir_integrity", "manifest 校验"),
    Anchor("A52", MAIN, "tilelang/cache/kernel_cache.py", 893, "return JITKernel.from_database(", "磁盘命中走 from_database"),
    Anchor("A53", MAIN, "tilelang/cache/binary_cache.py", 112, "def _finalize_key", "二进制缓存键"),
    Anchor("A54", MAIN, "tilelang/cache/cuda_binary_cache.py", 22, "def make_key", "CUDA 二进制键字段"),
    Anchor("A55", MAIN, "tilelang/cache/ascend_binary_cache.py", 17, 'cache_root_dir = "ascend-binaries"', "昇腾二进制目录"),
    Anchor("A56", MAIN, "tilelang/cache/ascend_binary_cache.py", 20, 'binary_format = "aibin"', "昇腾二进制格式"),
    Anchor("A57", MAIN, "tilelang/ascend/backend.py", 37, "cached_binary = AscendBinaryCache.load", "bisheng 之前先查缓存"),
    Anchor("A58", MAIN, "src/cuda/codegen/rt_mod_cuda.cc", 126, "tilelang_callback_cuda_compile", "codegen 末尾调编译回调"),
    # --- 5. host ABI ---
    Anchor("A59", MAIN, "src/transform/make_packed_api.cc", 317, "RequiresPackedAPI(const PrimFunc &func)", "哪些函数不改写"),
    Anchor("A60", MAIN, "src/transform/make_packed_api.cc", 342, "GetCalleeAllocatedOutputIndices", "输出 ABI 的读取"),
    Anchor("A61", MAIN, "src/transform/make_packed_api.cc", 386, "MakePackedAPI(PrimFunc func,", "改写主体"),
    Anchor("A62", MAIN, "src/transform/make_packed_api.cc", 423, "int num_args = static_cast<int>(func_ptr->params.size())", "num_args 的算法"),
    Anchor("A63", MAIN, "src/transform/make_packed_api.cc", 435, 'Var v_self_handle("self_handle"', "新签名的四个形参"),
    Anchor("A64", MAIN, "src/transform/make_packed_api.cc", 473, "MakeAssertEQ(v_num_packed_args, num_args", "参数个数断言"),
    Anchor("A65", MAIN, "src/transform/make_packed_api.cc", 481, "MakeAssertNotNull(v_packed_args", "args 非空断言"),
    Anchor("A66", MAIN, "src/transform/make_packed_api.cc", 713, "if (func_ptr->buffer_map.count(param))", "buffer 参数绑定"),
    Anchor("A67", MAIN, "src/transform/make_packed_api.cc", 721, "if (use_callee_allocated_output_abi)", "callee 分配分支"),
    Anchor("A68", MAIN, "src/transform/make_packed_api.cc", 777, "binder.BindDLTensors(", "DLTensor 检查的挂载点"),
    Anchor("A69", MAIN, "src/transform/make_packed_api.cc", 1006, "CallingConv::kCPackedFunc", "call_conv 改写"),
    Anchor("A70", MAIN, "src/transform/make_packed_api.cc", 1011, "Stmt body = ReturnRewriter(v_result)(func_ptr->body)", "返回值改写"),
    Anchor("A71", MAIN, "src/transform/make_packed_api.cc", 1072, "tvm::transform::Pass MakePackedAPI()", "pass 工厂"),
    Anchor("A72", MAIN, "src/transform/make_packed_api.cc", 1115, '"tl.MakePackedAPI"', "pass 名字"),
    Anchor("A73", MAIN, "src/transform/tvm_ffi_binder.cc", 330, "void TVMFFIABIBuilder::BindDLTensors", "ABI 检查生成器"),
    Anchor("A74", MAIN, "src/transform/tvm_ffi_binder.cc", 394, "is expected to have non-NULL pointer", "非空检查"),
    Anchor("A75", MAIN, "src/transform/tvm_ffi_binder.cc", 434, "kDLTensorNDim", "ndim 读取"),
    Anchor("A76", MAIN, "src/transform/tvm_ffi_binder.cc", 466, "AssertStmt(Or(is_null, ndim_ok)", "ndim 断言"),
    Anchor("A77", MAIN, "src/transform/tvm_ffi_binder.cc", 472, "kDLTensorTypeCode", "dtype 三元组检查"),
    Anchor("A78", MAIN, "src/transform/tvm_ffi_binder.cc", 583, "dtype mismatch, expected", "dtype 报错文案"),
    Anchor("A79", MAIN, "src/transform/tvm_ffi_binder.cc", 208, "Relax dtype check to allow FP8 E4M3 variants", "FP8 家族放宽"),
    Anchor("A80", MAIN, "src/transform/tvm_ffi_binder.h", 105, "BindDLTensors(", "binder 的公开契约"),
    Anchor("A81", MAIN, "src/transform/tvm_ffi_binder.h", 32, "DLTensor ABI validation", "错误经 FFI error slot 回传"),
    Anchor("A82", MAIN, "tilelang/transform/__init__.py", 190, "def MakePackedAPI", "Python 侧包装"),
    Anchor("A83", MAIN, "tilelang/cuda/pipeline.py", 273, "tilelang.transform.MakePackedAPI()", "CUDA pipeline 位置"),
    Anchor("A84", MAIN, "tilelang/ascend/pipeline.py", 184, "tilelang.transform.MakePackedAPI()", "Ascend pipeline 位置"),
    # --- 6. adapter ---
    Anchor("A85", MAIN, "tilelang/jit/adapter/__init__.py", 1, "BaseKernelAdapter, CachedTextSource", "adapter 导出面"),
    Anchor("A86", MAIN, "tilelang/jit/adapter/base.py", 24, "class BaseKernelAdapter", "adapter 基类"),
    Anchor("A87", MAIN, "tilelang/jit/adapter/base.py", 61, "def get_current_stream_functor", "stream 取值"),
    Anchor("A88", MAIN, "tilelang/jit/adapter/base.py", 104, "def get_current_device_functor", "device 取值"),
    Anchor("A89", MAIN, "tilelang/jit/adapter/base.py", 124, "def __call__(self", "adapter 转调用"),
    Anchor("A90", MAIN, "tilelang/jit/adapter/tvm_ffi.py", 48, "class TVMFFIKernelAdapter", "默认 adapter"),
    Anchor("A91", MAIN, "tilelang/jit/adapter/tvm_ffi.py", 212, "def _convert_torch_func", "两条 ABI 分支"),
    Anchor("A92", MAIN, "tilelang/jit/adapter/tvm_ffi.py", 261, "expected_inputs = len(self.params) - len(self.result_idx)", "入参个数校验"),
    Anchor("A93", MAIN, "tilelang/jit/adapter/tvm_ffi.py", 334, "tensor_list[i] = torch.empty", "输出在 Python 侧分配"),
    Anchor("A94", MAIN, "tilelang/jit/adapter/tvm_ffi.py", 348, "def _convert_ffi_callee_allocated_output_func", "callee 分配路径"),
    Anchor("A95", MAIN, "tilelang/jit/adapter/tvm_ffi.py", 424, "adapter.executable = runtime.load_module(kernel_lib_path)", "从 .so 重建"),
    Anchor("A96", MAIN, "tilelang/jit/adapter/cython/adapter.py", 60, "class CythonKernelAdapter", "cython adapter"),
    Anchor("A97", MAIN, "tilelang/jit/adapter/cython/adapter.py", 167, "self.cython_wrapper = CythonKernelWrapper", "cython 侧做了校验与输出分配"),
    Anchor("A98", MAIN, "tilelang/jit/adapter/cython/adapter.py", 443, "skip_tensor_validation", "可以跳过校验"),
    Anchor("A99", MAIN, "tilelang/jit/adapter/nvrtc/adapter.py", 27, "class NVRTCKernelAdapter", "nvrtc adapter"),
    Anchor("A100", MAIN, "tilelang/jit/adapter/nvrtc/adapter.py", 313, "if stream is None:", "nvrtc 的 stream 缺省"),
    Anchor("A101", MAIN, "tilelang/jit/adapter/torch/metal.py", 14, "class MetalKernelAdapter", "名字叫 torch 的 Metal adapter"),
    Anchor("A102", MAIN, "tilelang/jit/adapter/cutedsl/adapter.py", 23, "class CuTeDSLKernelAdapter", "CuTeDSL adapter"),
    Anchor("A103", MAIN, "tilelang/jit/adapter/cutedsl/kernel_cache.py", 14, 'kernel_lib_path = "kernel.py"', "CuTeDSL 缓存存 .py"),
    Anchor("A104", MAIN, "tilelang/jit/adapter/nvrtc/kernel_cache.py", 8, 'kernel_lib_path = "kernel.cubin"', "nvrtc 缓存存 cubin"),
    Anchor("A105", MAIN, "tilelang/jit/adapter/kernel_cache.py", 8, 'kernel_lib_path = "executable.so"', "tvm_ffi 缓存存 executable.so"),
    Anchor("A106", MAIN, "tilelang/ascend/kernel_cache.py", 11, 'device_kernel_path = "device_kernel.asc"', "昇腾缓存后缀与 CUDA 不同"),
    # --- 7. 后端策略与 env ---
    Anchor("A107", MAIN, "tilelang/backend/execution_backend.py", 19, "class ExecutionBackendSpec", "execution backend 契约"),
    Anchor("A108", MAIN, "tilelang/backend/execution_backend.py", 29, "supports_callee_allocated_outputs: bool = False", "输出 ABI 的开关"),
    Anchor("A109", MAIN, "tilelang/backend/module.py", 167, 'if requested_name in (None, "auto"):', "auto = 列表第一个可用"),
    Anchor("A110", MAIN, "tilelang/backend/module.py", 211, "def lower(self, mod: IRModule)", "上下文执行 pipeline"),
    Anchor("A111", MAIN, "tilelang/backend/module.py", 310, "def create_backend_context", "上下文工厂"),
    Anchor("A112", MAIN, "tilelang/cuda/execution_backend.py", 24, "CUDA_EXECUTION_BACKENDS = [", "CUDA 优先级"),
    Anchor("A113", MAIN, "tilelang/ascend/execution_backend.py", 7, "ASCEND_EXECUTION_BACKENDS = [", "Ascend 优先级"),
    Anchor("A114", MAIN, "tilelang/metal/execution_backend.py", 9, "EXECUTION_BACKENDS = [", "Metal 把 torch 排在第一位"),
    Anchor("A115", MAIN, "tilelang/env.py", 368, "TILELANG_CACHE_DIR", "缓存根目录"),
    Anchor("A116", MAIN, "tilelang/env.py", 435, "def is_cache_enabled", "缓存开关"),
    Anchor("A117", MAIN, "tilelang/env.py", 447, "def should_use_kernel_cache_lib_stamp", "库内容哈希开关"),
    Anchor("A118", MAIN, "tilelang/env.py", 459, "def is_jit_diagnostics_enabled", "JIT 阶段日志开关"),
    Anchor("A119", MAIN, "tilelang/env.py", 558, "enable_cache = env.enable_cache", "顶层 API"),
    Anchor("A120", MAIN, "tilelang/ascend/torch_exchange.py", 6, "def install_torch_npu_stream_exchange", "Ascend 的 stream 交接"),
    Anchor("A121", MAIN, "tilelang/ascend/torch_exchange.py", 40, "def npu_current_raw_stream", "取 NPU 当前 stream"),
    # --- 8. 启动开销 / graph ---
    Anchor("A122", MAIN, "tilelang/profiler/__init__.py", 220, "def do_bench(", "benchmark 入口"),
    Anchor("A123", MAIN, "tilelang/profiler/bench.py", 258, 'elif backend == "cudagraph":', "CUDA Graph 计时后端"),
    Anchor("A124", MAIN, "tilelang/profiler/torch_bench.py", 144, "def bench_with_cudagraph", "用图回放压低 launch 开销"),
    # --- 9. 昇腾适配器仓库（对照，不是分析基线） ---
    Anchor("A125", ASCEND, "tilelang/jit/__init__.py", 36, '"dlpack", "ctypes", "cython"', "独立仓库的 adapter 家族不同"),
    Anchor("A126", ASCEND, "tilelang/cache/__init__.py", 70, "def clear_cache", "独立仓库有顶层 clear_cache"),
    Anchor("A127", ASCEND, "examples/aclgraph/rms_rope_aclgraph.py", 271, "g = torch.npu.NPUGraph()", "ACLGraph 捕获"),
    # --- 10. 正文引用的区间端点（确保「a-b」两端的行号都成立） ---
    Anchor("A128", MAIN, "tilelang/cache/kernel_cache.py", 85, "@staticmethod", "库内容哈希辅助函数的起点"),
    Anchor("A129", MAIN, "tilelang/cache/kernel_cache.py", 137, "return None", "库内容哈希辅助函数的终点"),
    Anchor("A130", MAIN, "tilelang/cache/kernel_cache.py", 220, "def _cleanup_stale_staging_dirs", "staging 残骸清理（1 小时）"),
    Anchor("A131", MAIN, "tilelang/cache/kernel_cache.py", 473, "return os.path.join(self._get_cache_root(), key)", "key → 目录路径"),
    Anchor("A132", MAIN, "tilelang/cache/kernel_cache.py", 583, "os.makedirs(staging_path)", "先写 staging"),
    Anchor("A133", MAIN, "tilelang/cache/kernel_cache.py", 622, "KernelCache._fsync_dir(staging_path)", "发布前的耐久屏障"),
    Anchor("A134", MAIN, "tilelang/cache/kernel_cache.py", 636, "KernelCache._fsync_dir(self._get_cache_root())", "rename 之后的目录 fsync"),
    Anchor("A135", MAIN, "tilelang/cache/kernel_cache.py", 675, "return None", "不完整条目当作未命中"),
    Anchor("A136", MAIN, "tilelang/cache/kernel_cache.py", 680, "CUDA_ERROR_INVALID_IMAGE", "损坏缓存的真实症状"),
    Anchor("A137", MAIN, "tilelang/cache/kernel_cache.py", 848, "return exc.errno in {errno.EEXIST, errno.ENOTEMPTY}", "先发布者赢"),
    Anchor("A138", MAIN, "tilelang/cache/binary_cache.py", 121, "def get_path", "二进制缓存的路径约定"),
    Anchor("A139", MAIN, "tilelang/cache/binary_cache.py", 139, "Empty/short reads and missing metadata are always misses", "损坏即未命中，不删共享条目"),
    Anchor("A140", MAIN, "tilelang/cache/ascend_binary_cache.py", 42, "return cls._finalize_key(key_data)", "昇腾二进制 key 的收尾"),
    Anchor("A141", MAIN, "src/transform/tvm_ffi_binder.cc", 294, "RelaxedStrideCheck", "stride 检查"),
    Anchor("A142", MAIN, "src/transform/tvm_ffi_binder.cc", 375, "Create all is_null vars and shape buffers first", "非空检查的生成起点"),
    Anchor("A143", MAIN, "src/transform/tvm_ffi_binder.cc", 398, "Create all shape buffers before binding any shapes", "shape 指针短路"),
    Anchor("A144", MAIN, "src/transform/tvm_ffi_binder.cc", 592, "Bind symbolic variables from buffer shape", "逐维 shape 绑定"),
    Anchor("A145", MAIN, "src/transform/make_packed_api.cc", 234, "class AssumeRuntimeCheckExtractor", "tl.assume 检查的抽取"),
    Anchor("A146", MAIN, "src/transform/make_packed_api.cc", 615, "int packed_arg_index = 0", "逐参数检查的开始"),
    Anchor("A147", MAIN, "src/transform/make_packed_api.cc", 1058, "func_ptr->params = args", "新签名落地"),
    Anchor("A148", MAIN, "tilelang/jit/adapter/tvm_ffi.py", 40, "def _install_torch_stream_exchange", "NPU stream 交接的调用点"),
    Anchor("A149", MAIN, "tilelang/jit/adapter/tvm_ffi.py", 367, "__dlpack_c_exchange_api__", "callee-allocated 的前置条件"),
    Anchor("A150", MAIN, "tilelang/ascend/torch_npu_exchange.pyx", 50, "cdef int torch_npu_current_work_stream", "补丁后的 exchange API"),
    Anchor("A151", MAIN, "tilelang/profiler/torch_bench.py", 144, "def bench_with_cudagraph", "CUDA Graph 计时后端"),
    Anchor("A152", MAIN, "tilelang/engine/lower.py", 26, "def has_device_kernel_launch", "host/device 划分的判据"),
    Anchor("A153", MAIN, "tilelang/engine/lower.py", 89, "return context.codegen_host(host_mod)", "host codegen 出口"),
    Anchor("A154", MAIN, "tilelang/engine/lower.py", 106, "return context.codegen_device(device_mod, compile_device=False)", "只 codegen 不编译"),
    Anchor("A155", MAIN, "tilelang/jit/kernel.py", 295, "enable_host_codegen = self.execution_backend_spec.enable_host_codegen", "两个开关来自 backend 声明"),
    Anchor("A156", MAIN, "tilelang/jit/adapter/base.py", 90, "torch.npu.current_stream(device).npu_stream", "NPU stream 的兜底取值"),
    Anchor("A157", MAIN, "tilelang/jit/adapter/cython/adapter.py", 431, "def _forward_from_prebuild_lib", "cython 显式传 stream"),
    Anchor("A158", MAIN, "tilelang/language/eager/builder.py", 1269, 'pf = pf.with_attr("tilelang_out_idx", builder.out_idx)', "eager 模式在 TIR 上写 out_idx"),
    Anchor("A159", ASCEND, "examples/aclgraph/rms_rope_aclgraph.py", 59, "T.copy(x_ub, out[row_x : row_x + row_per_vec, :])", "官方 RMS Norm kernel 的最后一行"),
    Anchor("A160", ASCEND, "examples/aclgraph/README.md", 133, "capture model", "如何确认图模式生效"),
    Anchor("A161", MAIN, "tilelang/language/loop.py", 82, "def Persistent(", "手动的 T.Persistent 在 v0.1.15 里就有"),
    # --- 11. 动态形状 / 选择 adapter / 跨机复用（正文 4.10-4.12、5.1、5.4） ---
    Anchor("A162", MAIN, "tilelang/jit/adapter/tvm_ffi.py", 172, "def _process_dynamic_symbolic", "动态形状映射"),
    Anchor("A163", MAIN, "tilelang/jit/adapter/tvm_ffi.py", 192, "ordered = [i for i in range(len(params)) if i not in self.result_idx]", "输入优先访问"),
    Anchor("A164", MAIN, "tilelang/jit/adapter/tvm_ffi.py", 206, "stride_scale = 8 // element_bits if element_bits < 8 else 1", "亚字节 stride 换算"),
    Anchor("A165", MAIN, "tilelang/jit/adapter/tvm_ffi.py", 344, "return [tensor_list[i] for i in self.result_idx]", "输出返回形态"),
    Anchor("A166", MAIN, "tilelang/profiler/bench.py", 39, '"event", "cupti", "cudagraph"', "计时后端取值域"),
    Anchor("A167", MAIN, "tilelang/cuda/execution_backend.py", 29, "supports_callee_allocated_outputs=True", "CUDA 声明输出 ABI 能力"),
    Anchor("A168", MAIN, "tilelang/ascend/execution_backend.py", 18, "supports_callee_allocated_outputs=True", "Ascend 声明输出 ABI 能力"),
    Anchor("A169", MAIN, "tilelang/cache/binary_cache.py", 139, "Empty/short reads and missing metadata are always misses", "不删共享缓存条目"),
    Anchor("A170", MAIN, "src/transform/tvm_ffi_binder.cc", 1047, "}", "BindDLTensors 的收尾"),
    Anchor("A171", MAIN, "src/transform/make_packed_api.cc", 295, "};", "AssumeRuntimeCheckExtractor 的收尾"),
    Anchor("A172", MAIN, "tilelang/autotuner/param.py", 69, "return tilelang.compile(", "compile() 的第二个调用点：也只用关键字参数"),
)

TREE_PATHS = {
    MAIN: "_vendor/tilelang-v0.1.15",
    ASCEND: "_vendor/tilelang-ascend-v0.1.2.000",
}

# The reverse map (tree directory prefix -> our short name) is used by the
# citation scan, which sees paths as written in the prose.
TREE_PREFIXES = {path: name for name, path in TREE_PATHS.items()}

_CITATION = re.compile(r"([A-Za-z0-9_./\-]+\.(?:py|cc|h|pyx|txt)):(\d{1,4})(?:-(\d{1,4}))?")


def find_book_root(start: Path) -> Path:
    """Walk upwards until we find the book root that owns `_vendor/`."""
    configured = os.environ.get("TILELANG_TUTORIAL_BOOK_ROOT")
    start = Path(configured).resolve() if configured else start
    for candidate in (start, *start.parents):
        if (candidate / "_vendor").is_dir():
            return candidate
    raise SystemExit(
        f"could not find a directory containing `_vendor/` above {start}. "
        "Pass `--book-root /path/to/checkout` or set "
        "`TILELANG_TUTORIAL_BOOK_ROOT`. See VERSIONS.md."
    )


def check(root: Path, anchor: Anchor) -> tuple[bool, str]:
    path = root / TREE_PATHS[anchor.tree] / anchor.relpath
    if not path.is_file():
        return False, f"{anchor.relpath}: file missing"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if anchor.line > len(lines) or anchor.line < 1:
        return False, f"{anchor.relpath}:{anchor.line}: out of range (file has {len(lines)} lines)"
    text = lines[anchor.line - 1]
    if anchor.needle not in text:
        return False, f'{anchor.relpath}:{anchor.line}: expected {anchor.needle!r}, found {text.strip()!r}'
    return True, f"{anchor.relpath}:{anchor.line}"


def scan_citations(root: Path, chapter_dir: Path) -> list[str]:
    """Check every `path:line` citation in the chapter against the locked trees.

    This complements the anchor table: the table pins the lines the prose
    *argues from*, while the scan catches typos in the ranges quoted inline.
    """
    problems: list[str] = []
    checked = 0
    for doc in sorted(chapter_dir.rglob("*.md")) + sorted(chapter_dir.rglob("*.py")):
        if doc.name == Path(__file__).name:
            continue  # the anchor table itself is not prose
        for match in _CITATION.finditer(doc.read_text(encoding="utf-8", errors="replace")):
            raw_path, start, end = match.group(1), int(match.group(2)), match.group(3)
            relpath = tree = None
            for prefix, name in TREE_PREFIXES.items():
                marker = prefix + "/"
                idx = raw_path.find(marker)
                if idx >= 0:
                    relpath, tree = raw_path[idx + len(marker) :], name
                    break
            if relpath is None:
                continue  # a citation into the chapter's own files
            target = root / prefix_of(tree) / relpath
            doc_label = doc.relative_to(chapter_dir)
            if not target.is_file():
                problems.append(f"{doc_label}: {relpath} (file not found)")
                continue
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            for line_no in ([start] + ([int(end)] if end else [])):
                checked += 1
                if line_no < 1 or line_no > len(lines):
                    problems.append(
                        f"{doc_label}: {relpath}:{line_no} out of range (file has {len(lines)} lines)"
                    )
                elif not lines[line_no - 1].strip():
                    problems.append(f"{doc_label}: {relpath}:{line_no} points at a blank line")
    print(f"citation scan: {checked} line references checked")
    return problems


def prefix_of(tree: str) -> str:
    return TREE_PATHS[tree]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="print failures and the summary only")
    parser.add_argument("--list", action="store_true", dest="list_anchors", help="print the anchor table and exit")
    parser.add_argument("--no-scan", action="store_true", help="skip the citation scan and check only the anchor table")
    parser.add_argument(
        "--book-root",
        type=Path,
        default=None,
        help="directory containing _vendor/ (also settable via TILELANG_TUTORIAL_BOOK_ROOT)",
    )
    args = parser.parse_args()

    if args.list_anchors:
        for a in ANCHORS:
            print(f"{a.id}  {TREE_PATHS[a.tree]}/{a.relpath}:{a.line}  # {a.why}")
        print(f"\n{len(ANCHORS)} anchors")
        return 0

    root = (args.book_root.resolve() if args.book_root else find_book_root(Path(__file__).resolve().parent))
    failures: list[tuple[Anchor, str]] = []
    for a in ANCHORS:
        ok, detail = check(root, a)
        if ok:
            if not args.quiet:
                print(f"ok   {a.id}  {detail:64s} {a.why}")
        else:
            failures.append((a, detail))

    scan_problems: list[str] = []
    if not args.no_scan:
        scan_problems = scan_citations(root, Path(__file__).resolve().parent.parent.parent)

    print()
    if failures or scan_problems:
        print(f"FAIL: {len(failures)}/{len(ANCHORS)} anchors moved; {len(scan_problems)} citation problems")
        for a, detail in failures:
            print(f"  {a.id}  {detail}")
            print(f"      ({TREE_PATHS[a.tree]}/{a.relpath}:{a.line} was cited for: {a.why})")
        for problem in scan_problems:
            print(f"  scan: {problem}")
        return 1

    print(f"ok: {len(ANCHORS)}/{len(ANCHORS)} anchors verified; book root = {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
