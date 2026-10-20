# 图 1：双语结构，以及一条 pass 从 C++ 到 Python 的完整通路（自绘）

TileLang 的「写一个 pass」有两个入口：**C++ 入口**和 **Python 入口**。两者最终都
产出同一个东西——`tvm.transform.Pass`，然后被 backend pipeline 用
`mod = X()(mod)` 串起来。

```text
                        ①定义                          ②实现
  ┌───────────────────────────────────┐   ┌────────────────────────────────────┐
  │ C++ 入口                          │   │ Python 入口                        │
  │                                   │   │                                    │
  │ src/transform/*.cc       (42 个)  │   │ tilelang/transform/                │
  │   ├─ layout_inference/            │   │   ├─ add_bufstore_wrapper.py        │
  │   ├─ common/                      │   │   ├─ decouple_type_cast.py          │
  │   └─ pipeline/                    │   │   └─ hoist_broadcast_values.py      │
  │ src/ascend/transform/*.cc (31 个) │   │ tilelang/analysis/            (5 个)│
  │   └─ auto_schedule/               │   │ tilelang/ascend/analysis/     (1 个)│
  │                                   │   │ tilelang/metal/transform/     (2 个)│
  │ struct : StmtExprVisitor/Mutator  │   │ @tirx.functor.mutator / visitor     │
  │ Pass X() { CreatePrimFuncPass }   │   │ prim_func_pass(pass_fn, ...)        │
  └───────────────┬───────────────────┘   └───────────────┬────────────────────┘
                  │                                       │
                  ▼  ③注册（只有 C++ 需要）               │
     TVM_FFI_STATIC_INIT_BLOCK() {                        │
       refl::GlobalDef().def("tl.transform.X", X);        │
     }                                                    │
                  │                                       │
                  ▼  ④Python 侧自动生成属性               ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ tilelang/transform/_ffi_api.py:6                                        │
  │     tvm_ffi.init_ffi_api("tl.transform", __name__)                      │
  │   → 把每个注册名 "tl.transform.X" 变成模块属性 _ffi_api.X               │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ tilelang/transform/__init__.py:86-94、:97-108 …                          │
  │     def X(): return _ffi_api.X()        # 薄包装，不含逻辑              │
  │     from .hoist_broadcast_values import HoistBroadcastValues  # 纯 Python │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ 对外的唯一门面：tilelang.transform.*                                     │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ tilelang/cuda/pipeline.py:79,84,98,150        （CUDA 后端）              │
  │ tilelang/ascend/pipeline.py:58,61,76,125      （昇腾后端）              │
  │     mod = tilelang.transform.X()(mod)                                   │
  └─────────────────────────────────────────────────────────────────────────┘
```

三个必须记住的坐标：

* **`_ffi_api` 不写逻辑。** `tilelang/transform/_ffi_api.py` 一共 6 行，全部内容
  就是一次 `init_ffi_api`。逻辑要么在 C++，要么在 `tilelang/transform/` 下的
  Python 模块里。
* **注册名是唯一的契约。** C++ 侧的字符串 `"tl.transform.X"` 与 Python 侧
  `_ffi_api.X` 必须逐字对应；前缀 `tl.transform` 来自
  `init_ffi_api("tl.transform", ...)` 的第一个参数。改名会让所有调用点
  在 import 期或首次调用期炸掉。
* **挂载只有一行。** 没有注册表、没有插件机制：pipeline 函数就是一串
  `mod = pass()(mod)`。要改顺序，就改这串调用的位置。

## 对照：昇腾适配器仓库用的是旧接口

`_vendor/tilelang-ascend-v0.1.2.000/tilelang/transform/_ffi_api.py:5-8` 写的是
`tvm._ffi._init_api("tl.transform", __name__)`，而进主干的 v0.1.15 写的是
`tvm_ffi.init_ffi_api(...)`。C++ 侧的注册宏也从 `TVM_REGISTER_GLOBAL(...)`
搬到了 `TVM_FFI_STATIC_INIT_BLOCK()` + `refl::GlobalDef().def(...)`。
**这是两棵树上最容易抄错的一行**：把适配器里的一段 pass 代码直接拷进
v0.1.15，注册宏和 `refl` 命名空间往往需要改写。

来源：作者自绘。文中行号取自
`_vendor/tilelang-v0.1.15/`（2026-10-01 核对）。
