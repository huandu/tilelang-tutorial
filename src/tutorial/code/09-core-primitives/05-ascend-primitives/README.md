# 05 · 昇腾侧原语骨架

## 目的

把本章昇腾相关的三条主线放进可读的骨架里：

* `ascend_dual_copy_simtvf_sketch.py` —— 主干 v0.1.15 就有的
  `T.alloc_l1` / `T.alloc_l0c` / `T.gemm` / `T.dual_copy` / `T.SimtVF`；
  原型是 `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:60-155`。
* `ascend_tile_library_sketch.py` —— **适配器仓库**的 `T.tile.*` 向量内联函数；
  原型是 `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend_tile.py`。

**这两个文件都只做静态核对**：没有 NPU，也没有跑过。

## 运行前提

| 文件 | 前提 |
| --- | --- |
| `ascend_dual_copy_simtvf_sketch.py` | 昇腾 NPU + CANN + `torch_npu` + TileLang `v0.1.15`（主干自带的 `tilelang.ascend.language`） |
| `ascend_tile_library_sketch.py` | 昇腾 NPU + CANN + `torch_npu` + **tilelang-ascend 适配器 `v0.1.2.000`**（提供 `T.tile` 命名空间） |

## 运行命令

```bash
cd 05-ascend-primitives
python3 -m py_compile ascend_dual_copy_simtvf_sketch.py ascend_tile_library_sketch.py

# 有 NPU 的机器上：
python3 ascend_dual_copy_simtvf_sketch.py
python3 ascend_tile_library_sketch.py
```

## 期望输出（正确性判据，不是实测结果）

* `ascend_dual_copy_simtvf_sketch.py`：
  `torch.testing.assert_close(c, torch.relu(a.float() @ b.float().T), rtol=1e-2, atol=1e-2)`
  必须通过。
* `ascend_tile_library_sketch.py`：`T.tile.select` 的输出必须等于
  `torch.where(mask_bits.view_as(x) != 0, x, y)` 的等价写法；
  `T.tile.sort` 的输出必须满足“相邻两两一组、第一项降序、第二项是原始位置”。
* 性能口径（若要做）：用 `tilelang.profiler.do_bench(..., backend="msprof")`，
  并记录 NPU 型号、CANN 版本与 TileLang 版本。

## 作者验证级别

**仅静态核对**：两个文件都通过 `python3 -m py_compile`；其中每个 API 都逐条比对过：

* `T.alloc_l1` / `T.alloc_l0c` / `T.dual_copy` / `T.SimtVF` ——
  `tilelang/ascend/language/{allocate,copy_op,frame}.py`
* `T.tile.add` / `T.tile.select` / `T.tile.sort` / `T.tile.atomic_add` ——
  `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend_tile.py`

**未在 NPU 上运行。**

## 已知的静态核对结论

* `T.tile.*` **不在** TileLang 主干 v0.1.15 里：`tilelang/ascend/language/` 下没有
  `ascend_tile.py`，也没有 `as tile` 的导入（可用
  `../01-primitive-ir-map/primitive_ir_map.py` 复现）。
* `T.dual_copy` 是混合 kernel 原语：用户必须自己提供 Cube 侧的工作，编译器不会
  替你合成（`tilelang/ascend/language/copy_op.py:62-66`）。
