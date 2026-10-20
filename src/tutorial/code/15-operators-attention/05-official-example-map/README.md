# 示例 5：官方 attention 例子与 API 的行号映射

## 目的

正文第 3.4、4.4、4.6 节引用了大量「哪个文件、第几行」。这个示例把那些引用变成
**可以一键复核的输出**，做两件事：

* **A. 例子结构映射**：对九个 attention 族官方例子，打印本章关心的每一行
  （kernel 启动、流水循环、softmax 递推、mask、sink、split/combine……）；
* **B. API 出处映射**：对本章引用的 14 个 TileLang API，在锁定版本里定位定义行，
  并**与正文引用的行号比对**；不一致会打印 `DRIFT`。

`--strict` 时，任何探针找不到或任何行号漂移都会让程序非 0 退出——可以用它当作
「正文引用是否还成立」的回归检查。

## 运行前提

* Python 3.9+（只用标准库）；
* `_vendor/tilelang-v0.1.15/` 在书稿工作树上（脚本从自己的路径往上找到书稿根目录）。

**不 import tilelang**，所以没有安装 TileLang 的机器也能跑。

## 运行命令

```bash
python3 map_official_examples.py            # 打印映射
python3 map_official_examples.py --strict   # 有漂移则退出码 1
```

## 期望输出（写作机实际输出，退出码 0）

节选：

```text
examples/flash_attention/example_gqa_fwd_varlen.py  (211 lines)
    kv head index              :52
    causal mask with -1e9      :84
    empty-row guard            :118
    store guard                :123

examples/flash_decoding/example_mha_inference.py  (269 lines)
    split kernel               :30
    per-split lse store        :110
    combine kernel             :115
    combine max                :137
    combine weights            :149

examples/attention_sink/example_mha_sink_fwd_bhsd.py  (286 lines)
    window lower bound         :84
    window mask                :92
    check_inf flag             :104
    sink in denominator        :126

B. API definitions the chapter cites (line numbers are pinned)
  alloc_shared         tilelang/language/allocate.py:49
  reduce_max           tilelang/language/reduce_op.py:176
  gemm                 tilelang/language/gemm_op.py:160
  Pipelined            tilelang/language/loop.py:112
  exp2                 tilelang/language/tir/op.py:74
  infinity             tilelang/language/tir/op.py:361
  tcgen05_gemm         tilelang/cuda/language/gemm_op.py:264
  class GemmWarpPolicy tilelang/tileop/base.py:5

drift/miss count: 0
```

## 判据

* `drift/miss count: 0`；
* `--strict` 退出码 0。

## 注意事项

* 这个脚本**只读文本**：它用正则找 `with T.Kernel(` 这样的字符串，不做语法分析。
  探针命中不等于代码能编译。
* 探针里的行号是**锁定版本 v0.1.15** 的行号。换 tag 之后 `--strict` 会报 `DRIFT`，
  这是特性不是 bug——正文里的行号本来就绑定在版本上。
* 想扩展它：在 `EXAMPLES` 列表里加一条 `(相对路径, [(标签, 正则)])` 即可；
  API 行号在 `NV_API` 里加。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过
`--strict`，输出 `drift/miss count: 0`，退出码 0。
**没有编译、导入或运行任何 kernel。**
