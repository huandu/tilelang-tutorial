# 示例 03：复算 `TokenLayout` / `BufferLayout` 与 `get_buffer_size_hint`

## 目的

把 DeepEP-Ascend 里计算 buffer 容量的整数运算照抄成 Python，方便读者
对着源码读，并看清一个反直觉的点：**hint 为 SF 预留的空间是实际用量的 2 倍**。

对应源码：

* `deep_ep/include/deep_ep/layout/ep/token.hpp:36-43`（`TokenLayout::get_num_bytes`）
* `deep_ep/include/deep_ep/layout/ep/token.hpp:102-112`（`BufferLayout::get_num_bytes`）
* `csrc/buffers/ep.hpp:560-605`（`get_dispatch_buffer_size` /
  `get_combine_buffer_size` / `calculate_buffer_size`）

## 运行前提

Python 3.9+，仅标准库。

## 运行命令

```bash
python3 buffer_size.py
python3 buffer_size.py --ranks 128 --tokens 16384
```

## 期望输出（默认参数：ranks=8, T=4096, hidden=7168, topk=6）

```text
TokenLayout（每 token 字节）
  dispatch(with metadata, 112 个 SF pack): 7488 B
  combine (no metadata, 无 SF)           : 14368 B

get_buffer_size_hint(num_ranks=8, T=4096, hidden=7168, topk=6, use_fp8_dispatch=True)
  hint 用的 SF pack 数 = ceil(hidden/32) = 224 （实际是 112，多预留 112 个）
  dispatch 布局 = 252,706,816 B
  combine  布局 = 353,107,968 B
  max           = 353,107,968 B
  对齐到 2 MB   = 354,418,688 B = 338.00 MiB
```

## 三个结论

1. **容量 = `max(dispatch 布局, combine 布局)` 再对齐到 2 MB**。
   默认参数下 combine 布局更大，所以最终容量由 combine 决定。
2. **hint 的 SF 数量是 `ceil(hidden/32)`，而实际运行时是 `hidden/128×2 = hidden/64`**，
   差 2 倍（`csrc/buffers/ep.hpp:593-594` 的注释明确说是近似值）。
3. **这个偏大是设计上的安全余量**，不是 bug，也不影响实际流量——
   实际 dispatch 只搬 `num_sf_packs = ceil(hidden/64)` 个 int16。

## 验证级别

**CPU 逻辑验证**：作者在 macOS (Apple Silicon) / Python 3.11 上实际运行过。
公式与源码逐行对齐；**没有在任何设备上验证过分配是否成功**。
