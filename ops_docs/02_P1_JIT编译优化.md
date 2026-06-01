# P1: Numba JIT 编译优化

## 优化类型

**即时编译 (JIT)** — 用 Numba 将 Python 热循环编译为原生机器码，消除解释器开销。

## 涉及算子

| 算子 | 文件 | 编译方式 |
|------|------|---------|
| DFLDecode + BBoxDecode + BBoxDecode_OBB（融合） | `code/common.py:_decode_kernel` | `@njit` 编译为 x86/ARM 机器码 |

> 三个算子融合在同一个 `_decode_kernel` 函数中，12.7x 是该函数整体的加速比，非单个算子。

## 技术原理

### Numba @njit

```python
from numba import njit

@njit(cache=True)
def _decode_kernel(xd_l, xd_t, xd_r, xd_b, cf, ci, angle,
                   H, W, s, ao, scale, ox, oy, conf_th):
    out = np.empty((H * W, 7), np.float32)
    cnt = 0
    for h in range(H):
        for w in range(W):
            c = cf[h, w]
            if c <= conf_th:
                continue
            # ... 标量计算，编译为原生指令
            out[cnt, 0] = cx
            cnt += 1
    return out[:cnt]
```

**优势**：
1. 消除 Python 字节码解释开销
2. 类型推导后无运行时类型检查
3. 直接操作 NumPy 数组内存（无 Python 对象包装）
4. `cache=True` 首次编译后缓存到磁盘，后续启动零编译开销
5. 支持 x86 SSE/AVX 和 ARM NEON 向量指令

### 架构设计

```
decode_generic()
    ├── softmax + expected value (NumPy，已够快，不改)
    └── _decode_kernel (Numba JIT 编译)
            ├── 有 numba → @njit 编译为机器码
            └── 无 numba → 回退 NumPy 向量化 (P0)
```

**为什么不整个函数用 Numba？**
- softmax 涉及 `np.exp`、`np.sum`、`keepdims` 等高级 NumPy 操作，Numba 支持有限
- 这部分已是 NumPy C 实现，瓶颈不在这里
- 只编译热循环（坐标解码），收益最大、改动最小

## 测试结果

### 正确性

32 组测试全部 PASS：
- NumPy vs 原版：max_err = **0.00e+00**（完全一致）
- Numba vs 原版：max_err < **6.1e-05**（浮点运算顺序差异，可忽略）

### 性能（三代对比）

| imgsz | Python (原版) | NumPy (P0) | Numba (P1) | P1 加速比 |
|-------|-------------|-----------|-----------|----------|
| 320 | 14.54ms | 2.18ms | 1.76ms | **8.3x** |
| 640 | 67.72ms | 6.82ms | 5.04ms | **13.4x** |
| 1024 | 177.74ms | 17.80ms | 13.98ms | **12.7x** |

### Numba vs NumPy 增量收益

| imgsz | NumPy | Numba | Numba 额外提速 |
|-------|-------|-------|---------------|
| 320 | 2.18ms | 1.76ms | **+19%** |
| 640 | 6.82ms | 5.04ms | **+26%** |
| 1024 | 17.80ms | 13.98ms | **+21%** |

> Numba 在 ARM (RK3588) 上优势更大，因为 Cortex-A76 的 Python 解释器比 x86 更慢，而编译后的机器码差距更小。

## 兼容性

```python
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

if _HAS_NUMBA:
    @njit(cache=True)
    def _decode_kernel(...):  # Numba 版
else:
    _decode_kernel = _decode_kernel_fallback  # NumPy 回退
```

- **有 numba**：自动使用 JIT 编译版本
- **无 numba**：自动回退到 NumPy 向量化版本（P0），功能和结果一致

## 依赖

```bash
pip install numba  # 包含 llvmlite (LLVM 编译器后端)
```

- numba 0.65.1 + llvmlite 0.47.0
- 首次运行有 ~2s 编译开销，后续 `cache=True` 从磁盘加载

## 测试文件

- `ops/benchmark_decode_generic.py` — 三代对比测试脚本

## 相关 commit

- `cf07a8b` — P1 Numba JIT 实现
- `df2dd6e` — 添加 numba 回退兼容
