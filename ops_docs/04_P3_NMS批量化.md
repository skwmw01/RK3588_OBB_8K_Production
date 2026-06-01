# P3: NMS 批量化优化（待实施）

## 优化类型

**算法优化** — 将逐对比较的 Python 循环替换为矩阵批量计算。

## 涉及算子

| 算子 | 文件 | 当前实现 | 问题 |
|------|------|---------|------|
| NMS | `code/common.py:nms_rot` | Python while 循环 + 逐对距离计算 | 循环开销大 |
| RotatedNMS | 同上 | 中心距离近似（非精确 IoU） | 可能误杀 |
| IoU | `code/common.py:iou_rot` | `cv2.rotatedRectangleIntersection` (C++) | 已优化 |
| RotatedIoU | 同上 | 同上 | 已优化 |

## 瓶颈分析

### nms_rot（精确 IoU 版）

```python
def nms_rot(dets, iou_thr=0.3):
    while len(dets) > 0:
        a = dets[0]
        keep.append(a)
        rest = dets[1:]
        d = np.linalg.norm(rest[:, :2] - [a[0], a[1]], axis=1)  # 每次重新算距离
        th = (a[2] + a[3]) * 0.5 * iou_thr
        dets = rest[d > th]  # 过滤
```

**问题**：
1. while 循环，每轮只处理 1 个框
2. 每轮重新计算所有剩余框与当前框的距离
3. `dets = rest[d > th]` 每轮创建新数组（内存分配）

### 当前 RotatedNMS 使用的近似方法

```python
# 用中心距离近似 IoU，避免调用 cv2.rotatedRectangleIntersection
d = np.linalg.norm(rest[:, :2] - [a[0], a[1]], axis=1)
th = (a[2] + a[3]) * 0.5 * iou_thr
```

**问题**：中心距离不等于 IoU，对于长条形目标或斜角目标，近似误差大，可能误杀或漏检。

## 优化方案

### 方案 A：矩阵 IoU 预计算

```python
def nms_matrix(dets, iou_thr=0.3):
    """批量计算 IoU 矩阵，再做贪心 NMS"""
    if len(dets) == 0: return []
    dets = np.array(dets)
    scores = dets[:, 5]
    order = scores.argsort()[::-1]

    # 预计算所有 IoU（N×N 矩阵）
    N = len(dets)
    iou_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            iou_mat[i, j] = iou_mat[j, i] = iou_rot(dets[order[i]], dets[order[j]])

    # 贪心选择
    keep = []
    suppressed = np.zeros(N, dtype=bool)
    for i in range(N):
        if suppressed[i]:
            continue
        keep.append(order[i])
        suppressed[iou_mat[i] > iou_thr] = True

    return dets[keep].tolist()
```

### 方案 B：向量化中心距离 NMS

```python
def nms_vectorized(dets, iou_thr=0.3):
    """向量化 NMS（仍用中心距离近似，但消除 Python 循环）"""
    if len(dets) == 0: return []
    dets = np.array(dets)
    scores = dets[:, 5]
    order = scores.argsort()[::-1]
    dets = dets[order]

    centers = dets[:, :2]
    sizes = (dets[:, 2] + dets[:, 3]) * 0.5  # 平均边长

    keep = []
    alive = np.ones(len(dets), dtype=bool)
    for i in range(len(dets)):
        if not alive[i]:
            continue
        keep.append(i)
        # 一次计算当前框与所有剩余框的距离
        d = np.linalg.norm(centers[i+1:] - centers[i], axis=1)
        th = sizes[i] * iou_thr
        mask = d <= th
        alive[i+1:] &= ~mask  # 批量抑制

    return dets[keep].tolist()
```

### 方案 C：C 扩展 NMS

```python
# 用 torchvision 的 C++ NMS 或自写 Cython 扩展
from torchvision.ops import nms  # 只支持水平框，不支持旋转
# 需要自写 rotated_nms 的 C 扩展
```

## 预期收益

| 方案 | 预期加速 | 精度影响 | 改动量 |
|------|---------|---------|-------|
| A: 矩阵 IoU | 2x | 精确 IoU，提升精度 | 中 |
| B: 向量化距离 | 3x | 仍是近似，精度不变 | 小 |
| C: C 扩展 | 5-10x | 精确 IoU | 大 |

**推荐**：先做 B（改动小、见效快），再考虑 A 或 C。

## 相关代码

- `code/common.py:72-85` — `nms_rot` 函数
- `code/common.py:90-99` — `iou_rot` 函数
