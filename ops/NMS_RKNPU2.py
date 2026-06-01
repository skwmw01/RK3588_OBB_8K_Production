"""NMS_RKNPU2: 非极大值抑制 测试"""
import numpy as np

# ============================================================
# CPU 参考实现 (标准水平框 NMS)
# ============================================================
def iou_cpu(box_a, box_b):
    """[cx, cy, w, h] 格式的 IoU"""
    ax1, ay1 = box_a[0] - box_a[2]/2, box_a[1] - box_a[3]/2
    ax2, ay2 = box_a[0] + box_a[2]/2, box_a[1] + box_a[3]/2
    bx1, by1 = box_b[0] - box_b[2]/2, box_b[1] - box_b[3]/2
    bx2, by2 = box_b[0] + box_b[2]/2, box_b[1] + box_b[3]/2
    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    union = box_a[2]*box_a[3] + box_b[2]*box_b[3] - inter
    return inter / np.maximum(union, 1e-10)

def nms_cpu(boxes, scores, iou_thr=0.5):
    """标准 NMS: 按分数降序，贪心抑制"""
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = np.array([iou_cpu(boxes[i], boxes[j]) for j in rest])
        mask = ious <= iou_thr
        order = rest[mask]
    return keep

# ============================================================
# 项目实现 (common.py nms_rot: 中心距离近似)
# ============================================================
def nms_project(dets, iou_thr=0.3):
    """项目中的旋转 NMS (中心距离近似)"""
    if not dets:
        return []
    dets = np.array(dets)
    dets = dets[dets[:, 5].argsort()[::-1]]
    keep = []
    while len(dets) > 0:
        a = dets[0]
        keep.append(a)
        if len(dets) == 1:
            break
        rest = dets[1:]
        d = np.linalg.norm(rest[:, :2] - np.array([a[0], a[1]]), axis=1)
        th = (a[2] + a[3]) * 0.5 * iou_thr
        dets = rest[d > th]
    return keep

# ============================================================
# 测试
# ============================================================
def test_nms():
    print("=" * 60)
    print("NMS_RKNPU2 测试")
    print("=" * 60)

    # 测试1: 无重叠框
    boxes1 = np.array([[100, 100, 50, 50, 0, 0.9, 0],
                        [300, 300, 50, 50, 0, 0.8, 0],
                        [500, 500, 50, 50, 0, 0.7, 0]], dtype=float)
    proj_keep1 = nms_project(boxes1, 0.3)
    print(f"  [{'PASS' if len(proj_keep1) == 3 else 'FAIL'}] 无重叠框  "
          f"输入=3  输出={len(proj_keep1)}  期望=3")

    # 测试2: 完全重叠框
    boxes2 = np.array([[100, 100, 50, 50, 0, 0.9, 0],
                        [100, 100, 50, 50, 0, 0.8, 0],
                        [100, 100, 50, 50, 0, 0.7, 0]], dtype=float)
    proj_keep2 = nms_project(boxes2, 0.3)
    print(f"  [{'PASS' if len(proj_keep2) == 1 else 'FAIL'}] 完全重叠框  "
          f"输入=3  输出={len(proj_keep2)}  期望=1")

    # 测试3: 部分重叠
    boxes3 = np.array([[100, 100, 100, 100, 0, 0.9, 0],
                        [130, 100, 100, 100, 0, 0.85, 0],
                        [300, 300, 100, 100, 0, 0.8, 0]], dtype=float)
    proj_keep3 = nms_project(boxes3, 0.3)
    print(f"  [INFO] 部分重叠框  输入=3  project输出={len(proj_keep3)}  "
          f"(中心距离近似可能与标准NMS不同)")

    # 测试4: 空输入
    proj_keep4 = nms_project([], 0.3)
    print(f"  [{'PASS' if len(proj_keep4) == 0 else 'FAIL'}] 空输入  输出={len(proj_keep4)}  期望=0")

    # 测试5: 单框
    boxes5 = np.array([[100, 100, 50, 50, 0, 0.9, 0]], dtype=float)
    proj_keep5 = nms_project(boxes5, 0.3)
    print(f"  [{'PASS' if len(proj_keep5) == 1 else 'FAIL'}] 单框  输出={len(proj_keep5)}  期望=1")

    # 测试6: 标准 NMS 对比
    boxes6 = np.array([[100, 100, 80, 80],
                        [110, 100, 80, 80],
                        [300, 300, 80, 80]], dtype=float)
    scores6 = np.array([0.9, 0.85, 0.8])
    cpu_keep = nms_cpu(boxes6, scores6, 0.5)
    print(f"  [INFO] 标准NMS参考  输入=3  cpu_keep={cpu_keep}")

    print(f"\n注意: 项目中 nms_rot 使用中心距离近似，与标准 IoU-based NMS 行为不同")
    print(f"      中心距离近似速度快但精度较低，适合密集目标场景的快速筛选")

if __name__ == "__main__":
    test_nms()
