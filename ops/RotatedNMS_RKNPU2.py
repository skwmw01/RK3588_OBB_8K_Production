"""RotatedNMS_RKNPU2: 旋转框非极大值抑制 测试
基于中心距离近似的旋转框去重
"""
import numpy as np
import math
import cv2

# ============================================================
# CPU 参考实现 (精确旋转 IoU-based NMS)
# ============================================================
def rotated_iou(a, b):
    """[cx, cy, w, h, angle(rad)] 精确旋转 IoU"""
    try:
        ra = ((float(a[0]), float(a[1])), (float(a[2]), float(a[3])), math.degrees(float(a[4])))
        rb = ((float(b[0]), float(b[1])), (float(b[2]), float(b[3])), math.degrees(float(b[4])))
        ip = cv2.rotatedRectangleIntersection(ra, rb)
        if ip[0] == 0:
            return 0.0
        ia = cv2.contourArea(ip[1])
        return ia / (a[2]*a[3] + b[2]*b[3] - ia) if ia > 0 else 0.0
    except:
        return 0.0

def rotated_nms_cpu(dets, iou_thr=0.3):
    """精确旋转 IoU-based NMS"""
    if not dets:
        return []
    dets = np.array(dets)
    # 按 conf 降序
    order = dets[:, 5].argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = np.array([rotated_iou(dets[i], dets[j]) for j in rest])
        mask = ious <= iou_thr
        order = rest[mask]
    return [dets[i] for i in keep]

# ============================================================
# 项目实现 (common.py nms_rot: 中心距离近似)
# ============================================================
def rotated_nms_project(dets, iou_thr=0.3):
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
def test_rotated_nms():
    print("=" * 60)
    print("RotatedNMS_RKNPU2 测试")
    print("=" * 60)

    # 测试1: 无重叠
    print("\n--- 无重叠框 ---")
    dets1 = [[100, 100, 50, 30, 0, 0.9, 0],
             [300, 300, 50, 30, 0, 0.8, 0],
             [500, 500, 50, 30, 0, 0.7, 0]]
    proj1 = rotated_nms_project(dets1, 0.3)
    cpu1 = rotated_nms_cpu(dets1, 0.3)
    print(f"  [{'PASS' if len(proj1) == 3 and len(cpu1) == 3 else 'FAIL'}] "
          f"输入=3  project={len(proj1)}  cpu={len(cpu1)}  期望=3")

    # 测试2: 完全重叠
    print("\n--- 完全重叠框 ---")
    dets2 = [[100, 100, 50, 30, 0, 0.9, 0],
             [100, 100, 50, 30, 0, 0.8, 0],
             [100, 100, 50, 30, 0, 0.7, 0]]
    proj2 = rotated_nms_project(dets2, 0.3)
    cpu2 = rotated_nms_cpu(dets2, 0.3)
    print(f"  [{'PASS' if len(proj2) == 1 and len(cpu2) == 1 else 'FAIL'}] "
          f"输入=3  project={len(proj2)}  cpu={len(cpu2)}  期望=1")

    # 测试3: 部分重叠（两种方法可能不同）
    print("\n--- 部分重叠框 ---")
    dets3 = [[100, 100, 80, 40, 0, 0.9, 0],
             [120, 100, 80, 40, 0, 0.85, 0],
             [300, 300, 80, 40, 0, 0.8, 0]]
    proj3 = rotated_nms_project(dets3, 0.3)
    cpu3 = rotated_nms_cpu(dets3, 0.3)
    print(f"  [INFO] 输入=3  project={len(proj3)}  cpu={len(cpu3)}")
    print(f"  (中心距离近似与精确IoU NMS可能不同，这是预期的)")

    # 测试4: 旋转框
    print("\n--- 旋转框 ---")
    dets4 = [[100, 100, 80, 40, 0, 0.9, 0],
             [100, 100, 80, 40, math.pi/4, 0.85, 0],
             [100, 100, 80, 40, math.pi/2, 0.8, 0]]
    proj4 = rotated_nms_project(dets4, 0.3)
    cpu4 = rotated_nms_cpu(dets4, 0.3)
    print(f"  [INFO] 旋转框(同中心不同角度)  project={len(proj4)}  cpu={len(cpu4)}")

    # 测试5: 空输入
    print("\n--- 空输入 ---")
    proj5 = rotated_nms_project([], 0.3)
    cpu5 = rotated_nms_cpu([], 0.3)
    print(f"  [{'PASS' if len(proj5) == 0 and len(cpu5) == 0 else 'FAIL'}] "
          f"空输入  project={len(proj5)}  cpu={len(cpu5)}")

    # 测试6: 单框
    print("\n--- 单框 ---")
    dets6 = [[100, 100, 50, 30, 0, 0.9, 0]]
    proj6 = rotated_nms_project(dets6, 0.3)
    cpu6 = rotated_nms_cpu(dets6, 0.3)
    print(f"  [{'PASS' if len(proj6) == 1 and len(cpu6) == 1 else 'FAIL'}] "
          f"单框  project={len(proj6)}  cpu={len(cpu6)}")

    # 测试7: conf 排序验证
    print("\n--- conf 排序验证 ---")
    dets7 = [[100, 100, 50, 30, 0, 0.5, 0],
             [100, 100, 50, 30, 0, 0.9, 0],
             [100, 100, 50, 30, 0, 0.7, 0]]
    proj7 = rotated_nms_project(dets7, 0.3)
    if len(proj7) > 0:
        print(f"  [{'PASS' if proj7[0][5] == 0.9 else 'FAIL'}] "
              f"最高conf优先保留: conf={proj7[0][5]}")

    # 测试8: 大量框性能对比
    print("\n--- 大量框性能对比 ---")
    import time
    np.random.seed(42)
    n = 100
    dets_many = []
    for _ in range(n):
        cx, cy = np.random.uniform(50, 500, 2)
        w, h = np.random.uniform(20, 80, 2)
        ang = np.random.uniform(-math.pi, math.pi)
        conf = np.random.uniform(0.1, 1.0)
        dets_many.append([cx, cy, w, h, ang, conf, 0])

    t0 = time.perf_counter()
    proj_many = rotated_nms_project(dets_many, 0.3)
    t_proj = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cpu_many = rotated_nms_cpu(dets_many, 0.3)
    t_cpu = (time.perf_counter() - t0) * 1000

    print(f"  [INFO] n={n}  project: {len(proj_many)}框 {t_proj:.1f}ms  "
          f"cpu: {len(cpu_many)}框 {t_cpu:.1f}ms  加速比: {t_cpu/max(t_proj,0.001):.1f}x")

    print(f"\n注意: 项目中 nms_rot 使用中心距离近似，速度快但精度低于精确IoU NMS")
    print(f"      中心距离阈值 = (w+h)/2 * iou_thr，是IoU的粗糙近似")

if __name__ == "__main__":
    test_rotated_nms()
