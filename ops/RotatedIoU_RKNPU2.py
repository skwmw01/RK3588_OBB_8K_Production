"""RotatedIoU_RKNPU2: 旋转矩形交并比计算 测试
基于旋转矩形交集面积计算 IoU
"""
import numpy as np
import cv2
import math

# ============================================================
# CPU 参考实现 (cv2.rotatedRectangleIntersection)
# ============================================================
def rotated_iou_cpu(a, b):
    """a, b: [cx, cy, w, h, angle(rad)]"""
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

# ============================================================
# 项目实现 (common.py iou_rot)
# ============================================================
def rotated_iou_project(a, b):
    """项目中的旋转 IoU 实现"""
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

# ============================================================
# 测试
# ============================================================
def test_rotated_iou():
    print("=" * 60)
    print("RotatedIoU_RKNPU2 测试")
    print("=" * 60)

    all_pass = True

    test_cases = [
        ("完全重叠 0°",
         [100, 100, 50, 30, 0], [100, 100, 50, 30, 0], 1.0),
        ("完全不重叠",
         [100, 100, 50, 30, 0], [300, 300, 50, 30, 0], 0.0),
        ("相同位置 旋转90°",
         [100, 100, 50, 30, 0], [100, 100, 30, 50, math.pi/2], 1.0),
        ("相同位置 旋转45°",
         [100, 100, 50, 30, 0], [100, 100, 50, 30, math.pi/4], None),
        ("半重叠 水平",
         [100, 100, 100, 50, 0], [150, 100, 100, 50, 0], 0.5),
        ("角重叠",
         [100, 100, 100, 100, 0], [180, 180, 100, 100, 0], None),
        ("小框在大框内",
         [100, 100, 100, 100, 0], [100, 100, 30, 30, 0], None),
        ("旋转30°重叠",
         [100, 100, 80, 40, 0], [100, 100, 80, 40, math.pi/6], None),
    ]

    for name, a, b, expected in test_cases:
        cpu_iou = rotated_iou_cpu(a, b)
        proj_iou = rotated_iou_project(a, b)
        err = abs(cpu_iou - proj_iou)
        ok = err < 1e-6
        if not ok:
            all_pass = False

        if expected is not None:
            exp_err = abs(cpu_iou - expected)
            ok2 = exp_err < 0.01
            if not ok2:
                all_pass = False
            print(f"  [{'PASS' if ok and ok2 else 'FAIL'}] {name:20s}  "
                  f"cpu={cpu_iou:.4f}  proj={proj_iou:.4f}  expected={expected:.4f}  "
                  f"impl_err={err:.2e}  exp_err={exp_err:.4f}")
        else:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:20s}  "
                  f"cpu={cpu_iou:.4f}  proj={proj_iou:.4f}  impl_err={err:.2e}")

    # 对称性测试
    print("\n--- 对称性测试 ---")
    np.random.seed(42)
    n = 20
    boxes = []
    for _ in range(n):
        cx, cy = np.random.uniform(50, 300, 2)
        w, h = np.random.uniform(20, 100, 2)
        ang = np.random.uniform(-math.pi, math.pi)
        boxes.append([cx, cy, w, h, ang])

    sym_err_max = 0
    for i in range(n):
        for j in range(i+1, n):
            iou_ij = rotated_iou_project(boxes[i], boxes[j])
            iou_ji = rotated_iou_project(boxes[j], boxes[i])
            sym_err_max = max(sym_err_max, abs(iou_ij - iou_ji))

    print(f"  [{'PASS' if sym_err_max < 1e-6 else 'FAIL'}] IoU对称性  "
          f"max_err={sym_err_max:.2e}")

    # 自身IoU=1测试
    self_err_max = 0
    for i in range(n):
        iou_self = rotated_iou_project(boxes[i], boxes[i])
        self_err_max = max(self_err_max, abs(iou_self - 1.0))

    print(f"  [{'PASS' if self_err_max < 1e-6 else 'FAIL'}] 自身IoU=1  "
          f"max_err={self_err_max:.2e}")

    print(f"\n结论: RotatedIoU 项目实现与 CPU 参考 {'完全一致' if all_pass else '存在差异'}")
    return all_pass

if __name__ == "__main__":
    test_rotated_iou()
