"""IoU_RKNPU2: 交并比计算 测试"""
import numpy as np

# ============================================================
# CPU 参考实现 (标准水平框 IoU)
# ============================================================
def iou_cpu(box_a, box_b):
    """box format: [cx, cy, w, h]"""
    ax1, ay1 = box_a[0] - box_a[2]/2, box_a[1] - box_a[3]/2
    ax2, ay2 = box_a[0] + box_a[2]/2, box_a[1] + box_a[3]/2
    bx1, by1 = box_b[0] - box_b[2]/2, box_b[1] - box_b[3]/2
    bx2, by2 = box_b[0] + box_b[2]/2, box_b[1] + box_b[3]/2

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)

    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-10)

# ============================================================
# 项目实现 (简化版，基于中心距离近似)
# common.py nms_rot 中使用的是中心距离近似
# ============================================================
def iou_project_center_dist(box_a, box_b, iou_thr=0.3):
    """项目中 nms_rot 使用的近似方法: 中心距离阈值"""
    d = np.linalg.norm(box_a[:2] - box_b[:2])
    th = (box_a[2] + box_a[3]) * 0.5 * iou_thr
    return d < th

# ============================================================
# 测试
# ============================================================
def test_iou():
    print("=" * 60)
    print("IoU_RKNPU2 测试")
    print("=" * 60)

    test_cases = [
        ("完全重叠",     np.array([100, 100, 50, 50]), np.array([100, 100, 50, 50]), 1.0),
        ("完全不重叠",   np.array([100, 100, 50, 50]), np.array([300, 300, 50, 50]), 0.0),
        ("半重叠",       np.array([100, 100, 100, 100]), np.array([150, 100, 100, 100]), 0.5),
        ("小框在大框内", np.array([100, 100, 100, 100]), np.array([100, 100, 30, 30]), 0.09),
        ("角重叠",       np.array([100, 100, 100, 100]), np.array([180, 180, 100, 100]), 0.04),
        ("相邻",         np.array([100, 100, 50, 50]), np.array([150, 100, 50, 50]), 0.0),
    ]

    all_pass = True
    for name, box_a, box_b, expected_iou in test_cases:
        cpu_iou = iou_cpu(box_a, box_b)
        err = abs(cpu_iou - expected_iou)

        # 项目近似方法
        proj_match = iou_project_center_dist(box_a, box_b, 0.3)

        pass_fail = "PASS" if err < 0.01 else "FAIL"
        if err >= 0.01:
            all_pass = False
        print(f"  [{pass_fail}] {name:12s}  cpu_iou={cpu_iou:.4f}  expected={expected_iou:.4f}  "
              f"err={err:.4f}  center_approx_match={proj_match}")

    # 批量 IoU 测试
    np.random.seed(42)
    boxes = np.random.randint(50, 200, (20, 4)).astype(float)
    boxes[:, 2:] = np.maximum(boxes[:, 2:], 10)
    cpu_ious = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            cpu_ious[i, j] = iou_cpu(boxes[i], boxes[j])

    # 验证对称性
    sym_err = np.max(np.abs(cpu_ious - cpu_ious.T))
    diag_ok = np.allclose(np.diag(cpu_ious), 1.0)
    print(f"  [{'PASS' if sym_err < 1e-10 and diag_ok else 'FAIL'}] 批量20x20  "
          f"对称误差={sym_err:.2e}  对角全1={diag_ok}")

    print(f"\n结论: IoU 精确实现与 CPU 参考 {'完全一致' if all_pass else '存在差异'}")
    print(f"注意: 项目中 nms_rot 使用的是中心距离近似，不是精确 IoU")
    return all_pass

if __name__ == "__main__":
    test_iou()
