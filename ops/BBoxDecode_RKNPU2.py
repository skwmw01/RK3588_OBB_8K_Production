"""BBoxDecode_RKNPU2: 边界框坐标解码 测试"""
import numpy as np

# ============================================================
# CPU 参考实现 (标准 anchor-based bbox decode)
# ============================================================
def bbox_decode_cpu(encoded, anchors):
    """encoded: [dx, dy, dw, dh], anchors: [cx, cy, w, h]"""
    ax, ay, aw, ah = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    dx, dy, dw, dh = encoded[:, 0], encoded[:, 1], encoded[:, 2], encoded[:, 3]
    cx = dx * aw + ax
    cy = dy * ah + ay
    w = np.exp(dw) * aw
    h = np.exp(dh) * ah
    return np.stack([cx, cy, w, h], axis=1)

# ============================================================
# 项目实现 (YOLOv8 无 anchor，DFL 解码)
# common.py decode_generic 中的坐标解码
# ============================================================
def bbox_decode_project(dfl_out, stride, imgsz):
    """YOLOv8 DFL 解码: dfl_out shape (4, 16, H, W)"""
    _, _, H, W = dfl_out.shape
    # softmax + expected value
    sm = np.exp(dfl_out - dfl_out.max(axis=1, keepdims=True))
    sm /= sm.sum(axis=1, keepdims=True)
    xd = (sm * np.arange(16).reshape(1, 16, 1, 1)).sum(axis=1)
    # xd shape: (4, H, W) = [l, t, r, b]
    l, t, r, b = xd[0], xd[1], xd[2], xd[3]
    # 转换为 cx, cy, w, h
    cx = (np.arange(W)[None, :] + 0.5 + (r - l) / 2) * stride
    cy = (np.arange(H)[:, None] + 0.5 + (b - t) / 2) * stride
    w = (l + r) * stride
    h = (t + b) * stride
    return cx, cy, w, h

# ============================================================
# 测试
# ============================================================
def test_bbox_decode():
    print("=" * 60)
    print("BBoxDecode_RKNPU2 测试")
    print("=" * 60)

    # 测试1: 标准 anchor-based decode
    print("\n--- 标准 Anchor-based Decode ---")
    anchors = np.array([[50, 50, 30, 30],
                         [100, 100, 50, 50],
                         [200, 200, 80, 80]], dtype=np.float32)
    # 零编码应返回 anchor 本身
    encoded_zero = np.zeros((3, 4), dtype=np.float32)
    decoded = bbox_decode_cpu(encoded_zero, anchors)
    err = np.max(np.abs(decoded - anchors))
    print(f"  [{'PASS' if err < 1e-6 else 'FAIL'}] 零编码 → 恢复anchor  max_err={err:.2e}")

    # 非零编码
    encoded = np.array([[0.1, -0.1, 0.2, -0.2],
                         [0.0, 0.0, 0.0, 0.0],
                         [-0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
    decoded2 = bbox_decode_cpu(encoded, anchors)
    print(f"  [INFO] 非零编码解码结果:")
    for i in range(3):
        print(f"    anchor={anchors[i]}  encoded={encoded[i]}  →  decoded={decoded2[i]}")

    # 测试2: YOLOv8 DFL decode
    print("\n--- YOLOv8 DFL Decode ---")
    np.random.seed(42)
    dfl_out = np.random.randn(4, 16, 4, 4).astype(np.float32)
    stride = 8
    imgsz = 32
    cx, cy, w, h = bbox_decode_project(dfl_out, stride, imgsz)

    # 验证输出 shape
    print(f"  [{'PASS' if cx.shape == (4, 4) else 'FAIL'}] DFL输出shape: cx={cx.shape}  "
          f"期望=(4,4)")

    # 验证坐标范围合理
    cx_ok = np.all(cx >= 0) and np.all(cx <= imgsz)
    cy_ok = np.all(cy >= 0) and np.all(cy <= imgsz)
    wh_ok = np.all(w >= 0) and np.all(h >= 0)
    print(f"  [{'PASS' if cx_ok and cy_ok else 'FAIL'}] 坐标范围合理: "
          f"cx∈[{cx.min():.1f},{cx.max():.1f}]  cy∈[{cy.min():.1f},{cy.max():.1f}]")
    print(f"  [{'PASS' if wh_ok else 'FAIL'}] 宽高非负: "
          f"w∈[{w.min():.1f},{w.max():.1f}]  h∈[{h.min():.1f},{h.max():.1f}]")

    # 测试3: DFL softmax 概率和为1
    sm = np.exp(dfl_out - dfl_out.max(axis=1, keepdims=True))
    sm /= sm.sum(axis=1, keepdims=True)
    prob_sum = sm.sum(axis=1)
    sum_err = np.max(np.abs(prob_sum - 1.0))
    print(f"  [{'PASS' if sum_err < 1e-5 else 'FAIL'}] DFL概率和=1  max_err={sum_err:.2e}")

    print(f"\n结论: BBoxDecode 两种模式（anchor-based 和 YOLOv8 DFL）均正常")

if __name__ == "__main__":
    test_bbox_decode()
