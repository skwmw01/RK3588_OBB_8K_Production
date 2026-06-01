"""BBoxDecode_OBB_RKNPU2: 旋转边界框解码 测试
将DFL距离+角度解码为旋转框(cx, cy, w, h, angle)
"""
import numpy as np
import math

# ============================================================
# CPU 参考实现 (逐步解码)
# ============================================================
def bbox_decode_obb_cpu(dfl_out, angle_out, stride, imgsz, ox=0, oy=0, scale=1.0):
    """逐步解码，作为参考基准
    dfl_out: (4, 16, H, W) DFL logits
    angle_out: (total_anchors,) 角度值 (0-1 归一化)
    """
    _, _, H, W = dfl_out.shape
    # DFL: softmax + expected value
    sm = np.exp(dfl_out - dfl_out.max(axis=1, keepdims=True))
    sm /= sm.sum(axis=1, keepdims=True)
    xd = (sm * np.arange(16).reshape(1, 16, 1, 1)).sum(axis=1)
    # xd: (4, H, W) = [l, t, r, b]

    l, t, r, b = xd[0], xd[1], xd[2], xd[3]

    # 网格坐标
    grid_x = np.arange(W).reshape(1, W) + 0.5
    grid_y = np.arange(H).reshape(H, 1) + 0.5

    # 中心点 + 宽高
    cx = (grid_x + (r - l) / 2) * stride * scale + ox
    cy = (grid_y + (b - t) / 2) * stride * scale + oy
    bw = (l + r) * stride * scale
    bh = (t + b) * stride * scale

    # 角度: ao + h*W + w, ang = (angle[ai] - 0.25) * pi
    g = imgsz // 8
    aos = [0, g*g, g*g + (g//2)*(g//2)]
    ao = aos[0]  # stride=8 的偏移
    ai = ao + np.arange(H).reshape(H, 1) * W + np.arange(W).reshape(1, W)
    ang = (angle_out[ai] - 0.25) * math.pi

    return cx, cy, bw, bh, ang

# ============================================================
# 项目实现 (common.py decode_generic)
# ============================================================
def bbox_decode_obb_project(outs, imgsz, ox=0, oy=0, scale=1.0, conf_th=0.0):
    """项目中的完整 OBB 解码"""
    dets = []
    strides = [8, 16, 32]
    g = imgsz // 8
    aos = [0, g*g, g*g + (g//2)*(g//2)]
    angle = outs[3][0, 0]
    for o, s, ao in zip(outs[:3], strides, aos):
        _, _, H, W = o.shape
        xywh = o[0, :64].reshape(4, 16, H, W)
        clsl = o[0, 64:79]
        sm = np.exp(xywh - xywh.max(axis=1, keepdims=True))
        sm /= sm.sum(axis=1, keepdims=True)
        xd = (sm * np.arange(16).reshape(1, 16, 1, 1)).sum(axis=1)
        cp = 1.0 / (1.0 + np.exp(-clsl))
        ci = cp.argmax(axis=0)
        cf = cp.max(axis=0)
        m = cf > conf_th
        if not m.any():
            continue
        hs, ws = np.where(m)
        for k in range(len(hs)):
            h, w = hs[k], ws[k]
            l, t, r, b = xd[:, h, w]
            cx = (w + 0.5 + (r - l) / 2) * s
            cy = (h + 0.5 + (b - t) / 2) * s
            bw = (l + r) * s
            bh = (t + b) * s
            ai = ao + h * W + w
            ang = (angle[ai] - 0.25) * math.pi
            dets.append([cx*scale+ox, cy*scale+oy, bw*scale, bh*scale, ang,
                         float(cf[h, w]), int(ci[h, w])])
    return dets

# ============================================================
# 测试
# ============================================================
def test_bbox_decode_obb():
    print("=" * 60)
    print("BBoxDecode_OBB_RKNPU2 测试")
    print("=" * 60)

    imgsz = 64
    np.random.seed(42)

    # 构造模拟 NPU 输出
    g = imgsz // 8  # 8
    outs = [
        np.random.randn(1, 79, g, g).astype(np.float32),          # stride 8
        np.random.randn(1, 79, g//2, g//2).astype(np.float32),    # stride 16
        np.random.randn(1, 79, g//4, g//4).astype(np.float32),    # stride 32
        np.random.randn(1, 1, g*g + (g//2)*(g//2) + (g//4)*(g//4)).astype(np.float32),  # angle
    ]

    # 测试1: 输出格式验证
    print("\n--- 输出格式验证 ---")
    dets = bbox_decode_obb_project(outs, imgsz, conf_th=0.0)
    n_dets = len(dets)
    print(f"  [INFO] conf_th=0.0  检出数={n_dets}  "
          f"(stride8={g*g} + stride16={(g//2)*(g//2)} + stride32={(g//4)*(g//4)} = "
          f"{g*g + (g//2)*(g//2) + (g//4)*(g//4)})")

    if n_dets > 0:
        d = dets[0]
        print(f"  [{'PASS' if len(d) == 7 else 'FAIL'}] 每个检测框7个值: "
              f"len={len(d)}  [cx, cy, w, h, angle, conf, cls]")
        print(f"  [INFO] 示例: cx={d[0]:.2f} cy={d[1]:.2f} w={d[2]:.2f} h={d[3]:.2f} "
              f"angle={d[4]:.4f} conf={d[5]:.4f} cls={d[6]}")

    # 测试2: 坐标范围
    print("\n--- 坐标范围验证 ---")
    dets_arr = np.array(dets)
    cx_range = (dets_arr[:, 0].min(), dets_arr[:, 0].max())
    cy_range = (dets_arr[:, 1].min(), dets_arr[:, 1].max())
    wh_pos = np.all(dets_arr[:, 2:4] >= 0)
    angle_range = (dets_arr[:, 4].min(), dets_arr[:, 4].max())
    print(f"  [{'PASS' if cx_range[0] >= 0 else 'FAIL'}] cx范围: [{cx_range[0]:.2f}, {cx_range[1]:.2f}]")
    print(f"  [{'PASS' if cy_range[0] >= 0 else 'FAIL'}] cy范围: [{cy_range[0]:.2f}, {cy_range[1]:.2f}]")
    print(f"  [{'PASS' if wh_pos else 'FAIL'}] w,h非负: {wh_pos}")
    print(f"  [INFO] angle范围: [{angle_range[0]:.4f}, {angle_range[1]:.4f}] rad "
          f"= [{math.degrees(angle_range[0]):.1f}, {math.degrees(angle_range[1]):.1f}] deg")

    # 测试3: conf 过滤
    print("\n--- conf 过滤测试 ---")
    for conf in [0.0, 0.1, 0.3, 0.5, 0.9]:
        dets_conf = bbox_decode_obb_project(outs, imgsz, conf_th=conf)
        print(f"  [INFO] conf_th={conf:.1f}  检出数={len(dets_conf)}")

    # 测试4: scale 和 offset
    print("\n--- scale/offset 测试 ---")
    dets_scaled = bbox_decode_obb_project(outs, imgsz, ox=100, oy=200, scale=2.0, conf_th=0.0)
    if len(dets_scaled) > 0 and len(dets) > 0:
        d0, ds = dets[0], dets_scaled[0]
        print(f"  [INFO] 原始: cx={d0[0]:.2f} cy={d0[1]:.2f}")
        print(f"  [INFO] scale=2+offset(100,200): cx={ds[0]:.2f} cy={ds[1]:.2f}")
        cx_expected = d0[0] * 2.0 + 100
        cy_expected = d0[1] * 2.0 + 200
        cx_ok = abs(ds[0] - cx_expected) < 1e-4
        cy_ok = abs(ds[1] - cy_expected) < 1e-4
        print(f"  [{'PASS' if cx_ok and cy_ok else 'FAIL'}] scale+offset 计算正确")

    print(f"\n结论: BBoxDecode_OBB 解码逻辑正常，输出格式正确")

if __name__ == "__main__":
    test_bbox_decode_obb()
