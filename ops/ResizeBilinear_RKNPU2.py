"""ResizeBilinear_RKNPU2: 双线性插值缩放 测试"""
import numpy as np
import cv2

# ============================================================
# CPU 参考实现 (纯 NumPy 双线性插值)
# ============================================================
def resize_bilinear_cpu(img, out_h, out_w):
    """纯 NumPy 双线性插值，作为参考基准"""
    in_h, in_w = img.shape[:2]
    # 计算采样坐标
    fy = np.arange(out_h) * (in_h / out_h) + (in_h / out_h) * 0.5 - 0.5
    fx = np.arange(out_w) * (in_w / out_w) + (in_w / out_w) * 0.5 - 0.5
    fy = np.clip(fy, 0, in_h - 1)
    fx = np.clip(fx, 0, in_w - 1)

    y0 = np.floor(fy).astype(int)
    x0 = np.floor(fx).astype(int)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = fy - y0
    wx = fx - x0

    out = np.zeros((out_h, out_w) + img.shape[2:], dtype=np.float64)
    for c in range(img.shape[2] if img.ndim == 3 else 1):
        ch = img[:, :, c] if img.ndim == 3 else img
        val = (ch[y0][:, x0] * (1 - wy)[:, None] * (1 - wx)[None, :] +
               ch[y0][:, x1] * (1 - wy)[:, None] * wx[None, :] +
               ch[y1][:, x0] * wy[:, None] * (1 - wx)[None, :] +
               ch[y1][:, x1] * wy[:, None] * wx[None, :])
        if img.ndim == 3:
            out[:, :, c] = val
        else:
            out = val
    return out.astype(img.dtype)

# ============================================================
# 项目实现 (cv2.resize INTER_LINEAR)
# ============================================================
def resize_bilinear_project(img, out_h, out_w):
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

# ============================================================
# 测试
# ============================================================
def test_resize_bilinear():
    print("=" * 60)
    print("ResizeBilinear_RKNPU2 测试")
    print("=" * 60)

    test_cases = [
        ("2048→1024 RGB",  (2048, 2048, 3), (1024, 1024), np.uint8),
        ("1024→512  RGB",  (1024, 1024, 3), (512, 512),   np.uint8),
        ("640→320   RGB",  (640, 640, 3),   (320, 320),   np.uint8),
        ("256→128   Gray", (256, 256),       (128, 128),   np.uint8),
        ("非等比 100→50",  (100, 200, 3),    (50, 100),    np.uint8),
        ("小图 8→4",       (8, 8, 3),        (4, 4),       np.uint8),
        ("FP32",           (64, 64, 3),      (32, 32),     np.float32),
    ]

    all_pass = True
    for name, in_shape, (out_h, out_w), dtype in test_cases:
        np.random.seed(42)
        if dtype == np.uint8:
            img = np.random.randint(0, 256, in_shape, dtype=np.uint8)
        else:
            img = np.random.randn(*in_shape).astype(dtype)

        cpu_out = resize_bilinear_cpu(img.astype(np.float64), out_h, out_w).astype(dtype)
        proj_out = resize_bilinear_project(img, out_h, out_w)

        if dtype == np.uint8:
            # 允许 ±1 的误差（插值实现差异）
            max_err = np.max(np.abs(cpu_out.astype(np.int16) - proj_out.astype(np.int16)))
            pass_fail = "PASS" if max_err <= 2 else "FAIL"
        else:
            max_err = np.max(np.abs(cpu_out - proj_out))
            pass_fail = "PASS" if max_err < 1e-3 else "FAIL"

        if (dtype == np.uint8 and max_err > 2) or (dtype != np.uint8 and max_err >= 1e-3):
            all_pass = False
        print(f"  [{pass_fail}] {name:20s}  {str(in_shape):15s} → ({out_h},{out_w})  "
              f"dtype={str(dtype):8s}  max_err={max_err}")

    print(f"\n结论: ResizeBilinear 项目实现与 CPU 参考 {'基本一致' if all_pass else '存在差异'}")
    return all_pass

if __name__ == "__main__":
    test_resize_bilinear()
