"""ColorConvert_BGR2RGB_RKNPU2: 色彩空间转换(BGR转RGB) 测试"""
import numpy as np
import cv2

# ============================================================
# CPU 参考实现 (纯 NumPy)
# ============================================================
def bgr2rgb_cpu(img):
    return img[:, :, ::-1].copy()

# ============================================================
# 项目实现 (cv2.cvtColor)
# ============================================================
def bgr2rgb_project(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ============================================================
# 测试
# ============================================================
def test_bgr2rgb():
    print("=" * 60)
    print("ColorConvert_BGR2RGB_RKNPU2 测试")
    print("=" * 60)

    test_cases = [
        ("随机图 1024",    (1024, 1024, 3)),
        ("随机图 256",     (256, 256, 3)),
        ("小图 8x8",       (8, 8, 3)),
        ("单行",           (1, 1024, 3)),
        ("单列",           (1024, 1, 3)),
        ("纯红 BGR=(0,0,255)", None),
        ("纯绿 BGR=(0,255,0)", None),
        ("纯蓝 BGR=(255,0,0)", None),
    ]

    all_pass = True
    for name, shape in test_cases:
        if shape is not None:
            np.random.seed(42)
            img = np.random.randint(0, 256, shape, dtype=np.uint8)
        else:
            img = np.zeros((4, 4, 3), dtype=np.uint8)
            if "纯红" in name:
                img[:, :, 2] = 255  # BGR: R=255
            elif "纯绿" in name:
                img[:, :, 1] = 255  # BGR: G=255
            elif "纯蓝" in name:
                img[:, :, 0] = 255  # BGR: B=255

        cpu_out = bgr2rgb_cpu(img)
        proj_out = bgr2rgb_project(img)

        match = np.array_equal(cpu_out, proj_out)
        pass_fail = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
            max_err = np.max(np.abs(cpu_out.astype(int) - proj_out.astype(int)))
            print(f"  [{pass_fail}] {name:25s}  shape={str(img.shape):15s}  max_err={max_err}")
        else:
            print(f"  [{pass_fail}] {name:25s}  shape={str(img.shape):15s}  exact={match}")

    print(f"\n结论: BGR2RGB 项目实现与 CPU 参考 {'完全一致' if all_pass else '存在差异'}")
    return all_pass

if __name__ == "__main__":
    test_bgr2rgb()
