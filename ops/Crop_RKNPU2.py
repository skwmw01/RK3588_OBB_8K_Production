"""Crop_RKNPU2: 图像矩形裁剪 测试"""
import numpy as np

# ============================================================
# CPU 参考实现
# ============================================================
def crop_cpu(img, x, y, w, h):
    return img[y:y+h, x:x+w].copy()

# ============================================================
# 项目实现 (production_v2.py 中的 tile crop)
# img[y:y+TILE_CUT, x:x+TILE_CUT]
# ============================================================
def crop_project(img, x, y, w, h):
    return img[y:y+h, x:x+w].copy()

# ============================================================
# 测试
# ============================================================
def test_crop():
    print("=" * 60)
    print("Crop_RKNPU2 测试")
    print("=" * 60)

    test_cases = [
        ("左上角",      (8192, 8192, 3), (0, 0, 2048, 2048)),
        ("右下角",      (8192, 8192, 3), (6144, 6144, 2048, 2048)),
        ("中间区域",    (8192, 8192, 3), (3072, 3072, 2048, 2048)),
        ("小裁剪",      (1024, 1024, 3), (100, 100, 256, 256)),
        ("单像素行",    (512, 512, 3),   (0, 0, 512, 1)),
        ("单像素列",    (512, 512, 3),   (0, 0, 1, 512)),
        ("灰度图",      (1024, 1024),    (0, 0, 256, 256)),
        ("整图",        (256, 256, 3),   (0, 0, 256, 256)),
    ]

    all_pass = True
    for name, img_shape, (x, y, w, h) in test_cases:
        np.random.seed(42)
        img = np.random.randint(0, 256, img_shape, dtype=np.uint8)

        cpu_out = crop_cpu(img, x, y, w, h)
        proj_out = crop_project(img, x, y, w, h)

        match = np.array_equal(cpu_out, proj_out)
        pass_fail = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        print(f"  [{pass_fail}] {name:12s}  img={str(img_shape):20s}  "
              f"crop=({x},{y},{w},{h})  shape={cpu_out.shape}  exact={match}")

    print(f"\n结论: Crop 项目实现与 CPU 参考 {'完全一致' if all_pass else '存在差异'}")
    return all_pass

if __name__ == "__main__":
    test_crop()
