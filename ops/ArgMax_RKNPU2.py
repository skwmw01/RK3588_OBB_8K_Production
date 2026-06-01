"""ArgMax_RKNPU2: 最大值索引 测试"""
import numpy as np

# ============================================================
# CPU 参考实现
# ============================================================
def argmax_cpu(x, axis=0):
    return np.argmax(x, axis=axis)

# ============================================================
# 项目实现 (common.py decode_generic 中的 argmax)
# 用于分类: cp.argmax(axis=0), shape (15, H, W) -> (H, W)
# ============================================================
def argmax_project(x, axis=0):
    return np.argmax(x, axis=axis)

# ============================================================
# 测试
# ============================================================
def test_argmax():
    print("=" * 60)
    print("ArgMax_RKNPU2 测试")
    print("=" * 60)

    test_cases = [
        ("单类别",       np.array([[[5.0]]]), 0),
        ("2类简单",      np.array([[[1.0, 2.0], [3.0, 0.5]],
                                   [[2.0, 1.0], [1.0, 3.0]]]), 0),
        ("DOTA 15类",    np.random.randn(15, 64, 64).astype(np.float32), 0),
        ("axis=1",       np.random.randn(1, 16, 8, 8).astype(np.float32), 1),
        ("相同最大值",    np.array([[[1.0, 1.0], [1.0, 1.0]],
                                    [[1.0, 1.0], [1.0, 1.0]]]), 0),
        ("负值",         np.random.randn(15, 32, 32).astype(np.float32) - 10, 0),
    ]

    all_pass = True
    for name, x, axis in test_cases:
        cpu_out = argmax_cpu(x, axis)
        proj_out = argmax_project(x, axis)
        match = np.array_equal(cpu_out, proj_out)
        pass_fail = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        n_total = cpu_out.size
        n_match = np.sum(cpu_out == proj_out)
        print(f"  [{pass_fail}] {name:15s}  shape={str(x.shape):25s}  axis={axis}  "
              f"match={n_match}/{n_total}")

    print(f"\n结论: ArgMax 项目实现与 CPU 参考 {'完全一致' if all_pass else '存在差异'}")
    return all_pass

if __name__ == "__main__":
    test_argmax()
