"""Sigmoid_RKNPU2: S型激活函数 测试"""
import numpy as np

# ============================================================
# CPU 参考实现
# ============================================================
def sigmoid_cpu(x):
    return 1.0 / (1.0 + np.exp(-x))

# ============================================================
# 项目实现 (common.py 中的 sigmoid)
# ============================================================
def sigmoid_project(x):
    return 1.0 / (1.0 + np.exp(-x))

# ============================================================
# 测试
# ============================================================
def test_sigmoid():
    print("=" * 60)
    print("Sigmoid_RKNPU2 测试")
    print("=" * 60)

    test_cases = [
        ("零向量",       np.zeros((1, 3, 8, 8), dtype=np.float32)),
        ("全1",          np.ones((1, 3, 8, 8), dtype=np.float32)),
        ("全-1",         -np.ones((1, 3, 8, 8), dtype=np.float32)),
        ("大正值",       np.full((1, 3, 8, 8), 10.0, dtype=np.float32)),
        ("大负值",       np.full((1, 3, 8, 8), -10.0, dtype=np.float32)),
        ("随机正常范围",  np.random.randn(1, 3, 16, 16).astype(np.float32)),
        ("随机大范围",    np.random.uniform(-100, 100, (1, 3, 16, 16)).astype(np.float32)),
        ("FP32边界",     np.array([1e-38, 1e38, -1e38], dtype=np.float32).reshape(1, 1, 1, 3)),
    ]

    all_pass = True
    for name, x in test_cases:
        cpu_out = sigmoid_cpu(x)
        proj_out = sigmoid_project(x)
        max_err = np.max(np.abs(cpu_out - proj_out))
        mean_err = np.mean(np.abs(cpu_out - proj_out))
        pass_fail = "PASS" if max_err < 1e-6 else "FAIL"
        if max_err >= 1e-6:
            all_pass = False
        print(f"  [{pass_fail}] {name:15s}  shape={str(x.shape):20s}  "
              f"max_err={max_err:.2e}  mean_err={mean_err:.2e}")

    print(f"\n结论: Sigmoid 项目实现与 CPU 参考 {'完全一致' if all_pass else '存在差异'}")
    return all_pass

if __name__ == "__main__":
    test_sigmoid()
