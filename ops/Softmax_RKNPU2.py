"""Softmax_RKNPU2: 归一化指数函数概率化 测试"""
import numpy as np

# ============================================================
# CPU 参考实现 (数值稳定版)
# ============================================================
def softmax_cpu(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# ============================================================
# 项目实现 (common.py decode_generic 中的 softmax)
# 用于 DFL 16-bin 概率分布: shape (4, 16, H, W), axis=1
# ============================================================
def softmax_project(x, axis=1):
    x_max = x.max(axis=axis, keepdims=True)
    sm = np.exp(x - x_max)
    sm /= sm.sum(axis=axis, keepdims=True)
    return sm

# ============================================================
# 测试
# ============================================================
def test_softmax():
    print("=" * 60)
    print("Softmax_RKNPU2 测试")
    print("=" * 60)

    test_cases = [
        ("均匀输入",       np.ones((1, 4, 16, 8, 8), dtype=np.float32)),
        ("随机输入",       np.random.randn(1, 4, 16, 8, 8).astype(np.float32)),
        ("大数值",         np.random.uniform(100, 200, (1, 4, 16, 8, 8)).astype(np.float32)),
        ("负大数值",       np.random.uniform(-200, -100, (1, 4, 16, 8, 8)).astype(np.float32)),
        ("混合范围",       np.random.uniform(-50, 50, (1, 4, 16, 8, 8)).astype(np.float32)),
        ("单元素axis",     np.random.randn(1, 4, 1, 8, 8).astype(np.float32)),
        ("DFL实际shape",   np.random.randn(4, 16, 128, 128).astype(np.float32)),
    ]

    all_pass = True
    for name, x in test_cases:
        cpu_out = softmax_cpu(x, axis=1)
        proj_out = softmax_project(x, axis=1)
        max_err = np.max(np.abs(cpu_out - proj_out))
        # 验证概率和为1
        prob_sum = np.sum(proj_out, axis=1)
        sum_err = np.max(np.abs(prob_sum - 1.0))
        pass_fail = "PASS" if max_err < 1e-6 and sum_err < 1e-5 else "FAIL"
        if max_err >= 1e-6 or sum_err >= 1e-5:
            all_pass = False
        print(f"  [{pass_fail}] {name:15s}  shape={str(x.shape):25s}  "
              f"max_err={max_err:.2e}  sum_err={sum_err:.2e}")

    print(f"\n结论: Softmax 项目实现与 CPU 参考 {'完全一致' if all_pass else '存在差异'}")
    return all_pass

if __name__ == "__main__":
    test_softmax()
