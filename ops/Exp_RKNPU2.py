"""Exp_RKNPU2: 指数运算 (e为底) 测试"""
import numpy as np

# ============================================================
# CPU 参考实现
# ============================================================
def exp_cpu(x):
    return np.exp(x)

# ============================================================
# 项目实现 (用于 softmax 中的 exp)
# ============================================================
def exp_project(x):
    return np.exp(x)

# ============================================================
# 测试
# ============================================================
def test_exp():
    print("=" * 60)
    print("Exp_RKNPU2 测试")
    print("=" * 60)

    test_cases = [
        ("零向量",       np.zeros((1, 3, 8, 8), dtype=np.float32)),
        ("全1",          np.ones((1, 3, 8, 8), dtype=np.float32)),
        ("负值",         -np.ones((1, 3, 8, 8), dtype=np.float32)),
        ("小数值",       np.array([0.001, 0.01, 0.1], dtype=np.float32).reshape(1, 1, 1, 3)),
        ("正常范围",     np.random.uniform(-5, 5, (1, 3, 16, 16)).astype(np.float32)),
        ("较大值",       np.array([10.0, 20.0, 50.0], dtype=np.float32).reshape(1, 1, 1, 3)),
        ("接近0",        np.array([1e-10, -1e-10, 1e-38], dtype=np.float32).reshape(1, 1, 1, 3)),
    ]

    all_pass = True
    for name, x in test_cases:
        cpu_out = exp_cpu(x)
        proj_out = exp_project(x)
        # 用相对误差比较（exp 值域大时绝对误差也大）
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err = np.abs((cpu_out - proj_out) / np.where(cpu_out != 0, cpu_out, 1.0))
        max_rel_err = np.nanmax(rel_err)
        max_abs_err = np.nanmax(np.abs(cpu_out - proj_out))
        pass_fail = "PASS" if max_rel_err < 1e-5 else "FAIL"
        if max_rel_err >= 1e-5:
            all_pass = False
        print(f"  [{pass_fail}] {name:15s}  shape={str(x.shape):20s}  "
              f"max_rel_err={max_rel_err:.2e}  max_abs_err={max_abs_err:.2e}")

    print(f"\n结论: Exp 项目实现与 CPU 参考 {'完全一致' if all_pass else '存在差异'}")
    return all_pass

if __name__ == "__main__":
    test_exp()
