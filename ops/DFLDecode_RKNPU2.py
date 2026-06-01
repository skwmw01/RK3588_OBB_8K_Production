"""DFLDecode_RKNPU2: 分布焦点损失解码 测试
对16个距离bin做softmax后加权求和回归边框距离
"""
import numpy as np

# ============================================================
# CPU 参考实现 (标准 DFL 解码)
# ============================================================
def dfl_decode_cpu(logits, reg_max=16):
    """标准 DFL: softmax → weighted sum
    logits shape: (..., reg_max) 每个方向的距离分布
    """
    # softmax
    x_max = np.max(logits, axis=-1, keepdims=True)
    e_x = np.exp(logits - x_max)
    probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
    # weighted sum
    bins = np.arange(reg_max, dtype=np.float32)
    return np.sum(probs * bins, axis=-1)

# ============================================================
# 项目实现 (common.py decode_generic 中的 DFL)
# shape: (4, 16, H, W) → (4, H, W)
# ============================================================
def dfl_decode_project(logits_4d, reg_max=16):
    """项目中的 DFL 解码，输入 shape (4, 16, H, W)"""
    sm = np.exp(logits_4d - logits_4d.max(axis=1, keepdims=True))
    sm /= sm.sum(axis=1, keepdims=True)
    xd = (sm * np.arange(reg_max).reshape(1, reg_max, 1, 1)).sum(axis=1)
    return xd

# ============================================================
# 测试
# ============================================================
def test_dfl_decode():
    print("=" * 60)
    print("DFLDecode_RKNPU2 测试")
    print("=" * 60)

    all_pass = True

    # 测试1: 确定性分布 (one-hot)
    print("\n--- 确定性分布测试 ---")
    for bin_idx in [0, 7, 15]:
        logits = np.full((4, 16, 2, 2), -10.0, dtype=np.float32)
        logits[:, bin_idx, :, :] = 10.0
        result = dfl_decode_project(logits)
        expected = float(bin_idx)
        err = np.max(np.abs(result - expected))
        ok = err < 1e-4
        if not ok:
            all_pass = False
        print(f"  [{'PASS' if ok else 'FAIL'}] one-hot bin={bin_idx:2d}  "
              f"结果={result[0,0,0]:.6f}  期望={expected:.1f}  err={err:.2e}")

    # 测试2: 均匀分布
    print("\n--- 均匀分布测试 ---")
    logits_uniform = np.zeros((4, 16, 4, 4), dtype=np.float32)
    result_uniform = dfl_decode_project(logits_uniform)
    expected_uniform = 7.5  # (0+1+...+15)/16 = 7.5
    err = np.max(np.abs(result_uniform - expected_uniform))
    ok = err < 1e-4
    if not ok:
        all_pass = False
    print(f"  [{'PASS' if ok else 'FAIL'}] 均匀分布  结果={result_uniform[0,0,0]:.6f}  "
          f"期望={expected_uniform:.1f}  err={err:.2e}")

    # 测试3: 偏态分布
    print("\n--- 偏态分布测试 ---")
    logits_skew = np.zeros((4, 16, 2, 2), dtype=np.float32)
    logits_skew[:, :8, :, :] = 2.0   # 前半部分概率高
    logits_skew[:, 8:, :, :] = -2.0
    result_skew = dfl_decode_project(logits_skew)
    # 偏向小值方向
    print(f"  [INFO] 偏态分布(前半高)  结果={result_skew[0,0,0]:.4f}  期望<7.5")
    ok = result_skew[0, 0, 0] < 7.5
    if not ok:
        all_pass = False
    print(f"  [{'PASS' if ok else 'FAIL'}] 偏向小值方向")

    # 测试4: 与标准 DFL 参考对比
    print("\n--- 与标准参考对比 ---")
    np.random.seed(42)
    for i in range(5):
        logits = np.random.randn(4, 16, 8, 8).astype(np.float32)
        proj_result = dfl_decode_project(logits)
        # 用标准参考逐元素验证
        cpu_result = np.zeros((4, 8, 8), dtype=np.float32)
        for d in range(4):
            for y in range(8):
                for x in range(8):
                    cpu_result[d, y, x] = dfl_decode_cpu(logits[d, :, y, x])
        err = np.max(np.abs(proj_result - cpu_result))
        ok = err < 1e-5
        if not ok:
            all_pass = False
        print(f"  [{'PASS' if ok else 'FAIL'}] 随机样本{i+1}  shape=(4,16,8,8)  max_err={err:.2e}")

    # 测试5: 概率和验证
    print("\n--- 概率和验证 ---")
    np.random.seed(123)
    logits = np.random.randn(4, 16, 16, 16).astype(np.float32)
    sm = np.exp(logits - logits.max(axis=1, keepdims=True))
    sm /= sm.sum(axis=1, keepdims=True)
    prob_sum = sm.sum(axis=1)
    sum_err = np.max(np.abs(prob_sum - 1.0))
    ok = sum_err < 1e-5
    if not ok:
        all_pass = False
    print(f"  [{'PASS' if ok else 'FAIL'}] softmax概率和=1  max_err={sum_err:.2e}")

    # 测试6: 输出范围
    result_range = dfl_decode_project(logits)
    range_ok = np.all(result_range >= 0) and np.all(result_range <= 15)
    if not range_ok:
        all_pass = False
    print(f"  [{'PASS' if range_ok else 'FAIL'}] 输出范围 [0,15]: "
          f"[{result_range.min():.4f}, {result_range.max():.4f}]")

    print(f"\n结论: DFLDecode 项目实现与 CPU 参考 {'完全一致' if all_pass else '存在差异'}")
    return all_pass

if __name__ == "__main__":
    test_dfl_decode()
