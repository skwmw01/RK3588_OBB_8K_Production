"""decode_generic 向量化前后对比测试
验证正确性 + 测量加速比
"""
import numpy as np
import math
import time

# ============================================================
# 原版（Python 循环）
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def decode_generic_old(outs, imgsz, ox=0, oy=0, scale=1.0, conf_th=0.3):
    """原版：逐像素 Python 循环"""
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
        cp = sigmoid(clsl)
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
            dets.append([
                cx * scale + ox, cy * scale + oy,
                bw * scale, bh * scale,
                ang, float(cf[h, w]), int(ci[h, w])
            ])
    return dets

# ============================================================
# 新版（NumPy 向量化）
# ============================================================
def decode_generic_new(outs, imgsz, ox=0, oy=0, scale=1.0, conf_th=0.3):
    """新版：全向量化"""
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
        cp = sigmoid(clsl)
        ci = cp.argmax(axis=0)
        cf = cp.max(axis=0)
        m = cf > conf_th
        if not m.any():
            continue
        # 向量化
        hs, ws = np.where(m)
        l = xd[0, hs, ws]
        t = xd[1, hs, ws]
        r = xd[2, hs, ws]
        b = xd[3, hs, ws]
        cx = (ws + 0.5 + (r - l) / 2) * s * scale + ox
        cy = (hs + 0.5 + (b - t) / 2) * s * scale + oy
        bw = (l + r) * s * scale
        bh = (t + b) * s * scale
        ai = ao + hs * W + ws
        ang = (angle[ai] - 0.25) * math.pi
        conf_vals = cf[hs, ws]
        cls_vals = ci[hs, ws]
        det = np.stack([cx, cy, bw, bh, ang, conf_vals, cls_vals.astype(np.float32)], axis=1)
        dets.append(det)
    if dets:
        return np.concatenate(dets, axis=0).tolist()
    return []

# ============================================================
# 测试
# ============================================================
def make_mock_outs(imgsz, seed=42):
    """生成模拟 NPU 输出"""
    np.random.seed(seed)
    g = imgsz // 8
    outs = [
        np.random.randn(1, 79, g, g).astype(np.float32),
        np.random.randn(1, 79, g//2, g//2).astype(np.float32),
        np.random.randn(1, 79, g//4, g//4).astype(np.float32),
        np.random.randn(1, 1, g*g + (g//2)*(g//2) + (g//4)*(g//4)).astype(np.float32),
    ]
    return outs

def run_test():
    print("=" * 60)
    print("decode_generic 向量化对比测试")
    print("=" * 60)

    all_pass = True

    # --- 正确性测试 ---
    print("\n--- 正确性测试 ---")
    for imgsz in [64, 320, 640, 1024]:
        for conf_th in [0.0, 0.3, 0.5]:
            for seed in [42, 123, 999]:
                outs = make_mock_outs(imgsz, seed)
                old = decode_generic_old(outs, imgsz, ox=100, oy=200, scale=1.5, conf_th=conf_th)
                new = decode_generic_new(outs, imgsz, ox=100, oy=200, scale=1.5, conf_th=conf_th)

                ok = len(old) == len(new)
                if ok and len(old) > 0:
                    old_arr = np.array(old)
                    new_arr = np.array(new)
                    max_err = np.max(np.abs(old_arr - new_arr))
                    ok = max_err < 1e-4
                else:
                    max_err = 0.0

                if not ok:
                    all_pass = False
                status = "PASS" if ok else "FAIL"
                print(f"  [{status}] imgsz={imgsz}  conf={conf_th}  seed={seed}  "
                      f"n_old={len(old)}  n_new={len(new)}  max_err={max_err:.2e}")

    # --- 性能测试 ---
    print("\n--- 性能测试 ---")
    for imgsz in [320, 640, 1024]:
        outs = make_mock_outs(imgsz)
        n_iter = 20

        # 预热
        decode_generic_old(outs, imgsz, conf_th=0.3)
        decode_generic_new(outs, imgsz, conf_th=0.3)

        t0 = time.perf_counter()
        for _ in range(n_iter):
            decode_generic_old(outs, imgsz, conf_th=0.3)
        t_old = (time.perf_counter() - t0) / n_iter * 1000

        t0 = time.perf_counter()
        for _ in range(n_iter):
            decode_generic_new(outs, imgsz, conf_th=0.3)
        t_new = (time.perf_counter() - t0) / n_iter * 1000

        speedup = t_old / max(t_new, 0.001)
        print(f"  imgsz={imgsz}:  old={t_old:.2f}ms  new={t_new:.2f}ms  "
              f"加速比={speedup:.1f}x")

    print(f"\n结论: 向量化版本 {'完全正确' if all_pass else '存在差异'}")
    return all_pass

if __name__ == "__main__":
    run_test()
