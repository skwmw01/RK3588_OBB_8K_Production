"""decode_generic 三代对比测试
P0 原版(Python循环) vs P0 向量化(NumPy) vs P1 Numba JIT
"""
import numpy as np
import math
import time
from numba import njit

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
# P0: NumPy 向量化
# ============================================================
def decode_generic_numpy(outs, imgsz, ox=0, oy=0, scale=1.0, conf_th=0.3):
    """P0：NumPy 全向量化"""
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
# P1: Numba JIT
# ============================================================
@njit(cache=True)
def _decode_kernel(xd_l, xd_t, xd_r, xd_b, cf, ci, angle,
                   H, W, s, ao, scale, ox, oy, conf_th):
    """Numba JIT 解码内核"""
    out = np.empty((H * W, 7), np.float32)
    cnt = 0
    pi = 3.14159265358979
    for h in range(H):
        for w in range(W):
            c = cf[h, w]
            if c <= conf_th:
                continue
            l = xd_l[h, w]
            t = xd_t[h, w]
            r = xd_r[h, w]
            b = xd_b[h, w]
            cx = (w + 0.5 + (r - l) * 0.5) * s * scale + ox
            cy = (h + 0.5 + (b - t) * 0.5) * s * scale + oy
            bw = (l + r) * s * scale
            bh = (t + b) * s * scale
            ai = ao + h * W + w
            ang = (angle[ai] - 0.25) * pi
            out[cnt, 0] = cx
            out[cnt, 1] = cy
            out[cnt, 2] = bw
            out[cnt, 3] = bh
            out[cnt, 4] = ang
            out[cnt, 5] = c
            out[cnt, 6] = ci[h, w]
            cnt += 1
    return out[:cnt]

def decode_generic_numba(outs, imgsz, ox=0, oy=0, scale=1.0, conf_th=0.3):
    """P1：Numba JIT 加速"""
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
        ci = cp.argmax(axis=0).astype(np.int32)
        cf = cp.max(axis=0)
        det = _decode_kernel(xd[0], xd[1], xd[2], xd[3],
                             cf, ci, angle, H, W, s, ao, scale, ox, oy, conf_th)
        if len(det) > 0:
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
    print("=" * 65)
    print("decode_generic 三代对比: Python循环 vs NumPy向量化 vs Numba JIT")
    print("=" * 65)

    all_pass = True

    # --- 正确性测试 ---
    print("\n--- 正确性测试 ---")
    for imgsz in [64, 320, 640, 1024]:
        for conf_th in [0.0, 0.3]:
            for seed in [42, 999]:
                outs = make_mock_outs(imgsz, seed)
                old = decode_generic_old(outs, imgsz, ox=100, oy=200, scale=1.5, conf_th=conf_th)
                npy = decode_generic_numpy(outs, imgsz, ox=100, oy=200, scale=1.5, conf_th=conf_th)
                nmb = decode_generic_numba(outs, imgsz, ox=100, oy=200, scale=1.5, conf_th=conf_th)

                for name, res in [("NumPy", npy), ("Numba", nmb)]:
                    ok = len(old) == len(res)
                    if ok and len(old) > 0:
                        max_err = np.max(np.abs(np.array(old) - np.array(res)))
                        ok = max_err < 1e-4
                    else:
                        max_err = 0.0
                    if not ok:
                        all_pass = False
                    status = "PASS" if ok else "FAIL"
                    print(f"  [{status}] vs {name:5s}  imgsz={imgsz}  conf={conf_th}  seed={seed}  "
                          f"n={len(old)}/{len(res)}  max_err={max_err:.2e}")

    # --- 性能测试 ---
    print("\n--- 性能测试 ---")
    print(f"  {'imgsz':>6}  {'Python':>10}  {'NumPy':>10}  {'Numba':>10}  {'NumPy加速':>10}  {'Numba加速':>10}")
    for imgsz in [320, 640, 1024]:
        outs = make_mock_outs(imgsz)
        n_iter = 20

        # 预热（Numba 首次调用含编译开销）
        decode_generic_old(outs, imgsz, conf_th=0.3)
        decode_generic_numpy(outs, imgsz, conf_th=0.3)
        decode_generic_numba(outs, imgsz, conf_th=0.3)

        t0 = time.perf_counter()
        for _ in range(n_iter):
            decode_generic_old(outs, imgsz, conf_th=0.3)
        t_old = (time.perf_counter() - t0) / n_iter * 1000

        t0 = time.perf_counter()
        for _ in range(n_iter):
            decode_generic_numpy(outs, imgsz, conf_th=0.3)
        t_npy = (time.perf_counter() - t0) / n_iter * 1000

        t0 = time.perf_counter()
        for _ in range(n_iter):
            decode_generic_numba(outs, imgsz, conf_th=0.3)
        t_nmb = (time.perf_counter() - t0) / n_iter * 1000

        print(f"  {imgsz:>6}  {t_old:>8.2f}ms  {t_npy:>8.2f}ms  {t_nmb:>8.2f}ms  "
              f"{t_old/t_npy:>8.1f}x  {t_old/t_nmb:>8.1f}x")

    print(f"\n结论: {'全部正确 ✅' if all_pass else '存在差异 ❌'}")
    return all_pass

if __name__ == "__main__":
    run_test()
