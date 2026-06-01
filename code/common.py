"""
通用函数：decode / NMS / recall / GT 生成 / 并行推理
所有实验脚本共用这些函数。
"""
import cv2, numpy as np, threading, queue, math, time
from rknnlite.api import RKNNLite

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

# =========================================================
# 解码内核：优先 Numba JIT，回退 NumPy 向量化
# =========================================================
def _decode_kernel_fallback(xd_l, xd_t, xd_r, xd_b, cf, ci, angle,
                            H, W, s, ao, scale, ox, oy, conf_th):
    """NumPy 向量化回退（无 numba 时使用）"""
    m = cf > conf_th
    if not m.any():
        return np.empty((0, 7), np.float32)
    hs, ws = np.where(m)
    l = xd_l[hs, ws]; t = xd_t[hs, ws]
    r = xd_r[hs, ws]; b = xd_b[hs, ws]
    cx = (ws + 0.5 + (r - l) * 0.5) * s * scale + ox
    cy = (hs + 0.5 + (b - t) * 0.5) * s * scale + oy
    bw = (l + r) * s * scale
    bh = (t + b) * s * scale
    ai = ao + hs * W + ws
    ang = (angle[ai] - 0.25) * math.pi
    return np.stack([cx, cy, bw, bh, ang, cf[hs, ws], ci[hs, ws].astype(np.float32)], axis=1)

if _HAS_NUMBA:
    @njit(cache=True)
    def _decode_kernel(xd_l, xd_t, xd_r, xd_b, cf, ci, angle,
                       H, W, s, ao, scale, ox, oy, conf_th):
        """Numba JIT 加速：从 DFL 输出解码检测框"""
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
else:
    _decode_kernel = _decode_kernel_fallback

# =========================================================
# 路径常量
# =========================================================
MODEL_DIR = "/home/orangepi/ablation/models"
IMG_PATH = "/home/orangepi/ablation/test_8k.jpg"

MODEL_1024 = f"{MODEL_DIR}/yolov8n-obb_i8_1024_airockchip.rknn"
MODEL_2048 = f"{MODEL_DIR}/yolov8n-obb_i8_2048.rknn"
MODEL_2048_FP16 = f"{MODEL_DIR}/yolov8n-obb_fp16_2048.rknn"
MODEL_640 = f"{MODEL_DIR}/yolov8n-obb_i8_640.rknn"
MODEL_2048_KL = f"{MODEL_DIR}/yolov8n-obb_kl_channel_w8a8_2048.rknn"

# =========================================================
# Decode YOLOv8-OBB 输出
#   输出格式 (airockchip 导出): 
#     outs[0]: (1, 79, H/8,  W/8)    stride=8
#     outs[1]: (1, 79, H/16, W/16)   stride=16
#     outs[2]: (1, 79, H/32, W/32)   stride=32
#     outs[3]: (1, 1, total_anchors) angle (normalized 0-1)
#   79 = 64 (4 box dist × 16 bins) + 15 (DOTA 类别数)
# =========================================================
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def decode_generic(outs, imgsz, ox=0, oy=0, scale=1.0, conf_th=0.3):
    """解码 OBB 输出到 [cx, cy, w, h, angle, conf, cls]（Numba JIT 加速）"""
    dets = []
    strides = [8, 16, 32]
    g = imgsz // 8
    aos = [0, g*g, g*g + (g//2)*(g//2)]
    angle = outs[3][0, 0]
    for o, s, ao in zip(outs[:3], strides, aos):
        _, _, H, W = o.shape
        xywh = o[0, :64].reshape(4, 16, H, W)
        clsl = o[0, 64:79]
        # softmax + expected value on 16 bins（NumPy 向量化，已够快）
        sm = np.exp(xywh - xywh.max(axis=1, keepdims=True))
        sm /= sm.sum(axis=1, keepdims=True)
        xd = (sm * np.arange(16).reshape(1, 16, 1, 1)).sum(axis=1)
        cp = sigmoid(clsl)
        ci = cp.argmax(axis=0).astype(np.int32)
        cf = cp.max(axis=0)
        # --- Numba JIT 解码内核 ---
        det = _decode_kernel(xd[0], xd[1], xd[2], xd[3],
                             cf, ci, angle, H, W, s, ao, scale, ox, oy, conf_th)
        if len(det) > 0:
            dets.append(det)
    if dets:
        return np.concatenate(dets, axis=0).tolist()
    return []

# =========================================================
# 旋转 NMS（中心距离近似）
# =========================================================
def nms_rot(dets, iou_thr=0.3):
    if not dets: return []
    dets = np.array(dets)
    dets = dets[dets[:, 5].argsort()[::-1]]
    keep = []
    while len(dets) > 0:
        a = dets[0]
        keep.append(a)
        if len(dets) == 1: break
        rest = dets[1:]
        d = np.linalg.norm(rest[:, :2] - np.array([a[0], a[1]]), axis=1)
        th = (a[2] + a[3]) * 0.5 * iou_thr
        dets = rest[d > th]
    return keep

# =========================================================
# 真实旋转 IoU（cv2.rotatedRectangleIntersection）
# =========================================================
def iou_rot(a, b):
    try:
        ra = ((float(a[0]), float(a[1])), (float(a[2]), float(a[3])), math.degrees(float(a[4])))
        rb = ((float(b[0]), float(b[1])), (float(b[2]), float(b[3])), math.degrees(float(b[4])))
        ip = cv2.rotatedRectangleIntersection(ra, rb)
        if ip[0] == 0: return 0
        ia = cv2.contourArea(ip[1])
        return ia / (a[2] * a[3] + b[2] * b[3] - ia) if ia > 0 else 0
    except:
        return 0

def recall_iou(gt, pred, iou_thr=0.3):
    """召回率计算：按真实旋转 IoU 匹配"""
    if not gt: return 1.0, 0, 0
    hit = 0
    used = set()
    for g in gt:
        for j, p in enumerate(pred):
            if j in used: continue
            if iou_rot(g, p) >= iou_thr:
                hit += 1
                used.add(j)
                break
    return hit / len(gt), hit, len(gt)

# =========================================================
# Tile 生成
# =========================================================
def gen_tiles(W, H, tile, overlap):
    stride = tile - overlap
    xs = list(range(0, W - tile + 1, stride))
    if xs[-1] + tile < W: xs.append(W - tile)
    ys = list(range(0, H - tile + 1, stride))
    if ys[-1] + tile < H: ys.append(H - tile)
    return [(x, y) for y in ys for x in xs]

# =========================================================
# 多核并行推理
# =========================================================
def build_ctx_list(model_path, n_cores=3):
    """建立 n_cores 个 RKNNLite 实例，分别绑 NPU_CORE_0/1/2"""
    cores = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2][:n_cores]
    ctx_list = []
    for c in cores:
        r = RKNNLite()
        r.load_rknn(model_path)
        r.init_runtime(core_mask=c)
        ctx_list.append(r)
    return ctx_list

def infer_par(ctx_list, img, tiles, tile_size, conf=0.3, scale=1.0):
    """并行推理多个 tile，返回所有检测框（已映射到原图坐标）"""
    if not tiles: return []
    iq = queue.Queue(maxsize=16)
    oq = queue.Queue()
    n = len(ctx_list)

    def pre():
        for i, (x, y) in enumerate(tiles):
            c = img[y:y+tile_size, x:x+tile_size]
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            iq.put((i, x, y, np.expand_dims(rgb, 0)))
        for _ in range(n):
            iq.put(None)

    def inf(r):
        while True:
            it = iq.get()
            if it is None: return
            i, x, y, inp = it
            outs = r.inference(inputs=[inp], data_format="nhwc")
            oq.put((i, x, y, outs))

    tp = threading.Thread(target=pre); tp.start()
    ts = [threading.Thread(target=inf, args=(ctx_list[i],)) for i in range(n)]
    for t in ts: t.start()
    tp.join()
    for t in ts: t.join()

    dets = []
    while not oq.empty():
        _, x, y, outs = oq.get()
        dets.extend(decode_generic(outs, tile_size, ox=x, oy=y, scale=scale, conf_th=conf))
    return dets

def release_ctx(ctx_list):
    for r in ctx_list:
        r.release()

# =========================================================
# 生成 GT：@1024 全扫 81 tile INT8
# =========================================================
def get_gt(img, cache={}):
    """生成 GT（自动缓存，避免每个实验都重跑）"""
    if 'gt' in cache:
        return cache['gt']
    H, W = img.shape[:2]
    ctx_list = build_ctx_list(MODEL_1024, 3)
    tiles = gen_tiles(W, H, 1024, 100)
    dets = infer_par(ctx_list, img, tiles, 1024, conf=0.3)
    gt = nms_rot(dets)
    release_ctx(ctx_list)
    cache['gt'] = gt
    print(f"[GT] 全扫 {len(tiles)} tile → {len(gt)} 个目标")
    return gt

# =========================================================
# 统一实验报告格式
# =========================================================
def report(exp_id, name, total_ms, count, recall_pct, hit, total, extra=""):
    mark = "✅" if total_ms <= 1000 else ("🟡" if total_ms <= 1500 else "❌")
    line = (f"{exp_id:<5} {name:<50} total={total_ms:6.0f}ms  "
            f"count={count:3d}  recall={recall_pct:5.1f}% ({hit}/{total}) {mark}")
    if extra: line += f"  {extra}"
    print(line)
    return {
        "exp_id": exp_id, "name": name,
        "total_ms": round(total_ms, 1), "count": count,
        "recall_pct": round(recall_pct, 1),
        "hit": hit, "total_gt": total,
        "within_1s": total_ms <= 1000,
    }
