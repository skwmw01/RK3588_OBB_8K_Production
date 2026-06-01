"""V2 细节优化：resize 算法对比 + 预处理并行 + 8192 直接切"""
import sys, time, cv2, numpy as np, threading, json
sys.path.insert(0, '/tmp')
from common import (
    MODEL_1024, IMG_PATH, sigmoid, decode_generic,
    nms_rot, recall_iou, build_ctx_list, release_ctx, get_gt
)

OVERLAP = 200
TILE_CUT = 2048
TILE_MODEL = 1024
CONF = 0.2  # 最优

img_orig = cv2.imread(IMG_PATH)
H0, W0 = img_orig.shape[:2]

# 测两种输入尺寸: 8000 (用户规则) vs 8192 (原图)
print("[GT]", end=" ")
GT = get_gt(img_orig)
print(f"{len(GT)}")

def gen_tiles(W, H, tile, overlap):
    stride = tile - overlap
    xs = list(range(0, W-tile+1, stride))
    if xs[-1]+tile < W: xs.append(W-tile)
    ys = list(range(0, H-tile+1, stride))
    if ys[-1]+tile < H: ys.append(H-tile)
    return [(x,y) for y in ys for x in xs]

def infer_v2(ctx_list, img, tiles, tile_cut, tile_model, conf, interp=cv2.INTER_AREA):
    if not tiles: return [], 0
    n = len(ctx_list)
    chunks = [[] for _ in range(n)]
    for i, t in enumerate(tiles): chunks[i % n].append(t)
    all_dets = []; lock = threading.Lock()
    scale_back = tile_cut / tile_model
    def worker(r, my_tiles):
        local = []
        for x, y in my_tiles:
            crop = img[y:y+tile_cut, x:x+tile_cut]
            small = cv2.resize(crop, (tile_model, tile_model), interpolation=interp)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            outs = r.inference(inputs=[np.expand_dims(rgb,0)], data_format="nhwc")
            dets = decode_generic(outs, tile_model, ox=x, oy=y, scale=scale_back, conf_th=conf)
            local.extend(dets)
        with lock: all_dets.extend(local)
    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker, args=(ctx_list[i], chunks[i])) for i in range(n)]
    for t in ts: t.start()
    for t in ts: t.join()
    return all_dets, (time.perf_counter()-t0)*1000

ctx = build_ctx_list(MODEL_1024, 3)
dummy = np.zeros((1, TILE_MODEL, TILE_MODEL, 3), dtype=np.uint8)
for r in ctx:
    for _ in range(3): r.inference(inputs=[dummy], data_format="nhwc")

print("\n【输入尺寸对比】")
print("-"*80)
for img_size in [8000, 8192]:
    if img_size == 8192:
        img_t = img_orig  # 原图
    else:
        img_t = cv2.resize(img_orig, (img_size, img_size))
    # GT 缩放
    scale_gt = img_size / W0
    GT_t = [[g[0]*scale_gt, g[1]*scale_gt, g[2]*scale_gt, g[3]*scale_gt, g[4], g[5], g[6]] for g in GT]
    tiles = gen_tiles(img_size, img_size, TILE_CUT, OVERLAP)
    print(f"\n输入 {img_size}x{img_size} → {len(tiles)} tile")
    best_time = 999999
    for _ in range(3):  # 3 次取最优
        t0 = time.perf_counter()
        dets, wall = infer_v2(ctx, img_t, tiles, TILE_CUT, TILE_MODEL, CONF)
        final = nms_rot(dets)
        total = (time.perf_counter()-t0)*1000
        rec, hit, nt = recall_iou(GT_t, final, 0.3)
        if total < best_time:
            best_time, best_rec, best_hit, best_n = total, rec*100, hit, len(final)
        print(f"  run: {total:.0f}ms  recall={rec*100:.1f}% ({hit}/{nt})  count={len(final)}")
    print(f"  ★最优: {best_time:.0f}ms / {best_rec:.1f}%")

print("\n【Resize 算法对比 (8192 输入)】")
print("-"*80)
img_t = img_orig  # 8192
tiles = gen_tiles(8192, 8192, TILE_CUT, OVERLAP)
print(f"  tiles: {len(tiles)}")
for interp_name, interp in [('NEAREST', cv2.INTER_NEAREST), ('LINEAR', cv2.INTER_LINEAR),
                              ('AREA', cv2.INTER_AREA), ('CUBIC', cv2.INTER_CUBIC)]:
    best = 999999
    for _ in range(2):
        t0 = time.perf_counter()
        dets, wall = infer_v2(ctx, img_t, tiles, TILE_CUT, TILE_MODEL, CONF, interp=interp)
        final = nms_rot(dets)
        total = (time.perf_counter()-t0)*1000
        if total < best:
            best, rec, _, _ = total, recall_iou(GT, final, 0.3)[0]*100, 0, 0
    print(f"  {interp_name:10}: {best:.0f}ms  recall={rec:.1f}%")

print("\n【conf 再细扫 (8192 / LINEAR)】")
print("-"*80)
for conf in [0.10, 0.15, 0.17, 0.18, 0.20, 0.22, 0.25]:
    best = 999999
    for _ in range(2):
        t0 = time.perf_counter()
        dets, wall = infer_v2(ctx, img_orig, tiles, TILE_CUT, TILE_MODEL, conf, cv2.INTER_LINEAR)
        final = nms_rot(dets)
        total = (time.perf_counter()-t0)*1000
        if total < best:
            best, rec, h, n_ = total, recall_iou(GT, final, 0.3)[0]*100, recall_iou(GT, final, 0.3)[1], len(final)
    print(f"  conf={conf:.2f}: {best:.0f}ms  recall={rec:.1f}% ({h}/47)  count={n_}")

release_ctx(ctx)
