"""用户规则 V2: 切 2048×2048 → resize 1024×1024 → 用 1024 模型推理
8000×8000 图 / overlap 200 / 25 tile / resize 后推理"""
import sys, time, cv2, numpy as np, threading, json
sys.path.insert(0, '/tmp')
from common import (
    MODEL_1024, IMG_PATH, sigmoid, decode_generic,
    nms_rot, recall_iou, build_ctx_list, release_ctx, get_gt
)

IMG_SIZE = 8000
OVERLAP = 200
TILE_CUT = 2048      # 原图上切这么大
TILE_MODEL = 1024    # 推理时 resize 到这个尺寸

print("[load]", end=" "); t0 = time.perf_counter()
img_orig = cv2.imread(IMG_PATH)
print(f"{(time.perf_counter()-t0)*1000:.0f}ms, orig={img_orig.shape}")

img8k = cv2.resize(img_orig, (IMG_SIZE, IMG_SIZE))
print(f"[resize input] → {IMG_SIZE}x{IMG_SIZE}")

# GT（仍从原图 8192 提取，然后映射到 8000 空间）
print("[GT]", end=" ")
GT = get_gt(img_orig)
scale_gt = IMG_SIZE / img_orig.shape[0]
GT_8k = [[g[0]*scale_gt, g[1]*scale_gt, g[2]*scale_gt, g[3]*scale_gt, g[4], g[5], g[6]] for g in GT]
print(f"GT={len(GT_8k)} 个目标 (mapped to 8000)")

def gen_tiles(W, H, tile, overlap):
    stride = tile - overlap
    xs = list(range(0, W-tile+1, stride))
    if xs[-1]+tile < W: xs.append(W-tile)
    ys = list(range(0, H-tile+1, stride))
    if ys[-1]+tile < H: ys.append(H-tile)
    return [(x,y) for y in ys for x in xs]

tiles = gen_tiles(IMG_SIZE, IMG_SIZE, TILE_CUT, OVERLAP)
print(f"[plan] 2048 切片 {len(tiles)} 张 (期望 25)")
print(f"  第一行 x: {[t[0] for t in tiles[:5]]}")

def infer_static_resize(ctx_list, img, tiles, tile_cut, tile_model, conf=0.3):
    """切 tile_cut → resize tile_model → 推理 → 坐标还原到 tile_cut 空间再加 ox/oy"""
    if not tiles: return [], 0
    n = len(ctx_list)
    chunks = [[] for _ in range(n)]
    for i, t in enumerate(tiles): chunks[i % n].append(t)
    all_dets = []; lock = threading.Lock()
    t_per_core = [0.0]*n
    scale_back = tile_cut / tile_model   # resize 回 tile_cut 空间
    def worker(idx, r, my_tiles):
        local = []
        tw = time.perf_counter()
        for x, y in my_tiles:
            crop = img[y:y+tile_cut, x:x+tile_cut]
            # resize 到模型尺寸
            small = cv2.resize(crop, (tile_model, tile_model))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            outs = r.inference(inputs=[np.expand_dims(rgb,0)], data_format="nhwc")
            # decode 在 tile_model 空间，scale 回 tile_cut 空间
            dets = decode_generic(outs, tile_model, ox=x, oy=y, scale=scale_back, conf_th=conf)
            local.extend(dets)
        t_per_core[idx] = (time.perf_counter()-tw)*1000
        with lock: all_dets.extend(local)
    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker, args=(i, ctx_list[i], chunks[i])) for i in range(n)]
    for t in ts: t.start()
    for t in ts: t.join()
    wall = (time.perf_counter()-t0)*1000
    return all_dets, wall

# 建 1024 模型 3 核
ctx = build_ctx_list(MODEL_1024, 3)
dummy = np.zeros((1, TILE_MODEL, TILE_MODEL, 3), dtype=np.uint8)
for r in ctx:
    for _ in range(3): r.inference(inputs=[dummy], data_format="nhwc")

print("\n" + "="*80)
print("V2: 切 2048 → resize 1024 → 推理")
print("="*80)

# 跑 3 次取中位数，第 1 次可能 warm up 偏慢
runs = []
for i in range(3):
    t0 = time.perf_counter()
    dets, wall = infer_static_resize(ctx, img8k, tiles, TILE_CUT, TILE_MODEL, conf=0.3)
    final = nms_rot(dets)
    total = (time.perf_counter()-t0)*1000
    rec, hit, nt = recall_iou(GT_8k, final, 0.3)
    runs.append((total, wall, rec*100, len(final), hit))
    print(f"  run {i+1}: total={total:.0f}ms  wall={wall:.0f}ms  count={len(final)}  recall={rec*100:.1f}% ({hit}/{nt})")

# 中位数
med = sorted(runs, key=lambda x: x[0])[len(runs)//2]
print(f"\n  === 中位数: total={med[0]:.0f}ms  recall={med[2]:.1f}%  count={med[3]} ===")

# 对照：同样 25 tile，但直接用 @1024 model 切原图（需要的是 stride=~350 的 25 tile fit）
# 实际上 8000/5=1600，切 25 块 1024 tile 需要 overlap 很大，意义不大，跳过

# 对照: conf 不同
print("\n[扫 conf 阈值]")
for conf in [0.2, 0.25, 0.3, 0.35, 0.4]:
    t0 = time.perf_counter()
    dets, wall = infer_static_resize(ctx, img8k, tiles, TILE_CUT, TILE_MODEL, conf=conf)
    final = nms_rot(dets)
    total = (time.perf_counter()-t0)*1000
    rec, hit, nt = recall_iou(GT_8k, final, 0.3)
    print(f"  conf={conf:.2f}: {total:.0f}ms  recall={rec*100:.1f}%  count={len(final)}")

# 对照: 只用 2 核 vs 3 核
print("\n[核数对比]")
release_ctx(ctx)
for n_cores in [1, 2, 3]:
    ctx = build_ctx_list(MODEL_1024, n_cores)
    for r in ctx:
        for _ in range(2): r.inference(inputs=[dummy], data_format="nhwc")
    t0 = time.perf_counter()
    dets, wall = infer_static_resize(ctx, img8k, tiles, TILE_CUT, TILE_MODEL, conf=0.3)
    final = nms_rot(dets)
    total = (time.perf_counter()-t0)*1000
    rec, _, _ = recall_iou(GT_8k, final, 0.3)
    print(f"  {n_cores} 核: {total:.0f}ms  recall={rec*100:.1f}%")
    release_ctx(ctx)

with open('/tmp/v2_results.json', 'w') as f:
    json.dump({'runs': runs}, f, indent=2)
