"""按用户规则：8000×8000 图 + 2048×2048 tile + overlap 200，25 tile 全扫
同时测 1024×1024 tile 版本作为对照"""
import sys, time, cv2, numpy as np, threading, queue, json
sys.path.insert(0, '/tmp')
from common import (
    MODEL_1024, MODEL_2048, IMG_PATH, sigmoid, decode_generic,
    nms_rot, recall_iou, build_ctx_list, release_ctx, get_gt, report
)

# 用户规则
IMG_SIZE = 8000
OVERLAP = 200

# 粗略 resize 到 8000×8000 (原图是 8192×8192)
print("[load]", end=" "); t0 = time.perf_counter()
img_orig = cv2.imread(IMG_PATH)
print(f"{(time.perf_counter()-t0)*1000:.0f}ms, orig={img_orig.shape}")
H0, W0 = img_orig.shape[:2]

# 按用户规则：8000x8000 输入
img8k = cv2.resize(img_orig, (IMG_SIZE, IMG_SIZE))
print(f"[resize] → {IMG_SIZE}x{IMG_SIZE}")

# GT 用 8192 原图（保持可比）
print("[GT] 生成（在 8192 原图上全扫）...", end=" ")
GT = get_gt(img_orig)
print(f"{len(GT)} 目标 (原图 8192x8192)")

# 把 GT 映射到 8000 空间
scale_gt = IMG_SIZE / W0
GT_8k = [[g[0]*scale_gt, g[1]*scale_gt, g[2]*scale_gt, g[3]*scale_gt, g[4], g[5], g[6]] for g in GT]

def gen_tiles_user_rule(W, H, tile, overlap):
    """用户规则：stride = tile - overlap，横向 5 片第 5 片靠右边界"""
    stride = tile - overlap
    xs = list(range(0, W - tile + 1, stride))
    if xs[-1] + tile < W: xs.append(W - tile)
    ys = list(range(0, H - tile + 1, stride))
    if ys[-1] + tile < H: ys.append(H - tile)
    return [(x, y) for y in ys for x in xs]

tiles_2048 = gen_tiles_user_rule(IMG_SIZE, IMG_SIZE, 2048, OVERLAP)
tiles_1024 = gen_tiles_user_rule(IMG_SIZE, IMG_SIZE, 1024, OVERLAP)
print(f"\n[plan] 2048 tiles: {len(tiles_2048)} (期望 25)")
print(f"[plan] 1024 tiles: {len(tiles_1024)}")
print(f"[plan] 2048 第一行 x 列表: {[t[0] for t in tiles_2048[:5]]}")

def infer_static(ctx_list, img, tiles, tile_size, conf=0.3):
    if not tiles: return [], 0, 0
    n = len(ctx_list)
    chunks = [[] for _ in range(n)]
    for i, t in enumerate(tiles): chunks[i % n].append(t)
    all_dets = []; lock = threading.Lock()
    t_inf_total = [0.0]*n
    def w(idx, r, my_tiles):
        local = []
        tw = time.perf_counter()
        for x, y in my_tiles:
            crop = img[y:y+tile_size, x:x+tile_size]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            outs = r.inference(inputs=[np.expand_dims(rgb,0)], data_format="nhwc")
            dets = decode_generic(outs, tile_size, ox=x, oy=y, conf_th=conf)
            local.extend(dets)
        t_inf_total[idx] = (time.perf_counter()-tw)*1000
        with lock: all_dets.extend(local)
    t0_ = time.perf_counter()
    ts = [threading.Thread(target=w, args=(i, ctx_list[i], chunks[i])) for i in range(n)]
    for t in ts: t.start()
    for t in ts: t.join()
    wall = (time.perf_counter()-t0_)*1000
    return all_dets, wall, max(t_inf_total)

results = []

# ======= 方案 U1: 2048 × 25 tile 全扫 =======
print("\n" + "="*80)
print("U1: 2048 tile × 25 全扫（用户规则原版）")
print("="*80)
ctx2 = build_ctx_list(MODEL_2048, 3)
dummy2 = np.zeros((1,2048,2048,3), dtype=np.uint8)
for r in ctx2:
    for _ in range(2): r.inference(inputs=[dummy2], data_format="nhwc")

t0 = time.perf_counter()
dets, wall, max_per_core = infer_static(ctx2, img8k, tiles_2048, 2048, conf=0.3)
final = nms_rot(dets)
total = (time.perf_counter()-t0)*1000
rec, hit, nt = recall_iou(GT_8k, final, 0.3)
print(f"  total={total:.0f}ms   wall_infer={wall:.0f}ms   slowest_core={max_per_core:.0f}ms")
print(f"  count={len(final)}   recall={rec*100:.1f}% ({hit}/{nt})")
results.append({'id':'U1','name':'2048x25 全扫', 'time':total, 'recall':rec*100, 'count':len(final), 'hit':hit})
release_ctx(ctx2)

# ======= 方案 U1-single: 2048 单核纯推理时间（看理论）=======
print("\n[profile] 2048 tile 单核纯推理耗时（基线）")
ctx2 = build_ctx_list(MODEL_2048, 1)
for _ in range(3): ctx2[0].inference(inputs=[dummy2], data_format="nhwc")
ts_single = []
for _ in range(5):
    t = time.perf_counter()
    ctx2[0].inference(inputs=[dummy2], data_format="nhwc")
    ts_single.append((time.perf_counter()-t)*1000)
print(f"  单核 2048 median: {np.median(ts_single):.0f} ms")
print(f"  理论下限 25/3 * median = {25/3*np.median(ts_single):.0f} ms")
release_ctx(ctx2)

# ======= 方案 U2: 1024 × 81 tile (原版对照) =======
print("\n" + "="*80)
print("U2: 1024 tile 全扫（按同规则 stride=824 实际很多 tile）")
print("="*80)
# 这里用 1024 + overlap 200 → stride 824
print(f"  1024 tiles: {len(tiles_1024)}")

ctx1 = build_ctx_list(MODEL_1024, 3)
dummy1 = np.zeros((1,1024,1024,3), dtype=np.uint8)
for r in ctx1:
    for _ in range(2): r.inference(inputs=[dummy1], data_format="nhwc")

t0 = time.perf_counter()
dets, wall, max_per_core = infer_static(ctx1, img8k, tiles_1024, 1024, conf=0.3)
final = nms_rot(dets)
total = (time.perf_counter()-t0)*1000
rec, hit, nt = recall_iou(GT_8k, final, 0.3)
print(f"  total={total:.0f}ms   wall_infer={wall:.0f}ms   slowest_core={max_per_core:.0f}ms")
print(f"  count={len(final)}   recall={rec*100:.1f}%")
results.append({'id':'U2','name':f'1024x{len(tiles_1024)} 全扫', 'time':total, 'recall':rec*100, 'count':len(final), 'hit':hit})
release_ctx(ctx1)

# ======= 总结 =======
print("\n" + "="*80)
print("结果")
print("="*80)
print(f"{'ID':<6} {'策略':<35} {'时间':>8} {'召回':>10} {'检测数':>8}")
for r in results:
    mark = '✅' if r['time'] <= 1000 else ('🟡' if r['time']<=2000 else '❌')
    print(f"{r['id']:<6} {r['name']:<35} {r['time']:>6.0f}ms  {r['recall']:>5.1f}%   {r['count']:>5d}  {mark}")

with open('/tmp/user_rule_results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
