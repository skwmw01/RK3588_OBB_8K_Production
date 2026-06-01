"""追 1s 目标：3 组激进方案 + Pareto"""
import sys, time, cv2, numpy as np, threading, queue, json
sys.path.insert(0, '/tmp')
from common import (
    MODEL_1024, IMG_PATH, sigmoid, decode_generic, nms_rot, recall_iou,
    gen_tiles, build_ctx_list, infer_par, release_ctx, get_gt, report
)
TILE = 1024
STRIDE = 960

def coords_to_tiles(dets, W, H, expand=0, limit=None):
    tiles_with_score = []
    def snap(tx, ty):
        tx = (tx // STRIDE) * STRIDE; ty = (ty // STRIDE) * STRIDE
        tx = max(0, min(tx, W-TILE));  ty = max(0, min(ty, H-TILE))
        return (tx, ty)
    tile_scores = {}
    for d in dets:
        bx = min(max(0, int(d[0]-TILE/2)), W-TILE)
        by = min(max(0, int(d[1]-TILE/2)), H-TILE)
        center = snap(bx, by)
        tile_scores[center] = max(tile_scores.get(center, 0), d[5])
        if expand == 1:
            for dx in [-STRIDE, 0, STRIDE]:
                for dy in [-STRIDE, 0, STRIDE]:
                    if dx == 0 and dy == 0: continue
                    t = snap(bx+dx, by+dy)
                    tile_scores[t] = max(tile_scores.get(t, 0), d[5]*0.5)
        elif expand == 2:
            for dx, dy in [(-STRIDE,0),(STRIDE,0),(0,-STRIDE),(0,STRIDE)]:
                t = snap(bx+dx, by+dy)
                tile_scores[t] = max(tile_scores.get(t, 0), d[5]*0.5)
    sorted_tiles = sorted(tile_scores.items(), key=lambda x: -x[1])
    if limit is not None:
        sorted_tiles = sorted_tiles[:limit]
    return [t for t, _ in sorted_tiles]

def infer_static(ctx_list, img, tiles, tile_size, conf=0.3):
    """静态均分版：每核固定处理自己那份，无队列开销"""
    if not tiles: return []
    n = len(ctx_list)
    # 按 n 均分
    chunks = [[] for _ in range(n)]
    for i, t in enumerate(tiles):
        chunks[i % n].append(t)
    # 每核先串行 pre + infer 自己那份
    all_dets = []
    lock = threading.Lock()
    def w(r, my_tiles):
        for x, y in my_tiles:
            crop = img[y:y+tile_size, x:x+tile_size]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            outs = r.inference(inputs=[np.expand_dims(rgb,0)], data_format="nhwc")
            dets = decode_generic(outs, tile_size, ox=x, oy=y, conf_th=conf)
            with lock: all_dets.extend(dets)
    ts = [threading.Thread(target=w, args=(ctx_list[i], chunks[i])) for i in range(n)]
    for t in ts: t.start()
    for t in ts: t.join()
    return all_dets

# 准备
print("[load]", end=" "); t0 = time.perf_counter()
img = cv2.imread(IMG_PATH); print(f"{(time.perf_counter()-t0)*1000:.0f}ms")
H, W = img.shape[:2]

print("[GT] ...", end=" "); 
GT = get_gt(img); print(f"{len(GT)} 目标")

print("[build ctx]")
ctx = build_ctx_list(MODEL_1024, 3)
dummy = np.zeros((1,TILE,TILE,3), dtype=np.uint8)
for r in ctx:
    for _ in range(3): r.inference(inputs=[dummy], data_format="nhwc")

results = []

def run(exp_id, name, conf_c, expand, tile_limit=None):
    t0 = time.perf_counter()
    thumb = cv2.resize(img, (TILE, TILE))
    rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
    outs = ctx[0].inference(inputs=[np.expand_dims(rgb,0)], data_format="nhwc")
    dets_c = decode_generic(outs, TILE, scale=W/TILE, conf_th=conf_c)
    t_c = (time.perf_counter()-t0)*1000
    hot = coords_to_tiles(dets_c, W, H, expand=expand, limit=tile_limit)
    t_f0 = time.perf_counter()
    dets_f = infer_static(ctx, img, hot, TILE, conf=0.3)
    t_f = (time.perf_counter()-t_f0)*1000
    final = nms_rot(dets_c + dets_f)
    total = (time.perf_counter()-t0)*1000
    rec, hit, nt = recall_iou(GT, final, 0.3)
    r = report(exp_id, name, total, len(final), rec*100, hit, nt,
               f"c={t_c:.0f} f={t_f:.0f} hot={len(hot)}")
    results.append(r)
    return r

print("\n" + "="*90)
print("激进 1s 方案")
print("="*90)

# Z0: 裸 Y4 不扩（对照）
run('Z0', 'Y4 裸跑 conf=0.1 不扩', 0.1, 0)
run('Z0b', 'Y4 裸跑 conf=0.15 不扩', 0.15, 0)

# Z1: 限制 hot tile 数
run('Z1-15', 'conf=0.1 + 十字扩 限15 tile', 0.1, 2, 15)
run('Z1-20', 'conf=0.1 + 十字扩 限20 tile', 0.1, 2, 20)
run('Z1-25', 'conf=0.1 + 十字扩 限25 tile', 0.1, 2, 25)
run('Z1-30', 'conf=0.1 + 十字扩 限30 tile', 0.1, 2, 30)
run('Z1-35', 'conf=0.1 + 十字扩 限35 tile', 0.1, 2, 35)

# Z2: 更高 conf + 十字扩 (较少 hot)
run('Z2-20', 'conf=0.25 + 十字扩 限20', 0.25, 2, 20)
run('Z2-25', 'conf=0.25 + 十字扩 限25', 0.25, 2, 25)
run('Z2-30', 'conf=0.25 + 十字扩 限30', 0.25, 2, 30)

# Z3: 高 conf + 8邻居但限制 hot 数
run('Z3-25', 'conf=0.25 + 8邻居 限25', 0.25, 1, 25)
run('Z3-30', 'conf=0.25 + 8邻居 限30', 0.25, 1, 30)

print("\n" + "="*90)
print("按时间排序")
print("="*90)
for r in sorted(results, key=lambda x: x['total_ms']):
    star = "⭐" if r['total_ms'] <= 1000 and r['recall_pct'] >= 55 else ""
    print(f"{r['exp_id']:<7} {r['name'][:45]:<45} {r['total_ms']:>6.0f}ms  {r['recall_pct']:>5.1f}%  {star}")

print("\n" + "="*90)
print("Pareto")
print("="*90)
for r in sorted(results, key=lambda x: x['total_ms']):
    pass
sorted_t = sorted(results, key=lambda x: x['total_ms'])
best = 0; pareto = []
for r in sorted_t:
    if r['recall_pct'] > best:
        pareto.append(r); best = r['recall_pct']
for r in pareto:
    mark = "🥇 1s" if r['total_ms'] <= 1000 else ""
    print(f"  [{r['exp_id']}] {r['name'][:45]:<45} {r['total_ms']:>6.0f}ms  {r['recall_pct']:>5.1f}%  {mark}")

release_ctx(ctx)
with open('/tmp/e18_1s_results.json','w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
