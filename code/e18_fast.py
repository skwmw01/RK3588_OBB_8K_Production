"""E18 加速实验：多种策略对比，全部用真实 IoU recall 度量"""
import sys, time, cv2, numpy as np, threading, queue, json
sys.path.insert(0, '/tmp')
from common import (
    MODEL_1024, IMG_PATH, sigmoid, decode_generic, nms_rot, recall_iou,
    gen_tiles, build_ctx_list, infer_par, release_ctx, get_gt, report
)

TILE = 1024
STRIDE = 960  # 1024 - 64? 和日记一致，看 ablation_final

# =========================================================
# hot tile 规划函数（多种 expand 策略）
# =========================================================
def coords_to_tiles(dets, W, H, expand=0):
    """expand: 0=不扩, 1=8邻居(3x3), 2=4邻居(十字), 3=仅大目标8邻居"""
    tiles = set()
    def snap(tx, ty):
        tx = (tx // STRIDE) * STRIDE; ty = (ty // STRIDE) * STRIDE
        tx = max(0, min(tx, W-TILE));  ty = max(0, min(ty, H-TILE))
        return (tx, ty)
    for d in dets:
        bx = min(max(0, int(d[0]-TILE/2)), W-TILE)
        by = min(max(0, int(d[1]-TILE/2)), H-TILE)
        tx, ty = snap(bx, by); tiles.add((tx, ty))
        if expand == 1:  # 8 邻居
            for dx in [-STRIDE, 0, STRIDE]:
                for dy in [-STRIDE, 0, STRIDE]:
                    if dx == 0 and dy == 0: continue
                    tiles.add(snap(bx+dx, by+dy))
        elif expand == 2:  # 十字 4 邻居
            for dx, dy in [(-STRIDE,0),(STRIDE,0),(0,-STRIDE),(0,STRIDE)]:
                tiles.add(snap(bx+dx, by+dy))
        elif expand == 3:  # 仅大目标（w>120）扩 8 邻居
            bw, bh = d[2], d[3]
            if max(bw, bh) > 120:
                for dx in [-STRIDE, 0, STRIDE]:
                    for dy in [-STRIDE, 0, STRIDE]:
                        if dx == 0 and dy == 0: continue
                        tiles.add(snap(bx+dx, by+dy))
    return list(tiles)

# =========================================================
# 流水线并行版（预处理和推理 overlap）
# =========================================================
def infer_pipeline(ctx_list, img, tiles, tile_size, conf=0.3, scale=1.0):
    """CPU 预处理线程 + 3 NPU 推理线程，流水线 overlap"""
    if not tiles: return []
    iq = queue.Queue(maxsize=12)
    oq = queue.Queue()
    n = len(ctx_list)
    def pre():
        for i, (x, y) in enumerate(tiles):
            c = img[y:y+tile_size, x:x+tile_size]
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            iq.put((i, x, y, np.expand_dims(rgb, 0)))
        for _ in range(n): iq.put(None)
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

# =========================================================
# 准备
# =========================================================
print("[load] 读图")
t0 = time.perf_counter()
img = cv2.imread(IMG_PATH)
print(f"  read: {(time.perf_counter()-t0)*1000:.0f}ms")
H, W = img.shape[:2]
print(f"  size: {W}x{H}")

print("\n[GT] 生成 GT（全扫 81 tile）...")
GT = get_gt(img)
print(f"  GT 目标数: {len(GT)}")

print("\n[build] 建 3 核 INT8@1024 上下文")
ctx = build_ctx_list(MODEL_1024, 3)
# 预热
for r in ctx:
    r.inference(inputs=[np.zeros((1,TILE,TILE,3), dtype=np.uint8)], data_format="nhwc")

results = []

def run_expand_exp(exp_id, name, conf_coarse, expand):
    """一个完整的 E18 变种：thumb 粗扫 → expand → 精扫"""
    t0 = time.perf_counter()
    # 粗扫 thumb
    thumb = cv2.resize(img, (TILE, TILE))
    rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
    outs = ctx[0].inference(inputs=[np.expand_dims(rgb,0)], data_format="nhwc")
    dets_c = decode_generic(outs, TILE, scale=W/TILE, conf_th=conf_coarse)
    t_c = (time.perf_counter()-t0)*1000
    # hot tile
    hot = coords_to_tiles(dets_c, W, H, expand=expand)
    # 精扫（流水线版）
    t_f_start = time.perf_counter()
    dets_f = infer_pipeline(ctx, img, hot, TILE, conf=0.3)
    t_f = (time.perf_counter()-t_f_start)*1000
    # 合并
    final = nms_rot(dets_c + dets_f)
    total = (time.perf_counter()-t0)*1000
    rec, hit, nt = recall_iou(GT, final, 0.3)
    extra = f"coarse={t_c:.0f}ms fine={t_f:.0f}ms hot={len(hot)}"
    r = report(exp_id, name, total, len(final), rec*100, hit, nt, extra)
    results.append(r)
    return r

print("\n" + "="*90)
print("E18 加速实验")
print("="*90)

# Baseline E18 (conf=0.1, expand=1 8邻居)
run_expand_exp('E18', 'baseline: conf=0.1 + expand=8邻居', conf_coarse=0.1, expand=1)

# A1: 提高 conf
run_expand_exp('A1', 'conf=0.15 + expand=8邻居', conf_coarse=0.15, expand=1)
run_expand_exp('A2', 'conf=0.20 + expand=8邻居', conf_coarse=0.20, expand=1)
run_expand_exp('A3', 'conf=0.25 + expand=8邻居', conf_coarse=0.25, expand=1)

# B: 换 expand 策略
run_expand_exp('B1', 'conf=0.1 + expand=十字4邻居', conf_coarse=0.1, expand=2)
run_expand_exp('B2', 'conf=0.15 + expand=十字4邻居', conf_coarse=0.15, expand=2)
run_expand_exp('B3', 'conf=0.1 + 只大目标扩8邻居', conf_coarse=0.1, expand=3)
run_expand_exp('B4', 'conf=0.15 + 只大目标扩8邻居', conf_coarse=0.15, expand=3)

# C: 组合
run_expand_exp('C1', 'conf=0.2 + 十字4邻居', conf_coarse=0.2, expand=2)
run_expand_exp('C2', 'conf=0.2 + 只大目标扩', conf_coarse=0.2, expand=3)

# D: 不扩 (Y4 对照)
run_expand_exp('D1', 'Y4 对照: conf=0.25 不扩', conf_coarse=0.25, expand=0)
run_expand_exp('D2', 'Y4 对照: conf=0.1 不扩', conf_coarse=0.1, expand=0)

print("\n" + "="*90)
print("总结（按 total_ms 排序）")
print("="*90)
print(f"{'ID':<5} {'NAME':<40} {'Time':>8}  {'Recall':>8}  {'Count':>6}  {'Hot':>5}")
for r in sorted(results, key=lambda x: x['total_ms']):
    # parse extra
    print(f"{r['exp_id']:<5} {r['name'][:40]:<40} {r['total_ms']:>8.0f}  {r['recall_pct']:>6.1f}%  {r['count']:>6d}")

print("\n" + "="*90)
print("Pareto 前沿（时间 vs 召回）")
print("="*90)
sorted_time = sorted(results, key=lambda x: x['total_ms'])
best_rec = 0
pareto = []
for r in sorted_time:
    if r['recall_pct'] > best_rec:
        pareto.append(r)
        best_rec = r['recall_pct']
for r in pareto:
    print(f"  [{r['exp_id']}] {r['name'][:50]:<50} {r['total_ms']:>6.0f}ms  {r['recall_pct']:>5.1f}%")

release_ctx(ctx)
with open('/tmp/e18_fast_results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\n保存: /tmp/e18_fast_results.json")
