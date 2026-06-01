"""
Production 方案：8192×8192 → 2048 切 25 tile → resize 1024 → INT8@1024 → 63.8% recall
完整时间剖析：读图 / 预处理 / 推理 / 后处理 / NMS
每个阶段单独测 10 次取统计
"""
import sys, time, cv2, numpy as np, threading, json
sys.path.insert(0, '/tmp')
from common import (
    MODEL_1024, IMG_PATH, decode_generic, nms_rot, recall_iou,
    build_ctx_list, release_ctx, get_gt
)

TILE_CUT = 2048
TILE_MODEL = 1024
OVERLAP = 200
CONF = 0.15
RESIZE_ALGO = cv2.INTER_LINEAR
N_CORES = 3
N_RUNS = 10

def gen_tiles(W, H, tile, overlap):
    stride = tile - overlap
    xs = list(range(0, W-tile+1, stride))
    if xs[-1]+tile < W: xs.append(W-tile)
    ys = list(range(0, H-tile+1, stride))
    if ys[-1]+tile < H: ys.append(H-tile)
    return [(x,y) for y in ys for x in xs]

class ProductionPipeline:
    def __init__(self):
        self.ctx_list = build_ctx_list(MODEL_1024, N_CORES)
        dummy = np.zeros((1, TILE_MODEL, TILE_MODEL, 3), dtype=np.uint8)
        for r in self.ctx_list:
            for _ in range(3): r.inference(inputs=[dummy], data_format="nhwc")
        self.scale_back = TILE_CUT / TILE_MODEL

    def run(self, img, measure_detail=False):
        stats = {}
        H, W = img.shape[:2]
        t = time.perf_counter()
        tiles = gen_tiles(W, H, TILE_CUT, OVERLAP)
        stats['plan'] = (time.perf_counter()-t)*1000

        if measure_detail:
            t = time.perf_counter()
            for x, y in tiles:
                crop = img[y:y+TILE_CUT, x:x+TILE_CUT]
                small = cv2.resize(crop, (TILE_MODEL, TILE_MODEL), interpolation=RESIZE_ALGO)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                _ = np.expand_dims(rgb, 0)
            stats['pre_serial'] = (time.perf_counter()-t)*1000

        t = time.perf_counter()
        n = len(self.ctx_list)
        chunks = [[] for _ in range(n)]
        for i, tt in enumerate(tiles): chunks[i % n].append(tt)
        all_outs = []
        lock = threading.Lock()
        core_times = [0.0]*n
        core_pre = [0.0]*n
        core_inf = [0.0]*n

        def worker(idx, r, my_tiles):
            tw0 = time.perf_counter()
            local = []
            pre_sum = 0.0
            inf_sum = 0.0
            for x, y in my_tiles:
                t_a = time.perf_counter()
                crop = img[y:y+TILE_CUT, x:x+TILE_CUT]
                small = cv2.resize(crop, (TILE_MODEL, TILE_MODEL), interpolation=RESIZE_ALGO)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                inp = np.expand_dims(rgb, 0)
                t_b = time.perf_counter()
                outs = r.inference(inputs=[inp], data_format="nhwc")
                t_c = time.perf_counter()
                pre_sum += (t_b - t_a)*1000
                inf_sum += (t_c - t_b)*1000
                local.append((x, y, outs))
            core_pre[idx] = pre_sum
            core_inf[idx] = inf_sum
            core_times[idx] = (time.perf_counter()-tw0)*1000
            with lock: all_outs.extend(local)

        ts = [threading.Thread(target=worker, args=(i, self.ctx_list[i], chunks[i])) for i in range(n)]
        for t_ in ts: t_.start()
        for t_ in ts: t_.join()
        stats['wall_pre_inf'] = (time.perf_counter()-t)*1000
        stats['core_times'] = core_times
        stats['core_pre_sum'] = core_pre
        stats['core_inf_sum'] = core_inf

        t = time.perf_counter()
        all_dets = []
        for x, y, outs in all_outs:
            dets = decode_generic(outs, TILE_MODEL, ox=x, oy=y, scale=self.scale_back, conf_th=CONF)
            all_dets.extend(dets)
        stats['decode'] = (time.perf_counter()-t)*1000
        stats['n_raw_dets'] = len(all_dets)

        t = time.perf_counter()
        final = nms_rot(all_dets, iou_thr=0.3)
        stats['nms'] = (time.perf_counter()-t)*1000
        stats['n_final'] = len(final)
        return final, stats

    def close(self):
        release_ctx(self.ctx_list)

print("="*80)
print("Production Benchmark")
print("="*80)
print(f"config: cut={TILE_CUT} model={TILE_MODEL} overlap={OVERLAP} conf={CONF} resize=LINEAR cores={N_CORES}")

print("\n[0] read image")
t = time.perf_counter()
img_file = cv2.imread(IMG_PATH)
print(f"  cold read: {(time.perf_counter()-t)*1000:.0f} ms  shape={img_file.shape}")
H, W = img_file.shape[:2]

print("\n[GT]")
GT = get_gt(img_file)
print(f"  {len(GT)} targets")

print("\n[init] load 3-core RKNN + warmup")
pipe = ProductionPipeline()
print("  done")

print("\n[profile] detailed single-run")
final, stats = pipe.run(img_file, measure_detail=True)
print(f"  plan:            {stats['plan']:.2f} ms")
print(f"  pre (serial):    {stats['pre_serial']:.1f} ms   (crop+resize+cvt for 25 tiles)")
print(f"  wall pre+inf:    {stats['wall_pre_inf']:.1f} ms   (3-core parallel)")
print(f"    core pre_sum:  {[f'{x:.0f}' for x in stats['core_pre_sum']]}")
print(f"    core inf_sum:  {[f'{x:.0f}' for x in stats['core_inf_sum']]}")
print(f"    core wall:     {[f'{x:.0f}' for x in stats['core_times']]}")
print(f"  decode:          {stats['decode']:.2f} ms   ({stats['n_raw_dets']} raw)")
print(f"  nms:             {stats['nms']:.2f} ms   ({stats['n_final']} final)")
rec, hit, nt = recall_iou(GT, final, 0.3)
print(f"  recall:          {rec*100:.1f}% ({hit}/{nt})")

print(f"\n[bench] {N_RUNS} consecutive runs")
print("-"*80)
all_runs = []
for i in range(N_RUNS):
    t0 = time.perf_counter()
    final, stats = pipe.run(img_file)
    total = (time.perf_counter()-t0)*1000
    rec, hit, nt = recall_iou(GT, final, 0.3)
    all_runs.append({'run': i+1, 'total': total, 'plan': stats['plan'],
                     'wall_pre_inf': stats['wall_pre_inf'], 'decode': stats['decode'],
                     'nms': stats['nms'], 'recall': rec*100, 'n_final': len(final)})
    print(f"  run {i+1:2d}: total={total:6.1f}ms  pre+inf={stats['wall_pre_inf']:5.1f}  "
          f"decode={stats['decode']:4.1f}  nms={stats['nms']:3.1f}  recall={rec*100:.1f}%  count={len(final)}")

totals = [r['total'] for r in all_runs]
walls = [r['wall_pre_inf'] for r in all_runs]
decodes = [r['decode'] for r in all_runs]
nmss = [r['nms'] for r in all_runs]
print(f"\n--- stats (skip warmup run 1) ---")
print(f"{'stage':<16} {'min':>7} {'median':>7} {'mean':>7} {'max':>7}  (ms)")
for name, arr in [('total', totals[1:]), ('wall pre+inf', walls[1:]),
                   ('decode', decodes[1:]), ('nms', nmss[1:])]:
    print(f"{name:<16} {min(arr):>7.1f} {np.median(arr):>7.1f} {np.mean(arr):>7.1f} {max(arr):>7.1f}")

recalls = [r['recall'] for r in all_runs[1:]]
print(f"\n  recall: median {np.median(recalls):.1f}%  min {min(recalls):.1f}%  max {max(recalls):.1f}%")

pipe.close()

out = {
    'config': {'tile_cut': TILE_CUT, 'tile_model': TILE_MODEL, 'overlap': OVERLAP,
               'conf': CONF, 'resize': 'LINEAR', 'n_cores': N_CORES, 'input_size': f"{W}x{H}"},
    'profile_single': {k: v for k, v in stats.items() if k != 'core_times'},
    'runs': all_runs,
    'summary': {
        'total_median_ms': float(np.median(totals[1:])),
        'total_min_ms': float(min(totals[1:])),
        'total_max_ms': float(max(totals[1:])),
        'recall_median_pct': float(np.median(recalls)),
    },
}
with open('/tmp/production_bench.json', 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)
print("\nsaved: /tmp/production_bench.json")
