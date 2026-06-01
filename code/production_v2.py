"""
Production v2: 把 decode 放回推理线程里（3核并发解码 + 推理）
+ 看 conf 0.15/0.2/0.25 的真实耗时差（conf 越低 decode 越慢）
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

    def run(self, img, conf):
        """v2: pre + infer + decode 全部在 3 个核线程里，只剩 NMS 串行"""
        stats = {}
        H, W = img.shape[:2]
        t = time.perf_counter()
        tiles = gen_tiles(W, H, TILE_CUT, OVERLAP)
        stats['plan'] = (time.perf_counter()-t)*1000

        t = time.perf_counter()
        n = len(self.ctx_list)
        chunks = [[] for _ in range(n)]
        for i, tt in enumerate(tiles): chunks[i % n].append(tt)
        all_dets = []
        lock = threading.Lock()
        core_stats = [{'pre':0.0,'inf':0.0,'dec':0.0,'wall':0.0} for _ in range(n)]

        def worker(idx, r, my_tiles):
            tw0 = time.perf_counter()
            local = []
            pre_sum = inf_sum = dec_sum = 0.0
            for x, y in my_tiles:
                t_a = time.perf_counter()
                crop = img[y:y+TILE_CUT, x:x+TILE_CUT]
                small = cv2.resize(crop, (TILE_MODEL, TILE_MODEL), interpolation=RESIZE_ALGO)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                inp = np.expand_dims(rgb, 0)
                t_b = time.perf_counter()
                outs = r.inference(inputs=[inp], data_format="nhwc")
                t_c = time.perf_counter()
                dets = decode_generic(outs, TILE_MODEL, ox=x, oy=y, scale=self.scale_back, conf_th=conf)
                t_d = time.perf_counter()
                pre_sum += (t_b-t_a)*1000
                inf_sum += (t_c-t_b)*1000
                dec_sum += (t_d-t_c)*1000
                local.extend(dets)
            core_stats[idx] = {'pre':pre_sum,'inf':inf_sum,'dec':dec_sum,
                               'wall':(time.perf_counter()-tw0)*1000}
            with lock: all_dets.extend(local)

        ts = [threading.Thread(target=worker, args=(i, self.ctx_list[i], chunks[i])) for i in range(n)]
        for t_ in ts: t_.start()
        for t_ in ts: t_.join()
        stats['wall_all_parallel'] = (time.perf_counter()-t)*1000
        stats['core_stats'] = core_stats

        t = time.perf_counter()
        final = nms_rot(all_dets, iou_thr=0.3)
        stats['nms'] = (time.perf_counter()-t)*1000
        stats['n_raw'] = len(all_dets)
        stats['n_final'] = len(final)
        return final, stats

    def close(self):
        release_ctx(self.ctx_list)

print("="*80)
print("Production v2: decode in parallel threads")
print("="*80)

img = cv2.imread(IMG_PATH)
H, W = img.shape[:2]
print(f"image: {W}x{H}")
GT = get_gt(img)

pipe = ProductionPipeline()

# 三种 conf 各跑 10 次
for conf in [0.15, 0.20, 0.25, 0.30]:
    print(f"\n=== conf={conf} ===")
    runs = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        final, stats = pipe.run(img, conf)
        total = (time.perf_counter()-t0)*1000
        rec, hit, nt = recall_iou(GT, final, 0.3)
        runs.append((total, stats['wall_all_parallel'], stats['nms'], rec*100,
                     stats['n_raw'], stats['n_final']))
    # skip warmup run 0
    totals = [r[0] for r in runs[1:]]
    walls = [r[1] for r in runs[1:]]
    recs = [r[3] for r in runs[1:]]
    raws = [r[4] for r in runs[1:]]
    print(f"  total:  min={min(totals):.0f}  median={np.median(totals):.0f}  mean={np.mean(totals):.0f}  max={max(totals):.0f} ms")
    print(f"  wall:   min={min(walls):.0f}  median={np.median(walls):.0f}  mean={np.mean(walls):.0f}  max={max(walls):.0f} ms")
    print(f"  recall: {np.median(recs):.1f}%   raw_dets median={int(np.median(raws))}")
    # 首次的 core stats
    cs = stats['core_stats']
    print(f"  core stats (last run):")
    for i, c in enumerate(cs):
        print(f"    core {i}: pre={c['pre']:.0f}  inf={c['inf']:.0f}  dec={c['dec']:.0f}  wall={c['wall']:.0f}")

pipe.close()
