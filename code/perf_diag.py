"""找 infer_parallel 为什么比理论慢 2×：真实单核 vs 3 核并发"""
import sys, time, cv2, numpy as np, threading, queue
sys.path.insert(0, '/tmp')
from common import MODEL_1024, IMG_PATH, build_ctx_list, release_ctx
from rknnlite.api import RKNNLite

TILE = 1024
img = cv2.imread(IMG_PATH)
H, W = img.shape[:2]

# 建 3 核
ctx = build_ctx_list(MODEL_1024, 3)
# warmup
dummy = np.zeros((1, TILE, TILE, 3), dtype=np.uint8)
for r in ctx:
    for _ in range(3):
        r.inference(inputs=[dummy], data_format="nhwc")

# 测 1 tile 单核 baseline
print("=== 单 tile 单核 10 次 ===")
ts = []
for _ in range(10):
    t = time.perf_counter()
    ctx[0].inference(inputs=[dummy], data_format="nhwc")
    ts.append((time.perf_counter()-t)*1000)
print(f"  median: {np.median(ts):.1f} ms  min: {min(ts):.1f}  mean: {np.mean(ts):.1f}")

# 测 3 核真正并发（3 个线程各跑 10 次）
print("\n=== 3 核并发各 10 次（同一时刻都在跑）===")
results = [[], [], []]
def worker(idx, r):
    for _ in range(10):
        t = time.perf_counter()
        r.inference(inputs=[dummy], data_format="nhwc")
        results[idx].append((time.perf_counter()-t)*1000)
t0 = time.perf_counter()
threads = [threading.Thread(target=worker, args=(i, ctx[i])) for i in range(3)]
for t in threads: t.start()
for t in threads: t.join()
wall = (time.perf_counter()-t0)*1000
print(f"  core 0 median: {np.median(results[0]):.1f} ms")
print(f"  core 1 median: {np.median(results[1]):.1f} ms")
print(f"  core 2 median: {np.median(results[2]):.1f} ms")
print(f"  wall clock 30 推理: {wall:.0f} ms ({wall/30:.1f} ms/infer amortized)")

# 测 30 tile 走原 infer_par（切图+预处理+推理）看每一步耗时
print("\n=== 30 个真实 tile pipeline ===")
tiles = [(x*800, y*800) for y in range(5) for x in range(6)][:30]

# 纯推理时间（不含 pre）
preped = []
for x, y in tiles:
    c = img[y:y+TILE, x:x+TILE]
    rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    preped.append(np.expand_dims(rgb, 0))
print(f"  pre 30 tile: done")

# 方案1: 原 queue 模型
iq = queue.Queue()
for i, inp in enumerate(preped): iq.put((i, inp))
for _ in range(3): iq.put(None)
def w1(r):
    while True:
        it = iq.get()
        if it is None: return
        r.inference(inputs=[it[1]], data_format="nhwc")
t0 = time.perf_counter()
ts = [threading.Thread(target=w1, args=(r,)) for r in ctx]
for t in ts: t.start()
for t in ts: t.join()
t_queue = (time.perf_counter()-t0)*1000
print(f"  方案1 (queue 轮询): {t_queue:.0f} ms = {t_queue/30:.1f} ms/tile = 理论 {30/3*np.median(results[0]):.0f} ms")

# 方案2: 静态分配 (core 0 做 1-10, core 1 做 11-20, core 2 做 21-30)
def w2(r, chunk):
    for inp in chunk:
        r.inference(inputs=[inp], data_format="nhwc")
chunks = [preped[0:10], preped[10:20], preped[20:30]]
t0 = time.perf_counter()
ts = [threading.Thread(target=w2, args=(ctx[i], chunks[i])) for i in range(3)]
for t in ts: t.start()
for t in ts: t.join()
t_static = (time.perf_counter()-t0)*1000
print(f"  方案2 (静态均分): {t_static:.0f} ms = {t_static/30:.1f} ms/tile")

# 方案3: 看 NPU 核是否真能同时跑（全 3 核 mask 单推理）
print("\n=== 单推理使用全 3 核 mask ===")
r_all = RKNNLite()
r_all.load_rknn(MODEL_1024)
r_all.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
for _ in range(3): r_all.inference(inputs=[dummy], data_format="nhwc")  # warmup
ts_all = []
for _ in range(10):
    t = time.perf_counter()
    r_all.inference(inputs=[dummy], data_format="nhwc")
    ts_all.append((time.perf_counter()-t)*1000)
print(f"  3核同跑单推理 median: {np.median(ts_all):.1f} ms")
r_all.release()

release_ctx(ctx)
