"""E18 时间细分：看粗扫、hot 生成、精扫各花多少"""
import cv2, numpy as np, time, threading, queue, math
from rknnlite.api import RKNNLite

MODEL_I8_1024 = "/home/orangepi/ablation/models/yolov8n-obb_i8_1024_airockchip.rknn"
IMG = "/home/orangepi/ablation/test_8k.jpg"
TILE = 1024
STRIDE = 960
CONF_COARSE = 0.1
CONF_FINE = 0.3

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

cores = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
ctx = []
for c in cores:
    r = RKNNLite()
    r.load_rknn(MODEL_I8_1024)
    r.init_runtime(core_mask=c)
    # warmup
    r.inference(inputs=[np.zeros((1,TILE,TILE,3), dtype=np.uint8)], data_format="nhwc")
    ctx.append(r)
print("3 INT8 ctx ready")

def decode(outs, scale=1.0, ox=0, oy=0, conf_th=0.3):
    dets = []
    strides = [8, 16, 32]
    total = 0; offsets = []
    for o in outs[:3]:
        offsets.append(total); total += o.shape[2]*o.shape[3]
    ang = outs[3][0,0]
    for o, st, aoff in zip(outs[:3], strides, offsets):
        _,_,H,W = o.shape
        xywh = o[0,:64].reshape(4,16,H,W).astype(np.float32)
        cls = sigmoid(o[0,64:].astype(np.float32))
        sm = np.exp(xywh - xywh.max(axis=1, keepdims=True))
        sm /= sm.sum(axis=1, keepdims=True)
        dfl = (sm * np.arange(16).reshape(1,16,1,1)).sum(axis=1)
        cls_id = cls.argmax(axis=0); conf = cls.max(axis=0)
        mask = conf > conf_th
        if not mask.any(): continue
        hs, ws = np.where(mask)
        for k in range(len(hs)):
            h,w = hs[k], ws[k]
            l,t,r,b = dfl[:,h,w]
            cx = (w+0.5+(r-l)/2)*st*scale+ox
            cy = (h+0.5+(b-t)/2)*st*scale+oy
            bw = (l+r)*st*scale; bh=(t+b)*st*scale
            aidx = aoff+h*W+w
            angle = (ang[aidx]-0.25)*math.pi
            dets.append([cx,cy,bw,bh,angle,float(conf[h,w]),int(cls_id[h,w])])
    return dets

def coords_to_tiles(dets, W, H, expand=1):
    stride = STRIDE; tiles = set()
    def snap(tx,ty):
        tx=(tx//stride)*stride; ty=(ty//stride)*stride
        tx=max(0,min(tx,W-TILE)); ty=max(0,min(ty,H-TILE))
        return (tx,ty)
    for d in dets:
        bx=min(max(0,int(d[0]-TILE/2)), W-TILE)
        by=min(max(0,int(d[1]-TILE/2)), H-TILE)
        tx,ty=snap(bx,by); tiles.add((tx,ty))
        if expand:
            for dx in [-stride,0,stride]:
                for dy in [-stride,0,stride]:
                    if dx==0 and dy==0: continue
                    tiles.add(snap(bx+dx,by+dy))
    return list(tiles)

def infer_parallel(img, tile_list):
    """3 核 NPU 并发推理"""
    q_in = queue.Queue()
    q_out = queue.Queue()
    t_pre_start = time.perf_counter()
    # pre-crop + cvtColor on CPU (serialized)
    inputs = []
    for i,(x,y) in enumerate(tile_list):
        crop = img[y:y+TILE, x:x+TILE]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        inputs.append((i, x, y, np.expand_dims(rgb, 0)))
    t_pre = (time.perf_counter()-t_pre_start)*1000
    for inp in inputs: q_in.put(inp)
    all_dets = []
    lock = threading.Lock()
    def worker(r):
        while True:
            try: i,x,y,inp = q_in.get_nowait()
            except queue.Empty: return
            outs = r.inference(inputs=[inp], data_format="nhwc")
            dets = decode(outs, scale=1.0, ox=x, oy=y, conf_th=CONF_FINE)
            with lock: all_dets.extend(dets)
    t_inf_start = time.perf_counter()
    threads = [threading.Thread(target=worker,args=(r,)) for r in ctx]
    for t in threads: t.start()
    for t in threads: t.join()
    t_inf = (time.perf_counter()-t_inf_start)*1000
    return all_dets, t_pre, t_inf

# ===== 跑 E18 =====
print(f"\n[load] {IMG}")
t_read0 = time.perf_counter()
img = cv2.imread(IMG)
t_read = (time.perf_counter()-t_read0)*1000
H,W = img.shape[:2]
print(f"  size: {W}x{H}, read: {t_read:.1f}ms")

# Step 1: thumb 粗扫
t1 = time.perf_counter()
thumb = cv2.resize(img, (TILE, TILE))
rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
outs = ctx[0].inference(inputs=[np.expand_dims(rgb,0)], data_format="nhwc")
dets_c = decode(outs, scale=W/TILE, conf_th=CONF_COARSE)
t_coarse = (time.perf_counter()-t1)*1000
print(f"\n[Step1] 粗扫 thumb: {t_coarse:.1f}ms")
print(f"  raw dets (conf>={CONF_COARSE}): {len(dets_c)}")

# Step 2: hot tiles + expand
t2 = time.perf_counter()
hot = coords_to_tiles(dets_c, W, H, expand=1)
t_plan = (time.perf_counter()-t2)*1000
print(f"\n[Step2] hot tile 规划: {t_plan:.1f}ms")
print(f"  hot tiles: {len(hot)}")

# Step 3: 精扫并发
t3 = time.perf_counter()
dets_f, t_pre, t_inf = infer_parallel(img, hot)
t_fine = (time.perf_counter()-t3)*1000
print(f"\n[Step3] 精扫 {len(hot)} tile: {t_fine:.1f}ms")
print(f"  其中 pre(crop+cvtColor serial): {t_pre:.1f}ms")
print(f"  其中 inf(3核并发): {t_inf:.1f}ms")
print(f"  fine dets: {len(dets_f)}")

total = t_coarse + t_plan + t_fine
print(f"\n=== TOTAL: {total:.1f} ms ===")
print(f"  图读取(不计): {t_read:.1f} ms")
print(f"  粗扫: {t_coarse:.1f} ms ({t_coarse/total*100:.0f}%)")
print(f"  规划:  {t_plan:.1f} ms ({t_plan/total*100:.0f}%)")
print(f"  精扫: {t_fine:.1f} ms ({t_fine/total*100:.0f}%)")
print(f"    └ pre(CPU串行): {t_pre:.1f} ms")
print(f"    └ inf(3核并发): {t_inf:.1f} ms")

for r in ctx: r.release()
