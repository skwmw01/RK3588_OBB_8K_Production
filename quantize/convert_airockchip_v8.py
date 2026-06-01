"""airockchip yolov8-obb @ 1024 → INT8 RKNN (用官方推荐 normal 算法)"""
import os, time
from rknn.api import RKNN

ONNX = "/home/xd/yolo11-obb-work/yolov8n-obb_1024_airockchip.onnx"
DATASET = "/home/xd/yolo11-obb-work/dataset.txt"
OUT = "/home/xd/yolo11-obb-work/yolov8n-obb_i8_1024_airockchip.rknn"

r = RKNN(verbose=False)
print("[1/4] config (官方推荐)")
r.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform="rk3588",
    quantized_algorithm="normal",
    optimization_level=3,
)
print("[2/4] load onnx")
r.load_onnx(model=ONNX)
print("[3/4] build quantized")
t0 = time.time()
r.build(do_quantization=True, dataset=DATASET)
print(f"  build took {time.time()-t0:.0f}s")
print("[4/4] export")
r.export_rknn(OUT)
print(f"done: {os.path.getsize(OUT)/1e6:.2f} MB")
r.release()
