from ultralytics import YOLO
m = YOLO("yolov8n-obb.pt")
print(f"Loaded: {m.task} / classes={len(m.names)}")
m.export(format="onnx", imgsz=1024, opset=12, simplify=True, dynamic=False)
print("\n✅ Exported: yolov8n-obb.onnx")
