#!/bin/bash
# 一键 benchmark 脚本（板子上跑）
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "======================================"
echo "RK3588 OBB 8K Benchmark"
echo "======================================"
echo

# 检查依赖
python -c "import cv2, numpy, rknnlite" 2>&1 || { echo "❌ 依赖缺失：pip install opencv-python numpy rknn-toolkit-lite2"; exit 1; }

# 检查模型
[ -f "models/yolov8n-obb_i8_1024_airockchip.rknn" ] || { echo "❌ 模型缺失"; exit 1; }
[ -f "test_8k.jpg" ] || { echo "❌ 测试图缺失"; exit 1; }

# 检查路径引用（common.py 里写死了 /home/orangepi/ablation/... 默认路径）
# 用户需要改或 symlink：
if [ ! -d "/home/orangepi/ablation" ]; then
    echo "⚠️  common.py 期望的默认路径 /home/orangepi/ablation 不存在"
    echo "    方案 A：symlink: sudo mkdir -p /home/orangepi && sudo ln -s $SCRIPT_DIR /home/orangepi/ablation"
    echo "    方案 B：修改 code/common.py 顶部的 MODEL_DIR 和 IMG_PATH"
    echo
fi

# NPU 频率检查
NPU_GOV=$(cat /sys/class/devfreq/fdab0000.npu/governor 2>/dev/null || echo "unknown")
echo "NPU governor: $NPU_GOV"
[ "$NPU_GOV" != "performance" ] && echo "    建议: echo performance | sudo tee /sys/class/devfreq/fdab0000.npu/governor"
echo

# 运行 benchmark
cd code
python production_v2.py
