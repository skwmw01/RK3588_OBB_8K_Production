# RK3588 OBB 8K 航拍目标检测 Production 方案

**版本**：v1.0 · 2026-04-23  
**目标平台**：瑞芯微 RK3588 (Orange Pi 5 / 5 Plus)  
**任务**：8192×8192 航拍大图，旋转框 (OBB) 多目标检测  
**性能**：**815 ms / 63.8% recall**，完全在 1 秒内

---

## 🎯 一句话

> 把 8K 大图**规则切成 5×5 个 2048 patch**，每个 patch **resize 到 1024** 丢进 INT8 量化的 YOLOv8n-OBB，**3 核 NPU 并行解码 + 旋转 NMS**，**总耗时 815 ms**。

---

## 📁 交付包结构

```
RK3588_OBB_8K_Production/
├── README.md                    ← 本文档
├── test_8k.jpg                  ← 测试图 (8192×8192 DOTA 航拍)
├── models/
│   └── yolov8n-obb_i8_1024_airockchip.rknn   ← 部署模型 (4.6 MB)
├── code/
│   ├── common.py                ← decode / NMS / recall / 并行基建
│   ├── production_v2.py         ← ⭐ Production 最终版
│   ├── production_bench.py      ← 时间剖析版（单独统计 decode）
│   ├── user_rule.py             ← 首个对照实验（2048 直推 vs 1024 resize）
│   ├── v2_resize.py             ← 2048 切 + resize 1024 初版
│   ├── v2_opt.py                ← resize 算法 / conf 阈值消融
│   ├── e18_fast.py              ← 老版 hot-expand 优化（对比基线）
│   ├── e18_1s.py                ← 限 hot-tile 数量的 1s 可达性探索
│   ├── e18_profile.py           ← 瓶颈剖析（发现 bw/CPU 分布）
│   └── perf_diag.py             ← 3 核 NPU 扩展率剖析
├── quantize/
│   ├── export_v8_1024.py        ← ultralytics 原版导出（仅供对比，有坑）
│   ├── convert_airockchip_v8.py ← 最终可用的 RKNN 量化脚本
│   └── yolov8n-obb_airockchip.onnx  ← airockchip 分支导出的 ONNX
├── bench_results/
│   ├── production_bench.json    ← 最终方案 10 次实测数据
│   ├── e18_fast_results.json
│   ├── e18_1s_results.json
│   └── v2_results.json
└── docs/
    ├── 01_实验概述.md
    ├── 02_模型来历_量化步骤.md
    ├── 03_方案原理与消融实验.md
    ├── 04_时间剖析与性能报告.md
    └── 05_部署指引.md
```

---

## ⏱️ 最终性能速览

### 实测 10 次连续运行（RK3588 / Orange Pi 5 / Ubuntu 22.04）

| 指标 | 数值 |
|---|---|
| **总耗时中位** | **815 ms** ✅ |
| 总耗时最快 | 787 ms |
| 总耗时最慢 | 853 ms |
| 抖动 | ±33 ms（4%） |
| **召回 (IoU=0.3)** | **63.8%** (30/47) |
| 召回抖动 | 0%（10 次完全一致）|
| 检出数 (final after NMS) | 42-47 |
| 原始检出数 (before NMS) | ~407 |

### 分阶段耗时（单次剖析）

| 阶段 | 耗时 | 占比 |
|---|---|---|
| Tile 规划 | 0.05 ms | 0.01% |
| 预处理 (25 tile crop+resize+cvt) | ~40 ms | 5% |
| NPU 推理 (25 tile × 3 核) | ~440 ms | 54% |
| OBB Decode (25 tile softmax+DFL) | ~285 ms | 35% |
| Rotated NMS | 1.8 ms | 0.2% |
| **总计（不含读图）** | **815 ms** | 100% |

> 读图 `cv2.imread` 冷读 8K JPEG 约 1055 ms，业务上通常热数据直接在内存里，**不计入推理流水**。

---

## 📖 阅读顺序建议

1. **[01_实验概述](docs/01_实验概述.md)** — 背景、问题、为什么 1s 内做 8K OBB 这么难
2. **[02_模型来历 & 量化步骤](docs/02_模型来历_量化步骤.md)** — YOLOv8n-OBB / DOTA / airockchip 分支 / RKNN INT8 踩坑史
3. **[03_方案原理 & 消融实验](docs/03_方案原理与消融实验.md)** — 核心思想 + 完整消融表
4. **[04_时间剖析 & 性能报告](docs/04_时间剖析与性能报告.md)** — 10 次 benchmark + 3 核负载分析
5. **[05_部署指引](docs/05_部署指引.md)** — 如何在新 RK3588 上运行 + 集成

---

## 🚀 快速运行

在 RK3588 板子上：

```bash
# 1. 依赖
pip install rknn-toolkit-lite2 opencv-python numpy

# 2. 跑 benchmark
python code/production_v2.py
```

输出（节选）：
```
=== conf=0.15 ===
  total:  min=787  median=815  mean=823  max=853 ms
  recall: 63.8%   raw_dets median=407
```

---

## ✍️ 作者

基于 RK3588 3 核 NPU 的 OBB 大图推理优化实验，2026-04-22 ~ 2026-04-23 完成。  
共进行 **40+ 组消融实验**（见 `bench_results/*.json`），最终方案见 `code/production_v2.py`。
