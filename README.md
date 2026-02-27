# 系统运行流程与参数配置说明

## 运行流程（按以下步骤启动系统）

### 1. 准备阶段
将待处理的**大图**放入 `figure/` 文件夹。

### 2. 图像切块
运行脚本将大图切成小块（如 `64x64`），并存入 `test_images/` 目录供 BNN 处理：

```bash
python tile_64.py
```

### 3. 启动接收端（终端 A）
打开终端 A 运行：

```bash
python receive1.py
```

接收端将监听 UDP 端口，实时重构画布并进行 YOLO 检测。

### 4. 启动发送端（终端 B）
打开终端 B 运行：

```bash
python send1.py
```

发送端会调用 BNN 脚本评分，按重要程度排序后通过 UDP 发送图像块。

### 5. 数据分析
任务完成后运行：

```bash
python plot.py
```

分析 `result/` 文件夹下的 CSV 数据，生成置信度与 IoU 的变化图表。

---

##  参数配置详解
各程序开头均设有配置区域，您可以根据实验需求调整以下参数。

### 1. `tile_64.py`（图像预处理）
用于将原始高分辨率图像切分为固定尺寸的 Patch。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `INPUT_PATH` | 初始大图存放地址 | `~/Fire_BNN_Tool/figure/` |
| `OUT_BASE_DIR` | 切块小图存放的统一地址 | `~/Fire_BNN_Tool/test_images` |
| `TILE_SIZE` | 切块尺寸（像素） | `64` |
| `STRIDE` | 切块步长（像素） | `64` |
| `PAD` | 是否对边缘进行填充以整除 | `True` |
| `FMT` | 保存的图像格式 | `"jpg"` |

### 2. `send1.py`（发送端）
负责 BNN 评分调用、优先级排序及 ALF 协议封装发送。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `bnn_script` | BNN 评分脚本路径 | `~/Fire_BNN_Tool/batch_test.py` |
| `result_file` | BNN 评分结果输出文件 | `~/Fire_BNN_Tool/result.txt` |
| `dst_ip` | 接收端 IP 地址 | `127.0.0.1` |
| `dst_port` | 接收端 UDP 端口 | `5005` |
| `sub_tile` | 转发子窗口尺寸（将 `64x64` 再细分） | `16` |
| `send_interval_s` | 发送每个包的间隔时间（秒） | `0.05` |

### 3. `receive1.py`（接收端 & 检测核心）
负责 UDP 监听、画布重构、YOLO 逻辑判定及 SCCS 提前终止。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `MODEL_PATH` | YOLOv8 模型权重路径 | `best.pt` 路径 |
| `LISTEN_PORT` | UDP 监听端口 | `5005` |
| `CHECK_INTERVAL` | 决策频率（每收到多少个包进行一次推理） | `25` |
| `CONF_THRESHOLD` | YOLO 原始检测置信度阈值 | `0.25` |
| `FINAL_CONF_THR` | SCCS 锁定目标的判定置信度阈值 | `0.70` |
| `STABLE_IOU_THR` | 判定目标框稳定的 IoU 阈值 | `0.90` |
| `STABLE_PATIENCE` | 判定锁定所需的连续稳定次数 | `5` |
| `SAVE_EVERY_CHECK` | 是否在每次推理决策时都保存过程图片 | `True` |

### 4. `plot.py`（可视化分析）
读取实验结果并生成性能趋势图。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `target_dir` | 存放 CSV 监控指标的文件夹 | `result` |
| `output_dir` | 图像保存路径（函数内部定义） | `result` |

---

## 📂 项目结构

```text
.
├── figure/               # 存放待处理的原始大图
├── result/               # 存放生成的 CSV 指标、过程图片及分析图表
├── test_images/          # 存放切分后的 Patch（BNN 扫描对象）
├── batch_test.py         # BNN 运行程序
├── tile_64.py            # 切片脚本
├── send1.py              # 发送端脚本
├── receive1.py           # 接收端脚本
├── plot.py               # 绘图分析脚本
└── README.md             # 本说明文档
```

---

## ⚠️ 注意事项
- **路径匹配**：请确保 `tile_64.py` 生成的坐标格式（如 `_x128_y256`）与 `send1.py` 中的正则解析逻辑一致。  
- **网络环境**：UDP 传输可能丢包，大规模测试时可适当调整 `send_interval_s`。  
- **文件夹创建**：`receive1.py` 会自动在当前目录下创建 `result/` 文件夹用于存储输出。  
