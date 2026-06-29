# 激光散斑眼底视频帧图像配准算法代码文档

## 一、项目概述

本项目实现了基于深度学习的激光散斑眼底视频帧图像配准算法，主要包含以下核心模块：

| 模块 | 文件位置 | 功能描述 |
|------|----------|----------|
| 预处理模块 | `pre/get_base.py` | 图像质量过滤、基准帧选择、对比度增强 |
| 配准核心模块 | `predictor.py` | SuperRetina深度学习模型推理、特征匹配、单应性估计 |
| 配准流程模块 | `register_from_base.py` | 批量帧配准、匹配信息保存 |
| 评价模块 | `evaluate_registration.py` | NCC/DSC指标计算、血管掩码提取 |
| 可视化模块 | `visualize_and_evaluate.py` | 血管掩码可视化、配准效果展示 |
| GUI模块 | `app.py` / `gui_worker.py` | 图形界面操作、多线程任务管理 |

---

## 二、整体架构与数据流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        数据处理流程                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  原始图像 ──→ [预处理模块] ──→ filtered文件夹                       │
│      │                              │                               │
│      │                              ↓                               │
│      │                    [配准核心模块]                            │
│      │                              │                               │
│      │                              ↓                               │
│      │                    registered_filtered文件夹                 │
│      │                              │                               │
│      │                              ↓                               │
│      │                    [predictor预处理]                          │
│      │                              │                               │
│      │                              ↓                               │
│      │                    filtered_predictor_preprocessed文件夹      │
│      │                              │                               │
│      │                              ↓                               │
│      └──────────→ [评价模块] ←─────────┘                            │
│                              │                                      │
│                              ↓                                      │
│                    vessel_mask_visualization文件夹                   │
│                              │                                      │
│                              ↓                                      │
│                    registration_eval.csv / registration_stats.txt   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、核心模块详细说明

### 3.1 预处理模块 (`pre/get_base.py`)

**功能定位**：对原始视频帧进行质量过滤和基准帧选择

**处理流程**：

```python
# 1. 图像质量评估
- 模糊检测：使用拉普拉斯方差计算
- 过曝/欠曝检测：基于直方图统计
- 全黑帧检测：像素值统计

# 2. 质量评分与排序
- 综合评分 = 清晰度分数 + 对比度分数 + 曝光分数

# 3. 基准帧选择
- 选择质量最高的帧作为基准帧
- 保存基准帧信息到 frame_info.json

# 4. 图像预处理
- 对比度增强（CLAHE）
- 高斯模糊去噪（可选）
```

**关键输出文件**：
- `results/filtered/` - 过滤后的有效帧
- `results/frame_info.json` - 基准帧信息

---

### 3.2 配准核心模块 (`predictor.py`)

**功能定位**：使用SuperRetina深度学习模型进行特征提取和图像配准

**核心类：Predictor**

#### 3.2.1 初始化流程

```python
def __init__(self, config):
    # 1. 加载配置参数
    - device: 计算设备（GPU/CPU）
    - model_save_path: 预训练模型路径
    - nms_size/nms_thresh: NMS参数
    - knn_thresh: KNN匹配阈值
    
    # 2. 加载预训练模型
    - 初始化SuperRetina模型
    - 加载训练好的权重
    - 设置为评估模式
    
    # 3. 初始化匹配器
    - BFMatcher (L2距离)
```

#### 3.2.2 图像预处理 (`image_read`)

```python
def image_read(self, query_path, refer_path):
    # 1. 读取彩色图像，提取绿色通道
    query_image = query_image[:, :, 1]
    refer_image = refer_image[:, :, 1]
    
    # 2. 应用标准化预处理（common_util.pre_processing）
    #    - 直方图均衡化
    #    - Gamma校正
    #    - 对比度归一化
    
    # 3. 转换为 uint8 格式
```

#### 3.2.3 特征提取 (`model_run_pair`)

```python
def model_run_pair(self, query_tensor, refer_tensor):
    # 1. 模型推理
    detector_pred, descriptor_pred = self.model(inputs)
    
    # 2. NMS非极大值抑制
    scores = simple_nms(detector_pred, self.nms_size)
    
    # 3. 关键点提取
    keypoints = torch.nonzero(scores > threshold)
    
    # 4. 边界去除
    keypoints = remove_borders(keypoints, 4, h, w)
    
    # 5. 描述子采样
    descriptors = sample_keypoint_desc(keypoints, descriptors, 8)
    
    return keypoints, descriptors
```

#### 3.2.4 特征匹配 (`match`)

```python
def match(self, query_path, refer_path):
    # 1. 图像读取与预处理
    query_image, refer_image = self.image_read(query_path, refer_path)
    
    # 2. 特征提取
    keypoints, descriptors = self.model_run_pair(query_tensor, refer_tensor)
    
    # 3. KNN匹配（k=2）
    matches = self.knn_matcher.knnMatch(query_desc, refer_desc, k=2)
    
    # 4. Lowe's ratio test
    for m, n in matches:
        if m.distance < self.knn_thresh * n.distance:
            goodMatch.append(m)
    
    return goodMatch, keypoints, images
```

#### 3.2.5 单应性矩阵估计 (`compute_homography`)

```python
def compute_homography(self, query_path, refer_path):
    # 1. 获取匹配点
    goodMatch, cv_kpts_query, cv_kpts_refer = self.match(...)
    
    # 2. 准备匹配点坐标
    src_pts = [cv_kpts_query[m.queryIdx].pt for m in goodMatch]
    dst_pts = [cv_kpts_refer[m.trainIdx].pt for m in goodMatch]
    
    # 3. 使用LMEDS算法估计单应性矩阵
    H_m, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
    
    # 4. 计算内点率
    inliers_num_rate = num_inliers / len(mask.ravel())
    
    return H_m, inliers_num_rate, match_info
```

#### 3.2.6 图像配准 (`align_image_pair`)

```python
def align_image_pair(self, query_path, refer_path):
    # 1. 计算单应性矩阵
    H_m, inliers_num_rate, ... = self.compute_homography(...)
    
    # 2. 透视变换
    if H_m is not None and inliers_num_rate >= 0.1:
        query_align = cv2.warpPerspective(raw_query_image, H_m, (w, h))
    
    # 3. 构建匹配信息字典
    match_info = {
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        "inliers_rate": inliers_num_rate,
        "H": H_m.tolist()
    }
    
    return merged, match_info
```

---

### 3.3 配准流程模块 (`register_from_base.py`)

**功能定位**：批量处理视频帧，将所有浮动帧配准到基准帧

**处理流程**：

```python
def register_from_base(results_dir, source_folder="filtered"):
    # 1. 初始化Predictor
    config = yaml.safe_load(open(config_path))
    predictor = Predictor(config)
    
    # 2. 读取基准帧信息
    with open("frame_info.json") as f:
        info = json.load(f)
        reference_frame = info["reference_frame"]
        valid_files = info["valid_files"]
    
    # 3. 基准帧自配准
    ref_result = predictor.align_image_pair(refer_path, refer_path)
    
    # 4. 遍历所有浮动帧进行配准
    for fname in valid_files:
        query_path = os.path.join(frames_dir, fname)
        result = predictor.align_image_pair(query_path, refer_path)
        
        # 保存配准结果
        cv2.imwrite(os.path.join(out_dir, fname), aligned_bgr)
        
        # 收集匹配信息
        all_match_info[fname] = match_info
    
    # 5. 保存匹配信息到JSON
    with open("match_info.json", "w") as f:
        json.dump(all_match_info, f)
```

**关键输出**：
- `results/registered_filtered/` - 配准后的图像
- `results/match_info_filtered/match_info.json` - 匹配点信息

---

### 3.4 评价模块 (`evaluate_registration.py`)

**功能定位**：计算配准效果评价指标

**核心指标**：

| 指标 | 计算方式 | 意义 |
|------|----------|------|
| NCC | 归一化互相关 | 衡量图像相似度 [-1, 1] |
| DSC | Dice相似系数 | 衡量血管掩码重叠度 [0, 1] |

**血管掩码提取流程**：

```python
def extract_vessel_mask(gray, threshold=50):
    # 1. 预处理（与predictor相同）
    preprocessed = pre_processing(gray)
    
    # 2. 高斯模糊
    blurred = cv2.GaussianBlur(preprocessed, (5, 5), 0)
    
    # 3. 顶帽变换（突出血管结构）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    
    # 4. 阈值分割
    _, binary = cv2.threshold(tophat, threshold, 255, cv2.THRESH_BINARY)
    
    # 5. 形态学操作
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    # 6. 连通区域分析（去除小区域）
    num_labels, labels, stats = cv2.connectedComponentsWithStats(opened)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            opened[labels == i] = 0
    
    return opened > 0
```

**评价流程**：

```python
def evaluate(results_dir, threshold=50):
    # 1. 读取基准帧和浮动帧
    ref_pre = cv2.imread(ref_pre_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. 遍历所有浮动帧
    for fname in valid_files:
        pre_img = cv2.imread(pre_path)  # 配准前
        post_img = cv2.imread(post_path)  # 配准后
        
        # 3. 计算NCC
        ncc_before = ncc(pre_img, ref_pre)
        ncc_after = ncc(post_img, ref_post)
        
        # 4. 提取血管掩码并计算DSC
        vessel_mask_pre = extract_vessel_mask(pre_img, threshold)
        vessel_mask_post = extract_vessel_mask(post_img, threshold)
        
        dsc_before = dsc(vessel_mask_pre, vessel_mask_ref)
        dsc_after = dsc(vessel_mask_post, vessel_mask_ref)
        
        # 5. 保存血管掩码可视化
        cv2.imwrite(f"{fname}_pre_mask.png", vessel_mask_pre * 255)
        cv2.imwrite(f"{fname}_post_mask.png", vessel_mask_post * 255)
    
    # 6. 计算统计信息（均值、上下限）
    stats = {
        "ncc_before_mean": np.mean(ncc_before_list),
        "ncc_after_mean": np.mean(ncc_after_list),
        ...
    }
    
    # 7. 输出CSV和统计文件
```

---

### 3.5 可视化模块 (`visualize_and_evaluate.py`)

**功能定位**：提供血管掩码的可视化展示

**生成文件**：

| 文件类型 | 命名格式 | 说明 |
|----------|----------|------|
| 原始图像 | `*_raw.png` | 未预处理的原始图像 |
| 预处理图像 | `*_preprocessed.png` | 经过predictor预处理的图像 |
| 血管掩码 | `*_mask.png` | 阈值分割得到的血管区域 |
| 叠加效果 | `*_overlay.png` | 血管掩码叠加在预处理图像上 |

---

## 四、配置文件说明

### 4.1 `config/test.yaml`

```yaml
PREDICT:
  device: "cuda:0"              # 计算设备
  model_save_path: "model.pth"  # 预训练模型路径
  nms_size: 8                   # NMS窗口大小
  nms_thresh: 0.05              # NMS阈值
  knn_thresh: 0.7               # KNN匹配阈值
  model_image_width: 640        # 模型输入宽度
  model_image_height: 640       # 模型输入高度
```

---

## 五、运行方式

### 5.1 命令行方式

```bash
# 1. 预处理（质量过滤和基准帧选择）
python pre/01_get_base.py --input_dir /path/to/images --output_dir results

# 2. 配准（批量帧配准到基准帧）
python register_from_base.py

# 3. predictor预处理（用于血管掩码生成）
python preprocess_filtered.py

# 4. 评价（计算指标并生成血管掩码）
python evaluate_registration.py

# 或者一步完成（推荐）
python run_all.py
```

### 5.2 GUI方式

```bash
python app.py
```

**GUI操作流程**：
1. 选择图像目录
2. 选择输出目录
3. 点击"一键配准"按钮
4. 等待配准完成后点击"评价配准结果"

---

## 六、关键技术要点

### 6.1 SuperRetina模型架构

```
┌─────────────────────────────────────────────────────┐
│                  SuperRetina模型                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  输入图像 ──→ [共享编码器] ──→ 特征图               │
│                    │                                │
│                    ├────→ [检测器头] ──→ 关键点得分  │
│                    │                                │
│                    └────→ [描述子头] ──→ 特征描述子  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 6.2 单应性变换原理

单应性矩阵是一个3x3矩阵，用于描述两个平面之间的透视变换：

```
[ x' ]   [ h11 h12 h13 ] [ x ]
[ y' ] = [ h21 h22 h23 ] [ y ]
[ w' ]   [ h31 h32 h33 ] [ 1 ]

其中：x' = x/w', y' = y/w'
```

### 6.3 内点率计算

内点率是符合几何模型（单应性矩阵）的匹配点占总匹配点的比例：

```
内点率 = 内点数量 / 总匹配点数量

误匹配率 = 1 - 内点率
```

---

## 七、文件结构总结

```
project_SuperRetina_main/
├── common/                    # 通用工具函数
│   ├── common_util.py         # 预处理、NMS、关键点采样
│   ├── eval_util.py           # 评价指标计算
│   └── train_util.py          # 训练相关工具
├── config/                    # 配置文件
│   └── test.yaml              # 推理配置
├── model/                     # 深度学习模型
│   ├── super_retina.py        # SuperRetina主模型
│   ├── pke_module.py          # 关键点增强模块
│   └── record_module.py       # 记录模块
├── pre/                       # 预处理脚本
│   ├── get_base.py            # 基准帧选择核心逻辑
│   └── 01_get_base.py         # 预处理入口脚本
├── predictor.py               # 模型推理核心类
├── register_from_base.py      # 批量配准流程
├── evaluate_registration.py   # 配准评价模块
├── visualize_and_evaluate.py  # 可视化模块
├── app.py                     # GUI主界面
├── gui_worker.py              # GUI工作线程
└── results/                   # 输出目录
    ├── filtered/              # 过滤后的图像
    ├── registered_filtered/   # 配准后的图像
    ├── filtered_predictor_preprocessed/  # predictor预处理图像
    ├── vessel_mask_visualization/        # 血管掩码可视化
    ├── chessboard/            # 棋盘格对比图
    └── registration_eval.csv  # 评价结果CSV
```

---

## 八、扩展说明

### 8.1 阈值调整

血管掩码提取的阈值参数可在以下位置调整：

1. `evaluate_registration.py` 的 `evaluate()` 函数参数
2. `vessel_mask.py` 的 `extract_vessel_mask()` 函数参数

**阈值范围建议**：
- 较低阈值（30-50）：提取更多血管结构，但噪声较多
- 较高阈值（70-90）：提取较少血管结构，但噪声较少

### 8.2 预处理选项

在 `pre/get_base.py` 中可配置：
- 是否启用光斑去除（`remove_specular` 参数）
- CLAHE对比度增强参数
- 图像质量过滤阈值

### 8.3 配准参数

在 `config/test.yaml` 中可配置：
- NMS阈值（`nms_thresh`）
- KNN匹配阈值（`knn_thresh`）
- 模型输入尺寸（`model_image_width/height`）

---

**文档版本**: v1.0  
**生成日期**: 2026年5月  
**适用项目**: 激光散斑眼底视频帧图像配准算法研究