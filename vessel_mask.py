import cv2
import numpy as np
from common.common_util import pre_processing


def extract_vessel_mask(gray: np.ndarray, threshold=50) -> np.ndarray:
    """
    基于阈值分割的血管掩码提取函数
    使用与 predictor 相同的预处理步骤。
    返回值为 bool 类型的 2D 数组，True 表示血管区域。
    
    参数:
        gray: 单通道灰度图像
        threshold: 手动阈值（None 则使用自动阈值）
    """
    if gray.ndim != 2:
        raise ValueError("extract_vessel_mask 期望输入为单通道灰度图像。")

    # 1. 使用与 predictor 相同的预处理步骤
    # 预处理包括：标准化、CLAHE、Gamma校正
    preprocessed = pre_processing(gray)
    
    # 2. 转换为 uint8 格式用于阈值分割
    preprocessed = (preprocessed * 255).astype(np.uint8)
    
    # 3. 高斯模糊去噪
    blurred = cv2.GaussianBlur(preprocessed, (5, 5), 0)
    
    # 4. 顶帽变换，突出血管结构
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    
    # 5. 阈值分割
    if threshold is not None:
        # 使用手动阈值
        _, binary = cv2.threshold(tophat, threshold, 255, cv2.THRESH_BINARY)
    else:
        # 自动阈值 - 只使用Otsu阈值
        _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 8. 形态学操作
    # 闭运算连接断裂的血管
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    
    # 开运算去除小噪声
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel, iterations=1)
    
    # 9. 连通区域分析，去除小区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    min_area = 40  # 最小血管区域面积
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            opened[labels == i] = 0

    return opened > 0

