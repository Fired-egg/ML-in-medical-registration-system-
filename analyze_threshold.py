import cv2
import numpy as np
import os
from common.common_util import pre_processing

def analyze_threshold_range(image_path):
    """
    分析顶帽变换后的图像值分布，帮助确定合适的阈值范围
    
    参数:
        image_path: 输入图像路径
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 预处理
    preprocessed = pre_processing(img)
    preprocessed = (preprocessed * 255).astype(np.uint8)
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(preprocessed, (5, 5), 0)
    
    # 顶帽变换
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    
    # 分析顶帽变换后的图像
    print(f"\n图像: {os.path.basename(image_path)}")
    print(f"顶帽变换后的图像统计:")
    print(f"  最小值: {tophat.min()}")
    print(f"  最大值: {tophat.max()}")
    print(f"  平均值: {tophat.mean():.2f}")
    print(f"  中位数: {np.median(tophat):.2f}")
    print(f"  标准差: {tophat.std():.2f}")
    
    # 计算直方图
    hist = cv2.calcHist([tophat], [0], None, [256], [0, 256])
    hist = hist.flatten()
    
    # 找到直方图的峰值
    peak_value = np.argmax(hist)
    print(f"  直方图峰值位置: {peak_value}")
    
    # 建议阈值范围
    print(f"\n建议阈值范围:")
    print(f"  保守（检测主要血管）: {peak_value + 20} - {peak_value + 50}")
    print(f"  中等（检测更多血管）: {peak_value + 10} - {peak_value + 30}")
    print(f"  激进（检测细小血管）: {peak_value} - {peak_value + 20}")
    
    # 测试不同阈值的效果
    print(f"\n测试不同阈值的效果:")
    test_thresholds = [peak_value, peak_value + 10, peak_value + 20, peak_value + 30, peak_value + 50]
    for t in test_thresholds:
        _, binary = cv2.threshold(tophat, t, 255, cv2.THRESH_BINARY)
        vessel_pixels = np.sum(binary > 0)
        total_pixels = binary.size
        vessel_ratio = vessel_pixels / total_pixels * 100
        print(f"  阈值={t:3d}: 血管像素占比={vessel_ratio:5.2f}%")

def analyze_directory(input_dir, max_samples=3):
    """
    分析目录中的图像
    
    参数:
        input_dir: 输入目录
        max_samples: 最大分析样本数
    """
    os.makedirs(input_dir, exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"找到 {len(files)} 个图像文件")
    
    # 只分析前几个样本
    for i, file in enumerate(files[:max_samples]):
        image_path = os.path.join(input_dir, file)
        analyze_threshold_range(image_path)
    
    if len(files) > max_samples:
        print(f"\n...（还有 {len(files) - max_samples} 个图像未分析）")

if __name__ == "__main__":
    input_dir = "results/filtered"
    
    if os.path.exists(input_dir):
        analyze_directory(input_dir, max_samples=3)
    else:
        print(f"输入目录不存在: {input_dir}")