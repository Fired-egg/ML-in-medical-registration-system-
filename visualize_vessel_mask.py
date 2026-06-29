import cv2
import numpy as np
import os
from vessel_mask import extract_vessel_mask
from common.common_util import pre_processing

def visualize_vessel_mask(image_path, output_dir, threshold=50):
    """
    可视化血管掩码
    
    参数:
        image_path: 输入图像路径
        output_dir: 输出目录
        threshold: 手动阈值（None 则使用自动阈值）
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 预处理图像（与 predictor 相同的步骤）
    preprocessed = pre_processing(img)
    preprocessed_uint8 = (preprocessed * 255).astype(np.uint8)
    
    # 提取血管掩码（使用原始图像，函数内部会进行预处理）
    mask_from_raw = extract_vessel_mask(img, threshold=threshold)
    
    # 提取血管掩码（直接使用预处理后的图像）
    mask_from_preprocessed = extract_vessel_mask(preprocessed_uint8, threshold=threshold)
    
    # 转换为可视化格式
    mask_raw_visual = (mask_from_raw * 255).astype(np.uint8)
    mask_pre_visual = (mask_from_preprocessed * 255).astype(np.uint8)
    
    # 创建彩色叠加效果（使用预处理后的图像）
    img_color = cv2.cvtColor(preprocessed_uint8, cv2.COLOR_GRAY2BGR)
    mask_color = cv2.cvtColor(mask_pre_visual, cv2.COLOR_GRAY2BGR)
    mask_color[:, :, 0] = 0  # 蓝色通道设为0
    mask_color[:, :, 1] = 0  # 绿色通道设为0
    # 红色标记血管
    overlay = cv2.addWeighted(img_color, 0.7, mask_color, 0.3, 0)
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    # 保存原始图像（未预处理）
    cv2.imwrite(os.path.join(output_dir, f"{name}_raw{ext}"), img)
    
    # 保存预处理后的图像
    cv2.imwrite(os.path.join(output_dir, f"{name}_preprocessed{ext}"), preprocessed_uint8)
    
    # 保存基于原始图像的血管掩码
    cv2.imwrite(os.path.join(output_dir, f"{name}_mask_raw{ext}"), mask_raw_visual)
    
    # 保存基于预处理后图像的血管掩码
    cv2.imwrite(os.path.join(output_dir, f"{name}_mask_pre{ext}"), mask_pre_visual)
    
    # 保存叠加效果
    cv2.imwrite(os.path.join(output_dir, f"{name}_overlay{ext}"), overlay)
    
    print(f"血管掩码可视化已保存到: {output_dir}")

def visualize_directory(input_dir, output_dir, threshold=None):
    """
    可视化目录中所有图像的血管掩码
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        threshold: 手动阈值（None 则使用自动阈值）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理目录中的所有图像
    for file in os.listdir(input_dir):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, file)
            visualize_vessel_mask(image_path, output_dir, threshold=threshold)

if __name__ == "__main__":
    # 示例：可视化 results/filtered 目录中的图像
    input_dir = "results/filtered"
    output_dir = "results/vessel_mask_visualization"
    
    # 可以在这里设置手动阈值，例如 threshold=50
    threshold = None  # 使用自动阈值
    
    if os.path.exists(input_dir):
        visualize_directory(input_dir, output_dir, threshold=threshold)
        print(f"所有图像的血管掩码已可视化到: {output_dir}")
    else:
        print(f"输入目录不存在: {input_dir}")
        print("请先运行预处理，生成 filtered 文件夹")
