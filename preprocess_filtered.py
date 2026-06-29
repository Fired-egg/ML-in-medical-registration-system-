import cv2
import numpy as np
import os
import argparse
from common.common_util import pre_processing


def preprocess_images(input_dir, output_dir):
    """
    将输入目录中的图像按照predictor中的预处理步骤进行处理
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for file in os.listdir(input_dir):
        if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image_files.append(file)
    
    # 按数字排序
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    
    # 处理目录中的所有图像
    for i, file in enumerate(image_files):
        image_path = os.path.join(input_dir, file)
        
        # 读取图像（与predictor相同的步骤）
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"无法读取图像: {file}")
            continue
        
        # 只使用绿色通道（与predictor相同）
        img = img[:, :, 1]
        
        # 预处理（与predictor相同）
        preprocessed = pre_processing(img)
        
        # 转换为uint8格式（与predictor相同）
        preprocessed_uint8 = (preprocessed * 255).astype(np.uint8)
        
        # 保存预处理后的图像
        output_path = os.path.join(output_dir, file)
        cv2.imwrite(output_path, preprocessed_uint8)
        
        # 进度提示
        if (i + 1) % 20 == 0 or i + 1 == len(image_files):
            print(f"已处理 {i + 1}/{len(image_files)} 张图像")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量预处理图像，使用与predictor相同的预处理步骤')
    parser.add_argument('--input', '-i', type=str, default='results/filtered',
                        help='输入目录（包含待处理的图像）')
    parser.add_argument('--output', '-o', type=str, default='results/filtered_predictor_preprocessed',
                        help='输出目录（保存预处理后的图像）')
    
    args = parser.parse_args()
    
    input_dir = args.input
    output_dir = args.output
    
    if os.path.exists(input_dir):
        print(f"开始处理目录: {input_dir}")
        preprocess_images(input_dir, output_dir)
        print(f"\n所有图像已按照predictor的预处理步骤处理并保存到: {output_dir}")
    else:
        print(f"输入目录不存在: {input_dir}")