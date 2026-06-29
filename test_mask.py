import cv2
import os
import numpy as np
from vessel_mask import extract_vessel_mask

# 测试不同图像的掩码生成
def test_vessel_mask():
    # 测试图像目录
    test_dir = "results/filtered"
    
    # 获取测试图像列表
    test_images = [f for f in os.listdir(test_dir) if f.endswith('.png')][:5]  # 只测试前5张图像
    
    # 输出目录
    output_dir = "mask_test"
    os.makedirs(output_dir, exist_ok=True)
    
    print("测试不同图像的血管掩码生成:")
    print("-" * 60)
    
    for i, img_name in enumerate(test_images):
        # 读取图像
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"无法读取图像: {img_name}")
            continue
        
        # 提取血管掩码
        mask = extract_vessel_mask(img)
        
        # 计算掩码统计信息
        mask_sum = mask.sum()
        mask_size = mask.size
        mask_density = mask_sum / mask_size
        
        # 保存掩码图像
        mask_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_mask.png")
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        
        # 打印统计信息
        print(f"图像 {img_name}:")
        print(f"  掩码像素数: {mask_sum}")
        print(f"  图像总像素: {mask_size}")
        print(f"  掩码密度: {mask_density:.6f}")
        print("-" * 60)

if __name__ == "__main__":
    test_vessel_mask()