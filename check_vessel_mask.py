import os
import cv2
import numpy as np
from vessel_mask import extract_vessel_mask

# 检查血管掩码提取的问题
def check_vessel_mask():
    """
    检查血管掩码提取的问题，看看为什么DSC值总是1
    """
    # 1. 选择一个测试图像
    test_dir = "results/filtered"
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.png')][:5]  # 前5个文件
    
    print(f"测试目录: {test_dir}")
    print(f"测试文件: {test_files}")
    
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        print(f"\n测试图像: {test_file}")
        
        # 2. 读取图像
        img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  无法读取图像: {test_path}")
            continue
        
        print(f"  图像形状: {img.shape}")
        print(f"  图像范围: {img.min()} - {img.max()}")
        
        # 3. 提取血管掩码（使用默认阈值20）
        mask = extract_vessel_mask(img)
        print(f"  掩码形状: {mask.shape}")
        print(f"  掩码类型: {mask.dtype}")
        print(f"  掩码中True像素数: {np.sum(mask)}")
        print(f"  掩码中False像素数: {mask.size - np.sum(mask)}")
        print(f"  掩码密度: {np.sum(mask) / mask.size:.6f}")
        
        # 4. 检查不同阈值的效果
        thresholds = [10, 20, 30, 40, 50]
        print("  不同阈值的效果:")
        for t in thresholds:
            mask_t = extract_vessel_mask(img, threshold=t)
            true_pixels = np.sum(mask_t)
            density = true_pixels / mask_t.size
            print(f"    阈值={t}: True像素数={true_pixels}, 密度={density:.6f}")
        
        # 5. 检查顶帽变换后的图像
        from common.common_util import pre_processing
        
        # 预处理
        preprocessed = pre_processing(img)
        preprocessed_uint8 = (preprocessed * 255).astype(np.uint8)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(preprocessed_uint8, (5, 5), 0)
        
        # 顶帽变换
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
        
        print(f"  顶帽变换后图像范围: {tophat.min()} - {tophat.max()}")
        print(f"  顶帽变换后图像平均值: {tophat.mean():.2f}")
        print(f"  顶帽变换后图像标准差: {tophat.std():.2f}")
        
        # 6. 检查DSC计算
        # 用同一图像的两个掩码计算DSC
        mask1 = extract_vessel_mask(img)
        mask2 = extract_vessel_mask(img)
        
        # 计算DSC
        def dsc(mask1, mask2):
            mask1 = mask1.astype(bool)
            mask2 = mask2.astype(bool)
            intersection = np.sum(mask1 & mask2)
            union = np.sum(mask1) + np.sum(mask2)
            if union == 0:
                return 1.0
            return 2.0 * intersection / union
        
        dsc_value = dsc(mask1, mask2)
        print(f"  同一图像的两个掩码DSC: {dsc_value:.6f}")
        
        # 7. 检查不同图像的DSC
        if len(test_files) > 1:
            next_file = test_files[1] if test_file != test_files[1] else test_files[0]
            next_path = os.path.join(test_dir, next_file)
            next_img = cv2.imread(next_path, cv2.IMREAD_GRAYSCALE)
            if next_img is not None:
                mask_next = extract_vessel_mask(next_img)
                dsc_diff = dsc(mask, mask_next)
                print(f"  与{next_file}的DSC: {dsc_diff:.6f}")

if __name__ == "__main__":
    check_vessel_mask()