import os
import cv2
import numpy as np
from vessel_mask import extract_vessel_mask
from evaluate_registration import dsc

# 简短测试脚本：测试DSC值是否不再是1
def test_dsc_values():
    """
    测试修改后的DSC值是否不再是1
    """
    # 1. 选择测试图像
    test_dir = "results/filtered"
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.png')][:5]  # 前5个文件
    
    print(f"测试目录: {test_dir}")
    print(f"测试文件: {test_files}")
    
    # 2. 读取基准帧
    reference_frame = "100.png"  # 假设100.png是基准帧
    ref_path = os.path.join(test_dir, reference_frame)
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    
    if ref_img is None:
        print(f"无法读取基准帧: {ref_path}")
        return
    
    # 3. 提取基准帧的血管掩码
    ref_mask = extract_vessel_mask(ref_img)
    print(f"基准帧 {reference_frame} 掩码密度: {np.sum(ref_mask) / ref_mask.size:.6f}")
    
    # 4. 测试每个图像的DSC值
    print("\n测试DSC值:")
    print("frame, dsc_before")
    
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        
        if test_img is None:
            print(f"{test_file}, 无法读取")
            continue
        
        # 提取测试图像的血管掩码
        test_mask = extract_vessel_mask(test_img)
        
        # 计算DSC值
        dsc_value = dsc(test_mask, ref_mask)
        
        print(f"{test_file}, {dsc_value:.6f}")
        print(f"  测试图像掩码密度: {np.sum(test_mask) / test_mask.size:.6f}")

if __name__ == "__main__":
    test_dsc_values()