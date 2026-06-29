import sys
import os

# 添加父目录到路径，支持直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pre import get_base

# ===============================
# 主程序
# ===============================
if __name__ == "__main__":
    folder = r"D:\桌面文件\文件\Python机器学习NUS\university_image\4"
    results_folder = "results"
    
    # 运行预处理（启用光斑去除）
    # 参数说明：
    #   remove_specular=True: 启用光斑去除
    #   specular_threshold=200: 光斑检测阈值（灰度值>200视为光斑）
    #   specular_kernel_size=15: 形态学操作核大小
    #   use_clahe=True: 使用CLAHE增强对比度
    #   clahe_clip_limit=2.0: CLAHE裁剪限制
    result = get_base(
        folder, 
        results_folder=results_folder,
        remove_specular=True,           # 启用光斑去除
        specular_threshold=200,         # 光斑检测阈值
        specular_kernel_size=15,        # 形态学核大小
        use_clahe=True,                 # 使用CLAHE增强
        clahe_clip_limit=2.0            # CLAHE裁剪限制
    )
    
    # 如果启用了光斑去除，会返回5个值
    if len(result) == 5:
        json_path, filtered_folder, reference_frame, valid_count, no_specular_folder = result
        print(f"\n基准帧: {reference_frame}")
        print(f"有效图像数: {valid_count}")
        print(f"原始图像保存至: {filtered_folder}")
        print(f"去光斑图像保存至: {no_specular_folder}")
    else:
        json_path, filtered_folder, reference_frame, valid_count = result
        print(f"\n基准帧: {reference_frame}")
        print(f"有效图像数: {valid_count}")
        print(f"图像保存至: {filtered_folder}")