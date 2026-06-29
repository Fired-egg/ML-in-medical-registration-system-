import os
import sys
import shutil

# 导入必要的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pre.get_base import get_base
from register_from_base import register_from_base
from evaluate_registration import evaluate

def copy_original_images(src_dir, dst_dir):
    """将原始图像复制到目标文件夹"""
    os.makedirs(dst_dir, exist_ok=True)
    for file in os.listdir(src_dir):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(src_dir, file)
            dst = os.path.join(dst_dir, file)
            if not os.path.exists(dst):
                shutil.copy(src, dst)
    print(f"原始图像已复制到: {dst_dir}")

def run_preprocessing_comparison():
    """运行预处理前后的配准效果对比"""
    # 配置参数
    source_dir = "D:\\桌面文件\\文件\\Python机器学习NUS\\university_image\\4"
    results_dir = "results"
    original_dir = os.path.join(results_dir, "original")
    
    print("=" * 60)
    print("开始预处理前后配准效果对比")
    print("=" * 60)
    
    # 步骤 1: 复制原始图像
    print("\n1. 复制原始图像...")
    copy_original_images(source_dir, original_dir)
    
    # 步骤 2: 运行预处理
    print("\n2. 运行预处理...")
    try:
        json_path, filtered_dir, ref, n = get_base(
            source_dir, 
            results_folder=results_dir,
            remove_specular=False  # 禁用光斑去除，使用基本预处理
        )
        print(f"预处理完成，基准帧: {ref}")
    except Exception as e:
        print(f"预处理失败: {e}")
        return
    
    # 步骤 3: 运行原始图像配准
    print("\n3. 运行原始图像配准...")
    try:
        out_dir_original = register_from_base(results_dir, source_folder="original")
        print(f"原始图像配准完成，结果保存至: {out_dir_original}")
    except Exception as e:
        print(f"原始图像配准失败: {e}")
        return
    
    # 步骤 4: 运行预处理后图像配准
    print("\n4. 运行预处理后图像配准...")
    try:
        out_dir_filtered = register_from_base(results_dir, source_folder="filtered")
        print(f"预处理后图像配准完成，结果保存至: {out_dir_filtered}")
    except Exception as e:
        print(f"预处理后图像配准失败: {e}")
        return
    
    # 步骤 5: 评价原始图像配准效果
    print("\n5. 评价原始图像配准效果...")
    try:
        eval_result_original = evaluate(
            results_dir=results_dir,
            filtered_subdir="original",
            registered_subdir="registered_original",
            match_info_subdir="match_info_original",
            stats_filename="registration_stats_original.txt"  # 原始图像统计文件
        )
        print("原始图像配准评价完成")
    except Exception as e:
        print(f"原始图像评价失败: {e}")
    
    # 步骤 6: 评价预处理后图像配准效果
    print("\n6. 评价预处理后图像配准效果...")
    try:
        eval_result_filtered = evaluate(
            results_dir=results_dir,
            filtered_subdir="filtered",
            registered_subdir="registered_filtered",
            match_info_subdir="match_info_filtered",
            stats_filename="registration_stats_filtered.txt"  # 预处理后图像统计文件
        )
        print("预处理后图像配准评价完成")
    except Exception as e:
        print(f"预处理后图像评价失败: {e}")
    
    print("\n" + "=" * 60)
    print("预处理前后配准效果对比完成！")
    print("=" * 60)
    print("请查看 results 文件夹中的评价结果和统计文件")
    print("\n对比内容:")
    print("- 原始图像配准结果: results/registered_original/")
    print("- 预处理后图像配准结果: results/registered_filtered/")
    print("- 原始图像统计摘要: results/registration_stats_original.txt")
    print("- 预处理后图像统计摘要: results/registration_stats_filtered.txt")
    print("- 棋盘图: results/chessboard/")

if __name__ == "__main__":
    run_preprocessing_comparison()
