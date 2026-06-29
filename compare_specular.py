import os
import sys
from register_from_base import register_from_base
from evaluate_registration import evaluate

# ===============================
# 比较去除光斑和不去除光斑的配准效果
# ===============================
def compare_specular_effect():
    results_dir = "results"
    
    # 1. 运行原始图像的配准（filtered）
    print("\n=== 运行原始图像配准 ===")
    try:
        out_dir_original = register_from_base(results_dir, source_folder="filtered")
        print(f"原始图像配准完成，结果保存至: {out_dir_original}")
    except Exception as e:
        print(f"原始图像配准失败: {e}")
        return
    
    # 2. 运行去光斑图像的配准（filtered_no_specular）
    print("\n=== 运行去光斑图像配准 ===")
    try:
        out_dir_no_specular = register_from_base(results_dir, source_folder="filtered_no_specular")
        print(f"去光斑图像配准完成，结果保存至: {out_dir_no_specular}")
    except Exception as e:
        print(f"去光斑图像配准失败: {e}")
        return
    
    # 3. 评价原始图像的配准效果
    print("\n=== 评价原始图像配准效果 ===")
    try:
        eval_result_original = evaluate(
            results_dir=results_dir,
            filtered_subdir="filtered",
            registered_subdir="registered_filtered",
            match_info_subdir="match_info_filtered"
        )
        print("原始图像配准评价完成")
    except Exception as e:
        print(f"原始图像评价失败: {e}")
    
    # 4. 评价去光斑图像的配准效果
    print("\n=== 评价去光斑图像配准效果 ===")
    try:
        eval_result_no_specular = evaluate(
            results_dir=results_dir,
            filtered_subdir="filtered_no_specular",
            registered_subdir="registered_filtered_no_specular",
            match_info_subdir="match_info_filtered_no_specular"
        )
        print("去光斑图像配准评价完成")
    except Exception as e:
        print(f"去光斑图像评价失败: {e}")
    
    print("\n=== 配准效果比较完成 ===")
    print("请查看 results 文件夹中的评价结果和统计文件")

if __name__ == "__main__":
    compare_specular_effect()
