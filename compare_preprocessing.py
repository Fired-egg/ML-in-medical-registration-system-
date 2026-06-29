import os
from register_from_base import register_from_base
from evaluate_registration import evaluate

def compare_preprocessing_effect():
    results_dir = "results"
    
    # 1. 运行原始图像的配准（预处理前）
    print("\n=== 运行原始图像配准（预处理前）===")
    try:
        out_dir_original = register_from_base(results_dir, source_folder="original")
        print(f"原始图像配准完成，结果保存至: {out_dir_original}")
    except Exception as e:
        print(f"原始图像配准失败: {e}")
        return
    
    # 2. 运行预处理后图像的配准
    print("\n=== 运行预处理后图像配准 ===")
    try:
        out_dir_filtered = register_from_base(results_dir, source_folder="filtered")
        print(f"预处理后图像配准完成，结果保存至: {out_dir_filtered}")
    except Exception as e:
        print(f"预处理后图像配准失败: {e}")
        return
    
    # 3. 评价原始图像的配准效果
    print("\n=== 评价原始图像配准效果 ===")
    try:
        eval_result_original = evaluate(
            results_dir=results_dir,
            filtered_subdir="original",
            registered_subdir="registered_original",
            match_info_subdir="match_info_original"
        )
        print("原始图像配准评价完成")
    except Exception as e:
        print(f"原始图像评价失败: {e}")
    
    # 4. 评价预处理后图像的配准效果
    print("\n=== 评价预处理后图像配准效果 ===")
    try:
        eval_result_filtered = evaluate(
            results_dir=results_dir,
            filtered_subdir="filtered",
            registered_subdir="registered_filtered",
            match_info_subdir="match_info_filtered"
        )
        print("预处理后图像配准评价完成")
    except Exception as e:
        print(f"预处理后图像评价失败: {e}")
    
    print("\n=== 预处理前后配准效果对比完成 ===")
    print("请查看 results 文件夹中的评价结果和统计文件")

if __name__ == "__main__":
    compare_preprocessing_effect()
