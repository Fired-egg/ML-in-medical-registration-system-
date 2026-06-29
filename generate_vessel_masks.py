import os
import sys

# 运行预处理脚本
print("正在运行预处理脚本，生成filtered_predictor_preprocessed文件夹...")
preprocess_script = "preprocess_filtered.py"
if os.path.exists(preprocess_script):
    os.system(f"python {preprocess_script}")
else:
    print(f"错误：找不到 {preprocess_script} 文件")
    sys.exit(1)

# 运行评价脚本
print("\n正在运行评价脚本，生成血管掩码并计算评价指标...")
evaluate_script = "evaluate_registration.py"
if os.path.exists(evaluate_script):
    os.system(f"python {evaluate_script}")
else:
    print(f"错误：找不到 {evaluate_script} 文件")
    sys.exit(1)

print("\n所有操作已完成！")
print("生成的文件：")
print("- results/filtered_predictor_preprocessed/ - 经过predictor预处理的图像")
print("- results/vessel_mask_visualization/ - 血管掩码图像")
print("- results/registration_eval.csv - 评价结果")
print("- results/registration_stats.txt - 统计信息")