import os
import cv2
import json
import numpy as np

results_dir = r"d:\桌面文件\文件\Python机器学习NUS\project_SuperRetina_main\results"
filtered_dir = os.path.join(results_dir, "filtered")
registered_dir = os.path.join(results_dir, "registered_superretina")
match_info_path = os.path.join(results_dir, "match_info", "match_info.json")

with open(match_info_path, "r", encoding="utf-8") as f:
    all_match_info = json.load(f)

fname = "334.png"
ref_fname = "0.png"

# 读取原始 filtered 图像和参考帧
query_img = cv2.imread(os.path.join(filtered_dir, fname), cv2.IMREAD_GRAYSCALE)
ref_img = cv2.imread(os.path.join(filtered_dir, ref_fname), cv2.IMREAD_GRAYSCALE)
h, w = ref_img.shape

print(f"原始 query image shape: {query_img.shape}")
print(f"原始 refer image shape: {ref_img.shape}")

# 读取 H
info = all_match_info[fname]
H = np.array(info['H'])
print(f"\nH matrix:\n{H}")

# warp it
query_align = cv2.warpPerspective(query_img, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0))

print(f"\nwarped shape: {query_align.shape}")
print(f"warped min: {query_align.min()}, max: {query_align.max()}")
print(f"zero pixel count: {(query_align == 0).sum()}")

# 保存对比图像
out_path = os.path.join(results_dir, "test_warp_334.png")
vis = np.hstack((query_img, query_align))
cv2.imwrite(out_path, vis)
print(f"\n对比图已保存到: {out_path}")

print("\n对比一下 registered 文件夹里的图像:")
registered_img = cv2.imread(os.path.join(registered_dir, fname), cv2.IMREAD_GRAYSCALE)
print(f"registered image shape: {registered_img.shape}")
print(f"registered min: {registered_img.min()}, max: {registered_img.max()}")
