import os
import json
import cv2
import numpy as np
from evaluate_registration import create_chessboard

results_dir = "results"
frame_info_path = os.path.join(results_dir, "frame_info.json")
filtered_dir = os.path.join(results_dir, "filtered")

chessboard_dir = os.path.join(results_dir, "chessboard_test")
os.makedirs(chessboard_dir, exist_ok=True)

with open(frame_info_path, "r", encoding="utf-8") as f:
    info = json.load(f)

reference_frame = info["reference_frame"]
valid_files = info["valid_files"][:5]

ref_img = cv2.imread(os.path.join(filtered_dir, reference_frame), cv2.IMREAD_GRAYSCALE)

print(f"基准帧: {reference_frame}")
print(f"测试 {len(valid_files)} 张图像...")

for fname in valid_files:
    img_path = os.path.join(filtered_dir, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        continue
    
    h, w = min(img.shape[0], ref_img.shape[0]), min(img.shape[1], ref_img.shape[1])
    img_c = img[:h, :w]
    ref_c = ref_img[:h, :w]
    
    chessboard = create_chessboard(ref_c, img_c)
    save_path = os.path.join(chessboard_dir, f"{os.path.splitext(fname)[0]}_test.png")
    cv2.imwrite(save_path, chessboard)
    print(f"已保存: {save_path}")

print(f"\n测试完成！棋盘格图保存在: {chessboard_dir}")
