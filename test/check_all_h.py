import os
import json
import cv2
import numpy as np

results_dir = r"d:\桌面文件\文件\Python机器学习NUS\project_SuperRetina_main\results"
match_info_path = os.path.join(results_dir, "match_info", "match_info.json")

with open(match_info_path, "r", encoding="utf-8") as f:
    all_match_info = json.load(f)

print("检查所有帧的单应矩阵缩放因子 (H[0][0], H[1][1])")
print("="*80)

bad_frames = []

for fname, info in all_match_info.items():
    if info['H'] is not None:
        H = np.array(info['H'])
        h00 = H[0][0]
        h11 = H[1][1]
        
        if h00 < 0.5 or h00 > 1.5 or h11 < 0.5 or h11 > 1.5:
            bad_frames.append((fname, h00, h11))

print(f"发现 {len(bad_frames)} 个帧的缩放因子异常：")
for fname, h00, h11 in bad_frames:
    print(f"  {fname}: H[0][0]={h00:.6f}, H[1][1]={h11:.6f}")

if len(bad_frames) > 0:
    print("\n" + "="*80)
    print("检查这些异常帧的 warped 结果：")
    for fname, _, _ in bad_frames[:3]:
        print(f"\n  {fname}:")
        reg_img = cv2.imread(os.path.join(results_dir, "registered_superretina", fname), cv2.IMREAD_GRAYSCALE)
        if reg_img is not None:
            print(f"    形状: {reg_img.shape}")
            print(f"    零像素比例: {(reg_img == 0).sum() / (reg_img.shape[0] * reg_img.shape[1]) * 100:.2f}%")
