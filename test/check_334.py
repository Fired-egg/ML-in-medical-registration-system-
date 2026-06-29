import os
import cv2
import numpy as np

results_dir = r"d:\桌面文件\文件\Python机器学习NUS\project_SuperRetina_main\results"
registered_dir = os.path.join(results_dir, "registered_superretina")

fname = "334.png"
img_path = os.path.join(registered_dir, fname)

print(f"检查文件: {img_path}")
print(f"文件存在: {os.path.exists(img_path)}")

if os.path.exists(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print(f"图像形状: {img.shape}")
    print(f"数据类型: {img.dtype}")
    print(f"最小值: {img.min()}")
    print(f"最大值: {img.max()}")
    print(f"平均值: {img.mean():.2f}")
    
    zero_pixels = (img == 0).sum()
    total_pixels = img.shape[0] * img.shape[1]
    print(f"零像素比例: {zero_pixels/total_pixels*100:.2f}%")
    
    # 检查中间的图片对比一下
    print("\n对比一下中间的图片 (100.png):")
    img100 = cv2.imread(os.path.join(registered_dir, "100.png"), cv2.IMREAD_GRAYSCALE)
    print(f"100.png 形状: {img100.shape}")
