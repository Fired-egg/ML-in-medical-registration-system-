import os
import json
import numpy as np

results_dir = r"d:\桌面文件\文件\Python机器学习NUS\project_SuperRetina_main\results"
match_info_path = os.path.join(results_dir, "match_info", "match_info.json")

with open(match_info_path, "r", encoding="utf-8") as f:
    all_match_info = json.load(f)

# 查看最后一帧的信息
fname = "334.png"
if fname in all_match_info:
    info = all_match_info[fname]
    print(f"===== {fname} 的匹配信息 =====")
    print(f"匹配点数: {info['num_matches']}")
    print(f"内点匹配数: {info['num_inliers']}")
    print(f"内点率: {info['inliers_rate']*100:.2f}%")
    
    if info['H'] is not None:
        H = np.array(info['H'])
        print(f"\n单应矩阵 H:")
        print(H)
        
        print(f"\nH 的行列式: {np.linalg.det(H):.6f}")
        print(f"H[0,0]: {H[0,0]:.6f}, H[1,1]: {H[1,1]:.6f}")
        print(f"H[2,0]: {H[2,0]:.6f}, H[2,1]: {H[2,1]:.6f}")
else:
    print(f"{fname} 不在匹配信息中")

print("\n===== 随机挑 3 张中间图片对比 =====")
for f in ["100.png", "200.png", "300.png"]:
    if f in all_match_info:
        info = all_match_info[f]
        print(f"\n{f}:")
        print(f"  匹配点数: {info['num_matches']}")
        print(f"  内点匹配数: {info['num_inliers']}")
        if info['H'] is not None:
            H = np.array(info['H'])
            print(f"  H[0,0]: {H[0,0]:.6f}, H[1,1]: {H[1,1]:.6f}")
