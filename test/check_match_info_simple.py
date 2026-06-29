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
    print(f"===== {fname} =====")
    print(f"匹配点数: {info['num_matches']}")
    print(f"内点匹配数: {info['num_inliers']}")
    print(f"内点率: {info['inliers_rate']}")
    
    if info['H'] is not None:
        H = np.array(info['H'])
        print(f"\nH matrix:")
        print(repr(H))
        print(f"\ndet(H): {np.linalg.det(H)}")
        print(f"H[0,0]={H[0,0]}, H[1,1]={H[1,1]}")
        print(f"H[2,0]={H[2,0]}, H[2,1]={H[2,1]}")

# 查看 100.png 对比
fname2 = "100.png"
if fname2 in all_match_info:
    print(f"\n===== {fname2} =====")
    info2 = all_match_info[fname2]
    print(f"匹配点数: {info2['num_matches']}")
    print(f"内点匹配数: {info2['num_inliers']}")
    if info2['H'] is not None:
        H2 = np.array(info2['H'])
        print(f"H[0,0]={H2[0,0]}, H[1,1]={H2[1,1]}")
