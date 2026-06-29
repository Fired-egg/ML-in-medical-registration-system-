import os
import json
import numpy as np

results_dir = r"d:\桌面文件\文件\Python机器学习NUS\project_SuperRetina_main\results"
match_info_path = os.path.join(results_dir, "match_info", "match_info.json")
output_path = os.path.join(results_dir, "h_info.txt")

with open(match_info_path, "r", encoding="utf-8") as f:
    all_match_info = json.load(f)

with open(output_path, "w", encoding="utf-8") as f:
    # 334.png
    fname = "334.png"
    if fname in all_match_info:
        info = all_match_info[fname]
        f.write(f"===== {fname} =====\n")
        f.write(f"num_matches: {info['num_matches']}\n")
        f.write(f"num_inliers: {info['num_inliers']}\n")
        f.write(f"inliers_rate: {info['inliers_rate']}\n")
        
        if info['H'] is not None:
            H = np.array(info['H'])
            f.write(f"\nH matrix:\n")
            f.write(repr(H) + "\n")
            f.write(f"\ndet(H): {np.linalg.det(H)}\n")
            f.write(f"H[0][0]: {H[0][0]}\n")
            f.write(f"H[1][1]: {H[1][1]}\n")
            f.write(f"H[2][0]: {H[2][0]}\n")
            f.write(f"H[2][1]: {H[2][1]}\n")
    
    # 100.png
    fname2 = "100.png"
    if fname2 in all_match_info:
        f.write(f"\n===== {fname2} =====\n")
        info2 = all_match_info[fname2]
        f.write(f"num_matches: {info2['num_matches']}\n")
        f.write(f"num_inliers: {info2['num_inliers']}\n")
        if info2['H'] is not None:
            H2 = np.array(info2['H'])
            f.write(f"H[0][0]: {H2[0][0]}\n")
            f.write(f"H[1][1]: {H2[1][1]}\n")

print(f"Saved to {output_path}")
