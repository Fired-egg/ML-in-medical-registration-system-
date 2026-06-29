import os
import json
import numpy as np

results_dir = r"d:\桌面文件\文件\Python机器学习NUS\project_SuperRetina_main\results"
match_info_path = os.path.join(results_dir, "match_info", "match_info.json")

with open(match_info_path, "r", encoding="utf-8") as f:
    all_match_info = json.load(f)

f334 = all_match_info["334.png"]
H = np.array(f334["H"])
print("334.png H matrix:")
print(H)
print("\n334.png H[0][0] =", H[0][0])
print("334.png H[1][1] =", H[1][1])
print("334.png H[2][0] =", H[2][0])
print("334.png H[2][1] =", H[2][1])
print("\n334.png det =", np.linalg.det(H))

print("\n--- 100.png ---")
f100 = all_match_info["100.png"]
H100 = np.array(f100["H"])
print("H[0][0] =", H100[0][0])
print("H[1][1] =", H100[1][1])
