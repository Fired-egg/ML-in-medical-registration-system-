import os
import sys
import yaml
import cv2

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir)

from predictor import Predictor

config_path = os.path.join(this_dir, "config", "test.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("Initializing Predictor on GPU...")
predictor = Predictor(config)
print("✓ Predictor initialized successfully!")

results_dir = os.path.join(this_dir, "results", "filtered")
import json
frame_info = os.path.join(this_dir, "results", "frame_info.json")
with open(frame_info, "r", encoding="utf-8") as f:
    info = json.load(f)
ref_frame = info["reference_frame"]

ref_path = os.path.join(results_dir, ref_frame)
print(f"Testing with reference frame: {ref_frame}")

merged = predictor.align_image_pair(ref_path, ref_path, show=False)
if merged is not None:
    print("✓ Test successful! GPU is working!")
    print(f"Output shape: {merged.shape}")
else:
    print("✗ Test failed!")
