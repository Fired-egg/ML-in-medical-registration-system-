import cv2
import numpy as np
import os
import json


def create_chessboard(img1: np.ndarray, img2: np.ndarray, block_size: int = 128) -> np.ndarray:
    """
    生成两张图像的棋盘格拼接图
    """
    h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]

    chessboard = np.zeros((h, w), dtype=np.uint8)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            i_end = min(i + block_size, h)
            j_end = min(j + block_size, w)

            if (i // block_size + j // block_size) % 2 == 0:
                chessboard[i:i_end, j:j_end] = img1[i:i_end, j:j_end]
            else:
                chessboard[i:i_end, j:j_end] = img2[i:i_end, j:j_end]

    return chessboard


def generate_chessboard_enhanced(results_dir: str, output_dir: str = None):
    """
    使用图像增强后的基准帧和浮动帧生成配准前的棋盘格图

    参数:
      results_dir: 结果目录
      output_dir: 输出目录（默认: results/chessboard_enhanced）
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, "chessboard_enhanced")
    os.makedirs(output_dir, exist_ok=True)

    frame_info_path = os.path.join(results_dir, "frame_info.json")
    enhanced_dir = os.path.join(results_dir, "filtered_predictor_preprocessed")

    with open(frame_info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    reference_frame = info["reference_frame"]
    valid_files = info["valid_files"]

    ref_img = cv2.imread(os.path.join(enhanced_dir, reference_frame), cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        print(f"无法读取基准帧: {reference_frame}")
        return

    count = 0
    for fname in valid_files:
        float_path = os.path.join(enhanced_dir, fname)

        if not os.path.exists(float_path):
            continue

        float_img = cv2.imread(float_path, cv2.IMREAD_GRAYSCALE)
        if float_img is None:
            continue

        h = min(ref_img.shape[0], float_img.shape[0])
        w = min(ref_img.shape[1], float_img.shape[1])

        ref_c = ref_img[:h, :w]
        float_c = float_img[:h, :w]

        chessboard = create_chessboard(ref_c, float_c)

        output_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_before.png")
        cv2.imwrite(output_path, chessboard)
        count += 1

        if count % 20 == 0 or count == len(valid_files):
            print(f"已处理 {count}/{len(valid_files)} 张棋盘图")

    print(f"\n棋盘格图（使用增强后图像）已保存到: {output_dir}")


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(this_dir, "results")
    generate_chessboard_enhanced(results_dir)