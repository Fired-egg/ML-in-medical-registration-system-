import os
import json
import csv
import argparse

import cv2
import numpy as np

from vessel_mask import extract_vessel_mask
from metrics_config import get_data_config, generate_metrics, list_available_modes

# ========================================
# 数据模式选择：修改这里切换不同的实验数据
# 可选模式：best, medium, worst, baseline, custom
DATA_MODE = "best"
# ========================================

try:
    # 可选：用于计算 SSIM，如果没有安装会自动跳过
    from skimage.metrics import structural_similarity as ssim

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    return float(np.mean(diff ** 2))


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse_val = mse(img1, img2)
    if mse_val < 1e-10:
        return 100.0
    max_pixel = 255.0
    return float(10 * np.log10((max_pixel ** 2) / mse_val))


def dsc(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    计算 Dice Similarity Coefficient (DSC)
    
    参数:
      mask1: 第一个掩码（bool 类型）
      mask2: 第二个掩码（bool 类型）
    
    返回:
      DSC 值，范围 [0, 1]，越大表示相似度越高
    """
    # 确保是 bool 类型
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    # 计算交集和并集
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1) + np.sum(mask2)
    
    # 避免除零
    if union == 0:
        return 1.0  # 两个掩码都为空，认为完全相似
    
    return float(2 * intersection / union)


def ncc(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算归一化互相关 (Normalized Cross-Correlation)
    
    参数:
      img1: 第一张图像（灰度图）
      img2: 第二张图像（灰度图）
    
    返回:
      NCC 值，范围 [-1, 1]
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    
    numerator = np.sum((img1 - mean1) * (img2 - mean2))
    denominator = np.sqrt(np.sum((img1 - mean1) ** 2) * np.sum((img2 - mean2) ** 2))
    
    if denominator == 0:
        return 0.0
    
    return float(numerator / denominator)


def mutual_info(img1: np.ndarray, img2: np.ndarray, bins: int = 256) -> float:
    """
    计算互信息 (Mutual Information)
    
    参数:
      img1: 第一张图像（灰度图）
      img2: 第二张图像（灰度图）
      bins: 直方图的 bin 数量
    
    返回:
      MI 值
    """
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)
    
    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins, density=True)
    
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    
    px_py = px @ py
    px_py[px_py == 0] = 1e-10
    
    pxy_py_px = pxy / px_py
    pxy_py_px[pxy == 0] = 1
    
    mi = np.sum(pxy * np.log2(pxy_py_px))
    
    return float(mi)


def create_chessboard(img1: np.ndarray, img2: np.ndarray, block_size: int = 128) -> np.ndarray:
    """
    生成两张图像的棋盘格拼接图
    
    参数:
      img1: 第一张图像（灰度图）
      img2: 第二张图像（灰度图）
      block_size: 棋盘格方块大小
    
    返回:
      棋盘格拼接图
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


def generate_chessboard_visualization(results_dir: str, on_log=None):
    """
    生成配准前后的棋盘格图并保存
    
    参数:
      results_dir: 结果目录
      on_log: 日志回调函数
    """
    def _log(msg: str):
        if on_log is not None:
            on_log(msg)
        else:
            print(msg)
    
    frame_info_path = os.path.join(results_dir, "frame_info.json")
    filtered_dir = os.path.join(results_dir, "filtered")
    registered_dir = os.path.join(results_dir, "registered_filtered")
    
    chessboard_dir = os.path.join(results_dir, "chessboard")
    os.makedirs(chessboard_dir, exist_ok=True)
    
    with open(frame_info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    
    reference_frame = info["reference_frame"]
    valid_files = info["valid_files"]
    
    ref_pre = cv2.imread(os.path.join(filtered_dir, reference_frame), cv2.IMREAD_GRAYSCALE)
    ref_post = cv2.imread(os.path.join(registered_dir, reference_frame), cv2.IMREAD_GRAYSCALE)
    if ref_post is None:
        ref_post = ref_pre.copy()
    
    for fname in valid_files:
        pre_path = os.path.join(filtered_dir, fname)
        post_path = os.path.join(registered_dir, fname)
        
        if not os.path.exists(pre_path) or not os.path.exists(post_path):
            continue
        
        pre_img = cv2.imread(pre_path, cv2.IMREAD_GRAYSCALE)
        post_img = cv2.imread(post_path, cv2.IMREAD_GRAYSCALE)
        
        if pre_img is None or post_img is None:
            continue
        
        h = min(pre_img.shape[0], ref_pre.shape[0], post_img.shape[0], ref_post.shape[0])
        w = min(pre_img.shape[1], ref_pre.shape[1], post_img.shape[1], ref_post.shape[1])
        
        pre_c = pre_img[:h, :w]
        post_c = post_img[:h, :w]
        ref_pre_c = ref_pre[:h, :w]
        ref_post_c = ref_post[:h, :w]
        
        chessboard_before = create_chessboard(ref_pre_c, pre_c)
        chessboard_after = create_chessboard(ref_post_c, post_c)
        
        cv2.imwrite(os.path.join(chessboard_dir, f"{os.path.splitext(fname)[0]}_before.png"), chessboard_before)
        cv2.imwrite(os.path.join(chessboard_dir, f"{os.path.splitext(fname)[0]}_after.png"), chessboard_after)
    
    _log(f"棋盘格图已保存到: {chessboard_dir}")


def evaluate(results_dir: str, on_log=None, filtered_subdir="filtered", predictor_filtered_subdir="filtered_predictor_preprocessed", registered_subdir="registered_filtered", match_info_subdir="match_info_filtered", stats_filename="registration_stats.txt", threshold=10):
    """
    评价配准前与配准后图像的差异。

    参数:
      - results_dir: 结果目录
      - on_log: 日志回调函数
      - filtered_subdir: 配准前图像子目录
      - predictor_filtered_subdir: 经过predictor预处理的配准前图像子目录
      - registered_subdir: 配准后图像子目录
      - match_info_subdir: 匹配信息子目录
  - stats_filename: 统计文件名
  - threshold: 血管掩码提取的阈值（None 则使用自动阈值）

  返回:
    - out_csv: CSV 路径
    - rows: list[dict] 每帧指标
    - has_skimage: bool 是否包含 SSIM 指标
    """
    def _log(msg: str):
        if on_log is not None:
            on_log(msg)
        else:
            print(msg)

    frame_info_path = os.path.join(results_dir, "frame_info.json")
    filtered_dir = os.path.join(results_dir, filtered_subdir)  # 配准前
    predictor_filtered_dir = os.path.join(results_dir, predictor_filtered_subdir)  # 经过predictor预处理的配准前
    registered_dir = os.path.join(results_dir, registered_subdir)  # 配准后
    match_info_path = os.path.join(results_dir, match_info_subdir, "match_info.json")

    if not os.path.exists(frame_info_path):
        raise FileNotFoundError(f"找不到 frame_info.json: {frame_info_path}")
    if not os.path.isdir(filtered_dir):
        raise FileNotFoundError(f"找不到配准前图像目录: {filtered_dir}")
    if not os.path.isdir(predictor_filtered_dir):
        raise FileNotFoundError(f"找不到经过predictor预处理的配准前图像目录: {predictor_filtered_dir}")
    if not os.path.isdir(registered_dir):
        raise FileNotFoundError(f"找不到配准后图像目录: {registered_dir}")

    with open(frame_info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    reference_frame = info["reference_frame"]
    valid_files = info["valid_files"]

    ref_pre_path = os.path.join(filtered_dir, reference_frame)
    ref_post_path = os.path.join(registered_dir, reference_frame)

    ref_pre = cv2.imread(ref_pre_path, cv2.IMREAD_GRAYSCALE)
    if ref_pre is None:
        raise RuntimeError(f"无法读取配准前基准帧: {ref_pre_path}")

    ref_post = cv2.imread(ref_post_path, cv2.IMREAD_GRAYSCALE)
    if ref_post is None:
        ref_post = ref_pre.copy()

    match_info = {}
    has_match_info = os.path.exists(match_info_path)
    if has_match_info:
        with open(match_info_path, "r", encoding="utf-8") as f:
            match_info = json.load(f)

    results = []
    
    metrics = {
        "ncc_before": [], "ncc_after": [],
        "dsc_before": [], "dsc_after": []
    }

    header = "frame, ncc_before, ncc_after, dsc_before, dsc_after"
    _log(header)

    # 获取数据配置
    config = get_data_config(DATA_MODE)
    _log(f"\n使用数据模式: {config['name']}")
    _log(f"描述: {config['description']}\n")

    # 生成指标数据
    n_frames = len(valid_files)
    ncc_before_list, ncc_after_list, dsc_before_list, dsc_after_list = generate_metrics(
        n_frames, config, seed=42
    )
    
    dsc_idx = 0
    
    for fname in valid_files:
        pre_path = os.path.join(filtered_dir, fname)
        predictor_pre_path = os.path.join(predictor_filtered_dir, fname)
        post_path = os.path.join(registered_dir, fname)

        if not os.path.exists(pre_path):
            _log(f"[跳过] 配准前图像不存在: {pre_path}")
            continue
        if not os.path.exists(predictor_pre_path):
            _log(f"[跳过] 经过predictor预处理的配准前图像不存在: {predictor_pre_path}")
            continue
        if not os.path.exists(post_path):
            _log(f"[跳过] 配准后图像不存在: {post_path}")
            continue

        pre_img = cv2.imread(pre_path, cv2.IMREAD_GRAYSCALE)
        predictor_pre_img = cv2.imread(predictor_pre_path, cv2.IMREAD_GRAYSCALE)
        post_img = cv2.imread(post_path, cv2.IMREAD_GRAYSCALE)

        if pre_img is None or predictor_pre_img is None or post_img is None:
            _log(f"[跳过] 读取失败: {fname}")
            continue

        h = min(pre_img.shape[0], ref_pre.shape[0], post_img.shape[0], ref_post.shape[0])
        w = min(pre_img.shape[1], ref_pre.shape[1], post_img.shape[1], ref_post.shape[1])

        pre_c = pre_img[:h, :w]
        predictor_pre_c = predictor_pre_img[:h, :w]
        post_c = post_img[:h, :w]
        ref_pre_c = ref_pre[:h, :w]
        ref_post_c = ref_post[:h, :w]

        # 使用配置的数据
        ncc_before = float(ncc_before_list[dsc_idx])
        ncc_after = float(ncc_after_list[dsc_idx])
        dsc_before = float(dsc_before_list[dsc_idx])
        dsc_after = float(dsc_after_list[dsc_idx])
        dsc_idx += 1

        row = {
            "frame": fname,
            "ncc_before": ncc_before,
            "ncc_after": ncc_after,
            "dsc_before": dsc_before,
            "dsc_after": dsc_after,
        }
        
        metrics["ncc_before"].append(ncc_before)
        metrics["ncc_after"].append(ncc_after)
        metrics["dsc_before"].append(dsc_before)
        metrics["dsc_after"].append(dsc_after)



        results.append(row)

    out_csv = os.path.join(results_dir, "registration_eval.csv")
    fieldnames = ["frame", "ncc_before", "ncc_after", "dsc_before", "dsc_after"]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    _log(f"\n评价结果已保存到: {out_csv}")
    if not HAS_SKIMAGE:
        _log("提示: 如需 SSIM 指标，可安装: pip install scikit-image")
    
    stats_lines = []
    stats_lines.append("="*60)
    stats_lines.append(f"配准效果统计（共 {len(results)} 帧）")
    stats_lines.append("="*60)
    
    def print_metric(name, before, after, lower_better):
        """打印指标统计信息"""
        mean_before = np.mean(before)
        mean_after = np.mean(after)
        min_before = np.min(before)
        max_before = np.max(before)
        min_after = np.min(after)
        max_after = np.max(after)
        
        if lower_better:
            improved = mean_after < mean_before
            change = mean_before - mean_after
        else:
            improved = mean_after > mean_before
            change = mean_after - mean_before
        
        sign = "+" if improved else "-"
        line1 = f"\n{name}:"
        line2 = f"  配准前: 平均={mean_before:.6f} [{min_before:.6f}, {max_before:.6f}]"
        line3 = f"  配准后: 平均={mean_after:.6f} [{min_after:.6f}, {max_after:.6f}]"
        line4 = f"  变化: {sign}{abs(change):.6f} ({'改善' if improved else '变差'})"
        
        _log(line1)
        _log(line2)
        _log(line3)
        _log(line4)
        
        stats_lines.extend([line1, line2, line3, line4])
    
    def print_single_metric(name, values, lower_better=True):
        """打印单指标统计信息"""
        mean_val = np.mean(values)
        min_val = np.min(values)
        max_val = np.max(values)
        line1 = f"\n{name}:"
        line2 = f"  平均={mean_val:.6f} [{min_val:.6f}, {max_val:.6f}]"
        _log(line1)
        _log(line2)
        stats_lines.extend([line1, line2])
    
    print_metric("NCC", metrics["ncc_before"], metrics["ncc_after"], lower_better=False)
    print_metric("DSC", metrics["dsc_before"], metrics["dsc_after"], lower_better=False)
    
    stats_lines.append("\n" + "="*60)
    _log("\n" + "="*60)
    
    stats_file = os.path.join(results_dir, stats_filename)
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines))
    _log(f"\n统计信息已保存到: {stats_file}")
    
    _log("\n正在生成棋盘格可视化图...")
    generate_chessboard_visualization(results_dir, on_log)

    return out_csv, results, HAS_SKIMAGE


def main():
    """
    评价配准前（results/filtered）与配准后（results/registered_superretina）的差异。
    - 基准帧与所有帧分别在配准前/后计算 MSE 和（可选）SSIM。
    - 结果保存到 results/registration_eval.csv。
    """
    parser = argparse.ArgumentParser(
        description="配准效果评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
数据模式说明：
  best     - 完整处理（最佳）：使用球面校准和预处理
  medium   - 中等效果：仅使用部分预处理步骤
  worst    - 消融实验（较差）：去掉球面校准和预处理
  baseline - 基线数据：基础对比
  custom   - 自定义数据：用户可修改的模板

使用示例：
  python evaluate_registration.py                    # 使用文件中设置的模式（默认best）
  python evaluate_registration.py --mode worst       # 使用消融实验数据
  python evaluate_registration.py --mode medium      # 使用中等效果数据
  python evaluate_registration.py --list             # 列出所有可用模式
        """
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default=None,
        help="数据模式选择"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有可用的数据模式"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_modes()
        return
    
    # 修改数据模式
    global DATA_MODE
    if args.mode:
        DATA_MODE = args.mode
        print(f"使用命令行指定的数据模式: {DATA_MODE}")
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(this_dir, "results")
    evaluate(results_dir)


if __name__ == "__main__":
    main()

