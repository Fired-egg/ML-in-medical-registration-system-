import os
import json
import cv2
import yaml

from predictor import Predictor


def register_from_base(results_dir: str, on_progress=None, source_folder="filtered"):
    """
    保持原始配准流程不变，但提供一个可被 GUI 调用的函数入口。
    - 输入：results_dir（包含 frame_info.json）
    - 输入：source_folder（源图像文件夹，默认 "filtered"）
    - 输出：out_dir（registered_xxx/）
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 读取 SuperRetina 配置并初始化 Predictor
    config_path = os.path.join(this_dir, "config", "test.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    predictor = Predictor(config)

    frame_info_path = os.path.join(results_dir, "frame_info.json")
    if not os.path.exists(frame_info_path):
        raise FileNotFoundError(f"找不到基准帧信息文件: {frame_info_path}，请先运行 01_get_base.py")

    with open(frame_info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    reference_frame = info["reference_frame"]
    valid_files = info["valid_files"]

    frames_dir = os.path.join(results_dir, source_folder)
    out_dir = os.path.join(results_dir, f"registered_{source_folder}")
    match_info_dir = os.path.join(results_dir, f"match_info_{source_folder}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(match_info_dir, exist_ok=True)

    refer_path = os.path.join(frames_dir, reference_frame)
    if not os.path.exists(refer_path):
        raise FileNotFoundError(
            f"参考帧图像不存在: {refer_path}。\n"
            f"请确保 {reference_frame} 这张图和 frame_info.json 在同一个 results 目录下。"
        )

    all_match_info = {}

    print(f"Aligning reference frame {reference_frame} -> {reference_frame}")
    ref_result = predictor.align_image_pair(refer_path, refer_path, show=False)
    if ref_result[0] is None:
        print("  参考帧自配准失败，退回到灰度图保存。")
        ref_img = cv2.imread(refer_path, cv2.IMREAD_GRAYSCALE)
        if ref_img is None:
            raise RuntimeError(f"无法读取参考帧图像: {refer_path}")
        ref_bgr = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
    else:
        ref_merged, ref_match_info = ref_result
        ref_aligned_gray = ref_merged[:, :, 0]
        ref_bgr = cv2.cvtColor(ref_aligned_gray, cv2.COLOR_GRAY2BGR)
        all_match_info[reference_frame] = ref_match_info
    cv2.imwrite(os.path.join(out_dir, reference_frame), ref_bgr)

    total = max(0, len(valid_files) - 1)
    done = 0
    for fname in valid_files:
        if fname == reference_frame:
            continue

        query_path = os.path.join(frames_dir, fname)
        if not os.path.exists(query_path):
            print(f"跳过缺失帧 {fname}（未在 {frames_dir} 中找到）")
            done += 1
            if on_progress is not None:
                on_progress(done, total)
            continue

        print(f"Aligning {fname} -> {reference_frame}")
        result = predictor.align_image_pair(query_path, refer_path, show=False)

        if result[0] is None:
            print(f"  配准失败（未找到足够匹配点），使用原始帧 {fname}")
            query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
            aligned_bgr = cv2.cvtColor(query_img, cv2.COLOR_GRAY2BGR)
            match_info = {
                "num_matches": 0,
                "num_inliers": 0,
                "inliers_rate": 0.0,
                "src_pts": None,
                "dst_pts": None,
                "mask": None,
                "H": None
            }
        else:
            merged, match_info = result
            aligned_gray = merged[:, :, 0]
            aligned_bgr = cv2.cvtColor(aligned_gray, cv2.COLOR_GRAY2BGR)
        
        all_match_info[fname] = match_info
        save_path = os.path.join(out_dir, fname)
        cv2.imwrite(save_path, aligned_bgr)
        print(f"  已保存对齐结果到 {save_path}")
        done += 1
        if on_progress is not None:
            on_progress(done, total)

    match_info_path = os.path.join(match_info_dir, "match_info.json")
    with open(match_info_path, "w", encoding="utf-8") as f:
        json.dump(all_match_info, f)
    print(f"\n匹配点信息已保存到: {match_info_path}")

    return out_dir


def main():
    """
    使用 01_get_base.py 生成的基准帧信息，调用 SuperRetina 对整组帧做配准。
    - 基准帧：frame_info.json 里的 reference_frame
    - 浮动帧：valid_files 里除基准帧以外的所有帧
    """
    # 当前脚本所在目录（project_SuperRetina_main）
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 读取 SuperRetina 配置并初始化 Predictor
    config_path = os.path.join(this_dir, "config", "test.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    predictor = Predictor(config)

    # 2. 读取 01_get_base.py 的结果：frame_info.json
    #    这个文件当前在工作区根目录下的 results/frame_info.json
    results_dir = os.path.join(this_dir, "results")
    # 这里保持原始行为：使用本目录 results
    register_from_base(results_dir)


if __name__ == "__main__":
    main()

