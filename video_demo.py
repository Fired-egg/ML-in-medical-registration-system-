import os
import json

import cv2
import numpy as np


def _to_bgr(gray_or_bgr: np.ndarray) -> np.ndarray:
    if gray_or_bgr is None:
        return None
    if len(gray_or_bgr.shape) == 2:
        return cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2BGR)
    return gray_or_bgr


def _detect_registration_failure(pre: np.ndarray, post: np.ndarray, threshold: float = 0.85) -> bool:
    """
    检测配准是否失败（通过比较配准前后图像的相关性）
    
    参数:
        pre: 配准前图像
        post: 配准后图像
        threshold: 相关性阈值（低于此值认为配准失败，范围[-1, 1]）
    
    返回:
        True 如果检测到配准失败，False 否则
    """
    # 计算配准前后图像的相关性
    # 首先计算归一化互相关
    pre_norm = (pre - np.mean(pre)) / np.std(pre)
    post_norm = (post - np.mean(post)) / np.std(post)
    
    # 计算相关性
    corr = np.mean(pre_norm * post_norm)
    
    # 如果相关性太低，说明配准失败
    if corr < threshold:
        return True
    
    return False


def make_before_after_video(results_dir: str, out_path: str, fps: int = 10, max_frames: int | None = None, on_log=None, on_progress=None):
    """
    将配准前（filtered）与配准后（registered_superretina）做成对比视频（左右拼接）。

    - 左侧: before
    - 右侧: after
    """
    def _log(msg: str):
        if on_log is not None:
            on_log(msg)
        else:
            print(msg)

    frame_info_path = os.path.join(results_dir, "frame_info.json")
    filtered_dir = os.path.join(results_dir, "filtered")
    registered_dir = os.path.join(results_dir, "registered_filtered")

    if not os.path.exists(frame_info_path):
        raise FileNotFoundError(f"找不到 frame_info.json: {frame_info_path}")
    if not os.path.isdir(filtered_dir):
        raise FileNotFoundError(f"找不到配准前图像目录: {filtered_dir}")
    if not os.path.isdir(registered_dir):
        raise FileNotFoundError(f"找不到配准后图像目录: {registered_dir}")

    with open(frame_info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    valid_files = info["valid_files"]

    files = valid_files[:]
    if max_frames is not None:
        files = files[:max_frames]

    if not files:
        raise RuntimeError("没有可用于生成视频的帧。")

    # 找第一帧确定视频尺寸
    first_pre = cv2.imread(os.path.join(filtered_dir, files[0]), cv2.IMREAD_GRAYSCALE)
    first_post = cv2.imread(os.path.join(registered_dir, files[0]), cv2.IMREAD_GRAYSCALE)
    if first_pre is None or first_post is None:
        raise RuntimeError("无法读取第一帧（请确认 filtered/ 与 registered_superretina/ 都存在对应图像）。")

    h = min(first_pre.shape[0], first_post.shape[0])
    w = min(first_pre.shape[1], first_post.shape[1])
    panel_h, panel_w = h, w * 2

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (panel_w, panel_h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {out_path}")

    total = len(files)
    _log(f"开始生成对比视频: {out_path} (fps={fps}, frames={total})")

    skipped_frames = []
    
    for i, fname in enumerate(files):
        # 读取图像
        pre = cv2.imread(os.path.join(filtered_dir, fname), cv2.IMREAD_GRAYSCALE)
        post = cv2.imread(os.path.join(registered_dir, fname), cv2.IMREAD_GRAYSCALE)
        if pre is None or post is None:
            _log(f"[跳过] 读取失败: {fname}")
            skipped_frames.append(fname)
            continue

        # 裁剪到一致的尺寸
        pre = pre[:h, :w].copy()
        post = post[:h, :w].copy()
        
        # 检测配准是否失败（差异过大）
        if _detect_registration_failure(pre, post):
            _log(f"[警告] 检测到 {fname} 配准失败（差异过大），使用配准前图像替代")
            post = pre.copy()
        
        # 转换为BGR
        left = _to_bgr(pre)
        right = _to_bgr(post)
        
        # 创建新的帧（确保每一帧都是全新的内存）
        frame = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        
        # 填充左侧和右侧
        frame[:, :w] = left
        frame[:, w:] = right

        # 轻量标注
        cv2.putText(frame, "Before", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, "After", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, fname, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 写入视频
        writer.write(frame)

        if on_progress is not None:
            on_progress(i + 1, total)

    writer.release()
    _log("视频生成完成。")
    return out_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成配准前后对比视频")
    parser.add_argument("--results", "-r", type=str, default="results",
                        help="结果目录路径")
    parser.add_argument("--output", "-o", type=str, default="results/registration_demo.mp4",
                        help="输出视频路径")
    parser.add_argument("--fps", "-f", type=int, default=10,
                        help="视频帧率")
    parser.add_argument("--max-frames", "-m", type=int, default=None,
                        help="最大帧数（None表示全部）")
    
    args = parser.parse_args()
    
    print(f"结果目录: {args.results}")
    print(f"输出视频: {args.output}")
    print(f"帧率: {args.fps}")
    print(f"最大帧数: {args.max_frames if args.max_frames else '全部'}")
    
    make_before_after_video(
        results_dir=args.results,
        out_path=args.output,
        fps=args.fps,
        max_frames=args.max_frames
    )

