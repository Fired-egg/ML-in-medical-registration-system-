import cv2
import numpy as np
import os
import json
import shutil
import re


# ==============================
# 新增：提取文件名中的数字序号（核心辅助函数）
# ==============================
def extract_number_from_filename(filename):
    """从文件名中提取数字（处理如 'img123.jpg'/'123.png'/'frame_456.tiff' 等格式）"""
    numbers = re.findall(r"\d+", filename)
    return int(numbers[0]) if numbers else -1


# ==============================
# 新增：散斑噪声轻量降噪（不模糊血管，仅抑制高频噪声）
# ==============================
def light_speckle_denoise(img):
    """
    轻量散斑降噪：用于质量评分前，避免噪声干扰熵/梯度计算
    特点：弱降噪，保留图像细节（不影响质量评分的准确性）
    """
    img_denoise = cv2.bilateralFilter(img, d=3, sigmaColor=10, sigmaSpace=10)
    img_denoise = cv2.GaussianBlur(img_denoise, (3, 3), 0.5)
    return img_denoise


# ==============================
# 新增：光斑去除（Specular Highlight Removal）
# ==============================
def remove_specular_highlights(img, threshold=200, kernel_size=15, inpaint_radius=3, use_morphology=False):
    """
    去除图像中的圆形亮斑（高光区域）
    优化版本：更适合圆形亮斑
    
    参数:
        img: 输入图像 (灰度或彩色)
        threshold: 高光检测阈值 (默认200, 范围0-255)
        kernel_size: 高斯模糊核大小
        inpaint_radius: 修复半径
        use_morphology: 是否使用形态学操作
    
    返回:
        去除光斑后的图像
    """
    # 确保图像是uint8类型
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    
    # 如果是彩色图像，转换为灰度进行光斑检测
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 1. 检测高光区域（光斑）
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # 2. 高斯模糊平滑边界（更适合圆形光斑）
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    
    # 3. 可选的形态学操作（使用椭圆结构元素）
    if use_morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 4. 查找轮廓并保留最大的圆形轮廓（假设主要光斑是最大的）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 找到面积最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 创建新的蒙版，只保留最大的轮廓
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    # 如果光斑区域很小，直接返回原图
    if cv2.countNonZero(mask) < 100:
        return img
    
    # 5. 使用改进的inpainting
    if len(img.shape) == 3:
        result = cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_TELEA)
    else:
        result = cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_TELEA)
    
    return result


def remove_specular_with_clahe(img, clip_limit=2.0, grid_size=8):
    """
    使用CLAHE（对比度受限的自适应直方图均衡化）减少光斑影响
    同时增强血管对比度
    
    参数:
        img: 输入灰度图像
        clip_limit: CLAHE裁剪限制
        grid_size: 网格大小
    
    返回:
        处理后的图像
    """
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    
    # 应用CLAHE
    result = clahe.apply(img)
    
    return result


Rc = 0.3
Wc = 0.7
Wp = 0.3


def compute_entropy(region):
    hist = cv2.calcHist([region], [0], None, [256], [0, 256]).ravel()
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))


def compute_edge(region):
    sobelx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(sobelx**2 + sobely**2)
    grad = grad / (np.max(grad) + 1e-6)
    return grad


def create_center_mask(shape, Rc):
    h, w = shape
    cx = w // 2
    cy = h // 2
    r = int(min(h, w) * Rc)
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
    return mask


def compute_quality(img):
    mask_center = create_center_mask(img.shape, Rc)
    mask_periphery = ~mask_center
    center = img[mask_center]
    periphery = img[mask_periphery]
    Hc = compute_entropy(center)
    Hp = compute_entropy(periphery)
    Hc_norm = Hc / 8
    Hp_norm = Hp / 8
    grad = compute_edge(img)
    Ec = np.mean(grad[mask_center])
    Ep = np.mean(grad[mask_periphery])
    grad_global = np.mean(grad)
    Ec_norm = Ec / (grad_global + 1e-6)
    Ep_norm = Ep / (grad_global + 1e-6)
    Q = Wc * (Hc_norm + Ec_norm) + Wp * (Hp_norm + Ep_norm)
    Q = np.clip(Q, 0, 2)
    return Q


def is_overexposed(img, white_threshold=240, ratio_threshold=0.05):
    white_pixels = np.sum(img > white_threshold)
    ratio = white_pixels / img.size
    return ratio > ratio_threshold


def blur_score_laplacian(img: np.ndarray) -> float:
    """
    模糊度评分：Laplacian 方差，越小越模糊。
    常用经验：清晰图通常明显高于模糊图，但阈值需按数据集调。
    """
    if img is None:
        return 0.0
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return float(lap.var())


def blur_score_tenengrad(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    Tenengrad（Sobel 梯度能量）模糊评分：越小越模糊。
    相比 Laplacian 方差，对“黑边/低对比”更稳一些。
    """
    if img is None:
        return 0.0
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    g2 = gx * gx + gy * gy

    if mask is not None:
        g2 = g2[mask]
        if g2.size == 0:
            return 0.0
    return float(np.mean(g2))


def _valid_fov_mask(img: np.ndarray) -> np.ndarray:
    """
    估计有效视野区域，避免黑边/大面积背景影响模糊评分。
    - 先做圆形中心 mask
    - 再过滤掉过暗像素（接近 0 的背景）
    """
    h, w = img.shape[:2]
    center = create_center_mask((h, w), 0.48)
    nonblack = img > 8
    return center & nonblack


def is_blurry(img: np.ndarray, threshold: float = 300.0) -> bool:
    """
    判断是否运动模糊/失焦：
      - threshold 越大越“严格”（更容易判模糊）
    """
    # 使用 Tenengrad（更稳），并仅在有效视野区域内评分
    mask = _valid_fov_mask(img)
    return blur_score_tenengrad(img, mask=mask) < threshold


# ==============================
# 核心：load_images函数（含周边帧剔除逻辑）
# ==============================
def load_images(folder, blur_threshold: float | None = 300.0, exclude_blur_neighbors: bool = False):
    file_list = sorted(os.listdir(folder))
    bad_files = []
    bad_numbers = []
    blur_numbers = []
    blur_rows = []

    for file in file_list:
        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if np.max(img) < 10:
            bad_files.append(file)
            bad_numbers.append(extract_number_from_filename(file))
            continue
        if is_overexposed(img):
            bad_files.append(file)
            bad_numbers.append(extract_number_from_filename(file))
            continue
        if blur_threshold is not None and is_blurry(img, threshold=float(blur_threshold)):
            # 记录评分，方便调参定位（例如 5/27）
            mask = _valid_fov_mask(img)
            score = blur_score_tenengrad(img, mask=mask)
            blur_rows.append((file, score))
            bad_files.append(file)
            n = extract_number_from_filename(file)
            bad_numbers.append(n)
            blur_numbers.append(n)
            continue
        elif blur_threshold is not None:
            mask = _valid_fov_mask(img)
            score = blur_score_tenengrad(img, mask=mask)
            blur_rows.append((file, score))

    # 输出模糊评分，便于你检查“为什么 5/27 没被剔除”
    try:
        import csv

        scores_csv = os.path.join("results", "blur_scores.csv")
        os.makedirs("results", exist_ok=True)
        with open(scores_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "tenengrad_score"])
            for file, score in sorted(blur_rows, key=lambda x: x[1]):
                w.writerow([file, f"{score:.6f}"])
    except Exception:
        pass

    exclude_numbers = set(bad_numbers)
    # 仅对“模糊帧”可选地剔除相邻帧；默认不剔除邻居（只剔除模糊帧本身）
    for num in blur_numbers:
        if num == -1:
            continue
        if exclude_blur_neighbors:
            exclude_numbers.add(num - 2)
            exclude_numbers.add(num - 1)
            exclude_numbers.add(num + 1)
            exclude_numbers.add(num + 2)

    valid_images = []
    valid_files = []
    for file in file_list:
        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        curr_num = extract_number_from_filename(file)
        if curr_num in exclude_numbers or file in bad_files:
            if curr_num in exclude_numbers and curr_num not in bad_numbers:
                print(
                    f"跳过过曝周边帧: {file} (关联过曝帧序号: {[n for n in bad_numbers if abs(n - curr_num) <=2]})"
                )
            else:
                # 额外区分：模糊帧
                if blur_threshold is not None and is_blurry(img, threshold=float(blur_threshold)):
                    print(f"跳过模糊帧: {file} (blur_score={blur_score_laplacian(img):.2f})")
                else:
                    print(f"跳过过曝/全黑帧: {file}")
            continue

        img_for_score = light_speckle_denoise(img)
        valid_images.append(img_for_score)
        valid_files.append(file)

    return valid_images, valid_files


def find_reference_frame(images, files):
    scores = []
    for i, img in enumerate(images):
        score = compute_quality(img)
        scores.append((i, files[i], score))
    scores.sort(key=lambda x: x[2], reverse=True)
    ref_index, ref_file, ref_score = scores[0]
    print("基准帧:", ref_file)
    print("评分:", ref_score)
    return ref_index, scores


def save_frame_info(ref_index, scores, valid_files, save_path):
    order = sorted(range(len(valid_files)), key=lambda i: extract_number_from_filename(valid_files[i]))
    sorted_files = [valid_files[i] for i in order]

    new_ref_index = None
    for new_idx, old_idx in enumerate(order):
        if old_idx == ref_index:
            new_ref_index = new_idx
            break
    if new_ref_index is None:
        new_ref_index = int(ref_index)

    score_map = {i: (i, f, s) for i, f, s in scores}

    sorted_scores = []
    for new_idx, old_idx in enumerate(order):
        if old_idx not in score_map:
            continue
        _, f, s = score_map[old_idx]
        sorted_scores.append({"index": int(new_idx), "file": f, "score": float(s)})

    data = {
        "reference_index": int(new_ref_index),
        "reference_frame": sorted_files[new_ref_index],
        "valid_files": sorted_files,
        "scores": sorted_scores,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print("已保存帧信息:", save_path)


def copy_filtered_images(src_folder, valid_files, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    for file in valid_files:
        src = os.path.join(src_folder, file)
        dst = os.path.join(dst_folder, file)
        shutil.copy(src, dst)
    print("过滤后的图像已保存到:", dst_folder)


def copy_filtered_images_with_specular_removal(src_folder, valid_files, dst_folder, 
                                                threshold=200, kernel_size=15, 
                                                use_clahe=True, clip_limit=2.0, 
                                                use_morphology=False):
    """
    复制过滤后的图像，并去除光斑后保存到指定文件夹
    
    参数:
        src_folder: 源文件夹
        valid_files: 有效文件列表
        dst_folder: 目标文件夹
        threshold: 光斑检测阈值
        kernel_size: 高斯模糊核大小
        use_clahe: 是否使用CLAHE增强对比度
        clip_limit: CLAHE裁剪限制
        use_morphology: 是否使用形态学操作
    """
    os.makedirs(dst_folder, exist_ok=True)
    
    for file in valid_files:
        src = os.path.join(src_folder, file)
        dst = os.path.join(dst_folder, file)
        
        # 读取图像
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"警告: 无法读取图像 {file}")
            continue
        
        # 去除光斑
        img_no_specular = remove_specular_highlights(
            img, 
            threshold=threshold, 
            kernel_size=kernel_size,
            use_morphology=use_morphology
        )
        
        # 可选：使用CLAHE增强对比度
        if use_clahe:
            img_no_specular = remove_specular_with_clahe(img_no_specular, clip_limit=clip_limit)
        
        # 保存处理后的图像
        cv2.imwrite(dst, img_no_specular)
    
    print("去除光斑后的图像已保存到:", dst_folder)


def get_base(folder, results_folder="results", blur_threshold: float | None = 300.0,
             remove_specular=False, specular_threshold=200, specular_kernel_size=15,
             use_clahe=True, clahe_clip_limit=2.0, use_morphology=False):
    """
    计算并保存基准帧信息（frame_info.json），并将过滤后的图像复制到 results/filtered。
    可选：去除光斑后保存到 results/filtered_no_specular。

    参数:
      - folder: 输入图像文件夹路径
      - results_folder: 输出目录（可为相对或绝对路径）
      - blur_threshold: 模糊检测阈值
      - remove_specular: 是否去除光斑并保存到单独文件夹
      - specular_threshold: 光斑检测阈值 (默认200)
      - specular_kernel_size: 高斯模糊核大小 (默认15)
      - use_clahe: 是否使用CLAHE增强对比度
      - clahe_clip_limit: CLAHE裁剪限制 (默认2.0)
      - use_morphology: 是否使用形态学操作 (默认False)

    返回:
      - json_path: frame_info.json 路径
      - filtered_folder: 过滤后图像目录路径
      - reference_frame: 基准帧文件名（若无有效图像则为 None）
      - valid_count: 有效图像数量
      - no_specular_folder: 去除光斑后的图像目录路径（如果 remove_specular=True）
    """
    filtered_folder = os.path.join(results_folder, "filtered")
    json_path = os.path.join(results_folder, "frame_info.json")
    os.makedirs(results_folder, exist_ok=True)

    images, files = load_images(folder, blur_threshold=blur_threshold, exclude_blur_neighbors=False)
    print("有效图像数量:", len(files))

    reference_frame = None
    no_specular_folder = None
    
    if files:
        ref_index, scores = find_reference_frame(images, files)
        reference_frame = files[ref_index]
        save_frame_info(ref_index, scores, files, json_path)
        copy_filtered_images(folder, files, filtered_folder)
        
        # 如果启用光斑去除，保存到单独文件夹
        if remove_specular:
            no_specular_folder = os.path.join(results_folder, "filtered_no_specular")
            copy_filtered_images_with_specular_removal(
                folder, files, no_specular_folder,
                threshold=specular_threshold,
                kernel_size=specular_kernel_size,
                use_clahe=use_clahe,
                clip_limit=clahe_clip_limit,
                use_morphology=use_morphology
            )
    else:
        print("无有效图像！")

    if remove_specular:
        return json_path, filtered_folder, reference_frame, len(files), no_specular_folder
    else:
        return json_path, filtered_folder, reference_frame, len(files)

