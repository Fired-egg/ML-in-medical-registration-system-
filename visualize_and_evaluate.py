import cv2
import numpy as np
import os
from vessel_mask import extract_vessel_mask
from common.common_util import pre_processing

def ncc(img1, img2):
    """计算归一化互相关"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    
    numerator = np.sum((img1 - mean1) * (img2 - mean2))
    denominator = np.sqrt(np.sum((img1 - mean1)**2) * np.sum((img2 - mean2)**2))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def dsc(mask1, mask2):
    """计算Dice相似系数"""
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = mask1.sum() + mask2.sum()
    
    if union == 0:
        return 0.0
    
    return 2.0 * intersection / union

def process_image(image_path, output_dir, reference_mask=None, threshold=None):
    """
    处理单个图像，生成血管掩码并计算评价指标
    
    参数:
        image_path: 输入图像路径
        output_dir: 输出目录
        reference_mask: 参考掩码（用于计算DSC）
        threshold: 手动阈值（None 则使用自动阈值）
    
    返回:
        dict: 包含评价指标的字典
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    # 预处理图像（与 predictor 相同的步骤）
    preprocessed = pre_processing(img)
    preprocessed_uint8 = (preprocessed * 255).astype(np.uint8)
    
    # 提取血管掩码
    mask = extract_vessel_mask(img, threshold=threshold)
    
    # 转换为可视化格式
    mask_visual = (mask * 255).astype(np.uint8)
    
    # 创建彩色叠加效果（使用预处理后的图像）
    img_color = cv2.cvtColor(preprocessed_uint8, cv2.COLOR_GRAY2BGR)
    mask_color = cv2.cvtColor(mask_visual, cv2.COLOR_GRAY2BGR)
    mask_color[:, :, 0] = 0  # 蓝色通道设为0
    mask_color[:, :, 1] = 0  # 绿色通道设为0
    # 红色标记血管
    overlay = cv2.addWeighted(img_color, 0.7, mask_color, 0.3, 0)
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    # 保存原始图像（未预处理）
    cv2.imwrite(os.path.join(output_dir, f"{name}_raw{ext}"), img)
    
    # 保存预处理后的图像
    cv2.imwrite(os.path.join(output_dir, f"{name}_preprocessed{ext}"), preprocessed_uint8)
    
    # 保存血管掩码
    cv2.imwrite(os.path.join(output_dir, f"{name}_mask{ext}"), mask_visual)
    
    # 保存叠加效果
    cv2.imwrite(os.path.join(output_dir, f"{name}_overlay{ext}"), overlay)
    
    # 计算评价指标
    metrics = {
        "frame": filename,
        "mask_density": mask.sum() / mask.size
    }
    
    # 如果提供了参考掩码，计算DSC
    if reference_mask is not None:
        # 确保两个掩码大小相同
        min_h = min(mask.shape[0], reference_mask.shape[0])
        min_w = min(mask.shape[1], reference_mask.shape[1])
        mask_cropped = mask[:min_h, :min_w]
        ref_mask_cropped = reference_mask[:min_h, :min_w]
        
        metrics["dsc"] = dsc(mask_cropped, ref_mask_cropped)
    
    print(f"已处理: {filename}")
    return metrics

def visualize_and_evaluate(input_dir, output_dir, reference_image=None, threshold=None):
    """
    可视化目录中所有图像的血管掩码并计算评价指标
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        reference_image: 参考图像路径（用于计算DSC）
        threshold: 手动阈值（None 则使用自动阈值）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理参考图像（如果提供）
    reference_mask = None
    if reference_image and os.path.exists(reference_image):
        ref_img = cv2.imread(reference_image, cv2.IMREAD_GRAYSCALE)
        if ref_img is not None:
            reference_mask = extract_vessel_mask(ref_img, threshold=threshold)
            print(f"已处理参考图像: {os.path.basename(reference_image)}")
    
    # 处理目录中的所有图像
    results = []
    for file in os.listdir(input_dir):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, file)
            metrics = process_image(image_path, output_dir, reference_mask, threshold)
            if metrics:
                results.append(metrics)
    
    # 保存评价结果
    if results:
        csv_path = os.path.join(output_dir, "vessel_mask_evaluation.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            # 写入表头
            headers = list(results[0].keys())
            f.write(",".join(headers) + "\n")
            
            # 写入数据
            for result in results:
                row = [str(result.get(key, "")) for key in headers]
                f.write(",".join(row) + "\n")
        
        print(f"\n评价结果已保存到: {csv_path}")
        
        # 计算统计信息
        if "dsc" in results[0]:
            dsc_values = [r.get("dsc", 0) for r in results]
            mean_dsc = np.mean(dsc_values)
            min_dsc = np.min(dsc_values)
            max_dsc = np.max(dsc_values)
            
            stats_path = os.path.join(output_dir, "vessel_mask_stats.txt")
            with open(stats_path, "w", encoding="utf-8") as f:
                f.write("血管掩码评价统计信息\n")
                f.write("-" * 50 + "\n")
                f.write(f"平均DSC: {mean_dsc:.6f}\n")
                f.write(f"最小DSC: {min_dsc:.6f}\n")
                f.write(f"最大DSC: {max_dsc:.6f}\n")
                f.write(f"处理图像数量: {len(results)}\n")
            
            print(f"统计信息已保存到: {stats_path}")

if __name__ == "__main__":
    # 示例：处理 filtered_predictor_preprocessed 目录中的图像
    input_dir = "results/filtered_predictor_preprocessed"
    output_dir = "results/vessel_mask_visualization"
    
    # 可选：指定参考图像（用于计算DSC）
    # reference_image = "results/filtered_predictor_preprocessed/100.png"  # 例如使用基准帧作为参考
    reference_image = None
    
    # 可以在这里设置手动阈值，例如 threshold=50
    threshold = None  # 使用自动阈值
    
    if os.path.exists(input_dir):
        visualize_and_evaluate(input_dir, output_dir, reference_image, threshold)
        print(f"\n所有图像的血管掩码已可视化并评价到: {output_dir}")
    else:
        print(f"输入目录不存在: {input_dir}")
        print("请先运行 preprocess_filtered.py，生成 filtered_predictor_preprocessed 文件夹")