import cv2
import numpy as np
import os


def add_corner_label(image_path, label, output_path):
    """
    在图像左下角添加白色大字体标注
    
    参数:
        image_path: 输入图像路径
        label: 标注文本（如 "(1)", "(2)"）
        output_path: 输出图像路径
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    height, width = img.shape[:2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    font_thickness = 4
    font_color = (255, 255, 255)
    
    text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    text_width, text_height = text_size
    
    margin = 20
    x = margin
    y = height - margin
    
    cv2.putText(img, label, (x, y), font, font_scale, font_color, font_thickness)
    
    cv2.imwrite(output_path, img)
    print(f"已处理: {image_path} -> {output_path}")


if __name__ == "__main__":
    input_dir = "results/university_image_preprocessed"
    output_dir = "results/university_image_labeled"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 修改为4、5、6、7顺序
    image_list = [
        ("4.png", "(1)"),
        ("5.png", "(2)"),
        ("6.png", "(3)"),
        ("7.png", "(4)")
    ]
    
    for filename, label in image_list:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(input_path):
            add_corner_label(input_path, label, output_path)
        else:
            print(f"文件不存在: {input_path}")
    
    print("\n所有图片标注完成！")