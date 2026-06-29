import os
import shutil

def copy_original_images(src_dir, dst_dir):
    """将原始图像复制到目标文件夹"""
    os.makedirs(dst_dir, exist_ok=True)
    for file in os.listdir(src_dir):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(src_dir, file)
            dst = os.path.join(dst_dir, file)
            if not os.path.exists(dst):
                shutil.copy(src, dst)
    print(f"原始图像已复制到: {dst_dir}")

# 使用示例
if __name__ == "__main__":
    source_dir = "D:\\桌面文件\\文件\\Python机器学习NUS\\university_image\\4"
    output_dir = "D:\\桌面文件\\文件\\Python机器学习NUS\\project_SuperRetina_main\\results\\original"
    copy_original_images(source_dir, output_dir)
