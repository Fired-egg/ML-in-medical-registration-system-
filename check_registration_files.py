import os
import json

# 检查配准后图像的存在性
def check_registration_images():
    """
    检查配准后图像的存在性，找出为什么系统提示找不到配准图像
    """
    # 1. 检查results目录结构
    results_dir = "results"
    print(f"检查目录: {results_dir}")
    
    if not os.path.exists(results_dir):
        print(f"ERROR: Results目录不存在: {results_dir}")
        return
    
    # 2. 列出所有子目录
    print("\n子目录:")
    subdirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    for d in subdirs:
        print(f"  - {d}")
    
    # 3. 检查frame_info.json
    frame_info_path = os.path.join(results_dir, "frame_info.json")
    print(f"\n检查frame_info.json: {frame_info_path}")
    if not os.path.exists(frame_info_path):
        print("ERROR: frame_info.json不存在")
        return
    
    # 4. 读取frame_info.json
    with open(frame_info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    
    reference_frame = info.get('reference_frame', 'unknown')
    valid_files = info.get('valid_files', [])
    print(f"基准帧: {reference_frame}")
    print(f"有效文件数: {len(valid_files)}")
    print(f"前10个有效文件: {valid_files[:10]}")
    
    # 5. 检查配准后图像目录
    registered_dirs = [
        "registered_superretina",
        "registered_filtered",
        "registered_original",
        "registered_filtered_no_specular"
    ]
    
    for reg_dir in registered_dirs:
        reg_path = os.path.join(results_dir, reg_dir)
        if os.path.exists(reg_path):
            print(f"\n检查配准目录: {reg_path}")
            files = [f for f in os.listdir(reg_path) if os.path.isfile(os.path.join(reg_path, f))]
            print(f"  文件数: {len(files)}")
            print(f"  前10个文件: {files[:10]}")
        else:
            print(f"  目录不存在: {reg_path}")
    
    # 6. 检查具体文件的存在性
    print("\n检查具体文件的存在性:")
    registered_dir = os.path.join(results_dir, "registered_superretina")
    
    if os.path.exists(registered_dir):
        # 检查前5个有效文件
        for fname in valid_files[:5]:
            file_path = os.path.join(registered_dir, fname)
            if os.path.exists(file_path):
                print(f"  ✓ 存在: {file_path}")
            else:
                print(f"  ✗ 不存在: {file_path}")
                
                # 检查是否有类似的文件
                base_name = os.path.splitext(fname)[0]
                ext = os.path.splitext(fname)[1]
                print(f"  检查类似文件 (前缀: {base_name}):")
                for f in os.listdir(registered_dir):
                    if f.startswith(base_name):
                        print(f"    - {f}")
    else:
        print(f"  配准目录不存在: {registered_dir}")
    
    # 7. 检查文件扩展名
    print("\n检查文件扩展名:")
    if os.path.exists(registered_dir):
        ext_count = {}
        for f in os.listdir(registered_dir):
            if os.path.isfile(os.path.join(registered_dir, f)):
                ext = os.path.splitext(f)[1].lower()
                ext_count[ext] = ext_count.get(ext, 0) + 1
        print("  扩展名统计:")
        for ext, count in ext_count.items():
            print(f"    {ext}: {count}个文件")
    
    # 8. 检查大小写问题
    print("\n检查大小写问题:")
    if os.path.exists(registered_dir):
        # 检查是否有大小写不同的文件
        files_lower = [f.lower() for f in os.listdir(registered_dir) if os.path.isfile(os.path.join(registered_dir, f))]
        valid_files_lower = [f.lower() for f in valid_files]
        
        for f_lower, f_original in zip(valid_files_lower, valid_files):
            if f_lower in files_lower:
                # 找到实际的文件名（考虑大小写）
                for actual_f in os.listdir(registered_dir):
                    if actual_f.lower() == f_lower:
                        if actual_f != f_original:
                            print(f"  大小写不匹配: {f_original} -> {actual_f}")
                        break
            else:
                print(f"  文件不存在（小写匹配）: {f_lower}")

if __name__ == "__main__":
    check_registration_images()