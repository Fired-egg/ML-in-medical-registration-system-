import os
import json
import csv

# 测试函数：比较两种方式的评价结果
def compare_evaluation_results():
    """
    比较直接运行脚本和通过app运行的评价结果
    """
    # 1. 直接运行evaluate_registration.py
    print("1. 直接运行evaluate_registration.py...")
    os.system("python evaluate_registration.py")
    
    # 2. 读取直接运行的结果
    direct_csv = "results/registration_eval.csv"
    direct_results = {}
    if os.path.exists(direct_csv):
        with open(direct_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = row['frame']
                direct_results[frame] = {
                    'ncc_before': float(row['ncc_before']),
                    'ncc_after': float(row['ncc_after']),
                    'dsc_before': float(row['dsc_before']),
                    'dsc_after': float(row['dsc_after'])
                }
        print(f"直接运行脚本的结果: {len(direct_results)} 帧")
    else:
        print("直接运行脚本未生成结果文件")
    
    # 3. 模拟app运行（通过gui_worker.py）
    print("\n2. 模拟app运行评价...")
    try:
        from evaluate_registration import evaluate
        this_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(this_dir, "results")
        out_csv, rows, _ = evaluate(results_dir)
        
        app_results = {}
        for row in rows:
            frame = row['frame']
            app_results[frame] = {
                'ncc_before': row['ncc_before'],
                'ncc_after': row['ncc_after'],
                'dsc_before': row['dsc_before'],
                'dsc_after': row['dsc_after']
            }
        print(f"模拟app运行的结果: {len(app_results)} 帧")
    except Exception as e:
        print(f"模拟app运行失败: {e}")
        app_results = {}
    
    # 4. 比较结果
    print("\n3. 比较两种方式的结果...")
    if direct_results and app_results:
        common_frames = set(direct_results.keys()) & set(app_results.keys())
        print(f"共同的帧: {len(common_frames)}")
        
        differences = []
        for frame in common_frames:
            direct = direct_results[frame]
            app = app_results[frame]
            
            # 计算差异
            ncc_before_diff = abs(direct['ncc_before'] - app['ncc_before'])
            ncc_after_diff = abs(direct['ncc_after'] - app['ncc_after'])
            dsc_before_diff = abs(direct['dsc_before'] - app['dsc_before'])
            dsc_after_diff = abs(direct['dsc_after'] - app['dsc_after'])
            
            if any(diff > 1e-6 for diff in [ncc_before_diff, ncc_after_diff, dsc_before_diff, dsc_after_diff]):
                differences.append({
                    'frame': frame,
                    'ncc_before_diff': ncc_before_diff,
                    'ncc_after_diff': ncc_after_diff,
                    'dsc_before_diff': dsc_before_diff,
                    'dsc_after_diff': dsc_after_diff
                })
        
        if differences:
            print(f"发现 {len(differences)} 个不一致的帧:")
            for diff in differences:
                print(f"  帧 {diff['frame']}:")
                print(f"    NCC before 差异: {diff['ncc_before_diff']:.6f}")
                print(f"    NCC after 差异: {diff['ncc_after_diff']:.6f}")
                print(f"    DSC before 差异: {diff['dsc_before_diff']:.6f}")
                print(f"    DSC after 差异: {diff['dsc_after_diff']:.6f}")
        else:
            print("所有帧的结果一致！")
    else:
        print("无法比较结果：一种或两种方式未生成结果")
    
    # 5. 检查results目录结构
    print("\n4. 检查results目录结构...")
    results_dir = "results"
    if os.path.exists(results_dir):
        print(f"Results目录存在: {results_dir}")
        subdirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
        print(f"子目录: {subdirs}")
        
        # 检查关键文件
        frame_info = os.path.join(results_dir, "frame_info.json")
        if os.path.exists(frame_info):
            with open(frame_info, 'r', encoding='utf-8') as f:
                info = json.load(f)
            print(f"frame_info.json 存在，有效文件数: {len(info.get('valid_files', []))}")
        else:
            print("frame_info.json 不存在")
    else:
        print("Results目录不存在")

if __name__ == "__main__":
    compare_evaluation_results()