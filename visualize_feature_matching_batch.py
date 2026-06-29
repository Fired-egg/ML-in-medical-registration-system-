import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import yaml
import os
import json

# 设置字体支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def draw_feature_matching(query_image, refer_image, query_kpts, refer_kpts, matches, mask=None, 
                          title='特征点配准结果', save_path=None):
    """
    绘制特征点匹配可视化图
    """
    h1, w1 = query_image.shape[:2]
    h2, w2 = refer_image.shape[:2]
    
    vis_height = max(h1, h2)
    vis_width = w1 + w2
    vis = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
    
    if len(query_image.shape) == 2:
        query_rgb = cv2.cvtColor(query_image, cv2.COLOR_GRAY2RGB)
    else:
        query_rgb = query_image
    
    if len(refer_image.shape) == 2:
        refer_rgb = cv2.cvtColor(refer_image, cv2.COLOR_GRAY2RGB)
    else:
        refer_rgb = refer_image
    
    vis[0:h1, 0:w1] = query_rgb
    vis[0:h2, w1:] = refer_rgb
    
    query_pts = np.array([kp.pt for kp in query_kpts])
    refer_pts = np.array([kp.pt for kp in refer_kpts])
    refer_pts_shifted = refer_pts.copy()
    refer_pts_shifted[:, 0] += w1
    
    for (x, y) in query_pts:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
    
    for (x, y) in refer_pts_shifted:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
    
    if mask is None:
        mask = np.ones(len(matches), dtype=bool)
    
    for i, match in enumerate(matches):
        if mask[i]:
            query_idx = match.queryIdx
            refer_idx = match.trainIdx
            pt1 = (int(query_kpts[query_idx].pt[0]), int(query_kpts[query_idx].pt[1]))
            pt2 = (int(refer_kpts[refer_idx].pt[0]) + w1, int(refer_kpts[refer_idx].pt[1]))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    
    plt.figure(figsize=(16, 8), dpi=150)
    plt.imshow(vis)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    
    num_matches = len(matches)
    num_inliers = int(mask.sum()) if mask is not None else num_matches
    
    stats_text = f'匹配点数: {num_matches} | 内点数: {num_inliers} | 内点率: {num_inliers/num_matches*100:.1f}%'
    plt.annotate(stats_text, (10, vis_height - 10), xycoords='data', 
                 fontsize=12, color='white',
                 bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    
    plt.close()
    return vis


def run_batch_feature_matching(config_path='config/test.yaml', 
                               results_dir='results',
                               output_dir='results/feature_matching_batch',
                               max_frames=10):
    """
    批量处理所有浮动帧与基准帧的特征点匹配可视化
    
    参数:
        config_path: 配置文件路径
        results_dir: 结果目录
        output_dir: 输出目录
        max_frames: 最大处理帧数（None表示处理所有帧）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载帧信息
    frame_info_path = os.path.join(results_dir, 'frame_info.json')
    with open(frame_info_path, 'r', encoding='utf-8') as f:
        frame_info = json.load(f)
    
    reference_frame = frame_info['reference_frame']
    valid_files = frame_info['valid_files']
    
    # 导入Predictor
    from predictor import Predictor
    predictor = Predictor(config)
    
    # 基准帧路径
    filtered_dir = os.path.join(results_dir, 'filtered')
    reference_path = os.path.join(filtered_dir, reference_frame)
    
    # 统计信息
    stats_list = []
    
    # 处理帧（跳过基准帧本身）
    float_frames = [f for f in valid_files if f != reference_frame]
    
    # 如果指定了最大帧数，截取部分帧
    if max_frames is not None and max_frames > 0:
        float_frames = float_frames[:max_frames]
    
    print(f'开始处理 {len(float_frames)} 帧...')
    
    for i, float_frame in enumerate(float_frames):
        float_path = os.path.join(filtered_dir, float_frame)
        
        if not os.path.exists(float_path):
            print(f'跳过不存在的文件: {float_frame}')
            continue
        
        try:
            # 执行匹配
            goodMatch, cv_kpts_query, cv_kpts_refer, query_image, refer_image = predictor.match(
                float_path, reference_path, show=False
            )
            
            # 计算单应性矩阵获取内点掩码
            if len(goodMatch) >= 4:
                src_pts = np.float32([cv_kpts_query[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
                dst_pts = np.float32([cv_kpts_refer[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
                _, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
                mask = mask.ravel().astype(bool)
            else:
                mask = None
            
            # 保存特征点匹配图
            frame_name = os.path.splitext(float_frame)[0]
            output_path = os.path.join(output_dir, f'{frame_name}_feature_matching.png')
            
            draw_feature_matching(
                query_image, refer_image,
                cv_kpts_query, cv_kpts_refer,
                goodMatch, mask,
                title=f'帧 {frame_name} 与基准帧特征点匹配',
                save_path=output_path
            )
            
            # 记录统计信息
            num_matches = len(goodMatch)
            num_inliers = int(mask.sum()) if mask is not None else 0
            inlier_rate = num_inliers / num_matches * 100 if num_matches > 0 else 0
            
            stats_list.append({
                'frame': float_frame,
                'query_kpts': len(cv_kpts_query),
                'refer_kpts': len(cv_kpts_refer),
                'matches': num_matches,
                'inliers': num_inliers,
                'inlier_rate': inlier_rate
            })
            
            print(f'[{i+1}/{len(float_frames)}] 完成: {float_frame}')
            
        except Exception as e:
            print(f'[{i+1}/{len(float_frames)}] 处理失败 {float_frame}: {str(e)}')
    
    # 保存统计信息
    stats_path = os.path.join(output_dir, 'feature_matching_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_list, f, indent=2, ensure_ascii=False)
    
    # 生成汇总统计
    if stats_list:
        avg_matches = np.mean([s['matches'] for s in stats_list])
        avg_inliers = np.mean([s['inliers'] for s in stats_list])
        avg_inlier_rate = np.mean([s['inlier_rate'] for s in stats_list])
        
        print(f'\n=== 汇总统计 ===')
        print(f'处理帧数: {len(stats_list)}')
        print(f'平均匹配点数: {avg_matches:.1f}')
        print(f'平均内点数: {avg_inliers:.1f}')
        print(f'平均内点率: {avg_inlier_rate:.1f}%')
        print(f'统计信息已保存到: {stats_path}')
        print(f'特征点匹配图已保存到: {output_dir}')
    
    return stats_list


def create_summary_grid(results_dir='results/feature_matching_batch', grid_size=(4, 4)):
    """
    创建特征点匹配结果的汇总网格图
    """
    # 获取所有特征点匹配图
    matching_files = sorted([f for f in os.listdir(results_dir) if f.endswith('_feature_matching.png')])
    
    if not matching_files:
        print('未找到特征点匹配图')
        return
    
    # 选择部分图像
    num_images = grid_size[0] * grid_size[1]
    selected_files = matching_files[:num_images]
    
    # 创建网格图
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(16, 12))
    
    for i, (ax, file) in enumerate(zip(axes.flat, selected_files)):
        img_path = os.path.join(results_dir, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(os.path.splitext(file)[0], fontsize=8)
        ax.axis('off')
    
    plt.suptitle('特征点匹配结果汇总（部分帧）', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    summary_path = os.path.join(results_dir, 'summary_grid.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'汇总网格图已保存到: {summary_path}')


if __name__ == '__main__':
    # 批量处理前20帧
    stats = run_batch_feature_matching(max_frames=20)
    
    # 创建汇总网格图
    create_summary_grid()
