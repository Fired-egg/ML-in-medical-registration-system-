import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

# 设置字体支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def draw_feature_matching(query_image, refer_image, query_kpts, refer_kpts, matches, mask=None, 
                          title='特征点配准结果', save_path=None):
    """
    绘制特征点匹配可视化图，类似SIFT/GFTT等算法的配准展示
    
    参数:
        query_image: 查询图像 (灰度图)
        refer_image: 参考图像 (灰度图)
        query_kpts: 查询图像的特征点列表 [cv2.KeyPoint]
        refer_kpts: 参考图像的特征点列表 [cv2.KeyPoint]
        matches: 匹配结果列表
        mask: 内点掩码 (可选)
        title: 图像标题
        save_path: 保存路径
    """
    # 获取图像尺寸
    h1, w1 = query_image.shape[:2]
    h2, w2 = refer_image.shape[:2]
    
    # 创建可视化画布
    vis_height = max(h1, h2)
    vis_width = w1 + w2
    vis = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
    
    # 将灰度图转换为RGB
    if len(query_image.shape) == 2:
        query_rgb = cv2.cvtColor(query_image, cv2.COLOR_GRAY2RGB)
    else:
        query_rgb = query_image
    
    if len(refer_image.shape) == 2:
        refer_rgb = cv2.cvtColor(refer_image, cv2.COLOR_GRAY2RGB)
    else:
        refer_rgb = refer_image
    
    # 将图像放置到画布上
    vis[0:h1, 0:w1] = query_rgb
    vis[0:h2, w1:] = refer_rgb
    
    # 提取关键点坐标
    query_pts = np.array([kp.pt for kp in query_kpts])
    refer_pts = np.array([kp.pt for kp in refer_kpts])
    
    # 将参考图像的关键点坐标偏移
    refer_pts_shifted = refer_pts.copy()
    refer_pts_shifted[:, 0] += w1
    
    # 绘制所有关键点（红色圆点）
    for (x, y) in query_pts:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
    
    for (x, y) in refer_pts_shifted:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
    
    # 绘制匹配线
    if mask is None:
        mask = np.ones(len(matches), dtype=bool)
    
    for i, match in enumerate(matches):
        if mask[i]:
            # 获取匹配的关键点索引
            query_idx = match.queryIdx
            refer_idx = match.trainIdx
            
            # 获取坐标
            pt1 = (int(query_kpts[query_idx].pt[0]), int(query_kpts[query_idx].pt[1]))
            pt2 = (int(refer_kpts[refer_idx].pt[0]) + w1, int(refer_kpts[refer_idx].pt[1]))
            
            # 绘制连接线（绿色）
            cv2.line(vis, pt1, pt2, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    
    # 创建图像
    plt.figure(figsize=(16, 8), dpi=150)
    plt.imshow(vis)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 添加统计信息
    num_matches = len(matches)
    num_inliers = int(mask.sum()) if mask is not None else num_matches
    
    stats_text = f'匹配点数: {num_matches} | 内点数: {num_inliers} | 内点率: {num_inliers/num_matches*100:.1f}%'
    plt.annotate(stats_text, (10, vis_height - 10), xycoords='data', 
                 fontsize=12, color='white',
                 bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f'图像已保存到: {save_path}')
    
    return vis


def draw_side_by_side_comparison(query_image, refer_image, query_kpts, refer_kpts, matches, 
                                 mask=None, title1='配准前', title2='配准后', save_path=None):
    """
    绘制类似示例图的并排对比图
    
    参数:
        query_image: 查询图像
        refer_image: 参考图像
        query_kpts: 查询图像特征点
        refer_kpts: 参考图像特征点
        matches: 匹配结果
        mask: 内点掩码
        title1: 左图标题
        title2: 右图标题
        save_path: 保存路径
    """
    h1, w1 = query_image.shape[:2]
    h2, w2 = refer_image.shape[:2]
    
    # 转换为RGB
    if len(query_image.shape) == 2:
        query_rgb = cv2.cvtColor(query_image, cv2.COLOR_GRAY2RGB)
        refer_rgb = cv2.cvtColor(refer_image, cv2.COLOR_GRAY2RGB)
    else:
        query_rgb = query_image
        refer_rgb = refer_image
    
    # 创建并排显示的图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    
    # 左图：查询图像关键点
    ax1.imshow(query_rgb)
    ax1.scatter([kp.pt[0] for kp in query_kpts], [kp.pt[1] for kp in query_kpts], 
                s=8, c='red', alpha=0.8)
    ax1.set_title(title1, fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 右图：参考图像关键点
    ax2.imshow(refer_rgb)
    ax2.scatter([kp.pt[0] for kp in refer_kpts], [kp.pt[1] for kp in refer_kpts], 
                s=8, c='red', alpha=0.8)
    ax2.set_title(title2, fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'并排对比图已保存到: {save_path}')
    
    plt.close()


def run_feature_matching_visualization(config_path='config/test.yaml', 
                                       query_path='data/samples/query.jpg',
                                       refer_path='data/samples/refer.jpg',
                                       output_dir='results/feature_matching'):
    """
    运行特征点匹配可视化
    
    参数:
        config_path: 配置文件路径
        query_path: 查询图像路径
        refer_path: 参考图像路径
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 导入Predictor
    from predictor import Predictor
    
    # 创建预测器
    predictor = Predictor(config)
    
    # 执行匹配
    goodMatch, cv_kpts_query, cv_kpts_refer, query_image, refer_image = predictor.match(
        query_path, refer_path, show=False
    )
    
    # 计算单应性矩阵获取内点掩码
    if len(goodMatch) >= 4:
        src_pts = np.float32([cv_kpts_query[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        dst_pts = np.float32([cv_kpts_refer[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
        mask = mask.ravel().astype(bool)
    else:
        mask = None
    
    # 生成配准后的图像
    merged, _ = predictor.align_image_pair(query_path, refer_path, show=False)
    
    # 绘制特征点匹配图
    vis = draw_feature_matching(
        query_image, refer_image,
        cv_kpts_query, cv_kpts_refer,
        goodMatch, mask,
        title='SuperRetina 特征点配准结果',
        save_path=os.path.join(output_dir, 'feature_matching_result.png')
    )
    
    # 绘制并排对比图
    draw_side_by_side_comparison(
        query_image, refer_image,
        cv_kpts_query, cv_kpts_refer,
        goodMatch, mask,
        title1='查询图像 (特征点: {})'.format(len(cv_kpts_query)),
        title2='参考图像 (特征点: {})'.format(len(cv_kpts_refer)),
        save_path=os.path.join(output_dir, 'side_by_side_comparison.png')
    )
    
    # 如果有配准结果，保存配准后的图像
    if merged is not None:
        cv2.imwrite(os.path.join(output_dir, 'registered_merged.png'), merged)
        print(f'配准融合图已保存到: {os.path.join(output_dir, "registered_merged.png")}')
    
    print(f'\n特征点匹配可视化完成！')
    print(f'查询图像特征点数: {len(cv_kpts_query)}')
    print(f'参考图像特征点数: {len(cv_kpts_refer)}')
    print(f'匹配对数: {len(goodMatch)}')
    if mask is not None:
        print(f'内点数量: {int(mask.sum())}')
        print(f'内点率: {int(mask.sum())/len(goodMatch)*100:.1f}%')


if __name__ == '__main__':
    run_feature_matching_visualization()
