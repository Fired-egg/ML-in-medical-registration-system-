import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import os
import json

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def draw_feature_matching_inliers_only(query_image, refer_image, query_kpts, refer_kpts, matches,
                                      mask, title='特征点配准结果', save_path=None):
    """
    绘制特征点匹配可视化图，仅显示内点（通过单应性矩阵筛选的正确匹配）
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

    inlier_count = 0
    for i, match in enumerate(matches):
        if mask[i]:
            query_idx = match.queryIdx
            refer_idx = match.trainIdx
            pt1 = (int(query_kpts[query_idx].pt[0]), int(query_kpts[query_idx].pt[1]))
            pt2 = (int(refer_kpts[refer_idx].pt[0]) + w1, int(refer_kpts[refer_idx].pt[1]))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 1, lineType=cv2.LINE_AA)
            inlier_count += 1

    plt.figure(figsize=(16, 8), dpi=150)
    plt.imshow(vis)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f'已保存: {save_path}')

    plt.close()
    return vis, inlier_count


def create_selected_grid_inliers(selected_frames, config_path='config/test.yaml',
                                  results_dir='results',
                                  output_dir='results/feature_matching_paper'):
    """
    创建精选帧的网格图，仅显示内点
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    with open(os.path.join(results_dir, 'frame_info.json'), 'r', encoding='utf-8') as f:
        frame_info = json.load(f)

    reference_frame = frame_info['reference_frame']
    filtered_dir = os.path.join(results_dir, 'filtered')
    reference_path = os.path.join(filtered_dir, reference_frame)

    from predictor import Predictor
    predictor = Predictor(config)

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), dpi=150)

    for i, frame_num in enumerate(selected_frames):
        float_frame = f'{frame_num}.png'
        float_path = os.path.join(filtered_dir, float_frame)

        if not os.path.exists(float_path):
            print(f'文件不存在: {float_frame}')
            continue

        try:
            goodMatch, cv_kpts_query, cv_kpts_refer, query_image, refer_image = predictor.match(
                float_path, reference_path, show=False
            )

            if len(goodMatch) >= 4:
                src_pts = np.float32([cv_kpts_query[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
                dst_pts = np.float32([cv_kpts_refer[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
                _, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
                mask = mask.ravel().astype(bool)
            else:
                mask = np.ones(len(goodMatch), dtype=bool)

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

            query_pts = np.array([kp.pt for kp in cv_kpts_query])
            refer_pts = np.array([kp.pt for kp in cv_kpts_refer])
            refer_pts_shifted = refer_pts.copy()
            refer_pts_shifted[:, 0] += w1

            for (x, y) in query_pts:
                cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)

            for (x, y) in refer_pts_shifted:
                cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)

            inlier_count = 0
            for j, match in enumerate(goodMatch):
                if mask[j]:
                    query_idx = match.queryIdx
                    refer_idx = match.trainIdx
                    pt1 = (int(cv_kpts_query[query_idx].pt[0]), int(cv_kpts_query[query_idx].pt[1]))
                    pt2 = (int(cv_kpts_refer[refer_idx].pt[0]) + w1, int(cv_kpts_refer[refer_idx].pt[1]))
                    cv2.line(vis, pt1, pt2, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                    inlier_count += 1

            axes[i].imshow(vis)
            axes[i].set_title(f'Frame {frame_num}', fontsize=12, fontweight='bold')
            axes[i].axis('off')

            save_single = os.path.join(output_dir, f'{frame_num}_feature_matching.png')
            plt.imsave(save_single, vis)
            print(f'已保存单图: {save_single} (内点数: {inlier_count})')

        except Exception as e:
            print(f'处理帧 {frame_num} 失败: {str(e)}')

    plt.suptitle('Feature Point Matching Visualization (Reference Frame: Frame 0)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    grid_path = os.path.join(output_dir, 'selected_feature_matching_grid.png')
    plt.savefig(grid_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f'\n网格图已保存: {grid_path}')
    plt.close()


if __name__ == '__main__':
    selected_frames = [1, 2, 3, 8, 21]
    print(f'开始生成精选帧特征点匹配图（仅显示内点）: {selected_frames}')
    create_selected_grid_inliers(selected_frames)