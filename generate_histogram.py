import numpy as np
import matplotlib.pyplot as plt

# 设置字体支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 根据描述生成模拟数据（总帧数1000-1200）
np.random.seed(42)

# NCC数据 - 根据描述：配准前0.80~0.94，少数低于0.70；配准后0.88~0.97，0.80以下几乎消失
ncc_before = np.random.normal(0.87, 0.04, 1000)
ncc_before = np.append(ncc_before, np.random.uniform(0.55, 0.70, 80))
ncc_before = np.clip(ncc_before, 0.5, 1.0)

ncc_after = np.random.normal(0.92, 0.03, 1000)
ncc_after = np.append(ncc_after, np.random.uniform(0.82, 0.88, 80))
ncc_after = np.clip(ncc_after, 0.80, 1.0)

# DSC数据 - 根据描述：配准前0.71~0.89，主峰在0.75附近；配准后主峰移至0.80以上，高于0.85增多
dsc_before = np.random.normal(0.78, 0.04, 1080)
dsc_before = np.clip(dsc_before, 0.65, 0.92)

dsc_after = np.random.normal(0.83, 0.04, 1080)
dsc_after = np.clip(dsc_after, 0.70, 0.95)

# 创建图表
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

total_frames = len(ncc_before)

# NCC分布
ncc_bins = np.linspace(0.5, 1.0, 21)
ncc_before_counts, _, _ = axes[0].hist(ncc_before, bins=ncc_bins, alpha=0.5, label='配准前', density=False, edgecolor='black')
ncc_after_counts, _, _ = axes[0].hist(ncc_after, bins=ncc_bins, alpha=0.5, label='配准后', density=False, edgecolor='black')
axes[0].clear()
axes[0].bar(ncc_bins[:-1], ncc_before_counts/total_frames*100, width=(ncc_bins[1]-ncc_bins[0])*0.9, alpha=0.5, label='配准前', edgecolor='black')
axes[0].bar(ncc_bins[:-1], ncc_after_counts/total_frames*100, width=(ncc_bins[1]-ncc_bins[0])*0.9, alpha=0.5, label='配准后', edgecolor='black')
axes[0].set_xlabel('NCC值')
axes[0].set_ylabel('百分比 (%)')
axes[0].set_title('图4.3(a) NCC配准前后分布对比')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0.5, 1.02)
axes[0].set_ylim(0, 40)

# DSC分布
dsc_bins = np.linspace(0.6, 1.0, 21)
dsc_before_counts, _, _ = axes[1].hist(dsc_before, bins=dsc_bins, alpha=0.5, label='配准前', density=False, edgecolor='black')
dsc_after_counts, _, _ = axes[1].hist(dsc_after, bins=dsc_bins, alpha=0.5, label='配准后', density=False, edgecolor='black')
axes[1].clear()
axes[1].bar(dsc_bins[:-1], dsc_before_counts/total_frames*100, width=(dsc_bins[1]-dsc_bins[0])*0.9, alpha=0.5, label='配准前', edgecolor='black')
axes[1].bar(dsc_bins[:-1], dsc_after_counts/total_frames*100, width=(dsc_bins[1]-dsc_bins[0])*0.9, alpha=0.5, label='配准后', edgecolor='black')
axes[1].set_xlabel('DSC值')
axes[1].set_ylabel('百分比 (%)')
axes[1].set_title('图4.3(b) DSC配准前后分布对比')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0.6, 1.02)
axes[1].set_ylim(0, 40)

plt.tight_layout()
plt.savefig('results/ncc_dsc_distribution.png', dpi=300, bbox_inches='tight')
print('图表已保存到: results/ncc_dsc_distribution.png')
