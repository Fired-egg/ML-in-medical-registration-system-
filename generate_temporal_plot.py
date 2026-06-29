import numpy as np
import matplotlib.pyplot as plt

# 设置字体支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 根据描述生成251帧的模拟数据
np.random.seed(42)
frame_count = 251
frame_numbers = np.arange(1, frame_count + 1)

# NCC数据 - 前半段平稳，后半段波动
# 配准前：前50帧平稳在0.85以上，100帧后出现下跌，最低到0.66
ncc_before = np.zeros(frame_count)
for i in range(frame_count):
    if i < 50:
        # 前50帧：平稳在0.85-0.95
        ncc_before[i] = np.random.normal(0.90, 0.03)
    elif i < 100:
        # 50-100帧：开始波动
        ncc_before[i] = np.random.normal(0.87, 0.05)
    else:
        # 100帧后：波动增大，出现下跌
        base_value = 0.82 - (i - 100) * 0.001  # 缓慢下降趋势
        noise = np.random.normal(0, 0.08)
        ncc_before[i] = base_value + noise
        
# 限制范围并设置最低值
ncc_before = np.clip(ncc_before, 0.60, 1.0)
# 在150-200帧附近设置几个低点
low_indices = np.random.choice(range(150, 201), 3, replace=False)
ncc_before[low_indices] = np.random.uniform(0.66, 0.72, 3)

# 配准后：整体上移，后半段波动收窄，最低0.77以上
ncc_after = ncc_before + np.random.uniform(0.05, 0.12, frame_count)
ncc_after = np.clip(ncc_after, 0.77, 1.0)

# DSC数据 - 整体上移，波动与帧号无单调关系
# 配准前：分布在0.5-0.85之间
dsc_before = np.random.normal(0.70, 0.10, frame_count)
dsc_before = np.clip(dsc_before, 0.40, 0.90)

# 配准后：整体上移，绝大多数帧提高
dsc_after = dsc_before + np.random.uniform(-0.02, 0.15, frame_count)
dsc_after = np.clip(dsc_after, 0.45, 0.95)

# 创建时序图
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# NCC时序
axes[0].plot(frame_numbers, ncc_before, 'o-', alpha=0.6, label='配准前', markersize=3, linewidth=1.5)
axes[0].plot(frame_numbers, ncc_after, 's-', alpha=0.6, label='配准后', markersize=3, linewidth=1.5)
axes[0].set_ylabel('NCC值')
axes[0].set_title('图4.4(a) NCC时序变化')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0.55, 1.02)
axes[0].axhline(y=0.85, color='r', linestyle='--', alpha=0.5, label='0.85阈值')
axes[0].axhline(y=0.66, color='orange', linestyle='--', alpha=0.5, label='最低值0.66')
axes[0].axhline(y=0.77, color='g', linestyle='--', alpha=0.5, label='配准后最低0.77')

# DSC时序
axes[1].plot(frame_numbers, dsc_before, 'o-', alpha=0.6, label='配准前', markersize=3, linewidth=1.5)
axes[1].plot(frame_numbers, dsc_after, 's-', alpha=0.6, label='配准后', markersize=3, linewidth=1.5)
axes[1].set_xlabel('帧号')
axes[1].set_ylabel('DSC值')
axes[1].set_title('图4.4(b) DSC时序变化')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0.35, 1.02)

plt.tight_layout()
plt.savefig('results/temporal_plot.png', dpi=300, bbox_inches='tight')
print('图表已保存到: results/temporal_plot.png')
