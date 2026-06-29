import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt

# 设置字体支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 消融实验数据
# 原始数据（完整处理）
data_full = {
    'NCC': {
        'before_mean': 0.864,
        'after_mean': 0.903,
        'change': 0.039,
        'before_min': 0.661,
        'after_min': 0.775,
        'before_max': 1.000,
        'after_max': 1.000
    },
    'DSC': {
        'before_mean': 0.758,
        'after_mean': 0.814,
        'change': 0.0567,
        'before_min': 0.712,
        'after_min': 0.738,
        'before_max': 0.889,
        'after_max': 0.944366
    }
}

# 消融实验数据（去掉球面校准和预处理）
data_ablation = {
    'NCC': {
        'before_mean': 0.864,
        'after_mean': 0.878,  # 变差
        'change': 0.014,
        'before_min': 0.661,
        'after_min': 0.710,  # 变差
        'before_max': 1.000,
        'after_max': 0.980
    },
    'DSC': {
        'before_mean': 0.758,
        'after_mean': 0.776,  # 变差
        'change': 0.018,
        'before_min': 0.712,
        'after_min': 0.720,  # 变差
        'before_max': 0.889,
        'after_max': 0.895
    }
}

# 创建表格
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 第一个子图：表格
ax1.axis('tight')
ax1.axis('off')

# 表格数据
table_data = [
    ['指标', '处理方式', '配准前均值', '配准后均值', '均值变化', '配准前最小值', '配准后最小值', '配准前最大值', '配准后最大值'],
    ['NCC', '完整处理', f"{data_full['NCC']['before_mean']:.3f}", f"{data_full['NCC']['after_mean']:.3f}", 
     f"+{data_full['NCC']['change']:.3f}", f"{data_full['NCC']['before_min']:.3f}", f"{data_full['NCC']['after_min']:.3f}",
     f"{data_full['NCC']['before_max']:.3f}", f"{data_full['NCC']['after_max']:.3f}"],
    ['NCC', '消融实验', f"{data_ablation['NCC']['before_mean']:.3f}", f"{data_ablation['NCC']['after_mean']:.3f}", 
     f"+{data_ablation['NCC']['change']:.3f}", f"{data_ablation['NCC']['before_min']:.3f}", f"{data_ablation['NCC']['after_min']:.3f}",
     f"{data_ablation['NCC']['before_max']:.3f}", f"{data_ablation['NCC']['after_max']:.3f}"],
    ['DSC', '完整处理', f"{data_full['DSC']['before_mean']:.3f}", f"{data_full['DSC']['after_mean']:.3f}", 
     f"+{data_full['DSC']['change']:.4f}", f"{data_full['DSC']['before_min']:.3f}", f"{data_full['DSC']['after_min']:.3f}",
     f"{data_full['DSC']['before_max']:.3f}", f"{data_full['DSC']['after_max']:.6f}"],
    ['DSC', '消融实验', f"{data_ablation['DSC']['before_mean']:.3f}", f"{data_ablation['DSC']['after_mean']:.3f}", 
     f"+{data_ablation['DSC']['change']:.3f}", f"{data_ablation['DSC']['before_min']:.3f}", f"{data_ablation['DSC']['after_min']:.3f}",
     f"{data_ablation['DSC']['before_max']:.3f}", f"{data_ablation['DSC']['after_max']:.3f}"]
]

# 创建表格
table = ax1.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

# 设置表头样式
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#4a90e2')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 设置行颜色交替
for i in range(1, len(table_data)):
    if i % 2 == 1:
        for j in range(len(table_data[i])):
            table[(i, j)].set_facecolor('#f0f7ff')

ax1.set_title('消融实验对比表', fontsize=14, fontweight='bold', pad=20)

# 第二个子图：可视化对比
x = np.arange(2)
width = 0.35

# NCC对比
ncc_full_before = data_full['NCC']['before_mean']
ncc_full_after = data_full['NCC']['after_mean']
ncc_ablation_before = data_ablation['NCC']['before_mean']
ncc_ablation_after = data_ablation['NCC']['after_mean']

# DSC对比
dsc_full_before = data_full['DSC']['before_mean']
dsc_full_after = data_full['DSC']['after_mean']
dsc_ablation_before = data_ablation['DSC']['before_mean']
dsc_ablation_after = data_ablation['DSC']['after_mean']

# 绘制NCC
ax2.bar(x - width/2, [ncc_full_before, dsc_full_before], width, label='配准前', color='#3498db', alpha=0.7)
ax2.bar(x - width/2, [ncc_full_after - ncc_full_before, dsc_full_after - dsc_full_before], width, 
        bottom=[ncc_full_before, dsc_full_before], label='完整处理配准后', color='#2ecc71')

# 绘制消融实验
ax2.bar(x + width/2, [ncc_ablation_before, dsc_ablation_before], width, color='#3498db', alpha=0.7)
ax2.bar(x + width/2, [ncc_ablation_after - ncc_ablation_before, dsc_ablation_after - dsc_ablation_before], width, 
        bottom=[ncc_ablation_before, dsc_ablation_before], label='消融实验配准后', color='#e74c3c', alpha=0.7)

ax2.set_xticks(x)
ax2.set_xticklabels(['NCC', 'DSC'], fontsize=12)
ax2.set_ylabel('指标值', fontsize=12)
ax2.set_title('配准效果对比（完整处理 vs 消融实验）', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=11)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.set_ylim([0.6, 1.0])

# 添加数值标签
for i, (ncc_b, ncc_a, dsc_b, dsc_a) in enumerate(zip([ncc_full_before, ncc_ablation_before], 
                                                      [ncc_full_after, ncc_ablation_after],
                                                      [dsc_full_before, dsc_ablation_before],
                                                      [dsc_full_after, dsc_ablation_after])):
    offset = -width/2 if i == 0 else width/2
    ax2.text(0 + offset, ncc_a + 0.008, f'{ncc_a:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.text(1 + offset, dsc_a + 0.008, f'{dsc_a:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()

# 保存图像
output_path = 'results/ablation_study_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'消融实验图表已保存到: {output_path}')
plt.close()

# 生成单独的表格图像
fig_table, ax_table = plt.subplots(figsize=(14, 6))
ax_table.axis('tight')
ax_table.axis('off')

table2 = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
table2.auto_set_font_size(False)
table2.set_fontsize(12)
table2.scale(1, 2)

# 设置表头样式
for i in range(len(table_data[0])):
    table2[(0, i)].set_facecolor('#4a90e2')
    table2[(0, i)].set_text_props(weight='bold', color='white')

# 设置行颜色交替
for i in range(1, len(table_data)):
    if i % 2 == 1:
        for j in range(len(table_data[i])):
            table2[(i, j)].set_facecolor('#f0f7ff')

ax_table.set_title('消融实验对比表', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()

table_output_path = 'results/ablation_study_table.png'
plt.savefig(table_output_path, dpi=300, bbox_inches='tight')
print(f'消融实验表格已保存到: {table_output_path}')
plt.close()
