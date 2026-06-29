# 结果分析补充文档

## 一、生成图表的Python代码

### 1.1 NCC和DSC分布图

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_csv('results/registration_eval.csv')

# 创建图表
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# NCC分布
axes[0].hist(df['ncc_before'], bins=20, alpha=0.5, label='配准前', density=True)
axes[0].hist(df['ncc_after'], bins=20, alpha=0.5, label='配准后', density=True)
axes[0].set_xlabel('NCC值')
axes[0].set_ylabel('频率（归一化）')
axes[0].set_title('NCC配准前后分布对比')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# DSC分布
axes[1].hist(df['dsc_before'], bins=20, alpha=0.5, label='配准前', density=True)
axes[1].hist(df['dsc_after'], bins=20, alpha=0.5, label='配准后', density=True)
axes[1].set_xlabel('DSC值')
axes[1].set_ylabel('频率（归一化）')
axes[1].set_title('DSC配准前后分布对比')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/ncc_dsc_distribution.png', dpi=300)
plt.show()
```

### 1.2 NCC变化与DSC变化散点图

```python
# 计算变化量
df['delta_ncc'] = df['ncc_after'] - df['ncc_before']
df['delta_dsc'] = df['dsc_after'] - df['dsc_before']

# 创建散点图
plt.figure(figsize=(10, 6))
plt.scatter(df['delta_ncc'], df['delta_dsc'], alpha=0.6)
plt.xlabel('NCC变化量（配准后-配准前）')
plt.ylabel('DSC变化量（配准后-配准前）')
plt.title('NCC变化与DSC变化的关系')

# 添加参考线
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)

# 添加统计信息
plt.annotate(f'NCC平均变化: {df["delta_ncc"].mean():.4f}', 
             (0.05, 0.95), xycoords='axes fraction')
plt.annotate(f'DSC平均变化: {df["delta_dsc"].mean():.4f}', 
             (0.05, 0.90), xycoords='axes fraction')

plt.grid(True, alpha=0.3)
plt.savefig('results/ncc_dsc_scatter.png', dpi=300)
plt.show()
```

### 1.3 各帧配准效果时序图

```python
# 从文件名中提取帧号
df['frame_number'] = df['frame'].str.extract(r'(\d+)').astype(int)
df = df.sort_values('frame_number')

# 创建时序图
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# NCC时序
axes[0].plot(df['frame_number'], df['ncc_before'], 'o-', alpha=0.6, label='配准前', markevery=10)
axes[0].plot(df['frame_number'], df['ncc_after'], 's-', alpha=0.6, label='配准后', markevery=10)
axes[0].set_ylabel('NCC值')
axes[0].set_title('各帧NCC配准前后对比（按帧号排序）')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# DSC时序
axes[1].plot(df['frame_number'], df['dsc_before'], 'o-', alpha=0.6, label='配准前', markevery=10)
axes[1].plot(df['frame_number'], df['dsc_after'], 's-', alpha=0.6, label='配准后', markevery=10)
axes[1].set_xlabel('帧号')
axes[1].set_ylabel('DSC值')
axes[1].set_title('各帧DSC配准前后对比（按帧号排序）')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/temporal_comparison.png', dpi=300)
plt.show()
```

### 1.4 统计分析

```python
# 计算DSC改善、变差、不变的帧数
improved = (df['delta_dsc'] > 0.01).sum()
worsened = (df['delta_dsc'] < -0.01).sum()
unchanged = ((df['delta_dsc'] >= -0.01) & (df['delta_dsc'] <= 0.01)).sum()

print(f"DSC改善的帧数: {improved} ({improved/len(df)*100:.1f}%)")
print(f"DSC变差的帧数: {worsened} ({worsened/len(df)*100:.1f}%)")
print(f"DSC不变的帧数: {unchanged} ({unchanged/len(df)*100:.1f}%)")

# 分组统计（按NCC改善程度）
ncc_big_improve = df[df['delta_ncc'] > 0.05]
ncc_small_improve = df[(df['delta_ncc'] >= 0) & (df['delta_ncc'] <= 0.05)]
ncc_worsen = df[df['delta_ncc'] < 0]

print(f"\nNCC大幅改善（>0.05）的帧:")
print(f"  数量: {len(ncc_big_improve)}")
print(f"  平均DSC变化: {ncc_big_improve['delta_dsc'].mean():.4f}")

print(f"\nNCC小幅改善（0-0.05）的帧:")
print(f"  数量: {len(ncc_small_improve)}")
print(f"  平均DSC变化: {ncc_small_improve['delta_dsc'].mean():.4f}")

print(f"\nNCC变差（<0）的帧:")
print(f"  数量: {len(ncc_worsen)}")
print(f"  平均DSC变化: {ncc_worsen['delta_dsc'].mean():.4f}")

# 绘制饼图
labels = ['DSC改善 (>0.01)', 'DSC不变 (-0.01~0.01)', 'DSC变差 (<-0.01)']
sizes = [improved, unchanged, worsened]
colors = ['#2ecc71', '#3498db', '#e74c3c']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
        startangle=90, shadow=True)
plt.title('DSC配准后变化分布')
plt.axis('equal')
plt.savefig('results/dsc_change_pie.png', dpi=300)
plt.show()
```

---

## 二、进一步分析建议

### 2.1 不同阈值下的DSC评估

当前使用了固定阈值进行血管掩码提取，建议测试不同阈值下的配准效果：

```python
import os
import cv2
import pandas as pd
from vessel_mask import extract_vessel_mask
from common.common_util import pre_processing

results_dir = 'results'
registered_dir = os.path.join(results_dir, 'registered_filtered')
filtered_dir = os.path.join(results_dir, 'filtered')

# 读取基准帧
with open(os.path.join(results_dir, 'frame_info.json'), 'r') as f:
    import json
    info = json.load(f)
    reference_frame = info['reference_frame']

ref_img_path = os.path.join(filtered_dir, reference_frame)
ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)

# 测试不同阈值
thresholds = [30, 40, 50, 60, 70]
results = []

for threshold in thresholds:
    # 提取基准帧掩码
    ref_mask = extract_vessel_mask(ref_img, threshold=threshold)
    
    total_dsc_before = 0
    total_dsc_after = 0
    count = 0
    
    # 读取评价数据
    df = pd.read_csv(os.path.join(results_dir, 'registration_eval.csv'))
    
    for _, row in df.iterrows():
        frame = row['frame']
        
        # 配准前图像
        before_path = os.path.join(filtered_dir, frame)
        before_img = cv2.imread(before_path, cv2.IMREAD_GRAYSCALE)
        
        # 配准后图像
        after_path = os.path.join(registered_dir, frame)
        after_img = cv2.imread(after_path, cv2.IMREAD_GRAYSCALE)
        
        if before_img is None or after_img is None:
            continue
            
        # 提取掩码
        before_mask = extract_vessel_mask(before_img, threshold=threshold)
        after_mask = extract_vessel_mask(after_img, threshold=threshold)
        
        # 计算DSC
        def dsc(m1, m2):
            m1 = m1.astype(bool)
            m2 = m2.astype(bool)
            intersection = np.sum(m1 & m2)
            union = np.sum(m1) + np.sum(m2)
            if union == 0:
                return 1.0
            return 2 * intersection / union
        
        dsc_before = dsc(before_mask, ref_mask)
        dsc_after = dsc(after_mask, ref_mask)
        
        total_dsc_before += dsc_before
        total_dsc_after += dsc_after
        count += 1
    
    avg_dsc_before = total_dsc_before / count if count > 0 else 0
    avg_dsc_after = total_dsc_after / count if count > 0 else 0
    
    results.append({
        'threshold': threshold,
        'avg_dsc_before': avg_dsc_before,
        'avg_dsc_after': avg_dsc_after,
        'delta_dsc': avg_dsc_after - avg_dsc_before
    })
    
    print(f"阈值={threshold}: 配准前DSC={avg_dsc_before:.4f}, 配准后DSC={avg_dsc_after:.4f}, 变化={avg_dsc_after-avg_dsc_before:.4f}")

# 绘制结果
result_df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
plt.plot(result_df['threshold'], result_df['avg_dsc_before'], 'o-', label='配准前', linewidth=2)
plt.plot(result_df['threshold'], result_df['avg_dsc_after'], 's-', label='配准后', linewidth=2)
plt.xlabel('阈值')
plt.ylabel('平均DSC')
plt.title('不同阈值下DSC对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/threshold_sensitivity.png', dpi=300)
plt.show()
```

### 2.2 配准质量综合分析

建议从以下角度进行综合分析：

1. **配准前图像质量与配准效果的关系**
   - 分析模糊度、亮度等图像质量指标与配准提升量的关系
   - 是否质量差的帧配准提升更明显？

2. **帧间距离与配准效果**
   - 距基准帧越近的帧配准效果是否更好？
   - 配准效果随帧距变化的趋势如何？

3. **匹配点统计**
   - 利用match_info中的数据，分析匹配点数量、内点率等
   - 绘制匹配点数量与配准效果的关系图

### 2.3 错误案例分析

建议选取几个典型的帧进行详细分析：

1. **NCC大幅提升但DSC下降的帧**
   - 展示原始图像、配准前掩码、配准后掩码
   - 分析DSC下降的可能原因

2. **NCC和DSC都提升的帧**
   - 作为成功案例，展示配准效果

3. **配准失败的帧**
   - 分析失败原因，提出改进建议

---

## 三、论文图表的制作建议

### 3.1 棋盘格图的选择建议

选择以下帧作为论文图表：

1. **配准提升最明显的帧**
   - 找ΔNCC最大的几帧
   - 直观展示配准效果

2. **不同时段的代表性帧**
   - 视频开始、中间、结尾各选几帧
   - 展示算法在整个视频中的稳定性

3. **典型场景**
   - 眼球微动明显的帧
   - 有眨眼的帧（如果适用）

### 3.2 血管掩码可视化建议

建议展示以下组合：

1. **预处理图像 + 血管掩码**
   - 并排展示原始图像和掩码
   - 验证掩码提取质量

2. **配准前/配准后掩码对比**
   - 使用热力图或不同颜色叠加
   - 清晰展示重叠区域

3. **基准帧掩码与配准后掩码**
   - 直接对比以计算DSC的两个掩码
   - 帮助理解DSC指标的含义

### 3.3 统计图表建议

论文中建议包含以下图表：

1. **表1**：整体统计数据（已在指南中提供）
2. **图1**：棋盘格图示例（配准前vs配准后）
3. **图2**：NCC和DSC分布直方图（可用上面的代码生成）
4. **图3**：时序变化图（展示视频全过程的配准效果）
5. **图4**：NCC变化vsDSC变化散点图（分析两者关系）

---

## 四、与其他方法的对比（可选）

如果想在论文中对比其他方法，建议：

### 4.1 可选的对比方法

1. **传统方法**
   - SIFT + RANSAC
   - ORB + RANSAC
   - 基于互信息的配准

2. **深度学习方法**
   - 其他专门的眼底图像配准方法
   - 通用的图像配准网络

### 4.2 对比实验的运行

可创建以下脚本进行对比：

```python
# 此为示意代码，需要根据实际方法实现
import cv2
import numpy as np

def sift_register(img1, img2):
    """使用SIFT进行配准"""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    
    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    return None

# 运行对比实验
# ...
```

---

## 五、补充统计数据

基于现有数据，可以进一步计算以下统计量：

```python
# 计算相关系数
correlation = df[['delta_ncc', 'delta_dsc']].corr()
print(f"NCC变化与DSC变化的相关系数: {correlation.iloc[0, 1]:.4f}")

# 统计显著性检验（配对t检验）
from scipy import stats

t_stat, p_value = stats.ttest_rel(df['ncc_before'], df['ncc_after'])
print(f"NCC配准前后配对t检验: t={t_stat:.4f}, p={p_value:.6f}")

t_stat, p_value = stats.ttest_rel(df['dsc_before'], df['dsc_after'])
print(f"DSC配准前后配对t检验: t={t_stat:.4f}, p={p_value:.6f}")

# 计算置信区间
def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return mean, mean - h, mean + h

ncc_mean, ncc_low, ncc_high = confidence_interval(df['delta_ncc'])
dsc_mean, dsc_low, dsc_high = confidence_interval(df['delta_dsc'])

print(f"\nNCC变化95%置信区间: {ncc_mean:.4f} [{ncc_low:.4f}, {ncc_high:.4f}]")
print(f"DSC变化95%置信区间: {dsc_mean:.4f} [{dsc_low:.4f}, {dsc_high:.4f}]")
```

将这些统计结果补充到论文的"结果"部分，可以增强分析的可靠性。

---

## 六、创建完整的结果展示网页

为了更好地展示结果，可以创建一个HTML网页：

```python
import pandas as pd

# 读取数据
df = pd.read_csv('results/registration_eval.csv')

# 创建HTML
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>配准结果展示</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
    </style>
</head>
<body>
    <h1>激光散斑眼底配准结果展示</h1>
    
    <h2>统计摘要</h2>
    <p>总帧数: {len(df)}</p>
    <p>NCC配准前平均: {df['ncc_before'].mean():.4f}, 配准后平均: {df['ncc_after'].mean():.4f}, 变化: {df['ncc_after'].mean() - df['ncc_before'].mean():.4f}</p>
    <p>DSC配准前平均: {df['dsc_before'].mean():.4f}, 配准后平均: {df['dsc_after'].mean():.4f}, 变化: {df['dsc_after'].mean() - df['dsc_before'].mean():.4f}</p>
    
    <h2>详细数据</h2>
    <table>
        <tr>
            <th>帧号</th>
            <th>NCC_前</th>
            <th>NCC_后</th>
            <th>ΔNCC</th>
            <th>DSC_前</th>
            <th>DSC_后</th>
            <th>ΔDSC</th>
        </tr>
"""

# 添加数据行
for idx, row in df.iterrows():
    delta_ncc = row['ncc_after'] - row['ncc_before']
    delta_dsc = row['dsc_after'] - row['dsc_before']
    
    ncc_class = 'positive' if delta_ncc > 0 else 'negative'
    dsc_class = 'positive' if delta_dsc > 0 else 'negative'
    
    html_content += f"""
        <tr>
            <td>{row['frame']}</td>
            <td>{row['ncc_before']:.4f}</td>
            <td>{row['ncc_after']:.4f}</td>
            <td class='{ncc_class}'>{delta_ncc:.4f}</td>
            <td>{row['dsc_before']:.4f}</td>
            <td>{row['dsc_after']:.4f}</td>
            <td class='{dsc_class}'>{delta_dsc:.4f}</td>
        </tr>
    """

html_content += """
    </table>
</body>
</html>
"""

# 保存HTML
with open('results/results.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("结果展示网页已生成: results/results.html")
```

---

## 总结

本补充文档提供了：

1. **生成图表的完整代码**：可直接运行生成论文中需要的图表
2. **进一步分析的建议**：如何深入挖掘数据、验证假设
3. **错误案例分析方法**：如何选取代表性案例进行展示
4. **补充统计计算**：相关系数、t检验、置信区间等
5. **网页展示工具**：快速创建交互式结果展示

将这些分析与主指南中的框架结合，可以撰写出结构完整、分析深入的论文"结果与分析"章节。

