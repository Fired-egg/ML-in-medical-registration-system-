# Supplementary Results Analysis Document

## 1. Python Code for Generating Figures

### 1.1 NCC and DSC Distribution Plots

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
df = pd.read_csv('results/registration_eval.csv')

# Create plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# NCC distribution
axes[0].hist(df['ncc_before'], bins=20, alpha=0.5, label='Before registration', density=True)
axes[0].hist(df['ncc_after'], bins=20, alpha=0.5, label='After registration', density=True)
axes[0].set_xlabel('NCC value')
axes[0].set_ylabel('Frequency (normalized)')
axes[0].set_title('NCC Distribution Before and After Registration')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# DSC distribution
axes[1].hist(df['dsc_before'], bins=20, alpha=0.5, label='Before registration', density=True)
axes[1].hist(df['dsc_after'], bins=20, alpha=0.5, label='After registration', density=True)
axes[1].set_xlabel('DSC value')
axes[1].set_ylabel('Frequency (normalized)')
axes[1].set_title('DSC Distribution Before and After Registration')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/ncc_dsc_distribution.png', dpi=300)
plt.show()
```

### 1.2 Scatter Plot of NCC Change and DSC Change

```python
# Calculate changes
df['delta_ncc'] = df['ncc_after'] - df['ncc_before']
df['delta_dsc'] = df['dsc_after'] - df['dsc_before']

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['delta_ncc'], df['delta_dsc'], alpha=0.6)
plt.xlabel('NCC change (after registration - before registration)')
plt.ylabel('DSC change (after registration - before registration)')
plt.title('Relationship Between NCC Change and DSC Change')

# Add reference lines
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)

# Add statistics
plt.annotate(f'Mean NCC change: {df["delta_ncc"].mean():.4f}',
             (0.05, 0.95), xycoords='axes fraction')
plt.annotate(f'Mean DSC change: {df["delta_dsc"].mean():.4f}',
             (0.05, 0.90), xycoords='axes fraction')

plt.grid(True, alpha=0.3)
plt.savefig('results/ncc_dsc_scatter.png', dpi=300)
plt.show()
```

### 1.3 Temporal Plot of Per-Frame Registration Performance

```python
# Extract frame numbers from filenames
df['frame_number'] = df['frame'].str.extract(r'(\d+)').astype(int)
df = df.sort_values('frame_number')

# Create temporal plots
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# NCC over time
axes[0].plot(df['frame_number'], df['ncc_before'], 'o-', alpha=0.6, label='Before registration', markevery=10)
axes[0].plot(df['frame_number'], df['ncc_after'], 's-', alpha=0.6, label='After registration', markevery=10)
axes[0].set_ylabel('NCC value')
axes[0].set_title('Per-Frame NCC Before and After Registration (sorted by frame number)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# DSC over time
axes[1].plot(df['frame_number'], df['dsc_before'], 'o-', alpha=0.6, label='Before registration', markevery=10)
axes[1].plot(df['frame_number'], df['dsc_after'], 's-', alpha=0.6, label='After registration', markevery=10)
axes[1].set_xlabel('Frame number')
axes[1].set_ylabel('DSC value')
axes[1].set_title('Per-Frame DSC Before and After Registration (sorted by frame number)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/temporal_comparison.png', dpi=300)
plt.show()
```

### 1.4 Statistical Analysis

```python
# Count frames where DSC improved, worsened, or stayed unchanged
improved = (df['delta_dsc'] > 0.01).sum()
worsened = (df['delta_dsc'] < -0.01).sum()
unchanged = ((df['delta_dsc'] >= -0.01) & (df['delta_dsc'] <= 0.01)).sum()

print(f"Frames with improved DSC: {improved} ({improved/len(df)*100:.1f}%)")
print(f"Frames with worsened DSC: {worsened} ({worsened/len(df)*100:.1f}%)")
print(f"Frames with unchanged DSC: {unchanged} ({unchanged/len(df)*100:.1f}%)")

# Group statistics by NCC improvement level
ncc_big_improve = df[df['delta_ncc'] > 0.05]
ncc_small_improve = df[(df['delta_ncc'] >= 0) & (df['delta_ncc'] <= 0.05)]
ncc_worsen = df[df['delta_ncc'] < 0]

print(f"\nFrames with large NCC improvement (>0.05):")
print(f"  Count: {len(ncc_big_improve)}")
print(f"  Mean DSC change: {ncc_big_improve['delta_dsc'].mean():.4f}")

print(f"\nFrames with small NCC improvement (0-0.05):")
print(f"  Count: {len(ncc_small_improve)}")
print(f"  Mean DSC change: {ncc_small_improve['delta_dsc'].mean():.4f}")

print(f"\nFrames with worsened NCC (<0):")
print(f"  Count: {len(ncc_worsen)}")
print(f"  Mean DSC change: {ncc_worsen['delta_dsc'].mean():.4f}")

# Draw pie chart
labels = ['DSC improved (>0.01)', 'DSC unchanged (-0.01 to 0.01)', 'DSC worsened (<-0.01)']
sizes = [improved, unchanged, worsened]
colors = ['#2ecc71', '#3498db', '#e74c3c']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
        startangle=90, shadow=True)
plt.title('Distribution of DSC Changes After Registration')
plt.axis('equal')
plt.savefig('results/dsc_change_pie.png', dpi=300)
plt.show()
```

---

## 2. Suggestions for Further Analysis

### 2.1 DSC Evaluation Under Different Thresholds

The current method uses a fixed threshold for vessel-mask extraction. Testing registration performance under different thresholds is recommended:

```python
import os
import cv2
import pandas as pd
from vessel_mask import extract_vessel_mask
from common.common_util import pre_processing

results_dir = 'results'
registered_dir = os.path.join(results_dir, 'registered_filtered')
filtered_dir = os.path.join(results_dir, 'filtered')

# Read the reference frame
with open(os.path.join(results_dir, 'frame_info.json'), 'r') as f:
    import json
    info = json.load(f)
    reference_frame = info['reference_frame']

ref_img_path = os.path.join(filtered_dir, reference_frame)
ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)

# Test different thresholds
thresholds = [30, 40, 50, 60, 70]
results = []

for threshold in thresholds:
    # Extract the reference-frame mask
    ref_mask = extract_vessel_mask(ref_img, threshold=threshold)
    
    total_dsc_before = 0
    total_dsc_after = 0
    count = 0
    
    # Read evaluation data
    df = pd.read_csv(os.path.join(results_dir, 'registration_eval.csv'))
    
    for _, row in df.iterrows():
        frame = row['frame']
        
        # Image before registration
        before_path = os.path.join(filtered_dir, frame)
        before_img = cv2.imread(before_path, cv2.IMREAD_GRAYSCALE)
        
        # Image after registration
        after_path = os.path.join(registered_dir, frame)
        after_img = cv2.imread(after_path, cv2.IMREAD_GRAYSCALE)
        
        if before_img is None or after_img is None:
            continue
            
        # Extract masks
        before_mask = extract_vessel_mask(before_img, threshold=threshold)
        after_mask = extract_vessel_mask(after_img, threshold=threshold)
        
        # Calculate DSC
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
    
    print(f"Threshold={threshold}: DSC before registration={avg_dsc_before:.4f}, "
          f"DSC after registration={avg_dsc_after:.4f}, "
          f"change={avg_dsc_after-avg_dsc_before:.4f}")

# Plot results
result_df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
plt.plot(result_df['threshold'], result_df['avg_dsc_before'], 'o-', label='Before registration', linewidth=2)
plt.plot(result_df['threshold'], result_df['avg_dsc_after'], 's-', label='After registration', linewidth=2)
plt.xlabel('Threshold')
plt.ylabel('Mean DSC')
plt.title('DSC Comparison Under Different Thresholds')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/threshold_sensitivity.png', dpi=300)
plt.show()
```

### 2.2 Comprehensive Registration-Quality Analysis

A comprehensive analysis is recommended from the following perspectives:

1. **Relationship between pre-registration image quality and registration performance**
   - Analyze the relationship between image-quality metrics, such as blur and brightness, and registration improvement.
   - Determine whether lower-quality frames benefit more from registration.

2. **Frame distance and registration performance**
   - Check whether frames closer to the reference frame achieve better registration.
   - Analyze the trend of registration performance as frame distance changes.

3. **Matching-point statistics**
   - Use the data in `match_info` to analyze the number of matching points and the inlier rate.
   - Plot the relationship between matching-point count and registration performance.

### 2.3 Failure-Case Analysis

Select representative frames for detailed analysis:

1. **Frames with large NCC improvement but DSC decrease**
   - Show the original image, pre-registration mask, and post-registration mask.
   - Analyze possible reasons for DSC decrease.

2. **Frames where both NCC and DSC improve**
   - Use them as successful cases to demonstrate registration quality.

3. **Frames where registration fails**
   - Analyze failure causes and propose improvements.

---

## 3. Suggestions for Thesis Figures

### 3.1 Selecting Chessboard Figures

Select the following frames for thesis figures:

1. **Frames with the largest registration improvement**
   - Find several frames with the largest Delta NCC.
   - Use them to visually demonstrate registration performance.

2. **Representative frames from different time periods**
   - Select frames from the beginning, middle, and end of the video.
   - Demonstrate algorithm stability across the whole video.

3. **Typical scenarios**
   - Frames with obvious eye micro-motion.
   - Frames with blinking, if applicable.

### 3.2 Vessel-Mask Visualization Suggestions

The following combinations are recommended:

1. **Preprocessed image plus vessel mask**
   - Display the original image and mask side by side.
   - Validate vessel-mask extraction quality.

2. **Mask comparison before and after registration**
   - Use a heatmap or color overlay.
   - Clearly show overlapping regions.

3. **Reference-frame mask versus post-registration mask**
   - Directly compare the two masks used for DSC calculation.
   - Help readers understand what the DSC metric represents.

### 3.3 Statistical Figure Suggestions

The thesis should include the following tables and figures:

1. **Table 1**: overall statistics, already provided in the main guide.
2. **Figure 1**: chessboard example, before registration versus after registration.
3. **Figure 2**: NCC and DSC distribution histograms, generated using the code above.
4. **Figure 3**: temporal variation plot showing registration performance across the full video.
5. **Figure 4**: scatter plot of NCC change versus DSC change, used to analyze their relationship.

---

## 4. Comparison With Other Methods (Optional)

If comparison with other methods is needed in the thesis, the following options are recommended.

### 4.1 Candidate Comparison Methods

1. **Traditional methods**
   - SIFT + RANSAC
   - ORB + RANSAC
   - Mutual-information-based registration

2. **Deep learning methods**
   - Other fundus-image-specific registration methods
   - General-purpose image registration networks

### 4.2 Running Comparison Experiments

The following script can be created for comparison:

```python
# This is illustrative code and must be adapted to the actual method.
import cv2
import numpy as np

def sift_register(img1, img2):
    """Register images using SIFT."""
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

# Run comparison experiments
# ...
```

---

## 5. Supplementary Statistics

Based on the existing data, the following additional statistics can be calculated:

```python
# Calculate correlation coefficient
correlation = df[['delta_ncc', 'delta_dsc']].corr()
print(f"Correlation coefficient between NCC change and DSC change: {correlation.iloc[0, 1]:.4f}")

# Statistical significance test: paired t-test
from scipy import stats

t_stat, p_value = stats.ttest_rel(df['ncc_before'], df['ncc_after'])
print(f"Paired t-test for NCC before and after registration: t={t_stat:.4f}, p={p_value:.6f}")

t_stat, p_value = stats.ttest_rel(df['dsc_before'], df['dsc_after'])
print(f"Paired t-test for DSC before and after registration: t={t_stat:.4f}, p={p_value:.6f}")

# Calculate confidence intervals
def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return mean, mean - h, mean + h

ncc_mean, ncc_low, ncc_high = confidence_interval(df['delta_ncc'])
dsc_mean, dsc_low, dsc_high = confidence_interval(df['delta_dsc'])

print(f"\n95% confidence interval for NCC change: {ncc_mean:.4f} [{ncc_low:.4f}, {ncc_high:.4f}]")
print(f"95% confidence interval for DSC change: {dsc_mean:.4f} [{dsc_low:.4f}, {dsc_high:.4f}]")
```

Adding these statistics to the "Results" section of the thesis can strengthen the reliability of the analysis.

---

## 6. Creating a Complete Results Display Webpage

To present the results more clearly, an HTML webpage can be generated:

```python
import pandas as pd

# Read data
df = pd.read_csv('results/registration_eval.csv')

# Create HTML
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Registration Results Display</title>
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
    <h1>Laser Speckle Fundus Registration Results</h1>
    
    <h2>Statistical Summary</h2>
    <p>Total frames: {len(df)}</p>
    <p>Mean NCC before registration: {df['ncc_before'].mean():.4f}, mean NCC after registration: {df['ncc_after'].mean():.4f}, change: {df['ncc_after'].mean() - df['ncc_before'].mean():.4f}</p>
    <p>Mean DSC before registration: {df['dsc_before'].mean():.4f}, mean DSC after registration: {df['dsc_after'].mean():.4f}, change: {df['dsc_after'].mean() - df['dsc_before'].mean():.4f}</p>
    
    <h2>Detailed Data</h2>
    <table>
        <tr>
            <th>Frame</th>
            <th>NCC Before</th>
            <th>NCC After</th>
            <th>Delta NCC</th>
            <th>DSC Before</th>
            <th>DSC After</th>
            <th>Delta DSC</th>
        </tr>
"""

# Add data rows
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

# Save HTML
with open('results/results.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Results display webpage generated: results/results.html")
```

---

## Summary

This supplementary document provides:

1. **Complete code for figure generation**: directly runnable scripts for producing thesis figures.
2. **Suggestions for further analysis**: methods for exploring data more deeply and validating hypotheses.
3. **Failure-case analysis methods**: guidance on selecting and presenting representative cases.
4. **Supplementary statistical calculations**: correlation coefficients, t-tests, confidence intervals, and related metrics.
5. **Webpage display tool**: a quick way to create an interactive results display.

Combined with the framework in the main guide, these materials can support a complete and in-depth "Results and Analysis" chapter for the thesis.
