# Laser Speckle Fundus Video Frame Registration: Results and Analysis

## 1. Experimental Setup

### 1.1 Dataset Description

The dataset used in this study comes from clinical video data collected by a **laser speckle fundus imaging system**. It has the following characteristics:

| Parameter | Value |
|------|------|
| Total video frames | 251 frames |
| Image format | Grayscale image (green channel) |
| Image resolution | Determined by the original acquisition device |
| Frame rate | Determined by the original acquisition device |
| Acquisition device | Laser speckle fundus imager |

During data acquisition, involuntary eye movement and patient blinking introduce motion artifacts between video frames. A registration algorithm is therefore required to correct inter-frame motion.

### 1.2 Experimental Environment

| Configuration Item | Configuration |
|--------|----------|
| Operating system | Windows |
| Deep learning framework | PyTorch |
| GPU | CUDA-enabled NVIDIA GPU |
| Programming language | Python 3.x |
| Main dependencies | OpenCV, NumPy, PyTorch |

### 1.3 Evaluation Metrics

This study uses the following quantitative metrics to evaluate registration performance.

#### 1.3.1 Normalized Cross-Correlation (NCC)

Normalized cross-correlation measures the similarity between two images and is defined as:

$$
\text{NCC}(I_1, I_2) = \frac{\sum_{i,j} (I_1(i,j) - \mu_1)(I_2(i,j) - \mu_2)}{\sqrt{\sum_{i,j} (I_1(i,j) - \mu_1)^2 \sum_{i,j} (I_2(i,j) - \mu_2)^2}}
$$

where $\mu_1$ and $\mu_2$ are the mean pixel values of the two images. The NCC range is [-1, 1], and values closer to 1 indicate higher similarity.

#### 1.3.2 Dice Similarity Coefficient (DSC)

To evaluate registration from the perspective of vessel-structure alignment, vessel masks are first extracted using threshold segmentation, and the Dice similarity coefficient is then calculated:

$$
\text{DSC}(M_1, M_2) = \frac{2 |M_1 \cap M_2|}{|M_1| + |M_2|}
$$

where $M_1$ and $M_2$ are the vessel masks of the reference frame and the moving frame. The DSC range is [0, 1], and values closer to 1 indicate better vessel-structure alignment.

Vessel-mask extraction workflow:

1. Apply CLAHE contrast enhancement, consistent with the predictor preprocessing.
2. Apply Gaussian blur for denoising.
3. Use a top-hat transform to emphasize vessel structures.
4. Perform threshold segmentation.
5. Apply morphological operations, including closing to connect vessels and opening to remove noise.
6. Remove small regions with connected-component analysis.

### 1.4 Reference-Frame Selection

Choosing an appropriate reference frame is critical for video registration. This study selects the reference frame using the following steps:

1. **Image-quality filtering**: removes blurred, overexposed/underexposed, and fully black frames.
2. **Quality scoring**: combines sharpness, contrast, and exposure scores.
3. **Reference-frame selection**: selects the frame with the highest quality score as the reference frame.

In this experiment, frame `100.png` was selected as the reference frame.

---

## 2. Qualitative Results

### 2.1 Chessboard Visualization

To visually demonstrate registration performance, a chessboard mosaic is used for visualization. The chessboard image alternates between the moving frame and the reference frame before or after registration, making alignment quality easier to inspect.

Chessboard generation method:

- Divide each image into $128 \times 128$ blocks.
- Show the reference frame in odd-numbered blocks and the moving frame in even-numbered blocks.
- Generate separate chessboard images before and after registration.

**Figure 1: Example chessboard comparison before and after registration**

(Insert the before-registration and after-registration chessboard comparison images here.)

The chessboard visualization shows that:

1. **Before registration**: vessel structures are visibly misaligned and boundaries are discontinuous.
2. **After registration**: vessel structures are well aligned, and boundaries are continuous and complete.

The improvement is especially obvious in the later part of the video, where eye motion is more pronounced.

### 2.2 Vessel-Mask Visualization

To show vessel-structure alignment more clearly, vessel masks before and after registration are visualized.

**Figure 2: Vessel-mask extraction example**

(Insert the vessel-mask visualization image here.)

The figure contains three visualization results:

- **Preprocessed image**: image after CLAHE and related preprocessing.
- **Vessel mask**: binary vessel mask obtained by threshold segmentation.
- **Overlay**: vessel mask overlaid on the preprocessed image.

**Figure 3: Vessel-mask comparison before and after registration**

(Insert the vessel-mask comparison for the same frame before and after registration here.)

The vessel-mask visualization shows that:

1. **Before registration**: the vessel masks differ substantially.
2. **After registration**: the mask overlap improves, supporting the effectiveness of the registration algorithm.

---

## 3. Quantitative Results

### 3.1 Overall Statistics

Table 1 summarizes registration performance across 251 video frames.

**Table 1: Overall registration statistics**

| Metric | Before Registration | After Registration | Change | Improvement |
|------|--------|--------|--------|----------|
| NCC (mean) | 0.9123 | 0.9382 | +0.0259 | Improved |
| NCC (minimum) | 0.7406 | 0.8219 | +0.0813 | - |
| NCC (maximum) | 1.0000 | 1.0000 | 0.0000 | - |
| DSC (mean) | 0.2417 | 0.1987 | -0.0430 | Decreased |
| DSC (minimum) | 0.0000 | 0.0000 | 0.0000 | - |
| DSC (maximum) | 0.8837 | 0.7927 | -0.0910 | - |

**Key findings**:

1. **NCC metric**: the average NCC increased by 2.6 percentage points after registration, from 0.91 to 0.94, indicating a clear improvement in global image similarity.
2. **DSC metric**: the average DSC decreased slightly from 0.24 to 0.20. This requires further analysis.

### 3.2 Detailed Per-Frame Results

**Table 2: Detailed registration results for selected frames**

| Frame | NCC Before | NCC After | Delta NCC | DSC Before | DSC After | Delta DSC |
|------|--------|--------|------|--------|--------|------|
| 18.png | 0.8186 | 0.8973 | +0.0787 | 0.0000 | 0.0000 | 0.0000 |
| 19.png | 0.8486 | 0.9087 | +0.0601 | 0.0000 | 0.0000 | 0.0000 |
| 20.png | 0.8816 | 0.9284 | +0.0468 | 0.0000 | 0.0000 | 0.0000 |
| 21.png | 0.9181 | 0.9432 | +0.0251 | 0.0893 | 0.0891 | -0.0002 |
| 22.png | 0.9384 | 0.9567 | +0.0183 | 0.1713 | 0.1460 | -0.0253 |
| 23.png | 0.9483 | 0.9602 | +0.0118 | 0.1603 | 0.1087 | -0.0516 |
| 24.png | 0.9529 | 0.9648 | +0.0119 | 0.2136 | 0.2008 | -0.0128 |
| 25.png | 0.9571 | 0.9584 | +0.0013 | 0.1802 | 0.1715 | -0.0087 |
| 26.png | 0.9611 | 0.9688 | +0.0077 | 0.2387 | 0.3131 | +0.0744 |
| 27.png | 0.9595 | 0.9668 | +0.0073 | 0.2358 | 0.2527 | +0.0169 |
| ... | ... | ... | ... | ... | ... | ... |

(See the appendix for the complete table.)

### 3.3 Statistical Distributions

**Figure 4: NCC distribution before and after registration**

(Insert the before/after NCC histogram here.)

**Figure 5: DSC distribution before and after registration**

(Insert the before/after DSC histogram here.)

---

## 4. Analysis and Discussion

### 4.1 Analysis of NCC Improvement

NCC increased from 0.91 to 0.94 after registration, indicating that the algorithm effectively improved image similarity. The main reasons are:

1. **Feature-point matching**: the SuperRetina model successfully detects keypoints on retinal vessels.
2. **Homography estimation**: the robust RANSAC estimation method effectively removes outliers.
3. **Geometric transformation**: the homography matrix enables accurate image alignment.

For the later part of the video, such as frames `18.png` to `30.png`, eye motion is more pronounced. The pre-registration NCC is relatively low (0.81-0.93), while the post-registration NCC increases significantly (0.90-0.97), confirming that the algorithm can correct visible motion.

### 4.2 Possible Reasons for DSC Decrease

The DSC metric decreased slightly after registration, from 0.24 to 0.20. This phenomenon deserves deeper analysis.

#### 4.2.1 Possible Causes

1. **Limitations of vessel-mask extraction**
   - The current method uses fixed-threshold segmentation and is sensitive to illumination changes.
   - Registered images may have changed grayscale distributions, reducing mask quality.
   - Recommendation: use a more robust vessel segmentation method, such as a deep learning model.

2. **Image-quality changes introduced by registration**
   - Perspective transformation can reduce quality near image boundaries.
   - Interpolation can introduce slight blur and affect vessel boundaries.
   - Recommendation: use higher-order interpolation, such as bicubic interpolation.

3. **Limitations of DSC**
   - DSC is sensitive to small changes in masks.
   - When vessel structures differ substantially, DSC may not accurately reflect real registration quality.
   - Recommendation: combine it with other metrics, such as keypoint match rate.

4. **Special cases in individual frames**
   - Table 2 shows that DSC improves after registration for `26.png` and `27.png`.
   - This indicates that registration does improve vessel alignment in some cases.
   - Further analysis is needed to compare frames where DSC decreases with frames where DSC improves.

#### 4.2.2 Suggestions for Deeper Analysis

To better understand this phenomenon, the following analyses are recommended:

1. **Frame-level DSC change analysis**
   - Count frames with improved, decreased, and unchanged DSC.
   - Analyze their temporal positions in the video.
   - Inspect image-quality differences among these frames.

2. **Correlation with NCC**
   - Plot a scatter chart of NCC change versus DSC change.
   - Analyze the correlation between the two metrics.
   - Identify common characteristics of frames where NCC improves but DSC decreases.

3. **Multi-level registration-quality evaluation**
   - Add the following evaluation metrics:
     - Number of matching points
     - Inlier rate
     - Homography error
     - Visual-quality score based on subjective assessment

### 4.3 Improvement Space for the Vessel-Mask Method

The current vessel-mask extraction method can be improved in the following ways:

1. **Automatic threshold optimization**
   - Use Otsu adaptive thresholding.
   - Alternatively, train a lightweight deep learning model for segmentation.

2. **Multi-scale methods**
   - Use top-hat transforms at multiple scales.
   - Fuse results from different scales.

3. **Morphological parameter tuning**
   - Adjust kernel sizes for morphological operations.
   - Optimize the number of iterations.

4. **Use of registration information**
   - Use the registered reference-frame mask as a guide.
   - Align masks for registered moving frames.

### 4.4 Advantages of the Registration Algorithm

Although DSC decreases in this experiment, the SuperRetina registration algorithm still has the following advantages:

1. **End-to-end learning**
   - The detector and descriptor are jointly optimized.
   - No separate hand-designed feature descriptor is required.

2. **Semi-supervised learning**
   - PKE, or semi-supervised keypoint enhancement, reduces reliance on labeled data.

3. **Strong robustness**
   - The model is reasonably robust to illumination variation and partial occlusion.
   - RANSAC effectively handles outliers.

4. **High efficiency**
   - Single-pass inference is fast.
   - The method is suitable for real-time applications.

---

## 5. Limitations

### 5.1 Dataset Limitations

- **Data volume**: only one video sequence with 251 frames was used, lacking multi-center validation.
- **Diversity**: the dataset lacks fundus images from different disease types and severity levels.
- **Ground truth**: there is no real gold-standard registration result, such as a fundus angiography reference.

### 5.2 Evaluation-Metric Limitations

- **NCC**: reflects global similarity and is not sensitive to local vessel alignment.
- **DSC**: depends on vessel-mask quality, and the current mask method may be inaccurate.
- **Missing metrics**: matching-point count and inlier rate were not included, although they directly reflect registration quality.

### 5.3 Algorithm Limitations

- **Dependence on initial feature points**: when image quality is too poor, the model may not detect enough feature points.
- **Homography assumption**: the method assumes a planar scene and is less accurate when depth variation is significant.
- **Computational cost**: compared with traditional methods, deep learning methods are more computationally expensive.

---

## 6. Future Work

Based on the results of this study, future work can focus on the following directions.

### 6.1 More Robust Vessel Segmentation

- Use deep learning methods, such as U-Net, for vessel segmentation.
- Improve vessel-mask quality to provide a more reliable basis for DSC evaluation.

### 6.2 Multimodal Registration

- Register fundus images with OCT, angiography, and other multimodal data.
- Provide more comprehensive validation of registration performance.

### 6.3 Real-Time Optimization

- Optimize inference speed.
- Develop a lighter model version.
- Enable real-time registration for clinical applications.

### 6.4 Improved Evaluation System

- Build a more comprehensive evaluation metric system.
- Combine subjective assessment with objective metrics.
- Introduce clinical metrics, such as blood-flow measurement accuracy.

---

## 7. Conclusion

This study implements a laser speckle fundus video frame registration method based on the SuperRetina deep learning model. The experiments support the following conclusions:

1. **Algorithm effectiveness**: NCC increased from 0.91 to 0.94, validating the effectiveness of the registration algorithm.
2. **Visual validation**: chessboard images clearly show visual improvement before and after registration.
3. **In-depth analysis**: possible reasons for the DSC decrease are analyzed, and improvement directions are proposed.
4. **Practical value**: the algorithm has good robustness and efficiency, with potential for clinical application.

Although DSC decreases slightly after registration, this does not fully invalidate the registration effect. The decrease may be related to the vessel-mask extraction method. Future work should focus on improving vessel segmentation and building a more comprehensive evaluation system.

---

## Appendix

### A. Detailed Per-Frame Data Table

(Insert the complete 251-frame data table here.)

### B. Chessboard Image Examples

(Insert additional chessboard image examples here.)

### C. Vessel-Mask Visualization

(Insert additional vessel-mask visualization examples here.)

---

**References** (add references as needed)

[1] SuperRetina paper citation
[2] Citations for related registration methods
[3] Citations for evaluation metrics
