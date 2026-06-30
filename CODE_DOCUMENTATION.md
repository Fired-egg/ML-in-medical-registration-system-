# Code Documentation for the Laser Speckle Fundus Video Frame Registration Algorithm

## 1. Project Overview

This project implements a deep-learning-based registration algorithm for laser speckle fundus video frames. It mainly contains the following core modules:

| Module | File Location | Description |
|------|----------|----------|
| Preprocessing module | `pre/get_base.py` | Image-quality filtering, reference-frame selection, and contrast enhancement |
| Registration core module | `predictor.py` | SuperRetina model inference, feature matching, and homography estimation |
| Registration workflow module | `register_from_base.py` | Batch frame registration and matching-information export |
| Evaluation module | `evaluate_registration.py` | NCC/DSC metric calculation and vessel-mask extraction |
| Visualization module | `visualize_and_evaluate.py` | Vessel-mask visualization and registration-result display |
| GUI module | `app.py` / `gui_worker.py` | Graphical interface operation and multi-threaded task management |

---

## 2. Overall Architecture and Data Flow

```text
+---------------------------------------------------------------------+
|                         Data Processing Flow                        |
+---------------------------------------------------------------------+
|                                                                     |
|  Raw images --> [Preprocessing module] --> filtered folder          |
|      |                              |                               |
|      |                              v                               |
|      |                    [Registration core module]                 |
|      |                              |                               |
|      |                              v                               |
|      |                    registered_filtered folder                 |
|      |                              |                               |
|      |                              v                               |
|      |                    [Predictor preprocessing]                  |
|      |                              |                               |
|      |                              v                               |
|      |                    filtered_predictor_preprocessed folder     |
|      |                              |                               |
|      |                              v                               |
|      +------------> [Evaluation module] <------------+              |
|                              |                                      |
|                              v                                      |
|                    vessel_mask_visualization folder                  |
|                              |                                      |
|                              v                                      |
|                    registration_eval.csv / registration_stats.txt   |
|                                                                     |
+---------------------------------------------------------------------+
```

---

## 3. Core Module Details

### 3.1 Preprocessing Module (`pre/get_base.py`)

**Purpose**: performs quality filtering and reference-frame selection for raw video frames.

**Processing workflow**:

```python
# 1. Image-quality assessment
- Blur detection: calculated with Laplacian variance
- Overexposure/underexposure detection: based on histogram statistics
- Fully black frame detection: based on pixel-value statistics

# 2. Quality scoring and sorting
- Overall score = sharpness score + contrast score + exposure score

# 3. Reference-frame selection
- Select the frame with the highest quality score as the reference frame
- Save reference-frame information to frame_info.json

# 4. Image preprocessing
- Contrast enhancement (CLAHE)
- Gaussian denoising (optional)
```

**Key output files**:

- `results/filtered/` - valid frames after filtering
- `results/frame_info.json` - reference-frame information

---

### 3.2 Registration Core Module (`predictor.py`)

**Purpose**: uses the SuperRetina deep learning model for feature extraction and image registration.

**Core class: `Predictor`**

#### 3.2.1 Initialization Workflow

```python
def __init__(self, config):
    # 1. Load configuration parameters
    - device: compute device (GPU/CPU)
    - model_save_path: pretrained model path
    - nms_size/nms_thresh: NMS parameters
    - knn_thresh: KNN matching threshold

    # 2. Load the pretrained model
    - Initialize the SuperRetina model
    - Load trained weights
    - Set the model to evaluation mode

    # 3. Initialize matcher
    - BFMatcher (L2 distance)
```

#### 3.2.2 Image Preprocessing (`image_read`)

```python
def image_read(self, query_path, refer_path):
    # 1. Read color images and extract the green channel
    query_image = query_image[:, :, 1]
    refer_image = refer_image[:, :, 1]
    
    # 2. Apply standardized preprocessing (common_util.pre_processing)
    #    - Histogram equalization
    #    - Gamma correction
    #    - Contrast normalization
    
    # 3. Convert to uint8 format
```

#### 3.2.3 Feature Extraction (`model_run_pair`)

```python
def model_run_pair(self, query_tensor, refer_tensor):
    # 1. Model inference
    detector_pred, descriptor_pred = self.model(inputs)
    
    # 2. NMS non-maximum suppression
    scores = simple_nms(detector_pred, self.nms_size)
    
    # 3. Keypoint extraction
    keypoints = torch.nonzero(scores > threshold)
    
    # 4. Border removal
    keypoints = remove_borders(keypoints, 4, h, w)
    
    # 5. Descriptor sampling
    descriptors = sample_keypoint_desc(keypoints, descriptors, 8)
    
    return keypoints, descriptors
```

#### 3.2.4 Feature Matching (`match`)

```python
def match(self, query_path, refer_path):
    # 1. Read and preprocess images
    query_image, refer_image = self.image_read(query_path, refer_path)
    
    # 2. Extract features
    keypoints, descriptors = self.model_run_pair(query_tensor, refer_tensor)
    
    # 3. KNN matching (k=2)
    matches = self.knn_matcher.knnMatch(query_desc, refer_desc, k=2)
    
    # 4. Lowe's ratio test
    for m, n in matches:
        if m.distance < self.knn_thresh * n.distance:
            goodMatch.append(m)
    
    return goodMatch, keypoints, images
```

#### 3.2.5 Homography Matrix Estimation (`compute_homography`)

```python
def compute_homography(self, query_path, refer_path):
    # 1. Get matching points
    goodMatch, cv_kpts_query, cv_kpts_refer = self.match(...)
    
    # 2. Prepare matching-point coordinates
    src_pts = [cv_kpts_query[m.queryIdx].pt for m in goodMatch]
    dst_pts = [cv_kpts_refer[m.trainIdx].pt for m in goodMatch]
    
    # 3. Estimate the homography matrix using the LMEDS algorithm
    H_m, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
    
    # 4. Calculate the inlier rate
    inliers_num_rate = num_inliers / len(mask.ravel())
    
    return H_m, inliers_num_rate, match_info
```

#### 3.2.6 Image Registration (`align_image_pair`)

```python
def align_image_pair(self, query_path, refer_path):
    # 1. Compute the homography matrix
    H_m, inliers_num_rate, ... = self.compute_homography(...)
    
    # 2. Perspective transformation
    if H_m is not None and inliers_num_rate >= 0.1:
        query_align = cv2.warpPerspective(raw_query_image, H_m, (w, h))
    
    # 3. Build the matching-information dictionary
    match_info = {
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        "inliers_rate": inliers_num_rate,
        "H": H_m.tolist()
    }
    
    return merged, match_info
```

---

### 3.3 Registration Workflow Module (`register_from_base.py`)

**Purpose**: batch-processes video frames and registers all moving frames to the reference frame.

**Processing workflow**:

```python
def register_from_base(results_dir, source_folder="filtered"):
    # 1. Initialize Predictor
    config = yaml.safe_load(open(config_path))
    predictor = Predictor(config)
    
    # 2. Read reference-frame information
    with open("frame_info.json") as f:
        info = json.load(f)
        reference_frame = info["reference_frame"]
        valid_files = info["valid_files"]
    
    # 3. Self-register the reference frame
    ref_result = predictor.align_image_pair(refer_path, refer_path)
    
    # 4. Iterate through all moving frames and register them
    for fname in valid_files:
        query_path = os.path.join(frames_dir, fname)
        result = predictor.align_image_pair(query_path, refer_path)
        
        # Save registration result
        cv2.imwrite(os.path.join(out_dir, fname), aligned_bgr)
        
        # Collect matching information
        all_match_info[fname] = match_info
    
    # 5. Save matching information to JSON
    with open("match_info.json", "w") as f:
        json.dump(all_match_info, f)
```

**Key outputs**:

- `results/registered_filtered/` - registered images
- `results/match_info_filtered/match_info.json` - matching-point information

---

### 3.4 Evaluation Module (`evaluate_registration.py`)

**Purpose**: calculates evaluation metrics for registration quality.

**Core metrics**:

| Metric | Calculation | Meaning |
|------|----------|------|
| NCC | Normalized cross-correlation | Measures image similarity [-1, 1] |
| DSC | Dice similarity coefficient | Measures vessel-mask overlap [0, 1] |

**Vessel-mask extraction workflow**:

```python
def extract_vessel_mask(gray, threshold=50):
    # 1. Preprocess using the same method as predictor
    preprocessed = pre_processing(gray)
    
    # 2. Gaussian blur
    blurred = cv2.GaussianBlur(preprocessed, (5, 5), 0)
    
    # 3. Top-hat transform to emphasize vessel structures
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    
    # 4. Threshold segmentation
    _, binary = cv2.threshold(tophat, threshold, 255, cv2.THRESH_BINARY)
    
    # 5. Morphological operations
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    # 6. Connected-component analysis to remove small regions
    num_labels, labels, stats = cv2.connectedComponentsWithStats(opened)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            opened[labels == i] = 0
    
    return opened > 0
```

**Evaluation workflow**:

```python
def evaluate(results_dir, threshold=50):
    # 1. Read the reference frame and moving frames
    ref_pre = cv2.imread(ref_pre_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Iterate through all moving frames
    for fname in valid_files:
        pre_img = cv2.imread(pre_path)  # before registration
        post_img = cv2.imread(post_path)  # after registration
        
        # 3. Calculate NCC
        ncc_before = ncc(pre_img, ref_pre)
        ncc_after = ncc(post_img, ref_post)
        
        # 4. Extract vessel masks and calculate DSC
        vessel_mask_pre = extract_vessel_mask(pre_img, threshold)
        vessel_mask_post = extract_vessel_mask(post_img, threshold)
        
        dsc_before = dsc(vessel_mask_pre, vessel_mask_ref)
        dsc_after = dsc(vessel_mask_post, vessel_mask_ref)
        
        # 5. Save vessel-mask visualizations
        cv2.imwrite(f"{fname}_pre_mask.png", vessel_mask_pre * 255)
        cv2.imwrite(f"{fname}_post_mask.png", vessel_mask_post * 255)
    
    # 6. Calculate statistics (mean, lower bound, upper bound)
    stats = {
        "ncc_before_mean": np.mean(ncc_before_list),
        "ncc_after_mean": np.mean(ncc_after_list),
        ...
    }
    
    # 7. Export CSV and statistics files
```

---

### 3.5 Visualization Module (`visualize_and_evaluate.py`)

**Purpose**: provides visual presentation of vessel masks.

**Generated files**:

| File Type | Naming Format | Description |
|----------|----------|------|
| Raw image | `*_raw.png` | Original image without preprocessing |
| Preprocessed image | `*_preprocessed.png` | Image after predictor preprocessing |
| Vessel mask | `*_mask.png` | Vessel region produced by threshold segmentation |
| Overlay | `*_overlay.png` | Vessel mask overlaid on the preprocessed image |

---

## 4. Configuration File

### 4.1 `config/test.yaml`

```yaml
PREDICT:
  device: "cuda:0"              # Compute device
  model_save_path: "model.pth"  # Pretrained model path
  nms_size: 8                   # NMS window size
  nms_thresh: 0.05              # NMS threshold
  knn_thresh: 0.7               # KNN matching threshold
  model_image_width: 640        # Model input width
  model_image_height: 640       # Model input height
```

---

## 5. How to Run

### 5.1 Command Line

```bash
# 1. Preprocessing: quality filtering and reference-frame selection
python pre/01_get_base.py --input_dir /path/to/images --output_dir results

# 2. Registration: register frames to the reference frame in batch
python register_from_base.py

# 3. Predictor preprocessing for vessel-mask generation
python preprocess_filtered.py

# 4. Evaluation: calculate metrics and generate vessel masks
python evaluate_registration.py

# Or complete the post-processing workflow in one step (recommended)
python run_all.py
```

### 5.2 GUI

```bash
python app.py
```

**GUI workflow**:

1. Select the image directory.
2. Select the output directory.
3. Click the "One-click Registration" button.
4. After registration completes, click "Evaluate Registration Results".

---

## 6. Key Technical Points

### 6.1 SuperRetina Model Architecture

```text
+-----------------------------------------------------+
|                  SuperRetina Model                  |
+-----------------------------------------------------+
|                                                     |
|  Input image --> [Shared encoder] --> Feature maps  |
|                    |                                |
|                    +----> [Detector head] --> Keypoint scores |
|                    |                                |
|                    +----> [Descriptor head] --> Feature descriptors |
|                                                     |
+-----------------------------------------------------+
```

### 6.2 Homography Transformation Principle

A homography matrix is a 3x3 matrix used to describe a perspective transformation between two planes:

```text
[ x' ]   [ h11 h12 h13 ] [ x ]
[ y' ] = [ h21 h22 h23 ] [ y ]
[ w' ]   [ h31 h32 h33 ] [ 1 ]

where: x' = x/w', y' = y/w'
```

### 6.3 Inlier Rate Calculation

The inlier rate is the ratio of matching points that conform to the geometric model, namely the homography matrix:

```text
Inlier rate = Number of inliers / Total number of matching points

Mismatch rate = 1 - Inlier rate
```

---

## 7. File Structure Summary

```text
project_SuperRetina_main/
├── common/                    # Shared utility functions
│   ├── common_util.py         # Preprocessing, NMS, and keypoint sampling
│   ├── eval_util.py           # Evaluation metric calculation
│   └── train_util.py          # Training-related utilities
├── config/                    # Configuration files
│   └── test.yaml              # Inference configuration
├── model/                     # Deep learning model
│   ├── super_retina.py        # Main SuperRetina model
│   ├── pke_module.py          # Keypoint enhancement module
│   └── record_module.py       # Recording module
├── pre/                       # Preprocessing scripts
│   ├── get_base.py            # Core reference-frame selection logic
│   └── 01_get_base.py         # Preprocessing entry script
├── predictor.py               # Core model inference class
├── register_from_base.py      # Batch registration workflow
├── evaluate_registration.py   # Registration evaluation module
├── visualize_and_evaluate.py  # Visualization module
├── app.py                     # Main GUI interface
├── gui_worker.py              # GUI worker thread
└── results/                   # Output directory
    ├── filtered/              # Filtered images
    ├── registered_filtered/   # Registered images
    ├── filtered_predictor_preprocessed/  # Predictor-preprocessed images
    ├── vessel_mask_visualization/        # Vessel-mask visualizations
    ├── chessboard/            # Chessboard comparison images
    └── registration_eval.csv  # Evaluation results CSV
```

---

## 8. Extension Notes

### 8.1 Threshold Adjustment

The vessel-mask extraction threshold can be adjusted in the following locations:

1. The `evaluate()` function parameter in `evaluate_registration.py`
2. The `extract_vessel_mask()` function parameter in `vessel_mask.py`

**Recommended threshold ranges**:

- Lower thresholds (30-50): extract more vessel structures but include more noise.
- Higher thresholds (70-90): extract fewer vessel structures but include less noise.

### 8.2 Preprocessing Options

The following options can be configured in `pre/get_base.py`:

- Whether to enable specular-highlight removal through the `remove_specular` parameter
- CLAHE contrast-enhancement parameters
- Image-quality filtering thresholds

### 8.3 Registration Parameters

The following options can be configured in `config/test.yaml`:

- NMS threshold (`nms_thresh`)
- KNN matching threshold (`knn_thresh`)
- Model input size (`model_image_width/height`)

---

**Document version**: v1.0
**Generated date**: May 2026
**Applicable project**: Research on laser speckle fundus video frame image registration algorithms
