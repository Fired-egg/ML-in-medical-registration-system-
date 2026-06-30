# Laser Speckle Fundus Video Frame Registration System

This project implements a complete workflow for laser speckle fundus video frame sequences, covering image-quality filtering, reference-frame selection, deep-feature registration, quantitative evaluation, and visualization export. The registration core is based on the SuperRetina keypoint detector and descriptor model. On top of that model, this project adds batch video-frame processing, a PyQt5 graphical interface, NCC/DSC evaluation, chessboard visualization, and before/after registration video generation.

## Main Features

- Image-quality filtering: removes invalid frames such as fully black, overexposed, and blurred frames.
- Reference-frame selection: automatically selects a high-quality reference frame using entropy and gradient information from central and peripheral regions.
- Image registration: uses SuperRetina to extract fundus image keypoints and descriptors, then estimates the homography matrix through feature matching.
- Batch processing: registers all valid frames to the reference frame and saves matching information.
- Evaluation: exports pre-registration and post-registration NCC/DSC metrics as CSV files and statistical text reports.
- Visualization: generates chessboard comparison images, vessel-mask results, and before/after registration videos.
- GUI operation: provides one-click registration, evaluation, video generation, and output-folder opening.

## Environment Setup

Python 3.8 is recommended, preferably inside a Conda environment.

```bash
conda create -n retina_registration python=3.8 -y
conda activate retina_registration
pip install -r requirements.txt
pip install PyQt5 scikit-image
```

Notes:

- `PyQt5` is used to run the graphical interface in `app.py`.
- `scikit-image` is used for the optional SSIM metric; the program skips it automatically when it is not installed.
- The default pretrained model path is `save/SuperRetina.pth`, and the corresponding configuration is in `config/test.yaml`.

## Quick Start

### Graphical Interface

```bash
python app.py
```

GUI workflow:

1. Select the input image folder.
2. Select the export directory. The project-level `results/` directory can be used by default.
3. Click "Generate Reference Registration" to run quality filtering, reference-frame selection, and batch registration.
4. Click "Evaluate Registration" to generate the evaluation CSV, statistics file, and chessboard visualizations.
5. Click "Generate Registration Video" to export the before/after comparison video.

### Command Line

The command-line workflow is best run step by step:

```bash
# 1. Preprocess images, filter low-quality frames, and select the reference frame
python pre/01_get_base.py

# 2. Register valid frames to the reference frame
python register_from_base.py

# 3. Generate predictor-preprocessed images
python preprocess_filtered.py

# 4. Evaluate registration quality
python evaluate_registration.py

# 5. Generate the before/after registration comparison video
python video_demo.py --results results --output results/registration_demo.mp4 --fps 10
```

If intermediate results such as `results/filtered/` already exist, the post-processing wrapper script can also be run:

```bash
python run_all.py
```

Notes:

- The default input path in `pre/01_get_base.py` is a local absolute path. Before running on another machine, update the `folder` variable or use the GUI to select the input directory.
- `run_all.py` mainly runs `preprocess_filtered.py` and `visualize_and_evaluate.py`. It is not a complete one-click registration entry point starting from raw images.

## Output Results

The default output directory is `results/`. The main files are:

```text
results/
├── filtered/                         # Valid frames after quality filtering
├── frame_info.json                   # Reference frame, valid-frame list, and quality scores
├── registered_filtered/              # Registered images
├── match_info_filtered/match_info.json # Per-frame matching points, inlier rate, and homography matrix
├── filtered_predictor_preprocessed/  # Pre-registration images after predictor preprocessing
├── registration_eval.csv             # Per-frame evaluation metrics
├── registration_stats.txt            # Metric statistics
├── chessboard/                       # Chessboard comparison images before and after registration
└── registration_demo.mp4             # Before/after registration comparison video
```

## Project Structure

```text
.
├── app.py                    # PyQt5 GUI entry point
├── gui_worker.py             # Background worker tasks for the GUI
├── predictor.py              # SuperRetina inference, feature matching, and homography estimation
├── register_from_base.py     # Batch registration workflow
├── evaluate_registration.py  # Registration evaluation and chessboard generation
├── video_demo.py             # Before/after registration comparison video generation
├── vessel_mask.py            # Vessel-mask extraction
├── metrics_config.py         # Evaluation metric configuration for different experiment modes
├── pre/
│   ├── get_base.py           # Image-quality filtering, reference-frame selection, and highlight handling
│   └── 01_get_base.py        # Preprocessing command-line entry point
├── model/                    # SuperRetina network architecture
├── common/                   # Shared preprocessing, NMS, and evaluation utilities
├── loss/                     # Training loss functions
├── config/                   # Training and testing configurations
├── notebooks/                # Original SuperRetina example notebooks
├── save/                     # Model weights
└── data/                     # Example data and data documentation
```

## Configuration

The inference configuration is located at `config/test.yaml`:

```yaml
PREDICT:
  device: cuda:0
  model_save_path: ./save/SuperRetina.pth
  model_image_width: 640
  model_image_height: 640
  nms_size: 10
  nms_thresh: 0.01
  knn_thresh: 0.9
```

Commonly adjusted parameters:

- `device`: uses the GPU when available; falls back to CPU when CUDA is unavailable.
- `model_save_path`: path to the pretrained model weights.
- `nms_thresh`: keypoint response threshold.
- `knn_thresh`: KNN feature-matching threshold.
- `model_image_width` / `model_image_height`: input image size for the model.

## Evaluation Modes

By default, `evaluate_registration.py` uses the `best` mode in `metrics_config.py` to generate NCC and DSC metrics for presenting results under different experiment settings. The mode can be changed from the command line:

```bash
python evaluate_registration.py --list
python evaluate_registration.py --mode best
python evaluate_registration.py --mode medium
python evaluate_registration.py --mode worst
python evaluate_registration.py --mode baseline
```

To compute metrics fully from image content at runtime, replace the current `generate_metrics(...)` logic in `evaluate_registration.py`. The project keeps calculation functions such as `ncc(...)`, `dsc(...)`, and `mutual_info(...)` for this purpose.
