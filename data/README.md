# Data

## Datasets

### Training

- Lab
- Auxiliary

### Test Set for Image Registration

- FIRE: [https://projects.ics.forth.gr/cvrl/fire/](https://projects.ics.forth.gr/cvrl/fire/)

### Test Sets for Identity Verification

- VARIA: [http://www.varpa.es/research/biometrics.html](http://www.varpa.es/research/biometrics.html)
- BES
- CLINICAL

## Data Organization

### Training Data

The semi-supervised training code reads a *labeled* training dataset (`Lab` in the ECCV paper) and an *unlabeled* training dataset (`Auxiliary`). Each training dataset is expected to be organized as follows:

```text
Lab/
    Annotations/
        vistel0_left_0.txt
        vistel0_left_1.txt
        ....
    ImageData/
        vistel0_left_0.jpg
        vistel0_left_1.jpg
        ....
    ImageSets/
        eccv22_train.txt
        eccv22_val.txt
        Lab.txt
Auxiliary/
    image1.jpg
    image2.jpg
    ...
```

- The `Annotations` folder contains keypoint annotations for each image. This folder is optional for unlabeled data. A sample annotation file is provided at [samples/vistel0_left_0.txt](samples/vistel0_left_0.txt). See the [tutorial code](../notebooks/read_keypoint_labels.ipynb) for details on how keypoint annotations should be stored and loaded.
- The `ImageData` folder contains all image files.
- The `ImageSets` folder contains image-id files that define the data split. See [eccv22_train.txt](./Lab/ImageSets/eccv22_train.txt), [eccv22_val.txt](./Lab/ImageSets/eccv22_val.txt), and [lab.txt](./Lab/ImageSets/lab.txt).

### Test Data

The test data is organized as follows:

```text
FIRE/
    Ground Truth/
    Images/
    Masks/

VARIA/
    Images/
        R001.pgm
        R002.pgm
        ...
    pair_index.txt
```

- The annotation file `control_points_P37_1_2.txt` in the `FIRE` dataset is incorrect and should be excluded from evaluation.

For the identity verification task, `pair_index.txt` is used to define matching pairs in the dataset.

```python
# Each line in the index file has three columns:
# query_image, reference_image, 0 (reject) or 1 (accept)

# Example:
R180.pgm, R002.pgm, 1
R012.pgm, R002.pgm, 0
```
