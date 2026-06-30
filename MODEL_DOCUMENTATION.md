# Detailed Architecture of the SuperRetina Deep Learning Model

## 1. Model Overview

SuperRetina is a deep-learning-based model for fundus image keypoint detection and descriptor extraction. It is designed for laser speckle fundus video frame registration. The model uses **Point Keypoint Enhancement (PKE)**, a semi-supervised keypoint enhancement technique that can automatically learn effective keypoints without requiring large amounts of manual annotation.

---

## 2. Overall Architecture

### 2.1 Model Structure Diagram

```text
+-------------------------------------------------------------------------+
|                         SuperRetina Model Architecture                   |
+-------------------------------------------------------------------------+
|                                                                         |
|  Input image (1 x H x W)                                                |
|        |                                                                |
|        v                                                                |
|  +-------------------------------------------------+                    |
|  |              Shared Encoder                     |                    |
|  |  Conv1a -> Conv1b -> Pool -> Conv2a -> Conv2b   |                    |
|  |    v                                            |                    |
|  |  Pool -> Conv3a -> Conv3b -> Pool -> Conv4a -> Conv4b                |
|  +---------------+---------------------------------+                    |
|                  |                                                      |
|          +-------+-------+                                              |
|          v               v                                              |
|  +--------------+  +------------------+                                 |
|  | Detector Head|  | Descriptor Head  |                                 |
|  | (Detector)   |  | (Descriptor)     |                                 |
|  +--------------+  +------------------+                                 |
|  | Upsample     |  | ConvDa           |                                 |
|  | + Skip       |  | ConvDb (stride=2)|                                 |
|  | Connections  |  | ConvDc           |                                 |
|  |              |  | TransConv        |                                 |
|  | dconv_up3    |  | L2 Normalize     |                                 |
|  | dconv_up2    |  +------------------+                                 |
|  | dconv_up1    |          |                                           |
|  | conv_last    |          v                                           |
|  | Sigmoid      |   Descriptor output (256 dimensions)                  |
|  +------+-------+                                                      |
|         v                                                               |
|    Keypoint probability map (1 x H x W)                                 |
|                                                                         |
+-------------------------------------------------------------------------+
```

### 2.2 Module Functions

| Module | Function | Output |
|------|------|------|
| Shared encoder | Extracts hierarchical image features | Multi-scale feature maps |
| Detector head | Predicts keypoint-location probabilities | Keypoint probability map |
| Descriptor head | Generates feature descriptors for keypoints | 256-dimensional descriptors |
| PKE module | Performs semi-supervised keypoint enhancement | Enhanced keypoint labels |

---

## 3. Shared Encoder

### 3.1 Network Structure

```python
# Channel configuration
c1, c2, c3, c4 = 64, 64, 128, 128

# Encoder layers
self.conv1a = Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
self.conv1b = Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)

self.conv2a = Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
self.conv2b = Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)

self.conv3a = Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
self.conv3b = Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)

self.conv4a = Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
self.conv4b = Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
```

### 3.2 Feature Extraction Flow

| Level | Input Size | Operation | Output Size | Channels |
|------|----------|------|----------|--------|
| Conv1 | 1 x H x W | Conv -> ReLU -> Conv -> ReLU -> MaxPool | c1 x H/2 x W/2 | 64 |
| Conv2 | c1 x H/2 x W/2 | Conv -> ReLU -> Conv -> ReLU -> MaxPool | c2 x H/4 x W/4 | 64 |
| Conv3 | c2 x H/4 x W/4 | Conv -> ReLU -> Conv -> ReLU -> MaxPool | c3 x H/8 x W/8 | 128 |
| Conv4 | c3 x H/8 x W/8 | Conv -> ReLU -> Conv -> ReLU | c4 x H/8 x W/8 | 128 |

### 3.3 Design Characteristics

- **ReLU activation**: introduces nonlinearity and improves representation capacity.
- **MaxPool downsampling**: gradually reduces feature-map size and enlarges the receptive field.
- **Two-convolution stage design**: each stage contains two convolution layers, increasing model depth.

---

## 4. Detector Head

### 4.1 Network Structure

```python
# Upsampling decoder
self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

self.dconv_up3 = double_conv(c3 + c4, c3)  # 128+128 -> 128
self.dconv_up2 = double_conv(c2 + c3, c2)  # 64+128 -> 64
self.dconv_up1 = double_conv(c1 + c2, c1)  # 64+64 -> 64

self.conv_last = nn.Conv2d(c1, n_class, kernel_size=1)
```

### 4.2 Decoder Flow

```text
Conv4 feature map (128 x H/8 x W/8)
        |
        v Upsample (x2)
        |
        +--> concatenate with Conv3 feature map (128+128=256)
        v double_conv -> 128 channels
        |
        v Upsample (x2)
        |
        +--> concatenate with Conv2 feature map (128+64=192)
        v double_conv -> 64 channels
        |
        v Upsample (x2)
        |
        +--> concatenate with Conv1 feature map (64+64=128)
        v double_conv -> 64 channels
        |
        v conv_last (1x1) -> Sigmoid
        |
        v
    Keypoint probability map (1 x H x W)
```

### 4.3 Skip-Connection Mechanism

Skip connections concatenate intermediate feature maps from the encoder with the corresponding decoder layers, preserving fine-grained detail:

```python
cPa = self.upsample(x)           # Upsampling
cPa = torch.cat([cPa, conv3], dim=1)  # Concatenate with encoder features
cPa = self.dconv_up3(cPa)        # Convolutional fusion
```

---

## 5. Descriptor Head

### 5.1 Network Structure

```python
c5, d1, d2 = 256, 256, 256

self.convDa = Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)  # 128 -> 256
self.convDb = Conv2d(c5, d1, kernel_size=4, stride=2, padding=0)  # 256 -> 256
self.convDc = Conv2d(d1, d2, kernel_size=1, stride=1, padding=0)  # 256 -> 256

self.trans_conv = nn.ConvTranspose2d(d1, d2, 2, stride=2)  # Upsampling
```

### 5.2 Descriptor Generation Flow

```python
def network(self, x):
    # ... encoder forward pass ...
    
    # Descriptor head
    cDa = self.relu(self.convDa(x))      # Convolution 1
    cDb = self.relu(self.convDb(cDa))    # Convolution 2 with downsampling
    desc = self.convDc(cDb)              # Convolution 3 (1x1)
    
    # L2 normalization
    dn = torch.norm(desc, p=2, dim=1)
    desc = desc.div(torch.unsqueeze(dn, 1))
    
    # Transposed-convolution upsampling
    desc = self.trans_conv(desc)
    
    return semi, desc
```

### 5.3 Key Design Choices

| Operation | Role |
|------|------|
| L2 normalization | Ensures descriptors have unit length for easier similarity comparison |
| Transposed convolution | Restores the descriptor map to 1/8 of the original image size |
| 1x1 convolution | Adjusts the channel count while preserving spatial dimensions |

---

## 6. Point Keypoint Enhancement (PKE)

### 6.1 How PKE Works

PKE is the core innovation of SuperRetina. It automatically enhances keypoint labels through geometric transformation and content matching:

```text
+-------------------------------------------------------------+
|                         PKE Learning Flow                   |
+-------------------------------------------------------------+
|                                                             |
|  Original image --> [Detector] --> Keypoint probability map  |
|       |                              |                      |
|       v                              v                      |
|  [Affine transform]             [NMS extraction]             |
|       |                              |                      |
|       v                              v                      |
|  Affine image --> [Detector] --> Affine keypoint probability map |
|       |                              |                      |
|       +------------+-----------------+                      |
|                    v                                        |
|          +------------------+                               |
|          | Geometric filter | <-- Spatial-consistency check  |
|          +--------+---------+                               |
|                   v                                         |
|          +------------------+                               |
|          | Content filter   | <-- Descriptor-similarity check|
|          +--------+---------+                               |
|                   v                                         |
|          +------------------+                               |
|          | Value-map update | <-- Record learned keypoints   |
|          +--------+---------+                               |
|                   v                                         |
|          Enhanced keypoint labels                           |
|                                                             |
+-------------------------------------------------------------+
```

### 6.2 Geometric Filter

```python
def geometric_filter(affine_detector_pred, points, affine_points, 
                     max_num=1024, geometric_thresh=0.5):
    """
    Check whether each keypoint is also detected at the corresponding
    position after affine transformation.
    """
    geo_points = []
    for s, k in enumerate(affine_points):
        # Sample detection probabilities at corresponding positions in the affine image
        sample_aff_values = affine_detector_pred[s, 0, k[:, 1].long(), k[:, 0].long()]
        # Keep keypoints whose probabilities exceed the threshold
        check = sample_aff_values.squeeze() >= geometric_thresh
        geo_points.append(points[s][check][:max_num])
    
    return geo_points
```

### 6.3 Content Filter

```python
def content_filter(descriptor_pred, affine_descriptor_pred, geo_points, 
                   affine_geo_points, content_thresh=0.7, scale=8):
    """
    Check whether keypoints match their affine correspondences in descriptor space.
    """
    # Sample keypoint descriptors
    descriptors = [sample_keypoint_desc(k[None], d[None], scale)[0] 
                   for k, d in zip(geo_points, descriptor_pred)]
    
    # Calculate descriptor distances
    dist = [torch.norm(descriptors[d][:, None] - aff_descriptors[d], dim=2, p=2)
            for d in range(len(descriptors))]
    
    for i in range(len(dist)):
        D = dist[i]
        # Top-2 nearest neighbors
        val, ind = torch.topk(D, 2, dim=1, largest=False)
        
        # Rule 1: spatial correspondence, where the nearest neighbor is itself
        c1 = ind[:, 0] == arange
        
        # Rule 2: ratio test, where nearest distance < second-nearest distance * threshold
        c2 = val[:, 0] < val[:, 1] * content_thresh
        
        # Keep only points that satisfy both rules
        check = c2 * c1
```

### 6.4 Value Map

The value map records the learning history of keypoints and improves the stability of keypoint detection:

```python
def update_value_map(value_map, content_points, config):
    """
    Update the value map and record keypoint confidence.
    """
    # Increase confidence for newly detected keypoints
    # Decrease confidence for keypoints that are not detected
    # Keep only keypoints whose confidence exceeds the threshold
```

---

## 7. Loss Functions

### 7.1 Detector Loss (Dice Loss)

```python
loss_detector = loss_cal(detector_pred, enhanced_label)  # L_geo
loss_clf = loss_cal(detector_pred, affine_pred_inverse)  # L_clf
loss = loss_detector + loss_clf
```

- **L_geo**: Dice loss between detector prediction and enhanced labels.
- **L_clf**: Consistency loss between detector prediction and inverse affine-transformed prediction.

### 7.2 Descriptor Loss (Triplet Loss)

```python
def descriptor_loss(...):
    # Build triplets: anchor, positive, negative
    positive = affine_descriptor[:, 0, :].permute(1, 0)
    anchor = descriptor[:, :, 0].permute(1, 0)
    
    # Hard-negative mining: nearest neighbor that does not match
    with torch.no_grad():
        dis = torch.norm(descriptor - affine_descriptor, dim=0)
        dis[ar, ar] = dis.max() + 1  # Exclude positive samples
        neg_index1 = dis.argmin(axis=1)  # Hardest negative sample
    
    # Random negative samples
    neg_index2 = [random.randint(0, n-1) for _ in range(n)]
    
    # Triplet Margin Loss
    loss = triplet_margin_loss_gor(anchor, positive, negatives_hard, negatives_random, margin=0.8)
```

**Triplet construction strategy**:

- **Anchor**: keypoint descriptor on the original image.
- **Positive**: descriptor at the corresponding position after affine transformation.
- **Negative (Hard)**: hardest negative sample, which is closest but not corresponding.
- **Negative (Random)**: random negative sample.

---

## 8. Forward Pass

### 8.1 Training Stage

```python
def forward(self, x, label_point_positions=None, value_map=None, learn_index=None):
    # 1. Base network forward pass
    detector_pred, descriptor_pred = self.network(x)
    
    if label_point_positions is not None:
        # 2. Affine transformation augmentation
        with torch.no_grad():
            affine_x, grid, grid_inverse = affine_images(x, used_for='detector')
            affine_detector_pred, affine_descriptor_pred = self.network(affine_x)
        
        # 3. PKE keypoint enhancement if enabled
        if self.PKE_learn:
            loss_detector, number_pts, value_map_update, enhanced_label_pts, enhanced_label = \
                pke_learn(detector_pred[learn_index], descriptor_pred[learn_index],
                          grid_inverse[learn_index], affine_detector_pred[learn_index],
                          affine_descriptor_pred[learn_index], ...)
        
        # 4. Calculate descriptor loss
        loss_descriptor, _ = self.descriptor_loss(detector_pred_copy, label_point_positions,
                                                  descriptor_pred, affine_descriptor_pred_for_desc, ...)
        
        # 5. Total loss
        loss = loss_detector + loss_descriptor
        
        return loss, number_pts, loss_detector, loss_descriptor, ...
    
    # During inference, only return prediction results
    return detector_pred, descriptor_pred
```

### 8.2 Inference Stage

```python
def model_run_pair(self, query_tensor, refer_tensor):
    # 1. Model inference
    inputs = torch.cat((query_tensor.unsqueeze(0), refer_tensor.unsqueeze(0)))
    detector_pred, descriptor_pred = self.model(inputs)
    
    # 2. Extract keypoints with NMS
    scores = simple_nms(detector_pred, self.nms_size)
    keypoints = [torch.nonzero(s > self.nms_thresh) for s in scores]
    
    # 3. Remove border points
    keypoints, scores = remove_borders(keypoints, scores, 4, h, w)
    
    # 4. Sample descriptors
    descriptors = [sample_keypoint_desc(k[None], d[None], 8)[0].cpu()
                   for k, d in zip(keypoints, descriptor_pred)]
    
    return keypoints, descriptors
```

---

## 9. Key Configuration Parameters

| Parameter | Default | Role |
|------|--------|------|
| nms_size | 8 | NMS window size |
| nms_thresh | 0.05 | NMS threshold |
| knn_thresh | 0.7 | KNN matching threshold |
| geometric_thresh | 0.5 | Geometric matching threshold |
| content_thresh | 0.7 | Content matching threshold |
| gaussian_kernel_size | 15 | Gaussian kernel size |
| gaussian_sigma | 2.0 | Gaussian standard deviation |

---

## 10. Summary of Model Characteristics

### 10.1 Innovations

1. **Semi-supervised learning**: only a small amount of annotation is needed, while PKE automatically enhances keypoints.
2. **End-to-end training**: the detector and descriptor are jointly optimized.
3. **Geometric consistency**: affine transformation is used to ensure geometric stability of keypoints.
4. **Content consistency**: descriptor matching is used to ensure semantic consistency of keypoints.

### 10.2 Technical Advantages

| Aspect | Advantage |
|------|------|
| Data efficiency | Does not require large-scale manual keypoint annotation |
| Robustness | PKE enhancement improves keypoint detection stability |
| Accuracy | Detector and descriptor are optimized jointly |
| Generalization | Geometric and content constraints improve cross-frame matching |

### 10.3 Applicable Scenarios

- Laser speckle fundus video frame registration
- Retinal image keypoint detection
- Medical image registration tasks

---

**Document version**: v1.0
**Generated date**: May 2026
**Applicable model**: SuperRetina
