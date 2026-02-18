# Advanced-Object-Tracking-and-Detection-in-Video-Streams

Developed an end-to-end multi-object tracking system using Faster R-CNN for pedestrian detection and SORT (Kalman Filter + Hungarian matching) for identity tracking. Evaluated on MOT17 with mAP, MOTA, IDF1, precision, and recall. Achieved stable temporal consistency across 600 video frames.

---

## Project Overview

This project implements a complete end-to-end multi-object detection and tracking pipeline using:

- Faster R-CNN for pedestrian detection  
- SORT (Simple Online and Realtime Tracking) for identity tracking  
- Kalman Filter + Hungarian Algorithm for temporal consistency  
- Full evaluation on MOT17-02 (600 frames)  
- Standard MOT metrics: mAP, MOTA, IDF1, ID Switches, Precision, Recall  

The objective is to accurately detect and maintain consistent identities of multiple objects across video frames, even in dynamic and crowded environments.

---

## Dataset

Dataset used: MOT Challenge – MOT17  

- Sequence: MOT17-02-FRCNN  
- 600 frames  
- Pedestrian bounding boxes  
- Unique object IDs across frames  
- Ground truth annotations in `gt/gt.txt`  

Official dataset link:  
https://motchallenge.net/data/

---

## System Pipeline

The complete pipeline consists of five major components:

1. Data Preprocessing  
2. Faster R-CNN Model Development  
3. SORT-Based Multi-Object Tracking  
4. Evaluation using MOT Metrics  
5. Visualization and Performance Analysis  

---

## 1. Data Preprocessing

A custom `MOTDataset` class was implemented to:

- Load image frames from `img1/`  
- Read ground truth from `gt/gt.txt`  
- Filter valid annotations (`conf == 1`)  
- Convert bounding boxes into `[x1, y1, x2, y2]` format  
- Associate each bounding box with its corresponding object ID  

### Transformations Applied

```python
transforms.ToPILImage()
transforms.ToTensor()
```

Images are converted into tensors and normalized for Faster R-CNN input.

---

## 2. Faster R-CNN Model Development

### Model Architecture

The detection backbone used:

```python
fasterrcnn_resnet50_fpn(weights="DEFAULT")
```

### Key Components

- ResNet-50 Backbone  
- Feature Pyramid Network (FPN)  
- Region Proposal Network (RPN)  
- ROI Heads  
- Custom classification head (2 classes: background + pedestrian)  

The final classification layer is replaced using:

```python
FastRCNNPredictor(in_features, num_classes=2)
```

### Training Configuration

- Optimizer: Adam  
- Learning Rate: 1e-4  
- Gradient Clipping: 5.0  
- Epochs: 5  
- Batch Size: 2  

### Training Process

For each epoch:

- Forward pass through Faster R-CNN  
- Compute classification + regression losses  
- Backpropagation  
- Gradient clipping  
- Parameter updates  

Loss values are stored and plotted to verify convergence behavior.

---

## 3. Multi-Object Tracking with SORT

SORT (Simple Online and Realtime Tracking) is implemented fully from scratch.

The tracking system integrates:

- Kalman Filter for motion prediction  
- IOU-based matching  
- Hungarian Algorithm for optimal assignment  
- Track lifecycle management  

### Kalman Filter State Representation

Each object is modeled with a 7D state vector:

```python
[x, y, s, r, vx, vy, vs]
```

Where:

- x, y → center coordinates  
- s → bounding box scale (area)  
- r → aspect ratio  
- vx, vy, vs → velocity terms  

This enables motion prediction across frames and improves temporal stability.

---

### Data Association Strategy

- Predict positions of existing trackers  
- Compute IOU matrix between predictions and new detections  
- Apply Hungarian algorithm  
- Reject matches below IOU threshold  
- Create new trackers for unmatched detections  
- Remove trackers exceeding `max_age`  

Parameters used:

```python
max_age = 10
min_hits = 3
iou_threshold = 0.3
```

This ensures:

- Reduced identity switching  
- Temporal consistency  
- Stable bounding box smoothing  

---

## 4. Evaluation Metrics

The system evaluates both detection and tracking performance.

### Detection Metric

- mAP@0.5  

Computed using:

```python
average_precision_score
```

### Tracking Metrics (MOT Metrics)

Computed using:

```python
motmetrics
```

- MOTA (Multiple Object Tracking Accuracy)  
- IDF1 (Identity F1 Score)  
- ID Switches  
- Precision  
- Recall  

---

## Results

### Detection Performance

```python
mAP@0.5 = 0.9998
```

Since the model was trained and evaluated on the same sequence, detection performance is extremely high.

---

### Tracking Performance

| Metric | Score |
|--------|--------|
| MOTA | 0.8942 |
| IDF1 | 0.7846 |
| ID Switches | 81 |
| Precision | 0.9673 |
| Recall | 0.9301 |

### Interpretation

- High MOTA indicates strong overall tracking performance.  
- IDF1 shows reasonable identity preservation.  
- ID Switches indicate occasional identity inconsistencies in challenging scenarios.  
- High precision confirms minimal false positives.  
- Strong recall reflects effective object coverage.  

---

## Training Loss Analysis

The training loss decreases consistently across epochs, indicating:

- Stable convergence  
- Effective bounding box regression  
- Improved classification confidence  

---

## Visualization

The system visualizes tracked frames by:

- Drawing bounding boxes  
- Displaying track IDs  
- Maintaining identity continuity  

Each bounding box includes:

```python
ID: <track_number>
```

This verifies temporal identity tracking across frames.

---

## Technical Highlights

- Complete end-to-end detection + tracking pipeline  
- Full Kalman Filter implementation  
- Hungarian IOU matching  
- Full MOT evaluation  
- Numerical stability fixes for NumPy 2.x  
- Real multi-object lifecycle handling  
- Evaluated on full 600-frame sequence  

---

## Limitations

- Trained on only one MOT17 sequence  
- Limited epochs (5)  
- No appearance-based tracking (DeepSORT not used)  
- No learning rate scheduler  
- Minimal data augmentation  

---

## Future Improvements

- Train on full MOT17 dataset  
- Increase training epochs (10–20)  
- Integrate DeepSORT for appearance embeddings  
- Multi-scale data augmentation  
- Hyperparameter tuning  
- Replace IOU matching with learned association  

---

## Conclusion

This project successfully implements a complete advanced multi-object detection and tracking system using:

- Faster R-CNN for detection  
- SORT (Kalman + Hungarian) for tracking  
- Standard MOT evaluation metrics  
- Full-sequence analysis  

The system demonstrates strong detection precision, stable identity tracking, and a robust integration of deep learning with classical state estimation techniques.

