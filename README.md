
# 📏 Estimating Distance in 2D Images using Deep Learning

## 1️⃣ Introduction

The ability to accurately determine specific points or distances in 2D images is critical in a wide range of applications, from sports analytics to industrial measurement systems. In this project, our objective was to predict **two key x-coordinates (x1 and x2)** representing jump positions in a set of images. These coordinates are essential for calculating distances and evaluating object positions with high precision.

This work was conducted across **three datasets**:

1. **Array1** – The initial baseline, using simpler preprocessing and basic deep learning architecture.
2. **Array2** – Improved preprocessing pipeline with a refined ResNet50V2 backbone.
3. **Array3** – Optimized final pipeline, incorporating advanced data augmentation, label normalization, and a two-phase training approach.

Our **primary target** was to reduce the **Mean Absolute Error (MAE)** to **less than 2 pixels** on Array3, while also improving performance across Array1 and Array2.

---

## 2️⃣ Project Objectives

* **Predict** x1 and x2 coordinates accurately for each image.
* **Minimize MAE** between predicted and true coordinates.
* Improve from **baseline MAE (\~45 pixels)** to **< 2 pixels** on the final dataset.
* Experiment with **progressive improvements** across datasets.
* Maintain **generalization** while reducing overfitting.

---

## 3️⃣ Dataset Overview

### **3.1 Array1**

* **Description**: The starting dataset with basic cropped `.npy` images.
* **Shape**: Resized to smaller dimensions (initially 64×128 in some trials).
* **Labels**: Pixel positions of x1 and x2 coordinates.
* **Challenge**: Limited preprocessing and no label normalization.

### **3.2 Array2**

* **Description**: Refined version of Array1 with improved cropping and resizing.
* **Shape**: 256×256 RGB images.
* **Labels**: Still in pixel space, but better aligned with resized images.
* **Challenge**: Higher variance in predictions for x2.

### **3.3 Array3**

* **Description**: Final dataset with optimized processing pipeline.
* **Shape**: 384×384 RGB images.
* **Preprocessing**:

  * Albumentations for augmentation (flips, rotations, brightness adjustments).
  * Labels normalized by **width** (important for different image widths).
* **Challenge**: Achieve **MAE < 2 px** while keeping stability during training.

---

## 4️⃣ Methodology

Our methodology evolved through **three phases** aligned with each dataset.

---

### **4.1 Data Preprocessing**

**Steps for all datasets**:

* Load `.npy` grayscale images.
* Convert to RGB if required by model.
* Resize to target resolution.
* Normalize pixel values (0–1 range).
* Split into training and validation sets.

**Array-specific enhancements**:

* **Array2**: More accurate cropping based on bounding boxes.
* **Array3**: Width-normalized labels and Albumentations augmentations.

---

### **4.2 Model Architecture**

We used **ResNet50V2** as our backbone, pretrained on ImageNet, followed by:

* **GlobalAveragePooling2D** layer to reduce spatial dimensions.
* **Dense layers** for regression outputs.
* **Dropout** for regularization.
* **Output layer** with **linear activation** (regression).

**Loss functions**:

* MAE for baseline runs.
* Huber loss for stability in Array3.

**Optimizer**:

* Adam with ReduceLROnPlateau (adaptive learning rate).

---

### **4.3 Training Strategy**

**Phase 1 – Frozen Base**

* Freeze all ResNet50V2 layers.
* Train top layers for 50–100 epochs with higher learning rate.

**Phase 2 – Fine-Tuning**

* Unfreeze ResNet50V2 base.
* Train with low learning rate for fine-tuning.
* Early stopping to prevent overfitting.

---

### **4.4 Evaluation Metrics**

* **Mean Absolute Error (MAE)**: Primary metric for measuring positional accuracy.
* **Root Mean Squared Error (RMSE)**: Secondary metric for identifying larger errors.
* **Error Distribution**: Mean error, standard deviation, histograms.

---

## 5️⃣ Results and Analysis

---

### **5.1 Array1 – Final Evaluation**

| Metric | Value       |
| ------ | ----------- |
| MAE    | **4.02 px** |
| RMSE   | **6.67 px** |

**Interpretation**:

* MAE of \~4 pixels means predictions are close to true positions, especially compared to original MAE of \~45 pixels.
* RMSE suggests some larger errors, but most are within acceptable range.
* Significant improvement due to better preprocessing and model tuning.
<img width="821" height="453" alt="{10407359-AC23-4656-AECF-AD28DDF9C39F}" src="https://github.com/user-attachments/assets/15561150-546f-4d5a-979c-ab0eea7c68bd" />
<img width="735" height="558" alt="{2B8830E8-5034-4DE8-B704-7CDE9E63035C}" src="https://github.com/user-attachments/assets/7229b88a-55f7-4bf2-8734-d336b48541d5" />
<img width="162" height="108" alt="{55A6186D-7858-4D23-B353-C18079F21830}" src="https://github.com/user-attachments/assets/e9736607-a968-4159-bb4a-7c3eae35d8b3" />

---

### **5.2 Array2 – Final Evaluation**

| Metric      | Value       |
| ----------- | ----------- |
| MAE (x1)    | **3.16 px** |
| MAE (x2)    | **4.21 px** |
| Overall MAE | **3.69 px** |

**Interpretation**:

* x1 predictions are close to goal (<3 px).
* x2 still has slightly higher error but remains in strong range.
* Performance improved, but more optimization needed for x2.
<img width="834" height="559" alt="{56CEFC15-B0BB-4EA2-95CD-079DA16C7487}" src="https://github.com/user-attachments/assets/cac6c1e5-a7b6-42c8-b69f-092dbbcba472" />
<img width="657" height="304" alt="{CF280264-51B4-424A-8A02-32F9A9AD1E14}" src="https://github.com/user-attachments/assets/87d01e8a-befe-453d-b03e-6d63af92870e" />
<img width="304" height="118" alt="{1D65ADBC-A57A-4B46-BECF-13C53DFC1515}" src="https://github.com/user-attachments/assets/b6283ca4-7676-413b-8e82-99484258de9c" />

---

### **5.3 Array3 – Final Evaluation**

| Metric      | Value       |
| ----------- | ----------- |
| MAE (x1)    | **1.99 px** |
| MAE (x2)    | **1.79 px** |
| Overall MAE | **1.89 px** |

**Interpretation**:

* Target achieved: both coordinates under 2 px MAE.
* Predictions are typically within ±2 px of ground truth.
* No strong bias toward over/under-prediction.
* In resized 256 px images: \~0.74% relative error.
<img width="318" height="121" alt="{E2C51952-FDC3-43D2-B6C4-8DD5079CB16C}" src="https://github.com/user-attachments/assets/5a675a9d-6b9b-409f-8b79-12e67dff6811" />

---

## 6️⃣ Visualizations

* **ARRAY 3 FINAL VISUALISATIONS**
<img width="840" height="561" alt="{6BDEFF2A-CD92-4B09-B9E7-01AF04355AF7}" src="https://github.com/user-attachments/assets/08c81936-e2c8-41f2-a6cd-4c7a5d750c0d" />
<img width="672" height="550" alt="{F1CEDC76-B10C-438E-B36B-A0DF7D1BC3C4}" src="https://github.com/user-attachments/assets/646541fb-7f27-4763-affc-4c5fa0376c49" />
<img width="676" height="763" alt="{0406522D-2B72-4A20-8CE8-FA7C0D0CCFD4}" src="https://github.com/user-attachments/assets/f0b2e98a-14bb-475b-beca-f9107d27b7e9" />

---

## 7️⃣ Key Learnings

* **Label normalization** is crucial for handling varying image widths.
* **Two-phase training** improves performance without overfitting.
* **Albumentations** provides diverse augmentation while preserving geometry.
* A low MAE in resized image space translates to excellent accuracy in original resolution.

---

## 8️⃣ Future Work

* Integrate **YOLOv8** for bounding box prediction (direct detection of x1, x2).
* Combine **image + metadata** for richer feature learning.
* Extend pipeline to real-world datasets with greater variability.
* Experiment with EfficientNet variants for faster inference.

---

## 9️⃣ Conclusion

Through a series of iterative improvements across three datasets, we reduced MAE from **\~45 pixels** in early baselines to **1.89 pixels** on Array3. This was achieved via:

* Optimized preprocessing.
* Normalized labels.
* Two-phase training with fine-tuning.
* Strategic data augmentation.

The final model delivers **high precision** predictions and meets the ambitious sub-2 px target, making it suitable for deployment in real-world coordinate estimation tasks.

---


