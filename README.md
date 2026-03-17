# Brain Tumor Segmentation with 3D U-Net (BraTS 2020)

This project implements a 3D deep learning pipeline for **binary brain tumor segmentation** using multi-modal MRI scans from the BraTS 2020 dataset on Kaggle.

The model takes four MRI modalities as input:

- FLAIR  
- T1  
- T1ce  
- T2  

The original segmentation masks are converted into a **binary tumor mask**:

- 0 = background  
- 1 = tumor (any region)

---

## Pipeline Overview

The workflow consists of three stages:

1. Patient-level preprocessing  
2. Patch generation  
3. Model training with a 3D U-Net  

---
## Exploratory Data Analysis

An exploratory analysis notebook was used to better understand the dataset and validate preprocessing steps.

Key analyses included:

- visualization of MRI modalities and segmentation masks across slices
- identification of slices containing tumor regions
- intensity distribution analysis across modalities
- verification of cropping and normalization steps
- inspection of patch shapes and tumor voxel distribution
- validation of data augmentation effects

The notebook supports preprocessing decisions and helps ensure data consistency before training.

## Preprocessing

For each patient:

- load MRI volumes and segmentation mask (`.nii` files)
- remove low-intensity noise (values < 10 → 0)
- compute a brain mask from nonzero voxels
- crop to the brain region
- apply z-score normalization to nonzero voxels (per modality)

## Patch Generation

- segmentation masks are converted to **binary**
- volumes are padded if needed
- a **centered 128 × 128 × 128 patch** is extracted per patient

## Data Augmentation

Applied during training only:

**Spatial**
- random flip (one axis)
- random 90° rotation (random plane)

**Intensity**
- random intensity shift
- Gaussian noise

---

## Model

A custom **3D U-Net** implemented in PyTorch:

- 4 input channels  
- encoder–decoder architecture  
- skip connections  
- transposed convolution upsampling  
- 1 output channel (binary segmentation)

---

## Loss Function

Combined loss:

- Binary Cross Entropy (BCEWithLogitsLoss)  
- Dice loss  

---

## Training Configuration

- Epochs: 30  
- Batch size: 1  
- Learning rate: 1e-4  
- Train/validation split: 80/20 (patient-level)  
- Augmentation: enabled  
- Device: MPS / CUDA / CPU (auto-detected)

---

## Results

| Metric | Value |
|------|------|
| Best Validation Dice | **0.8848** |
| Final Training Dice | 0.9111 |
| Final Validation Dice | 0.8595 |

---

## Observations

- Training converges steadily across epochs  
- Validation performance is slightly lower but follows a similar trend  
- Dice fluctuations suggest variability across patient cases  
- Model generalizes reasonably well given limited patch-based training  

---

## Limitations

- Binary segmentation only (not multiclass BraTS labels)  
- Single centered patch per patient (may miss edge cases)  
- No cross-validation  
- Performance depends on preprocessing and tumor distribution  

---

## Future Work

- tumor-centered patch sampling  
- multiclass segmentation (ET, TC, WT)  
- class imbalance handling  
- cross-validation  
- architecture improvements  

---

## Tech Stack

- Python  
- PyTorch  
- NumPy  
- nibabel   
- Matplotlib  
- KaggleHub  

### Using Makefile

Run the full pipeline:


make all


#### Or run individual steps:


make preprocess
make patches
make train
make plot

## Setup

Install dependencies:

```bash
pip install -r requirements.txt