# Brain Tumor MRI Classification and Segmentation

## Project Overview

This project investigates brain tumor classification and approximate localization in MRI images using deep learning and computer vision.

The classification models are:

- Convolutional Neural Network (CNN)
- Vision Transformer (ViT)
- Swin Transformer

Grad-CAM and architecture-appropriate visualization techniques are explored for approximate localization. Segment Anything Model (SAM) is used to generate segmentation masks from bounding-box prompts.

**This is an academic research project and is not intended for clinical diagnosis.**

## Dataset

**Brain Tumor MRI Dataset (Kaggle)**

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Classes:
- Glioma
- Meningioma
- Pituitary
- No Tumor

Dataset organization used in this project:

| Split | Images |
|---|---:|
| Training | 4,480 |
| Validation | 1,120 |
| Testing | 1,600 |

The dataset is not included in this repository. Dataset splits and preprocessing details will be documented alongside the notebooks.

## Methods

### Classification

CNN, ViT and Swin Transformer are used for four-class MRI classification.

Evaluation metrics include accuracy, precision, recall, F1-score and confusion matrices.

### Approximate Localization

Grad-CAM-based activation maps are explored to identify image regions contributing to model predictions.

Classification accuracy does not imply localization accuracy.

### Segmentation with SAM

A pretrained SAM ViT-B model generates masks using bounding-box prompts.

Two approaches are investigated:

1. Automatic bounding boxes derived from Grad-CAM.
2. Manually specified bounding boxes for comparison.

SAM confidence scores are not treated as ground-truth segmentation accuracy.

## Current Swin Transformer Results

| Metric | Test Result |
|---|---:|
| Accuracy | 93.25% |
| Precision (macro) | 93.88% |
| Recall (macro) | 93.25% |
| F1-score (macro) | 93.19% |

The Swin Transformer was fine-tuned using pretrained weights.

## Project Status

- [x] Swin Transformer training and evaluation
- [x] Initial Grad-CAM experiments
- [x] Initial SAM segmentation experiments
- [ ] CNN and ViT results integration
- [ ] Localization and segmentation evaluation
- [ ] Final comparison and documentation

## Implementation

Development environment: Google Colab

Main framework: PyTorch

Additional libraries include timm, pytorch-grad-cam, OpenCV, scikit-learn and Segment Anything.

Runnable notebooks and detailed usage instructions will be added as the project progresses.

## Limitations

The classification dataset provides image-level class labels but does not provide ground-truth tumor segmentation masks.

Therefore, localization and segmentation performance cannot be established from classification accuracy or SAM's predicted mask-quality scores alone.

## Team

Two-person university course project.

## Submission

24 November 2026.
