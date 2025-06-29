# TFG – Interpretability Techniques on Social Media Images

## Overview

This project investigates how digital relational values (DRVs) related to nature can be extracted and explained from social media images using deep learning and interpretability techniques. It is structured into a multitask image classification pipeline combined with a comprehensive evaluation of interpretability methods to align model decisions with human reasoning.

## Project Abstract

Urbanisation and screen-based lifestyles are thought to erode the nature values that underlie environmental concern, yet social media’s role in fostering these values is largely overlooked. We addressed this gap with a multitask AI pipeline, fine-tuned on manually coded X posts, built on a ResNet-18 backbone that classifies Nature/Non-Nature, Biotic/Abiotic, Material/Immaterial, and Landscape Type. 

Of five interpretability methods, Grad-CAM and LIME delivered the most human-aligned heatmaps, remaining robust to perturbations. Binary tasks centred on faces or animals, while landscape predictions drew on broader context. EfficientNet-B0 matched ResNet-18’s accuracy but emphasised textures, showing that architecture shapes explanation quality as much as metrics. 

The pipeline transparently preselects relevant social-media content, reducing manual annotation and enabling DRV detection at scale. By revealing social media’s capacity to foster nature values, it points to new ways of designing online experiences that translate into real-world environmental stewardship.

## Objectives

This project is organized into two main phases:

1. **Automated Extraction of Nature-Related DRVs**  
   Develop a robust deep learning pipeline capable of classifying large-scale social media images into meaningful nature-related categories. This enables scalable, reproducible detection of Digital Relational Values (DRVs) in online content.

2. **Explainability and Human-Aligned Interpretation**  
   Apply and compare state-of-the-art interpretability techniques to analyze how models make decisions and ensure their alignment with human visual understanding.

## Repository Structure

```
TFG-Interpretability-Techniques-in-Social-Media-Images/
│
├── code/
│   ├── data_processing/             # Functions for image and coder label processing
│   │   ├── csv_process.py
│   │   └── dataset.py
│   │
│   ├── interpretability/           # Implementations of interpretability techniques
│   │   ├── gradcam_backprop.py
│   │   ├── gradcam.py
│   │   ├── guided_gradcam.py
│   │   ├── lime.py
│   │   ├── occlusion_sensitivity.py
│   │   ├── shap.py
│   │   └── xrai.py
│   │
│   ├── models/                     # Model definitions
│   │   ├── backbone.py
│   │   ├── multitask_model.py
│   │   └── wrappers.py
│   │
│   ├── utils/                      # Helper functions for visualization, metrics, and processing
│   │   ├── denormalize_image.py
│   │   ├── main_utils.py
│   │   ├── metrics_utils.py
│   │   └── visualization.py
│   │
│   ├── main.py                     # Main script to run training, evaluation, and interpretation
│
├── data_files/                    # CSV files with images and corresponding labels
│   ├── 39_20250401_0816.csv
│   ├── 43_3.csv
│   └── 51.csv
│
├── executes/                      # Shell scripts to execute the main pipeline with parameters
│   ├── execute.sh
│   └── execute_test.sh
│
├── output/                        # Folder where model outputs (images) are stored
│   ├── DenseNet121/
│   ├── EfficientNetB0/
│   └── ResNet18/
|
├── reports/                        # Folder with the submitted reports
│   ├── Final_report_30_06.pdf
│   ├── First_report_23_03.pdf
│   └── Second_report_18_05.pdf
|
├── trained_models/                # Pre-trained models (skip training from scratch)
│   ├── trained_DenseNet121_100epochs.pth
│   ├── trained_EfficientNetB0_100epochs.pth
│   └── trained_ResNet18_100epochs.pth
│
├── requirements.txt               # List of required libraries and their versions
```

## Usage

To run the project:

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Execute the main script via:
   ```bash
   sh executes/execute.sh
   ```

3. To switch the model used for predictions and interpretability, update the `MODEL_CHOICE` and `TARGET_LAYER` in `main.py`. All downstream processes will adapt automatically.

## Interpretability Methods Included

- **Grad-CAM**
- **Guided Grad-CAM**
- **Backprop Grad-CAM**
- **LIME**
- **SHAP**
- **Occlusion Sensitivity**
- **XRAI**

These techniques are used to generate heatmaps and insights on how each model interprets different tasks, enabling transparent analysis of DRV-related features in social media content.

## Results Summary
- Best human-aligned explanations: Grad-CAM and LIME
- Binary tasks (e.g., Biotic/Abiotic) focused on specific entities like faces or animals
- Landscape tasks relied on broader spatial context
- EfficientNet-B0 matched ResNet-18 accuracy, but emphasized different visual features like textures

These insights highlight the importance of model choice in interpretability—not just in performance.

## Author
Paula Feliu
Bachelor's Project - Artificial Intelligence Degree
UAB - Universitat Autònoma de Barcelona
GitHub: @paulafeliu
