# 🩺 Skin Cancer Detection using YOLOv9

## 📘 Project Overview
This project focuses on **automatic skin cancer detection** from dermoscopic images using **YOLOv9**, an advanced object detection and classification model.  
The dataset used is **HAM10000**, which contains dermoscopic images of various skin lesions classified into 7 categories.  
The main goal is to preprocess, analyze, and train a deep learning model that can accurately detect and classify skin cancer types.

---

## 🧠 Key Objectives
- Perform **Exploratory Data Analysis (EDA)** on HAM10000 metadata.
- Apply **image preprocessing**:
  - CLAHE (contrast normalization)
  - Median filtering for noise reduction
  - Resizing & padding to 640×640 (YOLO standard)
  - Data augmentation using Albumentations
- Train a **YOLOv9 model** for lesion classification.
- Evaluate the model with confusion matrix, precision–recall, and loss curves.

---

## 📂 Project Structure

├── main.ipynb                    # Main Jupyter Notebook (EDA + YOLO pipeline)
├── HAM10000_metadata.csv         # Metadata for the HAM10000 dataset
├── HAM10000_images/              # Original dataset images
├── HAM10000_yolo_cls/            # Preprocessed YOLO-ready images
├── eda_outputs/                  # Visualizations and CSV summaries
│   ├── age_distribution.png
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── metadata_describe.csv
│   ├── missing_values.csv
│   ├── sample_image_grid.jpg
│   ├── sex_counts.png
│   ├── top_localizations.png
├── .gitignore                    # Ignored files/folders
└── .ipynb_checkpoints/           # Auto-generated Jupyter checkpoints


<img width="800" height="500" alt="age_distribution" src="https://github.com/user-attachments/assets/b1e6b933-2d5d-48a3-a054-3054da1e756a" />
