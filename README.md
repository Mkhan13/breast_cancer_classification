# Breast Tissue Histopathology Classification

## Problem
Breast cancer is the most common form of cancer in women, and invasive ductal carcinoma (IDC) is the most common form of breast cancer. Accurate and early detection of breast cancer is critical for patient outcomes. Pathologists analyze histopathology images to distinguish benign from malignant tissue, but only have an accuracy of around 75%. This project develops a model to classify breast histopathology image patches as *benign* or *malignant* using machine learning and deep learning methods with the goal of improving the diagnostic accuracy compared to pathologists.

---

## Data Source
- **Invasive Ductal Carcinoma (IDC) Histology Image Dataset** on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)  
  - Binary classes: benign vs malignant tissue patches
  - Class distribution: 198,738 IDC negative, 78,786 IDC positive

---

## Review of Relevant Previous Efforts and Literature  
[This study](https://www.nature.com/articles/s41598-022-19278-2?fromPaywallRec=false), published in 2022, focuses on the multiclass classification of breast cancer tissue using deep learning methods using the Colsanitas dataset. [This 2021 study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8582388/#cancers-13-05368-t001) similarly focuses on using deep learning to classify breast cancer tissue as benign or malignant using a dataset they curated from a variety of hospitals.

**My Contribution:**  
While projects applying deep learning to breast cancer clasification have been conducted previously, my project provides direct comparisons against traditional machine learning models or naive baselines using a dataset that is not often used for research due to the size and complexity of the data.

---

## Model Evaluation Process & Metric Selection   
- **Metrics:**  
  - Accuracy  
  - Precision, Recall
  - F1-score  
  - ROC AUC  
- **Data Splits:** Stratified 80%/10%/10% split for train/validation/test 

All three approaches (naive, classical ML, and deep learning) are trained and evaluated on the same training, validation, and test sets. The results are compared directly against the naive baseline to quantify performance improvements

---

## Modeling Approach  
1. **Naive Baseline:** 
2. **Classical ML Approach:**  
3. **Deep Learning Approach:**  


### Data Processing Pipeline  
The raw dataset is organized by patient IDs, each containing subfolders `0/` (non-cancerous) and `1/` (cancerous). To avoid data leakage, patients (not individual images) are split into **train (80%)**, **validation (10%)**, and **test (10%)** sets.  

The images are then copied into a standardized folder structure under `data/processed/`:
data/processed/
├── train/{0,1}/
├── val/{0,1}/
└── test/{0,1}/

### Models Evaluated and Model Selected  
- **Evaluated Models:**  
  - Naive baseline  
  - Classical ML model
  - Deep Learning
- **Model Selected:**  

### Comparison to Naive Approach  
 

---

## Visual Interface Demo


---

## Results and Conclusions  
 

---

## Ethics Statement  
 

---

## Instructions on How to Run the Code
