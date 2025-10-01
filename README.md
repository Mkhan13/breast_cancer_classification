# Breast Tissue Histopathology Classification

## Problem
Breast cancer is the most common form of cancer in women, and invasive ductal carcinoma (IDC) is the most common form of breast cancer. Accurate and early detection of breast cancer is critical for patient outcomes. Pathologists analyze histopathology images to distinguish benign from malignant tissue, but only have an accuracy of around 75%. This project develops a model to classify breast histopathology image patches as *benign* or *malignant* using machine learning and deep learning methods with the goal of improving the diagnostic accuracy compared to pathologists.

---

## Data Source
- **[Invasive Ductal Carcinoma (IDC) Histology Image Dataset](https://academictorrents.com/details/e40bd59ab08861329ce3c418be191651f35e2ffa)**
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
1. **Naive Baseline:** Predict the majority class
2. **Classical ML Approach:**  Random Forest on extracted features with class weighting to handle imbalance
3. **Deep Learning Approach:**  CNN trained end-to-end


### Data Processing Pipeline  
The raw dataset is organized by patient IDs, each containing subfolders `0/` (non-cancerous) and `1/` (cancerous). To avoid data leakage, patients (not individual images) are split into **train (80%)**, **validation (10%)**, and **test (10%)** sets.  

The images are then copied into a standardized folder structure under `data/processed/`:  
```
data/processed/
├── train/
│   ├── 0/
│   └── 1/
├── val/
│   ├── 0/
│   └── 1/
└── test/
    ├── 0/
    └── 1/
```

### Models Evaluated and Model Selected  
- **Evaluated Models:**

| Approach             | Accuracy | Precision (Malignant) | Recall (Malignant) | F1-score (Malignant) | ROC-AUC | Notes                                                                 |
|----------------------|----------|------------------------|---------------------|-----------------------|---------|----------------------------------------------------------------------|
| **Naive Baseline**   | 73%      | 0.00                   | 0.00                | 0.00                  | 0.50    | Predicts all images as benign; fails to detect malignant tissue    |
| **Classical ML**     | 83%      | 0.715                  | 0.636                | 0.673                  | 0.888   | Stronger at detecting malignant cases than naive baseline |
| **Deep Learning (CNN)** | 85%   | 0.66                   | 0.92                | 0.77                  | 0.94    | Strong ROC-AUC, highest recall for malignant cases  |


- **Model Selected:**  Deep Learning (CNN)

### Comparison to Naive Approach  
The naive approach predicts all tissue samples as benign and completely fails to detect malignant cases and has an accuracy of 73% with zero precision or recall. The CNN has an accuracy of 85% with a recall of 0.92 meaning it successfully detected most positive cancerous samples but with less precision (66.3%). This model is sensitive to malignant cases which is preferable in medical applications where missing malignant tissue is more dangerous than false positives. The CNN has the most clinically relevant performance and is therefore the model best suited for this application. 

---

## Visual Interface Demo
<img width="997" height="729" alt="Visual Interface" src="https://github.com/user-attachments/assets/a4ed9d40-1de7-47bc-afc8-b930c988a21b" />

Video demo of the project can be found [here](https://drive.google.com/file/d/1h2BIcB1Hyp5ltOkDPBCCkxNpndm6q28v/view?usp=sharing)  
Streamlit site can be found [here](https://deep-learning-app-963698787646.us-central1.run.app/)

---

## Results and Conclusions  
The deep learning CNN achieved an accuracy of 85%, with a ROC-AUC of 0.94. Compared to the naive baseline (73%) and the classical machine learning approach (83%), the CNN consistently produced stronger performance, particularly in recall for malignant cases (92%). These results demonstrate that the model is able to capture more subtle and complex image features that the more simpler models miss.

Histopathology diagnosis by pathologists have documented error rates ranging from 1% to 9%, depending on the type of sample type and diagnosis. This CNN's performance falls within a range that's comparable to human pathologists.The CNN still misclassifies some benign tissues as malignant, or false positives, but in a medical setting this tradeoff is preferable to false negatives. A false positive can be cleared up with further testing, but a false negative, or a missed cancer diagnosis, can have more dangerous consequences.

While these models won't replace pathologists, they have the potential to serve as valuable and powerful assistive tools. Further validation in clinical settings would be needed before more widespread implementation of an AI breast cancer diagnostic tool can be implemented.

---

## Ethics Statement  
This project is intended solely for educational and research purposes. This model should not be used for clinical decision-making or as a substitute for professional medical judgment. While the results demonstrate promising accuracy, every diagnostic tool in healthcare must go through validation and regulatory approval before use on patients. 


---

## Instructions on How to Run the Code

1. Clone the Repository  
`git clone https://github.com/Mkhan13/breast_cancer_classification.git`  
`cd breast_cancer_classification`

3. Install Dependencies  
`pip install -r requirements.txt`

4. Run the Streamlit App  
`streamlit run main.py`  
The app will open in your browser  

6. Upload a tissue image as a PNG or JPG  

The model will output the predicted class (benign or malignant) and the prediction probability

