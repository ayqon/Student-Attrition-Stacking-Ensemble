# FAIDM Group Assessment - Group 2

## 1. Student Introduction

| Name | Student ID | Role/Contribution |
|------|------------|-------------------|
| Subodh Chandra | U5715404 | Data Preparation & PCA |
| Tyrone Fernandes | U5755613 | Data Preparation |
| Mohammad Noor Mohammad Irfan | U5741544 | Data Preparation & RFE |
| Ioannis Konstantinou | U5750302 | Data Preparation & Clustering Models |
| Ayan Paul | U5732939 | Data Preparation & Classification Models |
| Vanessa Rebecca Wiyono | U5729891 | Data Preparation & Clustering Models |
| Zheyu Wu | U5716921 | Data Preparation & Classification Models |

## 2. Project Overview

This project aims to predict student performance (Pass, Fail, Withdrawn) using the Open University Learning Analytics Dataset (OULAD). The solution involves extensive data preparation, feature engineering (including RFE and PCA), clustering analysis, and a sophisticated stacking ensemble classification model.

## 3. Data Processing

### 3.1 Data Preparation Notebook

The complete RFE and RFECV preparation process is documented in:

* `FAIDM_GROUP_2_Data_prep_without_exam_score_realrfewith3outcomes.ipynb`

### 3.2 Final Datasets

The project uses two processed CSV files located in the repository:

* `final_all_scaled_3class.csv` - The primary scaled dataset used for classification.
* `final_with_pca.csv` - The dataset transformed using PCA.

## 4. Classification Models

### 4.1 Model Architecture

This project implements a **Stacking Ensemble Model** that combines:

* **Base Learners**: Random Forest (RF), XGBoost (XGB), and Support Vector Machine (SVM).
* **Meta Learners**: Logistic Regression, RF, XGB, SVM.
* **Strategy**: Three distinct stacking strategies (RF-Only, XGB-Only, SVM-Only, and Hybrid combinations).

### 4.2 Scripts

There are two main scripts for classification:

1. **`final_classification_stacking.py` (Main)**
    * This is the primary script.
    * It implements a full stacking ensemble with 7 different versions (RF-Only, XGB-Only, SVM-Only, and combinations).
    * It performs smart model loading/saving and generates comprehensive comparison visualizations.

2. **`test_classification_stacking.py`**
    * This is a direct stacking implementation with Grid Search.
    * It focuses on optimizing base models and meta-learners directly.

### 4.3 How to Run

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the **Main Stacking Ensemble** (Recommended):

    ```bash
    python final_classification_stacking.py
    ```

    * Outputs models to: `Models/Stacking_Seven_Versions/`
    * Outputs results/plots to: `output/Stacking_Seven_Versions/`

3. Run the **Direct Stacking Test**:

    ```bash
    python test_classification_stacking.py
    ```

## 5. Clustering Models

### 5.1 Overview

Clustering analysis (K-Means) is performed to identify student groupings based on interaction and demographic data.

### 5.2 How to Run

1. Run the clustering script:

    ```bash
    python kmeans-clustering.py
    ```
