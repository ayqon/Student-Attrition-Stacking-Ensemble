# Student Academic Trajectory & Early Attrition Prediction Platform
## University of Warwick | Fundamentals of Artificial Intelligence and Data Mining (WM9QG-15)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/Stacking%20Ensemble-81.09%25%20Accuracy-emerald.svg)](final_classification_stacking.py)
[![Institution](https://img.shields.io/badge/Institution-University%20of%20Warwick-purple.svg)](https://warwick.ac.uk/)
[![Dataset](https://img.shields.io/badge/Dataset-OULAD%20(25%2C793%20Students)-blue.svg)](https://analyse.kmi.open.ac.uk/open-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-slate.svg)](LICENSE)

An end-to-end predictive learning analytics and public health-style educational surveillance platform developed for the **Open University Learning Analytics Dataset (OULAD)**. Designed following the **CRISP-DM** framework, the system enables proactive, early-semester identification of academically at-risk students prior to examination dates by coupling **unsupervised behavioral phenotyping (K-Means & DBSCAN)** with a **pairwise decomposed 9-submodel Stacking Ensemble (XGBoost, Random Forest, SVM)** achieving **81.09% classification accuracy**.

---

## Executive Summary & Problem Formulation

University retention teams require early indicators to detect students at risk of academic failure or withdrawal before final assessments occur. 

* **The Data Leakage Dilemma**: Including final exam scores artificially inflates model accuracy while defeating the purpose of early intervention. This project removes exam scores and predicts trajectories strictly from early demographic attributes and Virtual Learning Environment (VLE) interaction clickstreams.
* **Ghost Student Filtering**: Filtered **3,361 "Ghost" students** (registered records with zero assessment and zero VLE engagement), yielding an active modeling cohort of **25,793 students**.
* **Unsupervised Risk Stratification**: Segmented students into high-risk ($40.4\%$ fail rate) vs high-engagement ($7.1\%$ fail rate) behavioral cohorts without using outcome labels.
* **Supervised Stacking Performance**: Deployed a pairwise decomposition meta-learner stacking architecture delivering **81.09% multi-class accuracy** across `Pass`, `Fail`, and `Withdrawn`.

---

## Team Collaboration & Roles

Developed collaboratively by **Group 2** for the **Fundamentals of Artificial Intelligence and Data Mining (WM9QG-15)** module at the **University of Warwick**.

| Team Member | Module Focus | Core Engineering Contributions |
|:---|:---|:---|
| **Ioannis Konstantinou** | **Lead Data Engineering & Clustering** | Automated multi-table ETL across 7 relational OULAD tables (`data_preparation.py`). Designed unsupervised behavioral phenotyping (`kmeans_clustering.py`, `dbscan.ipynb`), multi-K Silhouette and Elbow sweeps ($K=2$ on 20 PCs), domain-specific **Academic Risk Metric** formulation, PCA 2D centroid risk mapping, and DBSCAN noise isolation ($3,078$ at-risk outlier students). |
| **Ayan Paul** | **Lead Classification Architecture** | Designed the pairwise decomposition stacking ensemble (`final_classification_stacking.py`), base learner parameter optimization (RF, XGB, SVM), and meta-model probability blending. |
| **Subodh Chandra** | **Feature Transformation & PCA** | Dimensionality reduction pipeline (`pca.py`), variance retention analysis ($21$ PCs retaining $95\%$ variance), and matrix scaling. |
| **Mohammad Noor Mohammad Irfan** | **Feature Selection & RFECV** | Recursive Feature Elimination (`feature_engineering_rfecv.ipynb`), cross-validated feature ranking (RFECV), and collinearity pruning. |
| **Vanessa Rebecca Wiyono** | **Clustering Analysis** | Cross-tabulation contingency mapping, demographic cluster composition, and intervention strategy profiling. |
| **Zheyu Wu** | **Model Evaluation & Benchmarking** | Model benchmarking, cross-validation metrics calculation, and evaluation scripts (`test_classification_stacking.py`). |
| **Tyrone Fernandes** | **Data Preparation & Cleaning** | Initial dataset inspection, missing value imputation, and categorical feature encoding pipelines. |

---

## Methodology & Architecture

### 1. Data Engineering & Preprocessing Pipeline
* **Source**: OULAD relational database (7 tables: `studentInfo`, `courses`, `assessments`, `studentRegistration`, `studentAssessment`, `studentVle`, `vle`).
* **Cohort Filtering**: $32,593 \to 25,793$ active students after removing $3,361$ unengaged ghost records.
* **Feature Engineering**: Extracted behavioral metrics including `total_attempt_ratio`, `active_weeks`, `attempted_core_score`, `core_completion_ratio`, and `performance_efficiency`.
* **Dimensionality Reduction (PCA)**: Selected **21 principal components** retaining **95% variance**, achieving a $50\%$ reduction in input dimensionality.

### 2. Unsupervised Behavioral Phenotyping (Clustering)
* **K-Means Clustering ($K=2$)**:
  * Evaluated across $K \in [1..10]$ using Inertia (Elbow) and Silhouette optimization on 20 Principal Components.
  * **Group 0 (At-Risk / Low Engagement)**: $17,252$ students ($66.91\%$) with a **40.4% Fail Rate**.
  * **Group 1 (Successful / High Engagement)**: $8,531$ students ($33.08\%$) with only a **7.1% Fail Rate**.
  * **Domain Risk Formula**:
    $$\text{Academic Risk Rate} = \frac{\text{Fail Count}}{\text{Total Enrolled} - \text{Withdrawn Count}}$$
* **DBSCAN Density-Based Outlier Detection**:
  * Parameters: $\varepsilon = 0.2, \text{MinPts} = 10$ selected via K-Distance curve.
  * Successfully flagged **3,078 irregular noise students** exhibiting severe engagement drop-off.

### 3. Multi-Model Stacking Ensemble Architecture
* **Pairwise Problem Decomposition**:
  * Decomposed the 3-class problem into 3 binary tasks: `Pass vs Fail`, `Fail vs Withdrawn`, and `Pass vs Withdrawn`.
  * Trained 3 base classifiers (Random Forest, XGBoost, Support Vector Machine RBF) on each sub-problem (9 total sub-models).
* **Meta-Learner Blending**:
  * Stacked all 9 sub-model probability outputs into an **XGBoost Meta-Learner**.
  * Delivered a final multi-class classification accuracy of **81.09%**.

```
                                  OULAD Relational Database
                          (32,593 Records -> 25,793 Active Cohort)
                                             │
                                             ▼
                                  Automated ETL Pipeline
                                (`data_preparation.py`)
                                             │
                        ┌────────────────────┴────────────────────┐
                        ▼                                         ▼
            Feature Selection & PCA                       Unsupervised Clustering
        (`feature_engineering_rfecv.ipynb`, `pca.py`)    (`kmeans_clustering.py`, `dbscan.ipynb`)
        - 21 PCs (95% Variance Retained)                 - K-Means (K=2): 40.4% vs 7.1% Risk
        - RFECV Feature Ranking                          - DBSCAN: 3,078 Outliers Isolated
                        │                                         │
                        ▼                                         ▼
                 Processed Dataset                       Behavioral Phenotypes
             (`final_all_scaled_3class.csv`)           (At-Risk vs Successful Cohorts)
                        │
                        ▼
            9-Submodel Pairwise Stacking
         ┌──────────────────┬──────────────────┬──────────────────┐
         ▼                  ▼                  ▼                  ▼
    Pass vs Fail      Fail vs Withdraw  Pass vs Withdraw       (RF, XGB, SVM)
         │                  │                  │                  │
         └──────────────────┼──────────────────┴──────────────────┘
                            ▼
                    Meta-Learner Blending
                    (XGBoost Meta-Classifier)
                            │
                            ▼
              Final Predictive Accuracy: 81.09%
                  (Pass | Fail | Withdrawn)
```

---

## Empirical Results Summary

| Model / Strategy | Configuration | Metric | Result |
|:---|:---|:---:|:---:|
| **K-Means Group 0** | Low Engagement Cohort | Fail Rate | **40.4%** |
| **K-Means Group 1** | High Engagement Cohort | Fail Rate | **7.1%** |
| **DBSCAN Clustering** | Density Outliers ($arepsilon=0.2, 	ext{MinPts}=10$) | Flagged Students | **3,078 Outliers** |
| **PCA Transformation** | 21 Components | Variance Retained | **95.0%** |
| **Baseline Classifier** | All Features | Macro F1 | **0.6940** |
| **Stacking Ensemble** | 9-Submodel Pairwise + XGBoost Meta | **Final Accuracy** | **81.09%** |

---

## Project Structure

```
.
├── data_preparation.py              # Automated multi-table OULAD ETL pipeline
├── kmeans_clustering.py             # K-Means clustering, silhouette sweeps & risk metrics
├── dbscan.ipynb                     # DBSCAN density-based clustering & outlier analysis
├── pca.py                           # Principal Component Analysis pipeline
├── feature_engineering_rfecv.ipynb  # RFECV feature selection & elimination notebook
├── final_classification_stacking.py # Main 7-configuration Stacking Ensemble
├── test_classification_stacking.py  # Grid search & direct stacking optimization
├── train_xgboost.py                 # Standalone XGBoost training & evaluation
├── train_rf.py                      # Standalone Random Forest training & evaluation
├── train_svm_rbf.py                 # Standalone SVM (RBF Kernel) training & evaluation
├── train_knn.py                     # Standalone K-Nearest Neighbors training & evaluation
├── final_all_scaled_3class.csv      # Scaled multi-class feature matrix
├── final_with_pca.csv               # PCA-transformed feature dataset
├── requirements.txt                 # Environment dependencies
└── README.md                        # System documentation
```

---

## Quickstart & Local Execution

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/ayqon/FAIDM-Group-Assessment.git
cd FAIDM-Group-Assessment

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Unsupervised Clustering & Risk Profiling

```bash
# Execute K-Means clustering with silhouette optimization and PCA plots
python kmeans_clustering.py
```

### 3. Run Stacking Ensemble Classification

```bash
# Train and evaluate the full stacking ensemble
python final_classification_stacking.py
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
