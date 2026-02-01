# FAIDM Group Assessment

## 1. Student Introduction

| Name | Student ID | Role/Contribution |
|------|------------|-------------------|
| Subodh Chandra | U5715404 | Data Preparation & PCA |
| Tyrone Fernandes | U5755613 | Data Preparation |
| Mohammad Noor Mohammad Irfan | U5741544 | Data Preparation & RFE |
| Ioannis Konstantinou | U5750302 | Data Preparation & Clustering Models |
| Ayan Paul | U5732939 | Data Preparation & Classification Models |
| Vanessa Wiyono | U5729891 | Data Preparation & Clustering Models |
| Zheyu Wu | U5716921 | Data Preparation & Classification Models |

## 2. Data Processing

### 2.1 Final Dataset
The project uses two CSV files located in the repository:
- `final_all_scaled_3class.csv` - Original Dataset
- `final_with_pca.csv` - Dataset after PCA

## 3. Classification Models

### 3.1 Model Selection
This project implements a **stacking ensemble model** with three classification algorithms:
- **Support Vector Machine (SVM)**
- **Random Forest (RF)**
- **XGBoost (XGB)**

### 3.2 How to Run

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the stacking ensemble model:
```bash
python final_classification_stacking.py
```

## 4. Clustering Models
1. Install required libraries (k-means):
```bash
pip install -r requirements.txt
```
2.  Run the clustering k-means model:
```bash
python kmeans-clustering.py
```
