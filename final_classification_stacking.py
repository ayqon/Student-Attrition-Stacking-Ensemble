import pandas as pd
import numpy as np
import pickle
import os
import time
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    recall_score, precision_score, classification_report,
    precision_recall_fscore_support
)
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("STACKING ENSEMBLE - SEVEN VERSIONS (SMART BASE MODELS LOADING)")
print("RF-Only | XGB-Only | SVM-Only | RF+XGB | RF+SVM | XGB+SVM | Hybrid")
print("=" * 80)

# ==================== Configuration ====================
RANDOM_STATE = 42

base_path = r"D:\Warwick\WM9QG-FAIDM\GROUP"
output_dir = os.path.join(base_path, "output", "Stacking_Seven_Versions")
models_dir = os.path.join(base_path, "Models", "Stacking_Seven_Versions")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

BASE_MODELS_PATH = os.path.join(models_dir, "trained_base_models.pkl")

POSSIBLE_ENSEMBLE_PATHS = [
    os.path.join(models_dir, "RF-Only", "ensemble.pkl"),
    os.path.join(models_dir, "XGB-Only", "ensemble.pkl"),
    os.path.join(models_dir, "SVM-Only", "ensemble.pkl"),
    os.path.join(models_dir, "RFXGB", "ensemble.pkl"),
    os.path.join(models_dir, "RFSVM", "ensemble.pkl"),
    os.path.join(models_dir, "XGBSVM", "ensemble.pkl"),
    os.path.join(models_dir, "Hybrid", "ensemble.pkl"),
]

# ==================== Step 1: Load Data ====================
print("\n" + "=" * 80)
print("STEP 1: LOAD AND PREPARE DATA")
print("=" * 80)

df = pd.read_csv(os.path.join(base_path, 'final_all_scaled_3class.csv'))
df['final_result'] = df['final_result'].str.lower()

X = df.drop(['final_result'], axis=1)
y = df['final_result'].map({'pass': 0, 'fail': 1, 'withdrawn': 2})

print(f"Dataset Info:")
print(f"Shape: {X.shape}")
print(f"Class Distribution:")
for cls, count in y.value_counts().sort_index().items():
    cls_name = ['Pass', 'Fail', 'Withdrawn'][cls]
    print(f"{cls_name}: {count} ({count / len(y) * 100:.1f}%)")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Train/Test Split (80/20, stratified):")
print(f"Training: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# ==================== Step 2: Prepare Binary Datasets ====================
print("\n" + "=" * 80)
print("STEP 2: PREPARE BINARY DATASETS (From Training Set)")
print("=" * 80)

df_train = pd.concat([
    X_train,
    y_train.map({0: 'pass', 1: 'fail', 2: 'withdrawn'})
], axis=1)
df_train.columns = df.columns

# Binary Dataset 1: Pass vs Fail
df_pf = df_train[df_train['final_result'].isin(['pass', 'fail'])].copy()
X_pf = df_pf.drop(['final_result'], axis=1)
y_pf = df_pf['final_result'].map({'fail': 0, 'pass': 1})
print(f"Pass vs Fail: {X_pf.shape[0]} samples (Fail={sum(y_pf == 0)}, Pass={sum(y_pf == 1)})")

# Binary Dataset 2: Fail vs Withdrawn
df_fw = df_train[df_train['final_result'].isin(['fail', 'withdrawn'])].copy()
X_fw = df_fw.drop(['final_result'], axis=1)
y_fw = df_fw['final_result'].map({'fail': 0, 'withdrawn': 1})
print(f"Fail vs Withdrawn: {X_fw.shape[0]} samples (Fail={sum(y_fw == 0)}, Withdrawn={sum(y_fw == 1)})")

# Binary Dataset 3: Pass vs Withdrawn
df_pw = df_train[df_train['final_result'].isin(['pass', 'withdrawn'])].copy()
X_pw = df_pw.drop(['final_result'], axis=1)
y_pw = df_pw['final_result'].map({'pass': 0, 'withdrawn': 1})
print(f"Pass vs Withdrawn: {X_pw.shape[0]} samples (Pass={sum(y_pw == 0)}, Withdrawn={sum(y_pw == 1)})")

# ==================== Step 3: Smart Load Base Models ====================
print("\n" + "=" * 80)
print("STEP 3: SMART LOAD BASE MODELS")
print("=" * 80)

base_models_loaded = False

# strategy 1: load trained_base_models.pkl
if os.path.exists(BASE_MODELS_PATH):
    print(f"\nStrategy 1: Found trained_base_models.pkl")
    print(f"Path: {BASE_MODELS_PATH}")
    print(f"Size: {os.path.getsize(BASE_MODELS_PATH) / 1024 / 1024:.2f} MB")
    print("\nLoading saved models...")

    try:
        with open(BASE_MODELS_PATH, 'rb') as f:
            saved_package = pickle.load(f)

        rf_pf = saved_package['rf_pf']
        rf_fw = saved_package['rf_fw']
        rf_pw = saved_package['rf_pw']
        xgb_pf = saved_package['xgb_pf']
        xgb_fw = saved_package['xgb_fw']
        xgb_pw = saved_package['xgb_pw']
        svm_pf = saved_package['svm_pf']
        svm_fw = saved_package['svm_fw']
        svm_pw = saved_package['svm_pw']

        print("All 9 base models loaded successfully!")

        if 'metadata' in saved_package:
            meta = saved_package['metadata']
            print(f"\nTraining Info:")
            print(f"Trained: {meta.get('train_date', 'N/A')}")
            print(f"Random State: {meta.get('random_state', 'N/A')}")

        base_models_loaded = True

    except Exception as e:
        print(f"Failed to load: {e}")
        print("Will try alternative strategies...")

# strategy 2: extract from ensemble.pkl
if not base_models_loaded:
    print(f"\nStrategy 2: Extract from existing ensemble.pkl")

    rf_pf = rf_fw = rf_pw = None
    xgb_pf = xgb_fw = xgb_pw = None
    svm_pf = svm_fw = svm_pw = None

    for ensemble_path in POSSIBLE_ENSEMBLE_PATHS:
        if os.path.exists(ensemble_path):
            print(f"\nFound: {ensemble_path}")
            print(f"Size: {os.path.getsize(ensemble_path) / 1024 / 1024:.2f} MB")
            print("Extracting base models...")

            try:
                with open(ensemble_path, 'rb') as f:
                    ensemble = pickle.load(f)

                base_models = ensemble['base_models']

                if rf_pf is None and base_models.get('rf_pf') is not None:
                    rf_pf = base_models['rf_pf']
                if rf_fw is None and base_models.get('rf_fw') is not None:
                    rf_fw = base_models['rf_fw']
                if rf_pw is None and base_models.get('rf_pw') is not None:
                    rf_pw = base_models['rf_pw']

                if xgb_pf is None and base_models.get('xgb_pf') is not None:
                    xgb_pf = base_models['xgb_pf']
                if xgb_fw is None and base_models.get('xgb_fw') is not None:
                    xgb_fw = base_models['xgb_fw']
                if xgb_pw is None and base_models.get('xgb_pw') is not None:
                    xgb_pw = base_models['xgb_pw']

                if svm_pf is None and base_models.get('svm_pf') is not None:
                    svm_pf = base_models['svm_pf']
                if svm_fw is None and base_models.get('svm_fw') is not None:
                    svm_fw = base_models['svm_fw']
                if svm_pw is None and base_models.get('svm_pw') is not None:
                    svm_pw = base_models['svm_pw']

                extracted_count = sum([
                    base_models.get('rf_pf') is not None,
                    base_models.get('rf_fw') is not None,
                    base_models.get('rf_pw') is not None,
                    base_models.get('xgb_pf') is not None,
                    base_models.get('xgb_fw') is not None,
                    base_models.get('xgb_pw') is not None,
                    base_models.get('svm_pf') is not None,
                    base_models.get('svm_fw') is not None,
                    base_models.get('svm_pw') is not None,
                ])
                print(f"Extracted {extracted_count} model(s) from this file")

            except Exception as e:
                print(f"Failed to extract: {e}")

    all_models = {
        'rf_pf': rf_pf, 'rf_fw': rf_fw, 'rf_pw': rf_pw,
        'xgb_pf': xgb_pf, 'xgb_fw': xgb_fw, 'xgb_pw': xgb_pw,
        'svm_pf': svm_pf, 'svm_fw': svm_fw, 'svm_pw': svm_pw
    }

    none_count = sum(1 for v in all_models.values() if v is None)

    if none_count == 0:
        print(f"\nSuccessfully collected all 9 base models from ensemble files!")

        print(f"Saving to trained_base_models.pkl for future use...")
        base_models_package = {
            'rf_pf': rf_pf, 'rf_fw': rf_fw, 'rf_pw': rf_pw,
            'xgb_pf': xgb_pf, 'xgb_fw': xgb_fw, 'xgb_pw': xgb_pw,
            'svm_pf': svm_pf, 'svm_fw': svm_fw, 'svm_pw': svm_pw,
            'metadata': {
                'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'random_state': RANDOM_STATE,
                'extracted_from': 'multiple ensemble files'
            }
        }

        with open(BASE_MODELS_PATH, 'wb') as f:
            pickle.dump(base_models_package, f)

        print(f"Saved to: {BASE_MODELS_PATH}")
        base_models_loaded = True

    else:
        print(f"\nOnly found {9 - none_count}/9 models across all ensembles")
        print(f"Missing models will be trained in Strategy 3...")

# strategy 3: retrain
if not base_models_loaded:
    print(f"\nStrategy 3: Train from scratch")
    print("No existing models found, will train new base models...\n")

    # ==================== Step 3.1: Hyperparameter Tuning ====================
    print("=" * 80)
    print("STEP 3.1: HYPERPARAMETER TUNING")
    print("=" * 80)

    tuning_tasks = [
        # ===== RF Models =====
        ("Pass_vs_Fail_RF", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
         X_pf, y_pf, {
             'n_estimators': [150, 200, 250],
             'max_depth': [12, 15, 18],
             'min_samples_split': [2, 5],
             'min_samples_leaf': [1, 2],
             'max_features': ['sqrt', 'log2']
         }),

        ("Fail_vs_Withdrawn_RF", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
         X_fw, y_fw, {
             'n_estimators': [150, 200, 250],
             'max_depth': [12, 15, 18],
             'min_samples_split': [2, 5],
             'min_samples_leaf': [1, 2],
             'max_features': ['sqrt', 'log2']
         }),

        ("Pass_vs_Withdrawn_RF", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
         X_pw, y_pw, {
             'n_estimators': [150, 200, 250],
             'max_depth': [12, 15, 18],
             'min_samples_split': [2, 5],
             'min_samples_leaf': [1, 2],
             'max_features': ['sqrt', 'log2']
         }),

        # ===== XGB Models =====
        ("Pass_vs_Fail_XGB", xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                               eval_metric='logloss', verbosity=0),
         X_pf, y_pf, {
             'n_estimators': [100, 150, 200],
             'max_depth': [5, 7, 9],
             'learning_rate': [0.05, 0.1, 0.15],
             'subsample': [0.8, 0.9],
             'colsample_bytree': [0.8, 0.9],
             'min_child_weight': [1, 3]
         }),

        ("Fail_vs_Withdrawn_XGB", xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                                    eval_metric='logloss', verbosity=0),
         X_fw, y_fw, {
             'n_estimators': [100, 150, 200],
             'max_depth': [5, 7, 9],
             'learning_rate': [0.05, 0.1, 0.15],
             'subsample': [0.8, 0.9],
             'colsample_bytree': [0.8, 0.9],
             'min_child_weight': [1, 3]
         }),

        ("Pass_vs_Withdrawn_XGB", xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                                    eval_metric='logloss', verbosity=0),
         X_pw, y_pw, {
             'n_estimators': [100, 150, 200],
             'max_depth': [5, 7, 9],
             'learning_rate': [0.05, 0.1, 0.15],
             'subsample': [0.8, 0.9],
             'colsample_bytree': [0.8, 0.9],
             'min_child_weight': [1, 3]
         }),

        # ===== SVM Models =====
        ("Pass_vs_Fail_SVM", SVC(random_state=RANDOM_STATE, probability=True),
         X_pf, y_pf, {
             'C': [0.1, 1, 10],
             'gamma': ['scale'],
             'kernel': ['rbf']
         }),

        ("Fail_vs_Withdrawn_SVM", SVC(random_state=RANDOM_STATE, probability=True),
         X_fw, y_fw, {
             'C': [0.1, 1, 10],
             'gamma': ['scale'],
             'kernel': ['rbf']
         }),

        ("Pass_vs_Withdrawn_SVM", SVC(random_state=RANDOM_STATE, probability=True),
         X_pw, y_pw, {
             'C': [0.1, 1, 10],
             'gamma': ['scale'],
             'kernel': ['rbf']
         })
    ]

    best_models = {}

    for name, model, X_data, y_data, param_grid in tuning_tasks:
        print(f"\nTuning {name}...")

        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"Combinations: {total_combinations}, Total fits: {total_combinations * 5}")

        start_time = time.time()

        search = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring='f1',
            n_jobs=-1,
            verbose=0,
            return_train_score=False
        )

        search.fit(X_data, y_data)
        elapsed = time.time() - start_time

        best_models[name] = search.best_estimator_

        print(f"Best F1: {search.best_score_:.4f} (Time: {elapsed:.1f}s)")
        print(f"Best parameters: {search.best_params_}")

    # ==================== Step 3.2: Train Final Base Models ====================
    print("\n" + "=" * 80)
    print("STEP 3.2: TRAIN FINAL BASE MODELS ON FULL TRAINING DATA")
    print("=" * 80)

    print("\nTraining RF base models...")
    rf_pf = best_models["Pass_vs_Fail_RF"]
    rf_pf.fit(X_pf, y_pf)
    print(f"RF Pass vs Fail")

    rf_fw = best_models["Fail_vs_Withdrawn_RF"]
    rf_fw.fit(X_fw, y_fw)
    print(f"RF Fail vs Withdrawn")

    rf_pw = best_models["Pass_vs_Withdrawn_RF"]
    rf_pw.fit(X_pw, y_pw)
    print(f"RF Pass vs Withdrawn")

    print("\nTraining XGB base models...")
    xgb_pf = best_models["Pass_vs_Fail_XGB"]
    xgb_pf.fit(X_pf, y_pf)
    print(f"XGB Pass vs Fail")

    xgb_fw = best_models["Fail_vs_Withdrawn_XGB"]
    xgb_fw.fit(X_fw, y_fw)
    print(f"XGB Fail vs Withdrawn")

    xgb_pw = best_models["Pass_vs_Withdrawn_XGB"]
    xgb_pw.fit(X_pw, y_pw)
    print(f"XGB Pass vs Withdrawn")

    print("\nTraining SVM base models...")
    svm_pf = best_models["Pass_vs_Fail_SVM"]
    svm_pf.fit(X_pf, y_pf)
    print(f"SVM Pass vs Fail")

    svm_fw = best_models["Fail_vs_Withdrawn_SVM"]
    svm_fw.fit(X_fw, y_fw)
    print(f"SVM Fail vs Withdrawn")

    svm_pw = best_models["Pass_vs_Withdrawn_SVM"]
    svm_pw.fit(X_pw, y_pw)
    print(f"SVM Pass vs Withdrawn")

    # ==================== Step 3.3: Save Base Models ====================
    print("\n" + "=" * 80)
    print("STEP 3.3: SAVE BASE MODELS FOR FUTURE USE")
    print("=" * 80)

    base_models_package = {
        'rf_pf': rf_pf, 'rf_fw': rf_fw, 'rf_pw': rf_pw,
        'xgb_pf': xgb_pf, 'xgb_fw': xgb_fw, 'xgb_pw': xgb_pw,
        'svm_pf': svm_pf, 'svm_fw': svm_fw, 'svm_pw': svm_pw,
        'metadata': {
            'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'random_state': RANDOM_STATE,
            'train_samples': {
                'pf': len(X_pf),
                'fw': len(X_fw),
                'pw': len(X_pw)
            },
            'feature_count': X_pf.shape[1]
        }
    }

    with open(BASE_MODELS_PATH, 'wb') as f:
        pickle.dump(base_models_package, f)

    print(f"\nBase models saved to:")
    print(f"{BASE_MODELS_PATH}")
    print(f"File size: {os.path.getsize(BASE_MODELS_PATH) / 1024 / 1024:.2f} MB")


# ==================== Step 4: Define Stacking Function ====================
def run_stacking_version(version_name, use_rf=False, use_xgb=False, use_svm=False):
    """
    Run one stacking version

    Args:
        version_name: "RF-Only", "XGB-Only", "SVM-Only", or "Hybrid"
        use_rf: Use RF base models
        use_xgb: Use XGB base models
        use_svm: Use SVM base models
    """
    print("\n" + "=" * 80)
    print(f"RUNNING STACKING: {version_name}")
    print("=" * 80)

    # Calculate meta-feature dimension
    n_features_per_model = 3  # pred + prob0 + prob1
    n_models = (3 if use_rf else 0) + (3 if use_xgb else 0) + (3 if use_svm else 0)
    meta_dim = n_features_per_model * n_models

    print(f"\nConfiguration:")
    model_types = []
    if use_rf: model_types.append('RF')
    if use_xgb: model_types.append('XGB')
    if use_svm: model_types.append('SVM')
    print(f"Base Models: {n_models} ({', '.join(model_types)})")
    print(f"Meta-features: {meta_dim} dimensions")

    # ==================== Generate Test Meta-Features ====================
    print(f"\nGenerating Test Set Meta-Features...")
    meta_X_test = []

    if use_rf:
        meta_X_test.extend([
            rf_pf.predict(X_test), rf_pf.predict_proba(X_test)[:, 0], rf_pf.predict_proba(X_test)[:, 1],
            rf_fw.predict(X_test), rf_fw.predict_proba(X_test)[:, 0], rf_fw.predict_proba(X_test)[:, 1],
            rf_pw.predict(X_test), rf_pw.predict_proba(X_test)[:, 0], rf_pw.predict_proba(X_test)[:, 1]
        ])

    if use_xgb:
        meta_X_test.extend([
            xgb_pf.predict(X_test), xgb_pf.predict_proba(X_test)[:, 0], xgb_pf.predict_proba(X_test)[:, 1],
            xgb_fw.predict(X_test), xgb_fw.predict_proba(X_test)[:, 0], xgb_fw.predict_proba(X_test)[:, 1],
            xgb_pw.predict(X_test), xgb_pw.predict_proba(X_test)[:, 0], xgb_pw.predict_proba(X_test)[:, 1]
        ])

    if use_svm:
        meta_X_test.extend([
            svm_pf.predict(X_test), svm_pf.predict_proba(X_test)[:, 0], svm_pf.predict_proba(X_test)[:, 1],
            svm_fw.predict(X_test), svm_fw.predict_proba(X_test)[:, 0], svm_fw.predict_proba(X_test)[:, 1],
            svm_pw.predict(X_test), svm_pw.predict_proba(X_test)[:, 0], svm_pw.predict_proba(X_test)[:, 1]
        ])

    meta_X_test = np.column_stack(meta_X_test)
    print(f"Test meta-features: {meta_X_test.shape}")

    # ==================== Generate Training Meta-Features (CV) ====================
    print(f"\nGenerating Training Set Meta-Features (5-fold CV)...")
    print(f"(Retraining base models in each fold)")

    meta_X_train = np.zeros((X_train.shape[0], meta_dim))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"Fold {fold}/5", end=" ")
        start_time = time.time()

        X_tr_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_tr_fold = y_train.iloc[train_idx]

        # Reconstruct binary datasets for this fold
        df_tr_fold = pd.concat([
            X_tr_fold,
            y_tr_fold.map({0: 'pass', 1: 'fail', 2: 'withdrawn'})
        ], axis=1)
        df_tr_fold.columns = df.columns

        # Pass vs Fail
        df_pf_fold = df_tr_fold[df_tr_fold['final_result'].isin(['pass', 'fail'])].copy()
        X_pf_fold = df_pf_fold.drop(['final_result'], axis=1)
        y_pf_fold = df_pf_fold['final_result'].map({'fail': 0, 'pass': 1})

        # Fail vs Withdrawn
        df_fw_fold = df_tr_fold[df_tr_fold['final_result'].isin(['fail', 'withdrawn'])].copy()
        X_fw_fold = df_fw_fold.drop(['final_result'], axis=1)
        y_fw_fold = df_fw_fold['final_result'].map({'fail': 0, 'withdrawn': 1})

        # Pass vs Withdrawn
        df_pw_fold = df_tr_fold[df_tr_fold['final_result'].isin(['pass', 'withdrawn'])].copy()
        X_pw_fold = df_pw_fold.drop(['final_result'], axis=1)
        y_pw_fold = df_pw_fold['final_result'].map({'pass': 0, 'withdrawn': 1})

        col = 0

        if use_rf:
            for (X_fold, y_fold), base_model in [
                ((X_pf_fold, y_pf_fold), rf_pf),
                ((X_fw_fold, y_fw_fold), rf_fw),
                ((X_pw_fold, y_pw_fold), rf_pw)
            ]:
                model_fold = RandomForestClassifier(**base_model.get_params())
                model_fold.fit(X_fold, y_fold)
                meta_X_train[val_idx, col] = model_fold.predict(X_val_fold)
                meta_X_train[val_idx, col + 1:col + 3] = model_fold.predict_proba(X_val_fold)
                col += 3

        if use_xgb:
            for (X_fold, y_fold), base_model in [
                ((X_pf_fold, y_pf_fold), xgb_pf),
                ((X_fw_fold, y_fw_fold), xgb_fw),
                ((X_pw_fold, y_pw_fold), xgb_pw)
            ]:
                model_fold = xgb.XGBClassifier(**base_model.get_params())
                model_fold.fit(X_fold, y_fold)
                meta_X_train[val_idx, col] = model_fold.predict(X_val_fold)
                meta_X_train[val_idx, col + 1:col + 3] = model_fold.predict_proba(X_val_fold)
                col += 3

        if use_svm:
            for (X_fold, y_fold), base_model in [
                ((X_pf_fold, y_pf_fold), svm_pf),
                ((X_fw_fold, y_fw_fold), svm_fw),
                ((X_pw_fold, y_pw_fold), svm_pw)
            ]:
                model_fold = SVC(**base_model.get_params())
                model_fold.fit(X_fold, y_fold)
                meta_X_train[val_idx, col] = model_fold.predict(X_val_fold)
                meta_X_train[val_idx, col + 1:col + 3] = model_fold.predict_proba(X_val_fold)
                col += 3

        print(f"({time.time() - start_time:.1f}s)")

    print(f"Training meta-features: {meta_X_train.shape}")

    # ==================== Scale Meta-Features ====================
    print(f"\nUsing meta-features directly (original data already scaled)...")
    meta_X_train_scaled = meta_X_train
    meta_X_test_scaled = meta_X_test

    # ==================== Train Meta-Learners ====================
    print(f"\nTraining Meta-Learners...")

    meta_learners = {}

    # Logistic Regression
    meta_lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    meta_lr.fit(meta_X_train_scaled, y_train)
    y_pred_lr = meta_lr.predict(meta_X_test_scaled)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr, average='weighted')
    meta_learners['Logistic Regression'] = {
        'model': meta_lr,
        'predictions': y_pred_lr,
        'accuracy': acc_lr,
        'f1': f1_lr
    }
    print(f"LR  - Acc: {acc_lr:.4f}, F1: {f1_lr:.4f}")

    # Random Forest
    meta_rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    meta_rf.fit(meta_X_train_scaled, y_train)
    y_pred_rf = meta_rf.predict(meta_X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
    meta_learners['Random Forest'] = {
        'model': meta_rf,
        'predictions': y_pred_rf,
        'accuracy': acc_rf,
        'f1': f1_rf
    }
    print(f"RF  - Acc: {acc_rf:.4f}, F1: {f1_rf:.4f}")

    # XGBoost
    meta_xgb_learner = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss', verbosity=0
    )
    meta_xgb_learner.fit(meta_X_train_scaled, y_train)
    y_pred_xgb = meta_xgb_learner.predict(meta_X_test_scaled)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')
    meta_learners['XGBoost'] = {
        'model': meta_xgb_learner,
        'predictions': y_pred_xgb,
        'accuracy': acc_xgb,
        'f1': f1_xgb
    }
    print(f"XGB - Acc: {acc_xgb:.4f}, F1: {f1_xgb:.4f}")

    # SVM
    meta_svm = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        random_state=RANDOM_STATE,
        probability=True
    )
    meta_svm.fit(meta_X_train_scaled, y_train)
    y_pred_svm = meta_svm.predict(meta_X_test_scaled)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
    meta_learners['SVM'] = {
        'model': meta_svm,
        'predictions': y_pred_svm,
        'accuracy': acc_svm,
        'f1': f1_svm
    }
    print(f"SVM - Acc: {acc_svm:.4f}, F1: {f1_svm:.4f}")

    # ==================== Select Best Meta-Learner ====================
    best_meta_name = max(meta_learners, key=lambda k: meta_learners[k]['accuracy'])
    best_meta = meta_learners[best_meta_name]

    print(f"\nBest Meta-Learner: {best_meta_name}")
    print(f"Accuracy: {best_meta['accuracy']:.4f}")
    print(f"F1-Score: {best_meta['f1']:.4f}")

    # ==================== Detailed Evaluation ====================
    class_names = ['Pass', 'Fail', 'Withdrawn']
    y_pred_best = best_meta['predictions']

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred_best)
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_test, y_pred_best, zero_division=0)

    # ==================== Save Models ====================
    version_dir = os.path.join(models_dir, version_name.replace(' ', '_').replace('+', '_').replace('(', '').replace(')', ''))
    os.makedirs(version_dir, exist_ok=True)

    ensemble_package = {
        'version': version_name,
        'base_models': {
            'rf_pf': rf_pf if use_rf else None,
            'rf_fw': rf_fw if use_rf else None,
            'rf_pw': rf_pw if use_rf else None,
            'xgb_pf': xgb_pf if use_xgb else None,
            'xgb_fw': xgb_fw if use_xgb else None,
            'xgb_pw': xgb_pw if use_xgb else None,
            'svm_pf': svm_pf if use_svm else None,
            'svm_fw': svm_fw if use_svm else None,
            'svm_pw': svm_pw if use_svm else None
        },
        'meta_learners': meta_learners,
        'best_meta': {
            'name': best_meta_name,
            'model': best_meta['model']
        },
        'metadata': {
            'accuracy': best_meta['accuracy'],
            'f1': best_meta['f1'],
            'test_size': len(y_test)
        }
    }

    with open(os.path.join(version_dir, 'ensemble.pkl'), 'wb') as f:
        pickle.dump(ensemble_package, f)

    print(f"\nModels saved to: {version_dir}")

    return {
        'version': version_name,
        'best_meta_name': best_meta_name,
        'accuracy': best_meta['accuracy'],
        'f1': best_meta['f1'],
        'predictions': y_pred_best,
        'confusion_matrix': cm,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class,
        'meta_learners': meta_learners
    }


# ==================== Step 5: Run All Seven Versions ====================
print("\n" + "=" * 80)
print("STEP 5: RUN ALL SEVEN STACKING VERSIONS")
print("=" * 80)

results = {}

# Version 1-3: Single model type
results['RF-Only'] = run_stacking_version("RF-Only", use_rf=True, use_xgb=False, use_svm=False)
results['XGB-Only'] = run_stacking_version("XGB-Only", use_rf=False, use_xgb=True, use_svm=False)
results['SVM-Only'] = run_stacking_version("SVM-Only", use_rf=False, use_xgb=False, use_svm=True)

# Version 4-6: Pairwise combinations
results['RF+XGB'] = run_stacking_version("RF+XGB", use_rf=True, use_xgb=True, use_svm=False)
results['RF+SVM'] = run_stacking_version("RF+SVM", use_rf=True, use_xgb=False, use_svm=True)
results['XGB+SVM'] = run_stacking_version("XGB+SVM", use_rf=False, use_xgb=True, use_svm=True)

# Version 7: Hybrid combination
results['Hybrid'] = run_stacking_version("Hybrid", use_rf=True, use_xgb=True, use_svm=True)

# ==================== Step 6: Comparison Visualizations ====================
print("\n" + "=" * 80)
print("STEP 6: GENERATING COMPARISON VISUALIZATIONS")
print("=" * 80)

class_names = ['Pass', 'Fail', 'Withdrawn']

# Visualization 1: Confusion Matrices Comparison
print("\nCreating confusion matrices comparison...")
fig, axes = plt.subplots(2, 4, figsize=(28, 12))
axes = axes.flatten()

for idx, (version_name, result) in enumerate(results.items()):
    ax = axes[idx]
    cm = result['confusion_matrix']

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names,
                yticklabels=class_names if idx % 4 == 0 else False,
                cbar=False)

    title = f"{version_name}\n"
    title += f"Meta: {result['best_meta_name']}\n"
    title += f"Acc: {result['accuracy']:.4f} | F1: {result['f1']:.4f}"

    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_xlabel('Predicted', fontsize=9)
    if idx % 4 == 0:
        ax.set_ylabel('True', fontsize=9)

axes[7].axis('off')
plt.suptitle('Stacking Ensemble Comparison - Confusion Matrices (7 Versions)',
             fontsize=14, fontweight='bold')
plt.tight_layout()

plt.savefig(os.path.join(output_dir, '1_confusion_matrices_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("1_confusion_matrices_comparison.png")

# Visualization 2: Performance Metrics Comparison
print("\nCreating performance metrics comparison...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Overall Accuracy & F1
ax = axes[0, 0]
versions = list(results.keys())
accuracies = [results[v]['accuracy'] for v in versions]
f1_scores = [results[v]['f1'] for v in versions]
x = np.arange(len(versions))
width = 0.35

bars1 = ax.bar(x - width / 2, accuracies, width, label='Accuracy', color='#4ECDC4', alpha=0.8)
bars2 = ax.bar(x + width / 2, f1_scores, width, label='F1-Score', color='#FF6B6B', alpha=0.8)

ax.set_ylabel('Score', fontsize=11)
ax.set_title('Overall Performance Comparison', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(versions, rotation=15, ha='right')
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Per-Class F1 Scores
ax = axes[0, 1]
x = np.arange(len(class_names))
width = 0.11

for idx, (version_name, result) in enumerate(results.items()):
    offset = (idx - 3) * width
    ax.bar(x + offset, result['f1_per_class'], width,
           label=version_name, alpha=0.8)

ax.set_ylabel('F1-Score', fontsize=11)
ax.set_title('Per-Class F1-Score Comparison', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.legend(fontsize=9)
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# Per-Class Precision
ax = axes[1, 0]
for idx, (version_name, result) in enumerate(results.items()):
    offset = (idx - 3) * width
    ax.bar(x + offset, result['precision_per_class'], width,
           label=version_name, alpha=0.8)

ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Per-Class Precision Comparison', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.legend(fontsize=9)
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# Per-Class Recall
ax = axes[1, 1]
for idx, (version_name, result) in enumerate(results.items()):
    offset = (idx - 3) * width
    ax.bar(x + offset, result['recall_per_class'], width,
           label=version_name, alpha=0.8)

ax.set_ylabel('Recall', fontsize=11)
ax.set_title('Per-Class Recall Comparison', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.legend(fontsize=9)
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Detailed Performance Metrics Comparison',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_metrics_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("2_metrics_comparison.png")

# Visualization 3: Summary Table
print("\nCreating summary table...")
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

table_data = [
    ['Version', 'Best Meta', 'Accuracy', 'F1-Score', 'Pass F1', 'Fail F1', 'Withdrawn F1']
]

for version_name, result in results.items():
    table_data.append([
        version_name,
        result['best_meta_name'],
        f"{result['accuracy']:.4f}",
        f"{result['f1']:.4f}",
        f"{result['f1_per_class'][0]:.4f}",
        f"{result['f1_per_class'][1]:.4f}",
        f"{result['f1_per_class'][2]:.4f}"
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.15, 0.18, 0.12, 0.12, 0.12, 0.12, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(7):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
colors = ['#E8F8F5', '#FADBD8', '#FCF3CF', '#E8DAEF', '#D5F4E6', '#FFE6E6', '#E6F3FF']
for i in range(1, len(table_data)):
    for j in range(7):
        table[(i, j)].set_facecolor(colors[i - 1])

ax.text(0.5, 0.95, 'Stacking Ensemble - Seven Versions Summary',
        ha='center', va='top', fontsize=14, fontweight='bold',
        transform=ax.transAxes)

ax.text(0.5, 0.02, f'Test Set: {len(y_test)} samples',
        ha='center', va='bottom', fontsize=9, style='italic',
        transform=ax.transAxes)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_summary_table.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("3_summary_table.png")

# Save CSV summary
summary_df = pd.DataFrame([
    {
        'Version': version_name,
        'Best_Meta_Learner': result['best_meta_name'],
        'Accuracy': result['accuracy'],
        'F1_Score': result['f1'],
        'Pass_Precision': result['precision_per_class'][0],
        'Pass_Recall': result['recall_per_class'][0],
        'Pass_F1': result['f1_per_class'][0],
        'Fail_Precision': result['precision_per_class'][1],
        'Fail_Recall': result['recall_per_class'][1],
        'Fail_F1': result['f1_per_class'][1],
        'Withdrawn_Precision': result['precision_per_class'][2],
        'Withdrawn_Recall': result['recall_per_class'][2],
        'Withdrawn_F1': result['f1_per_class'][2]
    }
    for version_name, result in results.items()
])
summary_df.to_csv(os.path.join(output_dir, 'summary_comparison.csv'), index=False)
print("summary_comparison.csv")

# ==================== Final Summary ====================
print("\n" + "=" * 80)
print("ALL SEVEN STACKING VERSIONS COMPLETE!")
print("=" * 80)

print("\nFinal Results Summary:")
print("-" * 80)
print(f"{'Version':<20} {'Best Meta':<20} {'Accuracy':>10} {'F1-Score':>10}")
print("-" * 80)
for version_name, result in results.items():
    print(f"{version_name:<20} {result['best_meta_name']:<20} "
          f"{result['accuracy']:>10.4f} {result['f1']:>10.4f}")
print("-" * 80)

best_version = max(results.keys(), key=lambda k: results[k]['accuracy'])
print(f"\nBest Overall Version: {best_version}")
print(f"Accuracy: {results[best_version]['accuracy']:.4f}")
print(f"F1-Score: {results[best_version]['f1']:.4f}")
print(f"Best Meta-Learner: {results[best_version]['best_meta_name']}")

print(f"\nAll outputs saved to:")
print(f"Models: {models_dir}")
print(f"Visualizations: {output_dir}")
print(f"\nTip: To retrain base models, delete:")
print(f"{BASE_MODELS_PATH}")

print("\n" + "=" * 80)