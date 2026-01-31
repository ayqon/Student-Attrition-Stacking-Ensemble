import pandas as pd
import numpy as np
import pickle
import os
import time
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_predict, GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    recall_score, precision_score, classification_report,
    precision_recall_fscore_support
)
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("DIRECT MULTI-CLASS STACKING ENSEMBLE WITH GRID SEARCH")
print("Base Models: LR, SVM, XGB, RF → Meta-Learner: Best of 4")
print("=" * 80)

# ==================== Configuration ====================
RANDOM_STATE = 42

base_path = r"D:\Warwick\WM9QG-FAIDM\GROUP"
output_dir = os.path.join(base_path, "output", "Direct_Stacking")
models_dir = os.path.join(base_path, "Models", "Direct_Stacking")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

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
    print(f"  {cls_name}: {count} ({count / len(y) * 100:.1f}%)")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain/Test Split (80/20, stratified):")
print(f"  Training: {X_train.shape[0]} samples")
print(f"  Test: {X_test.shape[0]} samples")

# ==================== Step 2: Define Base Models & Param Grids ====================
print("\n" + "=" * 80)
print("STEP 2: DEFINE BASE MODELS AND PARAMETER GRIDS")
print("=" * 80)

# Base model templates and their parameter grids
base_model_configs = {
    'LR': {
        'model': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        'param_grid': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'saga'],
            'penalty': ['l2']
        }
    },

    'SVM': {
        'model': SVC(
            probability=True,
            random_state=RANDOM_STATE
        ),
        'param_grid': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']
        }
    },

    'XGB': {
        'model': xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='mlogloss',
            verbosity=0
        ),
        'param_grid': {
            'n_estimators': [100, 150, 200],
            'max_depth': [5, 7, 9],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'min_child_weight': [1, 3]
        }
    },

    'RF': {
        'model': RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'param_grid': {
            'n_estimators': [150, 200, 250],
            'max_depth': [12, 15, 18],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
    }
}

print("Base Models Configuration:")
for name, config in base_model_configs.items():
    param_grid = config['param_grid']
    n_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"  {name}:")
    print(f"    Model: {config['model'].__class__.__name__}")
    print(f"    Param combinations: {n_combinations}")
    print(f"    Total CV fits: {n_combinations * 5}")

# ==================== Step 3: Grid Search for Base Models ====================
print("\n" + "=" * 80)
print("STEP 3: GRID SEARCH FOR OPTIMAL BASE MODELS")
print("=" * 80)

best_base_models = {}
grid_search_results = {}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, config in base_model_configs.items():
    print(f"\n{'=' * 60}")
    print(f"Grid Search: {name}")
    print(f"{'=' * 60}")

    model = config['model']
    param_grid = config['param_grid']

    print(f"Parameters to search:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )

    print(f"\nSearching...")
    grid_search.fit(X_train, y_train)

    elapsed = time.time() - start_time

    # Store best model
    best_base_models[name] = grid_search.best_estimator_

    # Store search results
    grid_search_results[name] = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_,
        'search_time': elapsed
    }

    print(f"\n✓ Best CV F1-Score: {grid_search.best_score_:.4f}")
    print(f"✓ Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"    {param}: {value}")
    print(f"✓ Search Time: {elapsed:.1f}s")

# ==================== Step 4: Train Final Base Models ====================
print("\n" + "=" * 80)
print("STEP 4: EVALUATE BASE MODELS ON TEST SET")
print("=" * 80)

trained_base_models = {}

print("\nBase models performance...")
print("-" * 80)
print(f"{'Model':<10} {'Test Acc':>10} {'Test F1':>10}")
print("-" * 80)

for name, model in best_base_models.items():
    # Evaluate on test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    trained_base_models[name] = model

    print(f"{name:<10} {acc:>10.4f} {f1:>10.4f}")

print("-" * 80)

# ==================== Step 5: Generate Meta-Features ====================
print("\n" + "=" * 80)
print("STEP 5: GENERATE META-FEATURES")
print("=" * 80)

# 5.1 Generate Training Meta-Features using Cross-Validation
print("\nGenerating Training Meta-Features (5-fold CV)...")
print("(Using cross_val_predict to avoid data leakage)")

n_classes = 3
n_base_models = len(trained_base_models)
meta_X_train = np.zeros((X_train.shape[0], n_base_models * n_classes))

for idx, (name, model) in enumerate(trained_base_models.items()):
    print(f"  {name}...", end=" ")
    start_time = time.time()

    # Use cross_val_predict to get out-of-fold predictions
    proba = cross_val_predict(
        model, X_train, y_train,
        cv=cv_strategy,
        method='predict_proba',
        n_jobs=-1
    )

    # Store probabilities for all 3 classes
    meta_X_train[:, idx * n_classes:(idx + 1) * n_classes] = proba

    elapsed = time.time() - start_time
    print(f"Done! ({elapsed:.1f}s)")

print(f"\nTraining meta-features shape: {meta_X_train.shape}")
print(f"  (4 base models × 3 class probabilities = 12 features)")

# 5.2 Generate Test Meta-Features
print("\nGenerating Test Meta-Features...")

meta_X_test = np.zeros((X_test.shape[0], n_base_models * n_classes))

for idx, (name, model) in enumerate(trained_base_models.items()):
    proba = model.predict_proba(X_test)
    meta_X_test[:, idx * n_classes:(idx + 1) * n_classes] = proba

print(f"Test meta-features shape: {meta_X_test.shape}")

# ==================== Step 6: Grid Search for Meta-Learners ====================
print("\n" + "=" * 80)
print("STEP 6: GRID SEARCH FOR OPTIMAL META-LEARNERS")
print("=" * 80)

meta_learner_configs = {
    'LR': {
        'model': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        'param_grid': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'saga']
        }
    },

    'SVM': {
        'model': SVC(
            probability=True,
            random_state=RANDOM_STATE
        ),
        'param_grid': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale'],
            'kernel': ['rbf']
        }
    },

    'XGB': {
        'model': xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='mlogloss',
            verbosity=0
        ),
        'param_grid': {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2]
        }
    },

    'RF': {
        'model': RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'param_grid': {
            'n_estimators': [100, 150, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5]
        }
    }
}

meta_results = {}
meta_search_results = {}

print("\nSearching for best meta-learner configurations...")
print("-" * 80)
print(f"{'Meta-Learner':<15} {'Best CV F1':>12} {'Test Acc':>10} {'Test F1':>10}")
print("-" * 80)

for name, config in meta_learner_configs.items():
    model = config['model']
    param_grid = config['param_grid']

    # Grid search on meta-features
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0,
        return_train_score=False
    )

    grid_search.fit(meta_X_train, y_train)

    # Get best model
    best_meta_model = grid_search.best_estimator_

    # Predict on test meta-features
    y_pred = best_meta_model.predict(meta_X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    meta_results[name] = {
        'model': best_meta_model,
        'predictions': y_pred,
        'accuracy': acc,
        'f1': f1,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_
    }

    meta_search_results[name] = {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }

    print(f"{name:<15} {grid_search.best_score_:>12.4f} {acc:>10.4f} {f1:>10.4f}")

print("-" * 80)

# Select best meta-learner based on test accuracy
best_meta_name = max(meta_results, key=lambda k: meta_results[k]['accuracy'])
best_meta = meta_results[best_meta_name]

print(f"\n🏆 Best Meta-Learner: {best_meta_name}")
print(f"   Best Parameters: {best_meta['best_params']}")
print(f"   CV F1-Score: {best_meta['cv_score']:.4f}")
print(f"   Test Accuracy: {best_meta['accuracy']:.4f}")
print(f"   Test F1-Score: {best_meta['f1']:.4f}")

# ==================== Step 7: Detailed Evaluation ====================
print("\n" + "=" * 80)
print("STEP 7: DETAILED EVALUATION OF BEST MODEL")
print("=" * 80)

class_names = ['Pass', 'Fail', 'Withdrawn']
y_pred_best = best_meta['predictions']

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=class_names))

cm = confusion_matrix(y_test, y_pred_best)
precision, recall, f1_per_class, support = precision_recall_fscore_support(
    y_test, y_pred_best, zero_division=0
)

print(f"\nConfusion Matrix:")
print(cm)

# ==================== Step 8: Save Models ====================
print("\n" + "=" * 80)
print("STEP 8: SAVE MODELS AND RESULTS")
print("=" * 80)

ensemble_package = {
    'base_models': trained_base_models,
    'base_model_search_results': grid_search_results,
    'meta_learners': meta_results,
    'meta_search_results': meta_search_results,
    'best_meta': {
        'name': best_meta_name,
        'model': best_meta['model'],
        'params': best_meta['best_params']
    },
    'metadata': {
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'random_state': RANDOM_STATE,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': X_train.shape[1],
        'accuracy': best_meta['accuracy'],
        'f1': best_meta['f1']
    }
}

model_path = os.path.join(models_dir, 'stacking_ensemble.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(ensemble_package, f)

print(f"Models saved to: {model_path}")
print(f"File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

# Save grid search results summary
grid_search_summary = []

print("\nBase Models - Best Parameters:")
for name, results in grid_search_results.items():
    print(f"\n{name}:")
    print(f"  CV F1: {results['best_score']:.4f}")
    print(f"  Parameters: {results['best_params']}")

    grid_search_summary.append({
        'Model_Type': 'Base',
        'Model_Name': name,
        'CV_F1_Score': results['best_score'],
        'Best_Params': str(results['best_params']),
        'Search_Time_s': results['search_time']
    })

print("\nMeta-Learners - Best Parameters:")
for name, results in meta_search_results.items():
    print(f"\n{name}:")
    print(f"  CV F1: {results['best_cv_score']:.4f}")
    print(f"  Parameters: {results['best_params']}")

    grid_search_summary.append({
        'Model_Type': 'Meta',
        'Model_Name': name,
        'CV_F1_Score': results['best_cv_score'],
        'Best_Params': str(results['best_params']),
        'Search_Time_s': None
    })

# Save to CSV
summary_df = pd.DataFrame(grid_search_summary)
summary_df.to_csv(os.path.join(output_dir, 'grid_search_summary.csv'), index=False)
print(f"\nGrid search summary saved to: grid_search_summary.csv")

# ==================== Step 9: Visualizations ====================
print("\n" + "=" * 80)
print("STEP 9: GENERATING VISUALIZATIONS")
print("=" * 80)

# Visualization 1: Confusion Matrix
print("\nCreating confusion matrix...")
fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})

ax.set_title(f'Stacking Ensemble - Confusion Matrix\n'
             f'Meta-Learner: {best_meta_name} | '
             f'Acc: {best_meta["accuracy"]:.4f} | F1: {best_meta["f1"]:.4f}',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('True', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ confusion_matrix.png")

# Visualization 2: Base Models Performance Comparison
print("\nCreating base models comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# CV scores vs Test scores
base_names = list(grid_search_results.keys())
cv_scores = [grid_search_results[m]['best_score'] for m in base_names]
test_scores = [f1_score(y_test, trained_base_models[m].predict(X_test), average='weighted')
               for m in base_names]

x = np.arange(len(base_names))
width = 0.35

bars1 = ax1.bar(x - width / 2, cv_scores, width, label='CV F1',
                color='#4ECDC4', alpha=0.8)
bars2 = ax1.bar(x + width / 2, test_scores, width, label='Test F1',
                color='#FF6B6B', alpha=0.8)

ax1.set_ylabel('F1-Score', fontsize=11)
ax1.set_title('Base Models: CV vs Test Performance', fontweight='bold', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(base_names)
ax1.legend()
ax1.set_ylim([0, 1])
ax1.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Meta-learners comparison
meta_names = list(meta_results.keys())
meta_cv_scores = [meta_results[m]['cv_score'] for m in meta_names]
meta_test_scores = [meta_results[m]['f1'] for m in meta_names]

x2 = np.arange(len(meta_names))

bars3 = ax2.bar(x2 - width / 2, meta_cv_scores, width, label='CV F1',
                color='#95E1D3', alpha=0.8)
bars4 = ax2.bar(x2 + width / 2, meta_test_scores, width, label='Test F1',
                color='#F38181', alpha=0.8)

ax2.set_ylabel('F1-Score', fontsize=11)
ax2.set_title('Meta-Learners: CV vs Test Performance', fontweight='bold', fontsize=12)
ax2.set_xticks(x2)
ax2.set_xticklabels(meta_names)
ax2.legend()
ax2.set_ylim([0, 1])
ax2.grid(axis='y', alpha=0.3)

for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ model_comparison.png")

# Visualization 3: Per-Class Performance
print("\nCreating per-class performance chart...")
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(class_names))
width = 0.25

bars1 = ax.bar(x - width, precision, width, label='Precision',
               color='#95E1D3', alpha=0.8)
bars2 = ax.bar(x, recall, width, label='Recall',
               color='#F38181', alpha=0.8)
bars3 = ax.bar(x + width, f1_per_class, width, label='F1-Score',
               color='#AA96DA', alpha=0.8)

ax.set_ylabel('Score', fontsize=11)
ax.set_title(f'Per-Class Performance (Meta: {best_meta_name})',
             fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'per_class_performance.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ per_class_performance.png")

# ==================== Final Summary ====================
print("\n" + "=" * 80)
print("STACKING ENSEMBLE WITH GRID SEARCH COMPLETE!")
print("=" * 80)

print(f"\nArchitecture:")
print(f"  Base Models (Layer 1): {len(trained_base_models)} models (LR, SVM, XGB, RF)")
print(f"  Each produces 3 probabilities → Total 12 meta-features")
print(f"  Meta-Learner (Layer 2): {best_meta_name}")

print(f"\nBase Models - Best CV F1 Scores:")
for name in base_names:
    print(f"  {name}: {grid_search_results[name]['best_score']:.4f}")

print(f"\nMeta-Learners - Best CV F1 Scores:")
for name in meta_names:
    print(f"  {name}: {meta_results[name]['cv_score']:.4f}")

print(f"\nFinal Test Results (Best Meta: {best_meta_name}):")
print(f"  Test Accuracy: {best_meta['accuracy']:.4f}")
print(f"  Test F1-Score: {best_meta['f1']:.4f}")

print(f"\nPer-Class F1-Scores:")
for i, cls in enumerate(class_names):
    print(f"  {cls}: {f1_per_class[i]:.4f}")

print(f"\nOutputs saved to:")
print(f"  Models: {models_dir}")
print(f"  Visualizations: {output_dir}")

print("\n" + "=" * 80)