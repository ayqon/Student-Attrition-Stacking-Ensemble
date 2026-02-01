"""
Random Forest Classification - 4-Class Student Outcome Prediction
Using finaltable_all_modules.csv with 80/20 stratified split
"""

import pandas as pd
import numpy as np
import os
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, recall_score, precision_score, precision_recall_fscore_support
)
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

# ==================== Configuration ====================
base_path = r"C:\Users\Ayan\Project_CNN\Project_Gesture_recognition\Ayan's_Assignments\Group_assignments"
output_dir = os.path.join(base_path, "output", "RF_outputs")
os.makedirs(output_dir, exist_ok=True)

data_path = r"C:\Users\Ayan\Project_CNN\Project_Gesture_recognition\Ayan's_Assignments\Group_assignments\output\Input\final_scaled_without_exam_score (1).csv"

print("=" * 80)
print("RANDOM FOREST CLASSIFICATION")
print("3-Class Student Outcome Prediction")
print("=" * 80)

# ==================== Load Data ====================
print("\n📂 Loading Data...")
df = pd.read_csv(data_path)

print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# Prepare features and target
X = df.drop(['final_result'], axis=1, errors='ignore')
y = df['final_result']

print(f"\n📊 Target Distribution:")
print(y.value_counts())

# Encode string labels to numeric
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = list(label_encoder.classes_)
print(f"   Classes: {class_names}")
print(f"   Label encoding: {dict(zip(class_names, range(len(class_names))))}")

# ==================== Train/Test Split ====================
print("\n📋 Splitting Data (80/20 Stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)

print(f"   Training set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")

# ==================== Scale Features ====================
print("\n⚙️ Scaling Features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Features scaled using StandardScaler")

# ==================== Baseline Random Forest Model ====================
print("\n🚀 Training Baseline Random Forest...")

baseline_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

start_time = time.time()
baseline_model.fit(X_train_scaled, y_train)
baseline_time = time.time() - start_time

y_pred_baseline = baseline_model.predict(X_test_scaled)
baseline_acc = accuracy_score(y_test, y_pred_baseline)
baseline_f1 = f1_score(y_test, y_pred_baseline, average='weighted')

print(f"   Baseline Accuracy: {baseline_acc:.4f}")
print(f"   Baseline F1-Score: {baseline_f1:.4f}")
print(f"   Training Time: {baseline_time:.2f}s")

# ==================== GridSearchCV ====================
print("\n🔍 GridSearchCV Hyperparameter Tuning...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

total_combinations = (len(param_grid['n_estimators']) * len(param_grid['max_depth']) * 
                      len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) *
                      len(param_grid['max_features']))
print(f"   Parameter combinations: {total_combinations}")
print(f"   3-fold CV: {total_combinations * 3} total fits")

rf_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

start_time = time.time()
grid_search.fit(X_train_scaled, y_train)
grid_time = time.time() - start_time

print(f"\n   ✅ GridSearch completed in {grid_time:.1f}s ({grid_time/60:.1f} min)")

# ==================== Best Model Evaluation ====================
print("\n📊 Best Model Evaluation:")
print(f"   Best Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"     {param}: {value}")

print(f"   Best CV Score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)

best_acc = accuracy_score(y_test, y_pred_best)
best_f1 = f1_score(y_test, y_pred_best, average='weighted')
best_precision = precision_score(y_test, y_pred_best, average='weighted')
best_recall = recall_score(y_test, y_pred_best, average='weighted')

print(f"\n   🏆 Test Set Performance:")
print(f"     Accuracy:  {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"     Precision: {best_precision:.4f}")
print(f"     Recall:    {best_recall:.4f}")
print(f"     F1-Score:  {best_f1:.4f}")

print(f"\n   📈 Improvement over Baseline:")
print(f"     Accuracy: {(best_acc - baseline_acc)*100:+.2f}%")
print(f"     F1-Score: {(best_f1 - baseline_f1)*100:+.2f}%")

print(f"\n   📋 Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=class_names))

cm = confusion_matrix(y_test, y_pred_best)
print(f"   Confusion Matrix:")
print(cm)

precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
    y_test, y_pred_best, zero_division=0
)

# ==================== Feature Importance ====================
print("\n📊 Extracting Feature Importance...")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("   Top 10 Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"     {row['Feature']}: {row['Importance']:.4f}")

# ==================== Save Model ====================
model_path = os.path.join(output_dir, 'rf_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"\n💾 Model saved: {model_path}")

scaler_path = os.path.join(output_dir, 'rf_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"💾 Scaler saved: {scaler_path}")

encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
with open(encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"💾 Label encoder saved: {encoder_path}")

# ==================== Visualizations ====================
print("\n📸 Generating Visualizations...")

# 1. Main Results (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
models_list = ['Baseline', 'GridSearch Best']
acc_list = [baseline_acc, best_acc]
colors = ['#FF6B6B', '#4ECDC4']
bars = ax1.bar(models_list, acc_list, color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_ylim([0, 1])
ax1.set_title('Random Forest - Model Comparison', fontweight='bold')
for bar, acc in zip(bars, acc_list):
    ax1.text(bar.get_x() + bar.get_width()/2, acc + 0.02, f'{acc:.4f}', 
             ha='center', va='bottom', fontweight='bold')

ax2 = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False,
            xticklabels=class_names, yticklabels=class_names)
ax2.set_ylabel('Actual', fontsize=12)
ax2.set_xlabel('Predicted', fontsize=12)
ax2.set_title(f'Confusion Matrix\nAccuracy: {best_acc:.4f}', fontweight='bold')

ax3 = axes[1, 0]
x = np.arange(len(class_names))
width = 0.35
bars1 = ax3.bar(x - width/2, precision_per_class, width, label='Precision', color='#4ECDC4', alpha=0.8)
bars2 = ax3.bar(x + width/2, recall_per_class, width, label='Recall', color='#FF6B6B', alpha=0.8)
ax3.set_ylabel('Score', fontsize=12)
ax3.set_xticks(x)
ax3.set_xticklabels(class_names, rotation=45, ha='right')
ax3.set_ylim([0, 1.1])
ax3.legend()
ax3.set_title('Precision & Recall by Class', fontweight='bold')

ax4 = axes[1, 1]
top_features = feature_importance.head(10)
ax4.barh(range(len(top_features)), top_features['Importance'], color='#95E1D3', alpha=0.8, edgecolor='black')
ax4.set_yticks(range(len(top_features)))
ax4.set_yticklabels(top_features['Feature'])
ax4.set_xlabel('Importance', fontsize=12)
ax4.set_title('Top 10 Feature Importance', fontweight='bold')
ax4.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rf_results.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ rf_results.png")

# 2. Overall Accuracy Summary Table
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
title_text = 'RANDOM FOREST - OVERALL ACCURACY SUMMARY\n4-Class Classification'
ax.text(0.5, 0.95, title_text, ha='center', va='top', fontsize=16, fontweight='bold',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

table_data = [
    ['Metric', 'Value'],
    ['Overall Accuracy', f'{best_acc:.4f} ({best_acc*100:.2f}%)'],
    ['Weighted Precision', f'{best_precision:.4f} ({best_precision*100:.2f}%)'],
    ['Weighted Recall', f'{best_recall:.4f} ({best_recall*100:.2f}%)'],
    ['Weighted F1-Score', f'{best_f1:.4f} ({best_f1*100:.2f}%)'],
    ['Best n_estimators', f'{grid_search.best_params_["n_estimators"]}'],
    ['Best max_depth', f'{grid_search.best_params_["max_depth"]}'],
    ['Best min_samples_split', f'{grid_search.best_params_["min_samples_split"]}'],
    ['Test Samples', f'{len(y_test):,}'],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.4, 0.4], bbox=[0.1, 0.3, 0.8, 0.55])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

for i in range(2):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(table_data)):
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E8F4F8')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')

footer_text = f'Training Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
ax.text(0.5, 0.05, footer_text, ha='center', va='bottom', fontsize=10,
        transform=ax.transAxes, style='italic')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'overall_accuracy_summary.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ overall_accuracy_summary.png")

# 3. Class-wise Performance Table
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
title_text = 'CLASS-WISE PERFORMANCE METRICS\nRandom Forest Classification'
ax.text(0.5, 0.95, title_text, ha='center', va='top', fontsize=16, fontweight='bold',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

class_data = [['Class', 'Precision', 'Recall', 'F1-Score', 'Support']]
for i, name in enumerate(class_names):
    class_data.append([name, f'{precision_per_class[i]:.4f}', f'{recall_per_class[i]:.4f}', 
                       f'{f1_per_class[i]:.4f}', f'{int(support_per_class[i])}'])

class_table = ax.table(cellText=class_data, cellLoc='center', loc='center',
                      colWidths=[0.18, 0.18, 0.18, 0.18, 0.15], bbox=[0.05, 0.25, 0.9, 0.6])
class_table.auto_set_font_size(False)
class_table.set_fontsize(11)
class_table.scale(1, 2.5)

for i in range(5):
    class_table[(0, i)].set_facecolor('#95E1D3')
    class_table[(0, i)].set_text_props(weight='bold', color='white')

colors_rows = ['#E8F8F5', '#FADBD8', '#FCF3CF', '#D5D8DC']
for i in range(1, len(class_data)):
    for j in range(5):
        class_table[(i, j)].set_facecolor(colors_rows[(i-1) % len(colors_rows)])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_wise_performance.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ class_wise_performance.png")

# ==================== Save CSV Reports ====================
print("\n💾 Saving CSV Reports...")

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Test Samples', 'Train Samples'],
    'Value': [best_acc, best_precision, best_recall, best_f1, len(y_test), len(y_train)]
})
metrics_df.to_csv(os.path.join(output_dir, 'performance_metrics.csv'), index=False)
print("   ✅ performance_metrics.csv")

cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv(os.path.join(output_dir, 'confusion_matrix.csv'))
print("   ✅ confusion_matrix.csv")

params_df = pd.DataFrame(list(grid_search.best_params_.items()), columns=['Parameter', 'Value'])
params_df.to_csv(os.path.join(output_dir, 'best_parameters.csv'), index=False)
print("   ✅ best_parameters.csv")

class_metrics_df = pd.DataFrame({
    'Class': class_names,
    'Precision': precision_per_class,
    'Recall': recall_per_class,
    'F1-Score': f1_per_class,
    'Support': support_per_class
})
class_metrics_df.to_csv(os.path.join(output_dir, 'class_wise_metrics.csv'), index=False)
print("   ✅ class_wise_metrics.csv")

predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Actual_Label': label_encoder.inverse_transform(y_test),
    'Predicted': y_pred_best,
    'Predicted_Label': label_encoder.inverse_transform(y_pred_best)
})
predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
print("   ✅ predictions.csv")

feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
print("   ✅ feature_importance.csv")

model_info_df = pd.DataFrame({
    'Property': ['Model Type', 'Best n_estimators', 'Best max_depth', 'Best min_samples_split',
                 'Best min_samples_leaf', 'Best max_features', 'Accuracy', 'F1-Score', 
                 'Train Samples', 'Test Samples', 'Num Classes', 'Classes', 'Training Date'],
    'Value': ['Random Forest', grid_search.best_params_['n_estimators'], 
              grid_search.best_params_['max_depth'], grid_search.best_params_['min_samples_split'],
              grid_search.best_params_['min_samples_leaf'], grid_search.best_params_['max_features'],
              f'{best_acc:.4f}', f'{best_f1:.4f}', len(y_train), len(y_test),
              len(class_names), ', '.join(class_names), datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
})
model_info_df.to_csv(os.path.join(output_dir, 'model_info.csv'), index=False)
print("   ✅ model_info.csv")

print("\n" + "=" * 80)
print("✨ RANDOM FOREST TRAINING COMPLETE!")
print("=" * 80)
print(f"\n📊 Final Results:")
print(f"   Accuracy:  {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"   F1-Score:  {best_f1:.4f}")
print(f"\n📁 All outputs saved to: {output_dir}")
print("=" * 80)
