import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.rcParams['figure.figsize'] = (12, 6)

# DATA LOADING
df_pca = pd.read_csv('final_with_pca.csv')

# Separate features and target
X = df_pca.drop('final_result', axis=1)
y = df_pca['final_result']

for result, count in y.value_counts().items():
    print(f"    • {result}: {count:,} ({count/len(y)*100:.1f}%)")

# ELBOW METHOD - FIND OPTIMAL K
K_range = range(1, 11)
inertias = []
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    if k >= 2:
        sil_score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(sil_score)
        print(f"  K={k}: Inertia={kmeans.inertia_:,.0f}, Silhouette={sil_score:.4f}")
    else:
        silhouette_scores.append(np.nan)  # Silhouette not defined for K=1
        print(f"  K={k}: Inertia={kmeans.inertia_:,.0f}, Silhouette=N/A")

# Plot elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
ax1.set_title('Elbow Method - Inertia', fontsize=14, fontweight='bold')
ax1.set_xticks(list(K_range))
ax1.grid(True, alpha=0.3)

ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score by K', fontsize=14, fontweight='bold')
ax2.set_xticks(list(K_range))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved to: elbow_silhouette.png")
plt.close()

# Find optimal K (highest silhouette score, ignoring NaN for K=1)
optimal_k_idx = np.nanargmax(silhouette_scores)
OPTIMAL_K = list(K_range)[optimal_k_idx]

# FIT FINAL K-MEANS MODEL
kmeans_final = KMeans(
    n_clusters=OPTIMAL_K, 
    random_state=42, 
    n_init=20,  
    max_iter=500
)
clusters = kmeans_final.fit_predict(X)

# Add cluster labels to dataframe
df_pca['cluster'] = clusters
cluster_counts = df_pca['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    print(f"  Cluster {cluster_id}: {count:,} students ({count/len(df_pca)*100:.1f}%)")

# EVALUATE CLUSTERING QUALITY
silhouette = silhouette_score(X, clusters)

# CLUSTER ANALYSIS
# Cross-tabulation of clusters and final results
cluster_result_crosstab = pd.crosstab(
    df_pca['cluster'], 
    df_pca['final_result'], 
    normalize='index'
) * 100

# Identify at-risk clusters (high failure rate)
fail_rates = cluster_result_crosstab.get('Fail', pd.Series(0, index=cluster_result_crosstab.index))
at_risk_threshold = 30  # 30% failure rate
at_risk_clusters = fail_rates[fail_rates > at_risk_threshold].index.tolist()

if at_risk_clusters:
    for cluster_id in at_risk_clusters:
        fail_pct = fail_rates[cluster_id]
        count = cluster_counts[cluster_id]

# Visualize cluster composition
# Colors mapped to alphabetical order: Distinction, Fail, Pass, Withdrawn
fig, ax = plt.subplots(figsize=(12, 6))
cluster_result_crosstab.plot(kind='bar', stacked=True, ax=ax, 
                             color=['#9b59b6',   # Distinction - purple (achievement)
                                    '#e74c3c',   # Fail - red (bad)
                                    '#2ecc71',   # Pass - green (good)
                                    '#f39c12'])  # Withdrawn - orange (neutral)
ax.set_title(f'Cluster Composition by Final Result (K={OPTIMAL_K})', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.legend(title='Final Result', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig('cluster_composition.png', dpi=300, bbox_inches='tight')
plt.close()

# Create PCA visualization for K=2
if OPTIMAL_K == 2:
    # Perform PCA for visualization (reduce to 2 components)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Calculate centroids in PCA space
    centroids = kmeans_final.cluster_centers_
    centroids_pca = pca.transform(centroids)
    
    # Calculate RISK RATE = Fail / (Total - Withdrawn)
    # Withdrawn students are excluded from consideration
    risk_rates_dict = {}
    cluster_names = {}
    
    for cluster_id in range(OPTIMAL_K):
        cluster_mask = df_pca['cluster'] == cluster_id
        cluster_data = df_pca[cluster_mask]
        fail_count = (cluster_data['final_result'] == 'Fail').sum()
        withdrawn_count = (cluster_data['final_result'] == 'Withdrawn').sum()
        completed = len(cluster_data) - withdrawn_count
        risk_rate = (fail_count / completed) if completed > 0 else 0
        risk_rates_dict[cluster_id] = risk_rate
    
    # Calculate overall risk rate (for reference line)
    total_fail = (df_pca['final_result'] == 'Fail').sum()
    total_withdrawn = (df_pca['final_result'] == 'Withdrawn').sum()
    total_completed = len(df_pca) - total_withdrawn
    overall_risk_rate = (total_fail / total_completed) if total_completed > 0 else 0
    
    # Determine which cluster is at-risk (higher risk rate)
    at_risk_cluster = 0 if risk_rates_dict[0] > risk_rates_dict[1] else 1
    successful_cluster = 1 - at_risk_cluster
    
    cluster_names[at_risk_cluster] = 'At-Risk'
    cluster_names[successful_cluster] = 'Successful'
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define colors for K=2: Green for Successful, Red for At-Risk
    colors_k2 = ['#2ecc71', '#e74c3c']
    
    # Plot 1: PCA Scatter
    for cluster_id in range(OPTIMAL_K):
        mask = df_pca['cluster'] == cluster_id
        color = '#e74c3c' if cluster_id == at_risk_cluster else '#2ecc71'
        ax = axes[0]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=color,
                   label=f'{cluster_names[cluster_id]} (n={mask.sum():,})',
                   alpha=0.6, s=40, edgecolor='white', linewidth=0.5)
    
    # Add centroids
    axes[0].scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='black', marker='X',
               s=300, edgecolor='white', linewidth=2, zorder=5, label='Centroids')
    
    axes[0].set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    axes[0].set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    axes[0].set_title('K=2 Clustering: At-Risk vs Successful Students', fontsize=14, fontweight='bold')
    axes[0].legend(title='Student Segment', fontsize=10, title_fontsize=11, loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: At-Risk Rate Comparison
    cluster_ids = list(range(OPTIMAL_K))
    risk_rates_list = [risk_rates_dict[i] * 100 for i in cluster_ids]
    bar_colors = ['#e74c3c' if i == at_risk_cluster else '#2ecc71' for i in cluster_ids]
    
    bars = axes[1].bar(cluster_ids, risk_rates_list, color=bar_colors, edgecolor='black', linewidth=2)
    axes[1].axhline(y=overall_risk_rate*100, color='gray', linestyle='--',
                    linewidth=2, label=f'Overall: {overall_risk_rate*100:.1f}%')
    axes[1].set_xlabel('Cluster', fontsize=12)
    axes[1].set_ylabel('At-Risk Rate (%)', fontsize=12)
    axes[1].set_title('At-Risk Rate by Cluster', fontsize=14, fontweight='bold')
    axes[1].set_xticks(cluster_ids)
    axes[1].set_xticklabels([f'{i}\n{cluster_names[i]}' for i in cluster_ids], fontsize=10)
    axes[1].legend(loc='upper right')
    
    for bar, rate in zip(bars, risk_rates_list):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{rate:.1f}%', ha='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('k2_clustering_results.png', dpi=150, bbox_inches='tight')
    plt.close()

# CALCULATE METRICS AND SAVE RESULTS

# Save clustered data
output_file = 'clustered_students.csv'
df_pca.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")

# Calculate FAIL RATE and RISK RATE for each cluster
# FAIL RATE = students who got "Fail" / total students in cluster
# RISK RATE = students who got "Fail" OR "Withdrawn" / total students in cluster

cluster_stats = []
for cluster_id in sorted(df_pca['cluster'].unique()):
    cluster_mask = df_pca['cluster'] == cluster_id
    cluster_data = df_pca[cluster_mask]
    total = len(cluster_data)
    
    # Count each outcome
    fail_count = (cluster_data['final_result'] == 'Fail').sum()
    withdrawn_count = (cluster_data['final_result'] == 'Withdrawn').sum()
    pass_count = (cluster_data['final_result'] == 'Pass').sum()
    distinction_count = (cluster_data['final_result'] == 'Distinction').sum()
    
    # Calculate RISK RATE = Fail / (Total - Withdrawn)
    # Withdrawn students are excluded from consideration
    completed_students = total - withdrawn_count
    risk_rate = (fail_count / completed_students) * 100 if completed_students > 0 else 0
    
    cluster_stats.append({
        'cluster': cluster_id,
        'student_count': total,
        'percentage': round((total / len(df_pca)) * 100, 2),
        'fail_count': fail_count,
        'withdrawn_count': withdrawn_count,
        'completed_count': completed_students,
        'pass_count': pass_count,
        'distinction_count': distinction_count,
        'risk_rate': round(risk_rate, 2)
    })

# Determine which cluster is At-Risk (higher risk rate = At-Risk)
if cluster_stats[0]['risk_rate'] > cluster_stats[1]['risk_rate']:
    cluster_stats[0]['label'] = 'At-Risk'
    cluster_stats[1]['label'] = 'Successful'
else:
    cluster_stats[0]['label'] = 'Successful'
    cluster_stats[1]['label'] = 'At-Risk'

# Create summary dataframe
summary_df = pd.DataFrame(cluster_stats)
summary_df.to_csv('cluster_summary.csv', index=False)
print(f"✓ Saved: cluster_summary.csv")

# Save metrics
metrics_df = pd.DataFrame({
    'metric': ['Optimal K', 'Silhouette Score'],
    'value': [OPTIMAL_K, round(silhouette, 4)]
})
metrics_df.to_csv('clustering_metrics.csv', index=False)
print(f"✓ Saved: clustering_metrics.csv")

# PRINT SUMMARY
print("\n" + "="*75)
print("                    CLUSTER PROFILE SUMMARY")
print("="*75)

# Get stats for easy access
c0 = cluster_stats[0]
c1 = cluster_stats[1]

# Pre-format values to ensure consistent column widths
c0_size = f"{c0['student_count']:,} ({c0['percentage']}%)"
c1_size = f"{c1['student_count']:,} ({c1['percentage']}%)"
c0_risk = f"{c0['risk_rate']}%"
c1_risk = f"{c1['risk_rate']}%"

print()
print("┌─────────────────────┬─────────────────────────┬─────────────────────────┐")
print("│ Metric              │ Cluster 0               │ Cluster 1               │")
print("├─────────────────────┼─────────────────────────┼─────────────────────────┤")
print(f"│ Label               │ {c0['label']:<23} │ {c1['label']:<23} │")
print(f"│ Size                │ {c0_size:<23} │ {c1_size:<23} │")
print(f"│ Risk Rate           │ {c0_risk:<23} │ {c1_risk:<23} │")
print("└─────────────────────┴─────────────────────────┴─────────────────────────┘")
print()

print("NOTES:")
print("-"*75)
print(f"• RISK RATE = Fail / (Total - Withdrawn)")
print(f"    Withdrawn students are excluded (not considered at-risk)")
print()
print(f"    Cluster 0: {c0['fail_count']:,} failed / ({c0['student_count']:,} - {c0['withdrawn_count']:,} withdrawn) = {c0['risk_rate']}%")
print(f"    Cluster 1: {c1['fail_count']:,} failed / ({c1['student_count']:,} - {c1['withdrawn_count']:,} withdrawn) = {c1['risk_rate']}%")
print()
print(f"• AT-RISK LABEL = Cluster with HIGHER risk rate is labeled 'At-Risk'")
print("-"*75)
print(f"\nSilhouette Score: {silhouette:.4f}")
print(f"Total Students: {len(df_pca):,}")
print("="*75)