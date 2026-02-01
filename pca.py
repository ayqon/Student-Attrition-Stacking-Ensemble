from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


dir = 'FAIDM/'

df_final_scaled = pd.read_csv(dir + 'final_all_scaled_without_exam_score.csv')

X = df_final_scaled.drop('final_result', axis=1)
pca = PCA() 

X_pca = pca.fit_transform(X)

print(pca.explained_variance_.shape)
# print(pca.components_)
cum_pca = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cum_pca) ### Selecting 22 components can give us 90% variance

pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(X_pca, columns=pca_cols)

print(f"Original columns: {X.shape[1]}")
print(f"Reduced columns: {df_pca.shape[1]}")

df_pca['final_result'] = df_final['final_result'].values
df_pca.to_csv('FAIDM/final_with_pca.csv', index=False)