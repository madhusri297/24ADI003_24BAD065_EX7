print("Madhusri S-24BAD065")

# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 2. Load dataset (sample data)
df = pd.DataFrame({
    'Annual Income': [15,16,17,18,19,20,60,62,64,65],
    'Spending Score': [39,81,6,77,40,76,55,60,65,70]
})

print("\nDataset:")
print(df)

# 3. Preprocessing (Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 4. Apply GMM
k = 3
gmm = GaussianMixture(n_components=k, random_state=42)
gmm.fit(X_scaled)

# 5. Predict probabilities
probabilities = gmm.predict_proba(X_scaled)

print("\nCluster Probabilities:")
print(probabilities)

# 6. Assign clusters (highest probability)
gmm_labels = np.argmax(probabilities, axis=1)
df['GMM_Cluster'] = gmm_labels

print("\nGMM Cluster Labels:")
print(df)

# 7. Evaluation Metrics
log_likelihood = gmm.score(X_scaled)
aic = gmm.aic(X_scaled)
bic = gmm.bic(X_scaled)
sil_score_gmm = silhouette_score(X_scaled, gmm_labels)

print("\nGMM Evaluation:")
print("Log-Likelihood:", log_likelihood)
print("AIC:", aic)
print("BIC:", bic)
print("Silhouette Score:", sil_score_gmm)

# 8. K-Means for comparison
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

df['KMeans_Cluster'] = kmeans_labels

sil_score_kmeans = silhouette_score(X_scaled, kmeans_labels)

print("\nK-Means Silhouette Score:", sil_score_kmeans)

# 9. Visualization

# GMM Clusters
plt.figure()
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=gmm_labels)
plt.title("GMM Clustering")
plt.xlabel("Income (scaled)")
plt.ylabel("Spending (scaled)")
plt.show()

# K-Means Clusters
plt.figure()
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=kmeans_labels)
plt.title("K-Means Clustering")
plt.xlabel("Income (scaled)")
plt.ylabel("Spending (scaled)")
plt.show()

# Probability Distribution (for first cluster)
plt.figure()
plt.hist(probabilities[:,0])
plt.title("Cluster 0 Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.show()