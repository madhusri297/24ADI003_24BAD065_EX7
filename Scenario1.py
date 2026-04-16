print("Madhusri S-24BAD065")

# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 2. Load dataset
# Replace with your dataset path
# df = pd.read_csv("Mall_Customers.csv")

# Sample dataset (if file not available)
df = pd.DataFrame({
    'Annual Income': [15,16,17,18,19,20,60,62,64,65],
    'Spending Score': [39,81,6,77,40,76,55,60,65,70]
})

print("\nDataset:")
print(df.head())

# 3. Preprocessing (check missing values)
print("\nMissing Values:\n", df.isnull().sum())

# 4. Select features
X = df[['Annual Income', 'Spending Score']]

# 5. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Elbow Method
inertia = []
K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure()
plt.plot(K, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.show()

# 7. Apply K-Means (choose optimal K, e.g., 3)
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# 8. Assign cluster labels
df['Cluster'] = clusters

print("\nClustered Data:")
print(df)

# 9. Silhouette Score
score = silhouette_score(X_scaled, clusters)
print("\nSilhouette Score:", score)

# 10. Visualization
plt.figure()
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters)
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            marker='X', s=200)
plt.title("K-Means Clusters")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.show()

# 11. Interpretation
print("\nCluster Interpretation:")
for i in range(k_optimal):
    print(f"Cluster {i}:")
    print(df[df['Cluster']==i].describe())