import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 🎯 Simulated purchase data (e.g., from a retail store)
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Annual_Spend': [15000, 16000, 30000, 12000, 14000, 32000, 35000, 10000, 9000, 40000],
    'Online_Spend': [2000, 2200, 7000, 1000, 1800, 8000, 8500, 900, 800, 9000]
}

df = pd.DataFrame(data)

# 🧼 Preprocessing
features = df[['Annual_Spend', 'Online_Spend']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 📊 Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# 🔍 Visualize the clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Annual_Spend', y='Online_Spend', hue='Cluster', palette='viridis', s=100)
plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Spend')
plt.ylabel('Online Spend')
plt.grid(True)
plt.show()

# 🧠 Optional: Inspect cluster centers (rescaled to original scale)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
print("\nCluster Centers (Original Scale):")
for i, center in enumerate(centroids):
    print(f"Cluster {i}: Annual Spend = {center[0]:.2f}, Online Spend = {center[1]:.2f}")
