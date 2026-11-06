import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

path = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")
print("Path to dataset files:", path)

df = pd.read_csv(f"{path}/Mall_Customers.csv")

X = df.iloc[:, [3, 4]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='#1f77b4')
plt.title('Метод "ліктя" для реального датасету Mall_Customers', fontsize=12, fontweight='bold')
plt.xlabel('Кількість кластерів (K)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('real_elbow_method.png', dpi=300)
plt.show()

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)
df['Cluster'] = y_kmeans

centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.figure(figsize=(10, 6))
colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6', '#f39c12']

for i in range(optimal_k):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=60, c=colors[i], label=f'Кластер {i}', alpha=0.7)

plt.scatter(centers[:, 0], centers[:, 1], s=250, c='black', marker='X', label='Центроїди', edgecolor='white')
plt.title('Кластеризація покупців (датасет Mall_Customers, K=5)', fontsize=12, fontweight='bold')
plt.xlabel('Річний дохід (Annual Income, k$)')
plt.ylabel('Оцінка витрат (Spending Score, 1-100)')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('real_kmeans_clusters.png', dpi=300)
plt.show()

stats = df.groupby('Cluster')[['Age', df.columns[3], df.columns[4]]].agg(['count', 'mean']).round(2)
print(stats)
