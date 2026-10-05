import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)
print("--- Перші 5 екземплярів датасету (Alcohol та Flavanoids) ---")
print(df[['alcohol', 'flavanoids']].head())

X = wine.data[:, [0, 6]] 
y_true = wine.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def visualize_clusters(X_data, true_labels, predicted_labels, centroids, k):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_data[:, 0], X_data[:, 1], c=true_labels, cmap='viridis', edgecolor='k')
    plt.title("Справжні класи (Original Labels)")
    plt.xlabel("Alcohol (стандартизовано)")
    plt.ylabel("Flavanoids (стандартизовано)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(X_data[:, 0], X_data[:, 1], c=predicted_labels, cmap='viridis', edgecolor='k')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=250, c='red', marker='X', label='Барицентри')
    plt.title(f"K-Means Кластеризація (k={k})")
    plt.xlabel("Alcohol (стандартизовано)")
    plt.ylabel("Flavanoids (стандартизовано)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

clusters_range = [2, 3, 4, 5, 6]
random_states = [0, 42, 100]

best_ari = -1
best_params = {}
results = []

for k in clusters_range:
    for rs in random_states:
        kmeans = KMeans(n_clusters=k, random_state=rs, n_init='auto')
        predicted_labels = kmeans.fit_predict(X_scaled)
        
        ari = adjusted_rand_score(y_true, predicted_labels)
        results.append({'Кластери (k)': k, 'random_state': rs, 'ARI Score': round(ari, 4)})
        
        if ari > best_ari:
            best_ari = ari
            best_params = {
                'k': k, 
                'rs': rs, 
                'labels': predicted_labels, 
                'centroids': kmeans.cluster_centers_
            }

results_df = pd.DataFrame(results)
print("\n--- Результати оцінки Adjusted Rand Index (ARI) ---")
print(results_df.to_string(index=False))
print(f"\nНайкраща комбінація знайдена при:")
print(f"Кількість кластерів (k): {best_params['k']}")
print(f"Параметр random_state: {best_params['rs']}")
print(f"Максимальний ARI = {best_ari:.4f}")

visualize_clusters(X_scaled, y_true, best_params['labels'], best_params['centroids'], best_params['k'])
