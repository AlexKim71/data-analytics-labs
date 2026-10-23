import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def func_active(sigma, W01, x):
    temp_val = []
    for i in range(W01.shape[0]):
        val = ((W01[i, 0] - x[0]) ** 2 + (W01[i, 1] - x[1]) ** 2) / sigma ** 2
        temp_val.append(math.exp(-val))
    return temp_val

class my_RBF:
    def __init__(self, W01, sigma, n_classes):
        self.W01 = W01
        self.sigma = sigma
        self.n_classes = n_classes
        self.temp_layer = np.zeros(W01.shape[0])
        self.output_layer = np.zeros(n_classes)

    def rbf(self, point):
        self.temp_layer = func_active(self.sigma, self.W01, point)
        for i in range(self.n_classes):
            i1 = int(5 * i)
            self.output_layer[i] = sum(self.temp_layer[i1:i1 + 5])

iris = load_iris()
X = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

W01 = []
for c in range(3):
    W01.extend(X_train[y_train == c][:5])
W01 = np.array(W01)

sigma = 0.1
n_classes = 3

rbf1 = my_RBF(W01, sigma, n_classes)

point = X_test[0]
print(f"Координати тестової точки: {point}")

rbf1.rbf(point)

print("\nПроміжний шар (активації):")
for i in range(len(rbf1.temp_layer)):
    if i < 15: 
        print(f"Активація нейрона {i + 1}: {rbf1.temp_layer[i]:.6e}")
print("... (активації інших нейронів прямують до нуля)")

print("\nВихідний шар (сумарна активація для кожного класу):")
for i in range(len(rbf1.output_layer)):
    print(f"Сумарна активація для класу {i + 1}: {rbf1.output_layer[i]:.6e}")

predicted_class = rbf1.output_layer.argmax() + 1
print(f"\nТочка належить до класу: {predicted_class}")
print(f"Реальний клас: {y_test[0] + 1}")

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.5, label='Тренувальні дані')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='k', marker='o', s=75, label='Тестові дані')
plt.scatter(point[0], point[1], c='red', marker='*', s=250, label='Класифікована точка')

plt.title('Класифікація даних Iris за допомогою RBF')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.grid(True)
plt.show()
