import numpy as np
import matplotlib.pyplot as plt

# 1. Вхідні дані (Варіант 2)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
Y = np.array([4.6, 4.6, 5.6, 6.8, 7.4, 8.8, 9.2, 10.4, 11.0])
n = len(X)

# ==========================================
# 2. МЕТОД НАЙМЕНШИХ КВАДРАТІВ (МНК)
# ==========================================
sum_x = np.sum(X)
sum_y = np.sum(Y)
sum_x2 = np.sum(X**2)
sum_xy = np.sum(X * Y)

m_ols = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
c_ols = (sum_y - m_ols * sum_x) / n
Y_pred_ols = m_ols * X + c_ols
mse_ols = np.mean((Y - Y_pred_ols)**2)

print(f"МНК: y = {m_ols:.4f}x + {c_ols:.4f} | MSE = {mse_ols:.4f}")

# ==========================================
# 3. НЕЙРОННА МЕРЕЖА (Дельта-правило)
# ==========================================
np.random.seed(42)
w_nn = np.random.randn()
b_nn = np.random.randn()
lr = 0.01  # Швидкість навчання
epochs = 500

for epoch in range(epochs):
    for i in range(n):
        y_pred = w_nn * X[i] + b_nn
        error = Y[i] - y_pred
        w_nn += lr * error * X[i]
        b_nn += lr * error

Y_pred_nn = w_nn * X + b_nn
mse_nn = np.mean((Y - Y_pred_nn)**2)
print(f"Нейромережа: y = {w_nn:.4f}x + {b_nn:.4f} | MSE = {mse_nn:.4f}")

# ==========================================
# 4. ВІЗУАЛІЗАЦІЯ
# ==========================================
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='black', label='Експериментальні дані (Y2)', zorder=5)
plt.plot(X, Y_pred_ols, 'b--', label=f'МНК: y = {m_ols:.3f}x + {c_ols:.3f}')
plt.plot(X, Y_pred_nn, 'r-', label=f'Нейромережа: y = {w_nn:.3f}x + {b_nn:.3f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Апроксимація даних: МНК vs Нейронна мережа (Варіант 2)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ==========================================
# 5. ДОДАТКОВЕ ЗАВДАННЯ: МЕТОД ХЕББА (Логічне "І")
# ==========================================
# Входи (X1, X2, Bias) та виходи (Y) для функції AND (біполярні)
X_hebb = np.array([[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]])
Y_hebb = np.array([1, -1, -1, -1])
w_hebb = np.zeros(3) # Початкові ваги

print("\nНавчання за методом Хебба (AND):")
for i in range(len(X_hebb)):
    w_hebb += X_hebb[i] * Y_hebb[i]  # Правило Хебба: Δw = x * y
    print(f"Крок {i+1}: ваги = {w_hebb}")
