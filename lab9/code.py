import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

X = np.arange(1, 31)
Y = np.array([3.0, 4.4, 5.0, 9.0, 8.2, 13.8, 15.0, 17.0, 16.0, 20.0,
              22.0, 21.4, 25.0, 24.0, 27.0, 30.8, 32.67, 34.53, 36.4,
              38.27, 40.13, 42.0, 43.87, 45.73, 47.6, 49.47, 51.33,
              53.2, 55.07, 56.93])

print("ЗАВДАННЯ 1")

p1 = np.polyfit(X, Y, 1)
p2 = np.polyfit(X, Y, 2)
Y1 = np.polyval(p1, X)
Y2 = np.polyval(p2, X)

print(f"Поліном 1 ст.: {p1[0]:.4f}·t + {p1[1]:.4f}")
print(f"Поліном 2 ст.: {p2[0]:.4f}·t² + {p2[1]:.4f}·t + {p2[2]:.4f}")
print(f"Макс. похибка (1 ст.): {np.max(np.abs(Y - Y1)):.4f}")
print(f"Макс. похибка (2 ст.): {np.max(np.abs(Y - Y2)):.4f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(X, Y, 'ko-', label='Вихідний ряд')
ax[0].plot(X, Y1, 'b--', label='Поліном 1 ст.')
ax[0].plot(X, Y2, 'r-.', label='Поліном 2 ст.')
ax[0].set_title('Вихідний ряд і поліноміальні моделі')
ax[0].set_xlabel('t'); ax[0].set_ylabel('Y')
ax[0].legend(); ax[0].grid(True)

ax[1].bar(X - 0.2, Y - Y1, width=0.4, alpha=0.7, label='Похибка 1 ст.', color='blue')
ax[1].bar(X + 0.2, Y - Y2, width=0.4, alpha=0.7, label='Похибка 2 ст.', color='red')
ax[1].set_title('Гістограма відхилень моделей')
ax[1].axhline(0, color='black', linewidth=0.8)
ax[1].legend(); ax[1].grid(True)
plt.tight_layout(); plt.show()

def s4(t, b, c):
    return np.exp(-b * t) + np.exp(-c * t)

t_norm = (X - X.min()) / (X.max() - X.min()) * 3.0

try:
    popt, _ = curve_fit(s4, t_norm, Y, p0=[0.2, 0.5], bounds=([0, 0], [5, 5]), maxfev=20000)
    Y4 = s4(t_norm, *popt)
    print(f"Модель s4: b = {popt[0]:.4f}, c = {popt[1]:.4f}")
    print(f"Макс. похибка s4: {np.max(np.abs(Y - Y4)):.4f}")
except Exception:
    Y4 = s4(t_norm, 0.5, 1.0)

plt.figure(figsize=(9, 4))
plt.plot(X, Y, 'ko-', label='Вихідний ряд')
plt.plot(X, Y4, 'm--', label='s4(t)=exp(-bt)+exp(-ct)')
plt.title('Модель s4(t) = exp(-bt) + exp(-ct)')
plt.legend(); plt.grid(True); plt.show()

arima_model = ARIMA(Y, order=(1, 1, 1)).fit()
forecast = arima_model.forecast(steps=5)
print(f"ARIMA(1,1,1) прогноз: {np.round(forecast.values, 3)}")

plt.figure(figsize=(10, 4))
plt.plot(X, Y, 'ko-', label='Вихідний ряд')
plt.plot(np.arange(31, 36), forecast.values, 'rs--', label='Прогноз ARIMA')
plt.title('Прогноз ARIMA(1,1,1) на 5 кроків')
plt.legend(); plt.grid(True); plt.show()

np.random.seed(42)
Y_noisy = Y + (np.random.rand(len(Y)) - 0.5)

p1n = np.polyfit(X, Y_noisy, 1)
p2n = np.polyfit(X, Y_noisy, 2)
Y1n = np.polyval(p1n, X)
Y2n = np.polyval(p2n, X)
print(f"Зашумлений ряд — макс. похибка (1 ст.): {np.max(np.abs(Y_noisy-Y1n)):.4f}")
print(f"Зашумлений ряд — макс. похибка (2 ст.): {np.max(np.abs(Y_noisy-Y2n)):.4f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(X, Y_noisy, 'ko', alpha=0.6, label='Зашумлений ряд')
ax[0].plot(X, Y1n, 'b--', label='Поліном 1 ст.')
ax[0].plot(X, Y2n, 'r-.', label='Поліном 2 ст.')
ax[0].set_title('Зашумлений ряд і поліноми')
ax[0].legend(); ax[0].grid(True)

ax[1].bar(X - 0.2, Y_noisy - Y1n, width=0.4, alpha=0.7, label='Похибка 1 ст.', color='blue')
ax[1].bar(X + 0.2, Y_noisy - Y2n, width=0.4, alpha=0.7, label='Похибка 2 ст.', color='red')
ax[1].set_title('Гістограма відхилень (зашумлений ряд)')
ax[1].axhline(0, color='black', linewidth=0.8)
ax[1].legend(); ax[1].grid(True)
plt.tight_layout(); plt.show()

arima_n = ARIMA(Y_noisy, order=(1, 1, 1)).fit()
forecast_n = arima_n.forecast(steps=5)
print(f"ARIMA прогноз (зашумлений ряд): {np.round(forecast_n.values, 3)}")

print("\nЗАВДАННЯ 2 (Варіант 2)")

a = 2
b = 5
c = 7

t = np.arange(0, 3.01, 0.01)
s1 = a + b * t
s2 = a + b * t + c * t**2

plt.figure(figsize=(10, 4))
plt.plot(t, s1, label='s1 = a + bt')
plt.plot(t, s2, label='s2 = a + bt + ct²')
plt.title(f'Моделі s1 та s2 (a={a}, b={b}, c={c})')
plt.xlabel('t'); plt.ylabel('s')
plt.legend(); plt.grid(True); plt.show()

b_vals = np.arange(0, 5.01, 0.05)
c_vals = np.arange(0, 5.01, 0.05)
B, C = np.meshgrid(b_vals, c_vals)
SSE = np.zeros_like(B)

for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        s2_test = a + B[i, j] * t_norm + C[i, j] * t_norm**2
        SSE[i, j] = np.sum((Y - s2_test) ** 2)

fig = plt.figure(figsize=(10, 6))
ax3d = fig.add_subplot(111, projection='3d')
surf = ax3d.plot_surface(B, C, SSE, cmap='viridis')
ax3d.set_xlabel('b'); ax3d.set_ylabel('c')
ax3d.set_zlabel('SSE')
ax3d.set_title('Поверхня помилки SSE(b, c)')
fig.colorbar(surf, shrink=0.6)
plt.show()

idx = np.unravel_index(np.argmin(SSE), SSE.shape)
b_opt, c_opt = B[idx], C[idx]
print(f"Оптимальні: b = {b_opt:.3f}, c = {c_opt:.3f}, SSE_min = {SSE[idx]:.4f}")

def moving_average(data, n=3):
    return np.convolve(data, np.ones(n) / n, mode='valid')

def lwma(data, n=3):
    w = np.arange(1, n + 1)
    return np.convolve(data, w / w.sum(), mode='valid')

def exp_smooth(data, alpha):
    out = np.zeros_like(data, dtype=float)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out

def rmse(a_, b_):
    return np.sqrt(np.mean((np.asarray(a_) - np.asarray(b_)) ** 2))

N = 3
alpha = 2 / (N + 1)
Y_ma   = moving_average(Y, N)
Y_lwma = lwma(Y, N)
Y_exp  = exp_smooth(Y, alpha)

r_ma   = rmse(Y[N-1:], Y_ma)
r_lwma = rmse(Y[N-1:], Y_lwma)
r_exp  = rmse(Y, Y_exp)

print(f"RMSE Ковзне середнє (N={N}):       {r_ma:.4f}")
print(f"RMSE Лінійно-зважене (N={N}):     {r_lwma:.4f}")
print(f"RMSE Експоненційне (α={alpha:.2f}): {r_exp:.4f}")

plt.figure(figsize=(11, 5))
plt.plot(X, Y, 'ko-', label='Вихідний ряд', lw=2)
plt.plot(X[N-1:], Y_ma, 'r--', label=f'Ковзне середнє (N={N})')
plt.plot(X[N-1:], Y_lwma, 'g-.', label=f'Лінійно-зважене (N={N})')
plt.plot(X, Y_exp, 'b:', label=f'Експоненційне (α={alpha:.2f})')
plt.title('Порівняння методів згладжування')
plt.xlabel('t'); plt.ylabel('Y')
plt.legend(); plt.grid(True); plt.show()
