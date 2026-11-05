import matplotlib.pyplot as plt
import numpy as np

# Настройка стиля графика
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(3, 1, figsize=(10, 14))

# =========================================================
# Завдання 1: Дискретні нечіткі множини та операції над ними
# =========================================================
x_labels = ['$x_1$', '$x_2$', '$x_3$', '$x_4$', '$x_5$']
x_indices = np.arange(len(x_labels))

a1 = [0.6, 0.8, 0.3, 1.0, 0.2]
a2 = [0.1, 1.0, 0.2, 0.5, 0.0]

union = [max(a1[i], a2[i]) for i in range(5)]
intersection = [min(a1[i], a2[i]) for i in range(5)]

width = 0.2
axs[0].bar(x_indices - 1.5*width, a1, width, label='$A_1$', color='#1f77b4')
axs[0].bar(x_indices - 0.5*width, a2, width, label='$A_2$', color='#ff7f0e')
axs[0].bar(x_indices + 0.5*width, union, width, label='$A_1 \\cup A_2$ (АБО / max)', color='#2ca02c')
axs[0].bar(x_indices + 1.5*width, intersection, width, label='$A_1 \\cap A_2$ (ТА / min)', color='#d62728')

axs[0].set_title('Завдання 1: Операції над дискретними нечіткими множинами', fontsize=12, fontweight='bold')
axs[0].set_xticks(x_indices)
axs[0].set_xticklabels(x_labels, fontsize=11)
axs[0].set_ylabel('Ступінь належності $\\mu(x)$', fontsize=11)
axs[0].set_ylim(0, 1.2)
axs[0].legend(loc='upper right', frameon=True)

# =========================================================
# Завдання 2: Лінгвістична змінна «Успішність»
# =========================================================
x2 = np.linspace(0, 100, 1000)

def mu_low(x):
    return np.piecewise(x, [x <= 50, (x > 50) & (x <= 60), x > 60],
                        [1.0, lambda x: (60 - x) / 10.0, 0.0])

def mu_satisfactory(x):
    return np.piecewise(x, [x < 50, (x >= 50) & (x <= 65), (x > 65) & (x <= 75), x > 75],
                        [0.0, lambda x: (x - 50) / 15.0, lambda x: (75 - x) / 10.0, 0.0])

def mu_good(x):
    return np.piecewise(x, [x < 70, (x >= 70) & (x <= 80), (x > 80) & (x <= 90), x > 90],
                        [0.0, lambda x: (x - 70) / 10.0, lambda x: (90 - x) / 10.0, 0.0])

def mu_excellent(x):
    return np.piecewise(x, [x <= 85, (x > 85) & (x <= 95), x > 95],
                        [0.0, lambda x: (x - 85) / 10.0, 1.0])

axs[1].plot(x2, mu_low(x2), label='Низька', color='#e74c3c', linewidth=2.5)
axs[1].plot(x2, mu_satisfactory(x2), label='Задовільна', color='#f39c12', linewidth=2.5)
axs[1].plot(x2, mu_good(x2), label='Добра', color='#3498db', linewidth=2.5)
axs[1].plot(x2, mu_excellent(x2), label='Відмінна', color='#2ecc71', linewidth=2.5)

axs[1].set_title('Завдання 2: Графічний опис терм-множини «Успішність»', fontsize=12, fontweight='bold')
axs[1].set_xlabel('Бали (X в межах [0, 100])', fontsize=11)
axs[1].set_ylabel('Ступінь належності $\\mu(x)$', fontsize=11)
axs[1].set_ylim(-0.05, 1.1)
axs[1].legend(loc='upper right', frameon=True)

# =========================================================
# Завдання 3: Нечіткі числа ~3 та ~5 і операції «ТА» / «АБО»
# =========================================================
x3 = np.linspace(1, 7, 1000)

def mu_3(x):
    return np.piecewise(x, [x <= 2, (x > 2) & (x <= 3), (x > 3) & (x <= 4), x > 4],
                        [0.0, lambda x: x - 2, lambda x: 4 - x, 0.0])

def mu_5(x):
    return np.piecewise(x, [x <= 4, (x > 4) & (x <= 5), (x > 5) & (x <= 6), x > 6],
                        [0.0, lambda x: x - 4, lambda x: 6 - x, 0.0])

y3 = mu_3(x3)
y5 = mu_5(x3)
y_union = np.maximum(y3, y5)
y_inter = np.minimum(y3, y5)

axs[2].plot(x3, y3, '--', label='Число ~3 ($\\mu_3$)', color='#8e44ad', linewidth=2)
axs[2].plot(x3, y5, '--', label='Число ~5 ($\\mu_5$)', color='#16a085', linewidth=2)
axs[2].plot(x3, y_union, label='Об\'єднання ($\\mu_3 \\cup \\mu_5$ / АБО)', color='#d35400', linewidth=2.5)
axs[2].fill_between(x3, y_inter, color='#c0392b', alpha=0.5, label='Перетин ($\\mu_3 \\cap \\mu_5$ / ТА)')

axs[2].set_title('Завдання 3: Нечіткі числа ~3 та ~5, їх перетин та об\'єднання', fontsize=12, fontweight='bold')
axs[2].set_xlabel('Універсум (X)', fontsize=11)
axs[2].set_ylabel('Ступінь належності $\\mu(x)$', fontsize=11)
axs[2].set_ylim(-0.05, 1.1)
axs[2].legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.show()
