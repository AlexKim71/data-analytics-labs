import numpy as np
import matplotlib.pyplot as plt
import random

a, b, c = 5, 5, -3

def F(x):
    return a + b*x + c*x**2

def dF(x):
    return b + 2*c*x

X = np.arange(0, 32, 1)
Fd = F(X)

print("ЗАВДАННЯ 1")
x_ext = -b / (2*c)
print(f"Функція: F(x) = {c}x^2 + {b}x + {a}")
print(f"Перша похідна: F'(x) = {2*c}x + {b}")
print(f"Корінь F'(x)=0: x = {x_ext:.4f}")
print(f"Друга похідна: F''(x) = {2*c} < 0 -> Максимум")
print(f"Максимум: F({x_ext:.4f}) = {F(x_ext):.4f}")
print(f"Мінімум на відрізку [0, 31]: x=31, F(31) = {F(31)}")

plt.figure(figsize=(8, 4))
plt.plot(X, Fd, 'b-o', label='Fd(x)')
plt.scatter(x_ext, F(x_ext), color='red', zorder=5, label='Max')
plt.scatter(31, F(31), color='green', zorder=5, label='Min')
plt.title('Аналітичний пошук')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.legend()
plt.grid(True)
plt.show()

print("\nЗАВДАННЯ 2")
def gradient_ascent(start_x, lr, steps):
    x = start_x
    path = [x]
    for _ in range(steps):
        x += lr * dF(x)
        x = np.clip(x, 0, 31)
        path.append(x)
    return path

def gradient_descent(start_x, lr, steps):
    x = start_x
    path = [x]
    for _ in range(steps):
        x -= lr * dF(x)
        x = np.clip(x, 0, 31)
        path.append(x)
    return path

path_max = gradient_ascent(10, 0.1, 30)
path_min = gradient_descent(10, 0.1, 30)

print(f"Чисельний максимум: x = {path_max[-1]:.4f}, F(x) = {F(path_max[-1]):.4f}")
print(f"Чисельний мінімум: x = {path_min[-1]:.4f}, F(x) = {F(path_min[-1]):.4f}")

plt.figure(figsize=(8, 4))
plt.plot(X, Fd, 'b-', label='Fd(x)')
plt.plot(path_max, F(np.array(path_max)), 'r-o', label='Градієнтний підйом')
plt.plot(path_min, F(np.array(path_min)), 'g-x', label='Градієнтний спуск')
plt.title('Чисельний метод')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.legend()
plt.grid(True)
plt.show()

print("\nЗАВДАННЯ 3")
def decode(ch):
    return int("".join(map(str, ch)), 2)

def create_pop(size):
    return [[random.randint(0, 1) for _ in range(5)] for _ in range(size)]

def crossover(p1, p2):
    pt = random.randint(1, 4)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

def mutate(ch):
    idx = random.randint(0, 4)
    ch[idx] = 1 - ch[idx]
    return ch

POP_SIZE = 4
GENS = 15

pop_max = create_pop(POP_SIZE)
hist_max = []
hist_mean = []

for gen in range(GENS):
    fits = [F(decode(ch)) for ch in pop_max]
    hist_max.append(max(fits))
    hist_mean.append(sum(fits) / POP_SIZE)
    new_pop = []
    for _ in range(POP_SIZE // 2):
        i1, i2 = random.sample(range(POP_SIZE), 2)
        p1 = pop_max[i1] if fits[i1] > fits[i2] else pop_max[i2]
        i3, i4 = random.sample(range(POP_SIZE), 2)
        p2 = pop_max[i3] if fits[i3] > fits[i4] else pop_max[i4]
        c1, c2 = crossover(p1, p2)
        new_pop.extend([c1, c2])
    m_idx = random.randint(0, POP_SIZE - 1)
    new_pop[m_idx] = mutate(new_pop[m_idx])
    pop_max = new_pop

best_max = max(pop_max, key=lambda ch: F(decode(ch)))
print(f"ГА Максимум: x = {decode(best_max)}, F(x) = {F(decode(best_max))}")

pop_min = create_pop(POP_SIZE)
hist_min_max = []
hist_min_mean = []

for gen in range(GENS):
    fits = [-F(decode(ch)) for ch in pop_min]
    hist_min_max.append(max(fits))
    hist_min_mean.append(sum(fits) / POP_SIZE)
    new_pop = []
    for _ in range(POP_SIZE // 2):
        i1, i2 = random.sample(range(POP_SIZE), 2)
        p1 = pop_min[i1] if fits[i1] > fits[i2] else pop_min[i2]
        i3, i4 = random.sample(range(POP_SIZE), 2)
        p2 = pop_min[i3] if fits[i3] > fits[i4] else pop_min[i4]
        c1, c2 = crossover(p1, p2)
        new_pop.extend([c1, c2])
    m_idx = random.randint(0, POP_SIZE - 1)
    new_pop[m_idx] = mutate(new_pop[m_idx])
    pop_min = new_pop

best_min = max(pop_min, key=lambda ch: -F(decode(ch)))
print(f"ГА Мінімум: x = {decode(best_min)}, F(x) = {F(decode(best_min))}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(hist_max, 'r-o', label='Max')
plt.plot(hist_mean, 'g-x', label='Mean')
plt.title('ГА (Пошук максимуму)')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(hist_min_max, 'r-o', label='Max(-F)')
plt.plot(hist_min_mean, 'g-x', label='Mean(-F)')
plt.title('ГА (Пошук мінімуму)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nЗАВДАННЯ 4")
np.random.seed(42)
noise = np.random.normal(0, 50, len(X))
Fd_noisy = Fd + noise

plt.figure(figsize=(8, 4))
plt.plot(X, Fd, 'b-', label='Оригінальна Fd(x)')
plt.plot(X, Fd_noisy, 'r.', label='Зашумлена Fd(x)')
plt.title('Функція з шумом')
plt.legend()
plt.grid(True)
plt.show()

def F_noisy(x):
    return F(x) + np.random.normal(0, 20)

path_max_noisy = gradient_ascent(10, 0.1, 30)
print(f"Чисельний максимум (з шумом): x = {path_max_noisy[-1]:.4f}")

pop_noisy = create_pop(POP_SIZE)
for gen in range(GENS):
    fits = [F_noisy(decode(ch)) for ch in pop_noisy]
    new_pop = []
    for _ in range(POP_SIZE // 2):
        i1, i2 = random.sample(range(POP_SIZE), 2)
        p1 = pop_noisy[i1] if fits[i1] > fits[i2] else pop_noisy[i2]
        i3, i4 = random.sample(range(POP_SIZE), 2)
        p2 = pop_noisy[i3] if fits[i3] > fits[i4] else pop_noisy[i4]
        c1, c2 = crossover(p1, p2)
        new_pop.extend([c1, c2])
    m_idx = random.randint(0, POP_SIZE - 1)
    new_pop[m_idx] = mutate(new_pop[m_idx])
    pop_noisy = new_pop

best_noisy = max(pop_noisy, key=lambda ch: F_noisy(decode(ch)))
print(f"ГА Максимум (з шумом): x = {decode(best_noisy)}")
