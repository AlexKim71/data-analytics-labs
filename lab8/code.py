!pip install datasets librosa -q

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from datasets import load_dataset

SR = 16000  # частота дискретизації

# Завантажуємо датасет БЕЗ cast_column: у цьому датасеті колонка audio
# вже містить готові масиви семплів, а не файли для декодування
dataset = load_dataset("MLSpeech/hillenbrand_vowels", split="train")
print(dataset.features)


def get_audio(item):
    """Повертає аудіо як numpy-масив незалежно від формату колонки."""
    a = item['audio']
    if isinstance(a, dict) and 'array' in a:
        return np.array(a['array'], dtype=np.float32)
    if isinstance(a, str):
        # у датасеті аудіо збережене як текст '[0.0, -0.0078, ...]'
        s = a.replace('[', ' ').replace(']', ' ').replace('...', ' ')
        parts = [p for p in s.replace('\n', ' ').split(',') if p.strip()]
        return np.array([float(p) for p in parts], dtype=np.float32)
    return np.array(a, dtype=np.float32)


unique_vowels = dataset.unique('vowel')
target_vowels = unique_vowels[:3]
print(f"Будемо класифікувати голосні: {target_vowels[0]}, {target_vowels[1]}, {target_vowels[2]}")

sound_1, sound_2, sound_3 = [], [], []

for item in dataset:
    vowel = item['vowel']
    if vowel not in target_vowels:
        continue

    audio = get_audio(item)

    if len(audio) < 1000:
        continue

    if vowel == target_vowels[0] and len(sound_1) < 5:
        sound_1.append(audio)
    elif vowel == target_vowels[1] and len(sound_2) < 5:
        sound_2.append(audio)
    elif vowel == target_vowels[2] and len(sound_3) < 5:
        sound_3.append(audio)

    if len(sound_1) == 5 and len(sound_2) == 5 and len(sound_3) == 5:
        break

print(f"Завантажено реальних зразків: {target_vowels[0]}={len(sound_1)}, "
      f"{target_vowels[1]}={len(sound_2)}, {target_vowels[2]}={len(sound_3)}")

# ==========================================
# ГРАФІК 1: Сигнал у часовій області
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(sound_1[0])
plt.title(f'Сигнал у часовій області (Звук {target_vowels[0]})')
plt.xlabel('Зразки')
plt.ylabel('Амплітуда')
plt.show()

# ==========================================
# ГРАФІК 2: Спектр сигналу
# ==========================================
N = len(sound_1[0])
fft_data = fft(sound_1[0])
frequencies = np.fft.fftfreq(N, 1 / SR)

plt.figure(figsize=(8, 5))
plt.plot(frequencies[:N // 2], np.abs(fft_data[:N // 2]))
plt.title(f'Спектр сигналу (Звук {target_vowels[0]})')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплітуда')
plt.show()

# ==========================================
# ГРАФІК 3: Переходи через нуль
# ==========================================
num_segments = 10
step = len(sound_1[0]) // num_segments
zero_crossings = [
    np.sum(np.diff(np.sign(sound_1[0][i:i + step])) != 0)
    for i in range(0, step * num_segments, step)  # рівно 10 сегментів
]

zero_crossings = np.array(zero_crossings)
normalized_crossings = (zero_crossings / zero_crossings.max()
                        if zero_crossings.max() > 0 else zero_crossings)

plt.figure(figsize=(8, 5))
plt.plot(normalized_crossings, 'b-o')
plt.xlabel('Індекс сегменту')
plt.ylabel('Нормалізовані переходи через нуль')
plt.title('Кількість переходів через нуль у сегментах')
plt.grid(True)
plt.show()

# ==========================================
# ВИДІЛЕННЯ ОЗНАК ТА НАВЧАННЯ RBF-МЕРЕЖІ
# ==========================================
def extract_features(data, sample_rate=SR):
    mean_val = np.mean(data)
    std_val = np.std(data)
    energy = np.sum(data ** 2)
    zcr = np.sum(np.diff(np.sign(data)) != 0)
    n = len(data)
    fft_vals = fft(data)
    freqs = np.fft.fftfreq(n, 1 / sample_rate)
    max_freq = freqs[np.argmax(np.abs(fft_vals[:n // 2]))]
    return [mean_val, std_val, zcr, energy, max_freq]


X = []
y = []

for audio in sound_1:
    X.append(extract_features(audio))
    y.append(0)
for audio in sound_2:
    X.append(extract_features(audio))
    y.append(1)
for audio in sound_3:
    X.append(extract_features(audio))
    y.append(2)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rbf_net = SVC(kernel='rbf', gamma='scale', C=1.0)
rbf_net.fit(X_train_scaled, y_train)

y_pred = rbf_net.predict(X_test_scaled)

print("=" * 40)
print("РЕЗУЛЬТАТИ КЛАСИФІКАЦІЇ")
print("=" * 40)
print(f"Точність: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nМатриця помилок:")
print(confusion_matrix(y_test, y_pred))
