import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

def generate_synthetic_audio(freq, duration=0.5, sr=16000):
    t = np.linspace(0, duration, int(sr * duration))
    signal = 0.5 * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
    return signal

data = generate_synthetic_audio(300)

plt.figure(figsize=(8, 5))
plt.plot(data[:1000])
plt.title('Сигнал у часовій області (Звук У)')
plt.xlabel('Зразки')
plt.ylabel('Амплітуда')
plt.show()

N = len(data)
fft_data = fft(data)
frequencies = np.fft.fftfreq(N, 1/16000)

plt.figure(figsize=(8, 5))
plt.plot(frequencies[:N//2], np.abs(fft_data[:N//2]))
plt.title('Спектр сигналу (Частотна область)')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплітуда')
plt.show()

num_segments = 10
step = len(data) // num_segments

zero_crossings = [
    np.sum(np.diff(np.sign(data[i:i + step])) != 0)
    for i in range(0, len(data), step)
]

zero_crossings = np.array(zero_crossings)
normalized_crossings = zero_crossings / zero_crossings.max()

plt.figure(figsize=(8, 5))
plt.plot(normalized_crossings, 'b-o')
plt.xlabel('Індекс сегменту')
plt.ylabel('Нормалізовані переходи через нуль')
plt.title('Кількість переходів через нуль у сегментах')
plt.grid(True)
plt.show()

def extract_features(data, sample_rate=16000):
    mean_val = np.mean(data)
    std_val = np.std(data)
    energy = np.sum(data**2)
    zcr = np.sum(np.diff(np.sign(data)) != 0)
    N = len(data)
    fft_data = fft(data)
    freqs = np.fft.fftfreq(N, 1/sample_rate)
    max_freq = freqs[np.argmax(np.abs(fft_data[:N//2]))]
    return [mean_val, std_val, zcr, energy, max_freq]

sound_U = [generate_synthetic_audio(300) for _ in range(5)]
sound_Yu = [generate_synthetic_audio(500) for _ in range(5)]
sound_E = [generate_synthetic_audio(700) for _ in range(5)]

X = []
y = []

for audio in sound_U:
    X.append(extract_features(audio))
    y.append(0)
for audio in sound_Yu:
    X.append(extract_features(audio))
    y.append(1)
for audio in sound_E:
    X.append(extract_features(audio))
    y.append(2)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
