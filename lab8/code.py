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

sound_U = [generate_synthetic_audio(300) for _ in range(5)]
sound_Yu = [generate_synthetic_audio(500) for _ in range(5)]
sound_E = [generate_synthetic_audio(700) for _ in range(5)]

def extract_features(data, sample_rate=16000):
    mean_val = np.mean(data)
    std_val = np.std(data)
    energy = np.sum(data**2)
    zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
    
    N = len(data)
    fft_data = fft(data)
    freqs = np.fft.fftfreq(N, 1/sample_rate)
    max_freq_idx = np.argmax(np.abs(fft_data[:N//2]))
    max_freq = freqs[max_freq_idx]
    
    return [mean_val, std_val, zero_crossings, energy, max_freq]

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

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(sound_U[0][:1000])
plt.title('Сигнал в часовій області (Звук У)')
plt.xlabel('Зразки')
plt.ylabel('Амплітуда')

plt.subplot(1, 2, 2)
fft_example = fft(sound_U[0])
freqs_example = np.fft.fftfreq(len(sound_U[0]), 1/16000)
plt.plot(freqs_example[:len(freqs_example)//2], np.abs(fft_example[:len(freqs_example)//2]))
plt.title('Спектр сигналу (Звук У)')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплітуда')

plt.tight_layout()
plt.show()

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
