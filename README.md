# KalmansFilter

![image](https://github.com/user-attachments/assets/c74198d6-92aa-49a9-9039-6e882abcc183)

import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x):
        self.F = F  # Матриця переходу стану
        self.H = H  # Матриця вимірювання
        self.Q = Q  # Коваріація шуму процесу
        self.R = R  # Коваріація шуму вимірювання
        self.P = P  # Коваріація початкової оцінки
        self.x = x  # Початкова оцінка стану

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        K = np.dot(self.P, self.H.T) / (np.dot(self.H, np.dot(self.P, self.H.T)) + self.R)
        self.x = self.x + K * (z - np.dot(self.H, self.x))
        self.P = (np.eye(len(self.P)) - K * self.H) @ self.P
        return self.x

# Параметри для сигналу
frequency = 1
amplitude = 5
offset = 10
sampling_interval = 0.001
total_time = 1

# Параметри шуму
noise_variance = 16
noise_std_dev = np.sqrt(noise_variance)

# Параметри фільтра
F = np.array([[1]])  # Матриця переходу стану
H = np.array([[1]])  # Матриця вимірювання
Q = np.array([[1]])  # Коваріація шуму процесу
R = np.array([[10]])  # Коваріація шуму вимірювання
P = np.array([[1]])  # Початкова коваріація оцінки
x = np.array([[0]])  # Початкова оцінка стану

# Ініціалізація фільтра Калмана
kf = KalmanFilter(F, H, Q, R, P, x)

# Генерація сигналу
time_steps = np.arange(0, total_time, sampling_interval)
true_signal = offset + amplitude * np.sin(2 * np.pi * frequency * time_steps)
noisy_signal = [val + np.random.normal(0, noise_std_dev) for val in true_signal]

# Застосування фільтра Калмана
kalman_estimates = []
for measurement in noisy_signal:
    kf.predict()
    estimate = kf.update(measurement)
    kalman_estimates.append(estimate[0][0])

# Візуалізація результатів
plt.figure(figsize=(12, 6))
plt.plot(time_steps, noisy_signal, label='Шумний сигнал', color='orange', alpha=0.6)
plt.plot(time_steps, true_signal, label='Реальний сигнал', linestyle='--', color='blue')
plt.plot(time_steps, kalman_estimates, label='Оцінка фільтра Калмана', color='green')
plt.xlabel('Час (с)')
plt.ylabel('Значення')
plt.title('Фільтр Калмана для шумного синусоїдального сигналу')
plt.legend()
plt.grid()
plt.show()
