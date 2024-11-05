# KalmansFilter

![image](https://github.com/user-attachments/assets/c74198d6-92aa-49a9-9039-6e882abcc183)

## Опис коду

Імпорт бібліотек:

numpy: використовується для чисельних обчислень, зокрема для роботи з масивами та матрицями.
matplotlib.pyplot: використовується для візуалізації даних.
Клас KalmanFilter:

Це основний клас, який реалізує алгоритм фільтра Калмана.
Методи класу:

__init__: конструктор, що ініціалізує параметри фільтра, такі як матриця переходу стану (F), матриця вимірювання (H), коваріації шуму (Q і R), коваріацію початкової оцінки (P) та початкову оцінку стану (x).

predict(): виконує прогнозування наступного стану на основі поточного стану та матриці переходу. Оновлює коваріацію оцінки.

update(z): виконує оновлення оцінки стану на основі нового вимірювання z. Використовує формулу для обчислення коефіцієнта Калмана (K) та оновлює стан і коваріацію оцінки.

Параметри для сигналу:

Визначаються параметри синусоїдального сигналу: частота (frequency), амплітуда (amplitude), зсув (offset), інтервал дискретизації (sampling_interval) та загальний час (total_time).
Параметри шуму:

Визначаються параметри шуму, включаючи дисперсію шуму (noise_variance) та стандартне відхилення (noise_std_dev).
Параметри фільтра:

Визначаються матриці та початкові значення для фільтра Калмана:
F: матриця переходу стану (для одномірного випадку).
H: матриця вимірювання (для одномірного випадку).
Q: коваріація шуму процесу.
R: коваріація шуму вимірювання.
P: початкова коваріація оцінки.
x: початкова оцінка стану.
Ініціалізація фільтра Калмана:

Створюється об'єкт kf класу KalmanFilter з заданими параметрами.

Генерація сигналу:

Створюється масив часових кроків (time_steps).
Генерується реальний синусоїдальний сигнал (true_signal) на основі заданих параметрів.
Генерується шумний сигнал (noisy_signal), до якого додається нормальний шум з заданим стандартним відхиленням.
Застосування фільтра Калмана:

Для кожного вимірювання з шумного сигналу:
Виконується прогнозування наступного стану.
Оновлюється оцінка стану на основі нового вимірювання.
Зберігається оцінка в списку kalman_estimates.
Візуалізація результатів:

Використовуючи matplotlib, створюється графік, на якому:
Показується шумний сигнал (помаранчевий).
Показується реальний сигнал (синій, пунктирний).
Показується оцінка фільтра Калмана (зелений).
Додаються підписи до осей, заголовок, легенда та сітка.
