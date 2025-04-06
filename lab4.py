import numpy as np 
import matplotlib.pyplot as plt 

# Читання даних з файлу та заміна ком на крапки 
file_path = '3.txt' 
with open(file_path, 'r') as file: 
    lines = file.readlines()[1:]  # Пропускаємо перший рядок (заголовки)

data = [] 
for line in lines: 
    line = line.replace(',', '.') 
    data.append([float(x) for x in line.strip().split('\t')]) 

data = np.array(data) 

# Зчитування перших двох колонок 
L = data[:, 0] 
V = data[:, 1] 

# Побудова графіка 
plt.figure(figsize=(8, 7)) 

# Графік у звичайному форматі 
plt.subplot(2, 2, 1) 
plt.plot(V, L, '.') 
plt.title('Звичайний графік') 
plt.xlabel('V') 
plt.ylabel('L') 

# Графік у подвійному логарифмічному форматі 
plt.subplot(2, 2, 2) 
plt.loglog(V, L, '.') 
plt.title('Подвійний логарифмічний графік') 
plt.xlabel('log(V)') 
plt.ylabel('log(L)') 

log_V = np.log(V) 
log_L = np.log(L) 

# Побудова лінійної апроксимації у подвійному логарифмічному масштабі 
coefficients = np.polyfit(log_V, log_L, 1) 
p = np.poly1d(coefficients) 

# Виведення коефіцієнта нахилу 
print("Коефіцієнт нахилу лінії регресії:", coefficients[0]-0.4) 

L_pred = np.exp(p(log_V)) 
ss_res = np.sum((L - L_pred) ** 2) 
ss_tot = np.sum((L - np.mean(L)) ** 2) 
r_squared = 1 - (ss_res / ss_tot) 
print("Коефіцієнт детермінації R-квадрат:", r_squared) 

# Розрахунок середньої квадратичної помилки (RMSE) 
rmse = np.sqrt(np.mean((L - L_pred) ** 2)) 

# Розрахунок нормованого середньоквадратичного відхилення (NRMSE) 
nrmse = rmse / (np.max(L) - np.min(L)) 
print("Нормоване середньоквадратичне відхилення:", nrmse) 

# Графік у подвійному логарифмічному масштабі 
plt.subplot(2, 2, (3, 4)) 
plt.loglog(V, L, 'o', label='Дані') 
plt.loglog(np.exp(log_V), np.exp(p(log_V)), 'r-', label='Лінійна апроксимація') 
plt.title('Лінійна апроксимація у подвійному логарифмічному масштабі') 
plt.xlabel('log(V)') 
plt.ylabel('log(L)') 
plt.legend() 

plt.tight_layout() 
plt.show()

print("Hello world!")
