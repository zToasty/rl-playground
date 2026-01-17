import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. Настройка среды
# is_slippery=False делает мир предсказуемым (детерминированным)
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array")

# 2. Гиперпараметры (Математическая база)
learning_rate = 0.1   # Alpha (скорость изменения весов)
gamma = 0.95          # Коэффициент дисконтирования (ценность будущих наград)
epsilon = 1.0         # Параметр исследования (Exploration)
epsilon_decay = 0.001 # Скорость уменьшения рандома
min_epsilon = 0.01
episodes = 5000       # Сколько раз агент пройдет игру

# 3. Инициализация Q-таблицы (16 состояний на 4 действия)
# 
q_table = np.zeros([env.observation_space.n, env.action_space.n])

rewards_per_episode = []

print("--- Начинаю обучение агента ---")

# 4. Основной цикл обучения
for i in tqdm(range(episodes)):
    state, info = env.reset()
    done = False
    total_rewards = 0
    
    while not done:
        # Epsilon-greedy стратегия выбора действия
        if np.random.random() < epsilon:
            action = env.action_space.sample() # Рандом
        else:
            action = np.argmax(q_table[state, :]) # Лучшее из опыта
        
        # Делаем шаг в среде
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Bellman Equation (Update Rule) ---
        # Мы обновляем текущее Q в сторону разности между целью и реальностью
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])
        
        new_value = old_value + learning_rate * (reward + gamma * next_max - old_value)
        q_table[state, action] = new_value
        # --------------------------------------
        
        state = next_state
        total_rewards += reward
        
    # Уменьшаем epsilon, чтобы агент со временем переходил к стратегии использования опыта
    epsilon = max(min_epsilon, epsilon - epsilon_decay)
    rewards_per_episode.append(total_rewards)

print("\n--- Обучение завершено! ---")

# 5. Визуализация прогресса
# 
plt.figure(figsize=(10, 5))
plt.plot(np.convolve(rewards_per_episode, np.ones(100)/100, mode='valid'))
plt.title('Процент успешных заходов (Скользящее среднее)')
plt.xlabel('Эпизоды')
plt.ylabel('Успех (1.0 = дошел до цели)')
plt.grid(True)
plt.show()

# 6. Финальная проверка: посмотрим на результат
print("\nДемонстрация обученного агента:")
test_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")
state, info = test_env.reset()
done = False

while not done:
    action = np.argmax(q_table[state, :])
    state, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated

test_env.close()

print("\nИтоговая Q-таблица:")
print(np.round(q_table, 3))