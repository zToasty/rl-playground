import gymnasium as gym

# Создаем среду (FrozenLake — это классика для начала)
# render_mode="human" заставит окно открыться
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")

observation, info = env.reset()

for _ in range(20):
    # Выбираем случайное действие
    action = env.action_space.sample() 
    
    # Делаем шаг
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()