import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dqn_agent import DQNAgent  # Menggunakan agen yang mendukung target network

# Inisialisasi environment MountainCar-v0
env_name = "MountainCar-v0"
env = gym.make(env_name, render_mode='rgb_array')

# Inisialisasi parameter
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Inisialisasi agen DQN dengan target network
agent = DQNAgent(state_size, action_size)  
agent.epsilon = 0.01  # Minimalkan eksplorasi saat testing

# Simpan skor tiap episode
scores = []

def visualize_episode():
    frames = []
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    
    total_reward = 0
    for time in range(500):
        frame = env.render()
        frames.append(frame)  # Simpan frame untuk animasi
        
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # Gabungkan status selesai
        state = np.reshape(next_state, [1, state_size])
        total_reward += reward
        
        if done:
            print(f"Test Episode Score ({env_name}): {total_reward}")
            break
    
    scores.append(total_reward)  # Simpan skor episode
    return frames, total_reward

def create_animation(frames):
    fig, ax = plt.subplots()
    ax.axis('off')
    img = ax.imshow(frames[0])
    
    def update(frame):
        img.set_array(frame)
        return img,
    
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
    plt.show()

def plot_scores(scores):
    """Menampilkan grafik skor episode."""
    plt.figure(figsize=(8, 5))
    plt.plot(scores, label="Total Reward per Episode", marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Performance of DQN Agent in {env_name}")
    plt.legend()
    plt.grid()
    plt.show()

# Jalankan pengujian pada MountainCar-v0
episodes = 10  # Jalankan beberapa episode untuk hasil lebih baik
for _ in range(episodes):
    frames, score = visualize_episode()

# Tampilkan animasi episode terakhir
create_animation(frames)

# Tampilkan grafik skor
plot_scores(scores)

# Simpan hasil eksperimen
with open(f"result_{env_name}.txt", "w") as file:
    file.write(f"Environment: {env_name}\n")
    file.write(f"Scores: {scores}\n")

env.close()
