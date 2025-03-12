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
    """Menjalankan satu episode dan menyimpan total reward."""
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
    return frames

def create_animation_and_plot(frames, scores):
    """Membuat animasi episode terakhir dan menampilkan grafik skor."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Animasi
    ax1.axis('off')
    img = ax1.imshow(frames[0])
    
    def update(frame):
        img.set_array(frame)
        return img,
    
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
    
    # Grafik skor
    ax2.plot(scores, label="Total Reward per Episode", marker="o", linestyle="-", color="b")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")
    ax2.set_title(f"Performance of DQN Agent in {env_name}")
    ax2.legend()
    ax2.grid()
    
    plt.show()

# Jalankan pengujian pada MountainCar-v0
episodes = 10  # Jalankan beberapa episode untuk hasil lebih baik
for episode in range(episodes):
    frames = visualize_episode()
    print(f"Episode: {episode + 1}, Score: {scores[-1]}, Epsilon: {agent.epsilon}")

# Tampilkan animasi dan grafik skor
create_animation_and_plot(frames, scores)

# Simpan hasil eksperimen ke file
with open(f"result_{env_name}.txt", "w") as file:
    file.write(f"Environment: {env_name}\n")
    file.write(f"Scores: {scores}\n")

env.close()
