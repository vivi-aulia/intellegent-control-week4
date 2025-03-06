import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import random
from collections import deque

# Definisi kelas DQN dengan Target Network
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, learning_rate=0.001, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Faktor diskon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0  # Faktor eksplorasi
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Model utama dan target
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_dim=self.state_size),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                future_q = np.amax(self.target_model.predict(next_state, verbose=0))
                target[0][action] = reward + self.gamma * future_q
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Fungsi untuk melatih agen di environment tertentu
def train_agent(env_name, episodes=1000):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    update_target_freq = 10  # Update target network setiap 10 episode
    
    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                print(f"Episode {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.4f}")
                break
        
        agent.replay()
        if e % update_target_freq == 0:
            agent.update_target_model()
    
    env.close()

# Uji agen pada beberapa environment
for env_name in ["LunarLander-v2", "MountainCar-v0"]:
    print(f"\nTraining on {env_name}...")
    train_agent(env_name, episodes=500)
