"""
agent.py
实现三种 Q-learning Agent：
1. LinearQLearningAgent  —— 线性函数近似（可解释性基线）
2. DQNAgent              —— 深度 Q 网络
3. DoubleDQNAgent        —— Double DQN（抑制 Q 值高估）
"""

import os
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ==================== 通用配置 ====================

@dataclass
class QLearningConfig:
    feature_dim: int = 28
    n_actions: int = 6          # MAX_HAND_SIZE(5) + 1(end turn)
    alpha: float = 0.01         # 线性 Q-learning 学习率
    gamma: float = 0.95         # 折扣因子
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 50_000  # 步数衰减到 epsilon_end


@dataclass
class DQNConfig:
    feature_dim: int = 28
    n_actions: int = 6
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    lr: float = 1e-3
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 80_000
    batch_size: int = 64
    memory_size: int = 50_000
    target_update_freq: int = 500   # 步数
    min_replay_size: int = 1_000    # 开始学习前最小经验数


# ==================== Linear Q-Learning ====================

class LinearQLearningAgent:
    """
    Q(s,a) = w_a^T * φ(s)
    每个动作一个独立权重向量，共 n_actions 个。
    """

    def __init__(self, cfg: QLearningConfig):
        self.cfg = cfg
        # shape: (n_actions, feature_dim)
        self.weights = np.zeros((cfg.n_actions, cfg.feature_dim), dtype=np.float64)
        self.total_steps = 0

    @property
    def epsilon(self) -> float:
        ratio = min(self.total_steps / self.cfg.epsilon_decay, 1.0)
        return self.cfg.epsilon_start + ratio * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self.weights @ state  # (n_actions,)

    def act(self, state: np.ndarray, legal_actions: List[int], epsilon: Optional[float] = None) -> int:
        eps = epsilon if epsilon is not None else self.epsilon
        if random.random() < eps:
            return random.choice(legal_actions)
        qs = self.q_values(state)
        legal_qs = {a: qs[a] for a in legal_actions}
        return max(legal_qs, key=legal_qs.get)

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool, legal_next_actions: List[int]) -> float:
        self.total_steps += 1
        qs_next = self.q_values(next_state)
        if done or not legal_next_actions:
            target = reward
        else:
            best_next = max(qs_next[a] for a in legal_next_actions)
            target = reward + self.cfg.gamma * best_next

        current = self.weights[action] @ state
        td_error = target - current
        self.weights[action] += self.cfg.alpha * td_error * state
        return abs(td_error)

    def save(self, path: str) -> None:
        np.save(path, self.weights)

    def load(self, path: str) -> None:
        self.weights = np.load(path)


# ==================== DQN 网络 ====================

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# ==================== DQN Agent ====================

class DQNAgent:
    """标准 DQN，使用经验回放和目标网络。"""

    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(cfg.feature_dim, cfg.n_actions, cfg.hidden_sizes).to(self.device)
        self.target_net = QNetwork(cfg.feature_dim, cfg.n_actions, cfg.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.memory_size)
        self.total_steps = 0
        self.loss_history: List[float] = []

    @property
    def epsilon(self) -> float:
        ratio = min(self.total_steps / self.cfg.epsilon_decay, 1.0)
        return self.cfg.epsilon_start + ratio * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def act(self, state: np.ndarray, legal_actions: List[int], epsilon: Optional[float] = None) -> int:
        eps = epsilon if epsilon is not None else self.epsilon
        if random.random() < eps:
            return random.choice(legal_actions)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            qs = self.policy_net(s).squeeze(0).cpu().numpy()
        legal_qs = {a: qs[a] for a in legal_actions}
        return max(legal_qs, key=legal_qs.get)

    def store(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self) -> Optional[float]:
        if len(self.memory) < self.cfg.min_replay_size:
            return None
        self.total_steps += 1
        states, actions, rewards, next_states, dones = self.memory.sample(self.cfg.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        q_current = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(1)[0]
            q_target = rewards_t + self.cfg.gamma * q_next * (1 - dones_t)

        loss = nn.MSELoss()(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        if self.total_steps % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def save(self, path: str) -> None:
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)


# ==================== Double DQN Agent ====================

class DoubleDQNAgent(DQNAgent):
    """
    Double DQN：动作选择用 policy_net，Q 值估计用 target_net，
    有效抑制 Q 值的正向偏差。
    """

    def learn(self) -> Optional[float]:
        if len(self.memory) < self.cfg.min_replay_size:
            return None
        self.total_steps += 1
        states, actions, rewards, next_states, dones = self.memory.sample(self.cfg.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        q_current = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN 核心：policy_net 选动作，target_net 算值
            best_actions = self.policy_net(next_states_t).argmax(1, keepdim=True)
            q_next = self.target_net(next_states_t).gather(1, best_actions).squeeze(1)
            q_target = rewards_t + self.cfg.gamma * q_next * (1 - dones_t)

        loss = nn.MSELoss()(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        if self.total_steps % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val