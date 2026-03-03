"""
train.py
统一训练入口，支持 linear / dqn / double_dqn 三种算法。
GPU 加速版本：
  - 自动检测 CUDA / MPS / CPU
  - --device 参数强制指定设备
  - Replay Buffer 使用 pin_memory 加速 CPU->GPU 传输
  - AMP 混合精度（CUDA 自动开启，约提速 30%）
  - 支持 torch.compile（PyTorch 2.0+ 进一步提速）

使用方法：
    python train.py --algo dqn    --episodes 8000
    python train.py --algo double --episodes 8000
    python train.py --algo linear --episodes 5000
    python train.py --algo dqn    --device cuda:1   # 多卡指定
    python train.py --algo dqn    --compile          # 启用 torch.compile
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from agent import (
    QLearningConfig, DQNConfig,
    LinearQLearningAgent, DQNAgent, DoubleDQNAgent
)
from env import CardGameEnv
from game_config import MAX_HAND_SIZE


# ==================== 设备检测 ====================

def get_device(device_arg: Optional[str] = None) -> torch.device:
    """优先级：命令行指定 > CUDA > MPS (Apple Silicon) > CPU"""
    if device_arg:
        device = torch.device(device_arg)
        print(f"[Device] 使用指定设备: {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        print(f"[Device] 检测到 CUDA GPU: {props.name}")
        print(f"         显存: {props.total_memory / 1024**3:.1f} GB  |  "
              f"SM 数量: {props.multi_processor_count}  |  "
              f"CUDA 版本: {torch.version.cuda}")
        return device

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] 检测到 Apple MPS (Metal) GPU")
        return device

    print("[Device] 未检测到 GPU，使用 CPU 训练")
    return torch.device("cpu")


def print_gpu_memory(device: torch.device):
    """打印当前 GPU 显存使用情况（仅 CUDA）。"""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved  = torch.cuda.memory_reserved(device)  / 1024**2
        print(f"[GPU Mem] 已分配: {allocated:.1f} MB  |  已保留: {reserved:.1f} MB")


# ==================== GPU 加速 Replay Buffer ====================

class PinnedReplayBuffer:
    """
    预分配 numpy 数组存储经验，采样时用 pin_memory + non_blocking
    实现异步 CPU->GPU 传输，减少 GPU 等待时间。
    """

    def __init__(self, capacity: int, feature_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0
        self.use_pin = (device.type == "cuda")

        self.states      = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        idx = self.pos % self.capacity
        self.states[idx]      = state
        self.next_states[idx] = next_state
        self.actions[idx]     = action
        self.rewards[idx]     = reward
        self.dones[idx]       = float(done)
        self.pos += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)

        def to_tensor(arr, dtype=torch.float32):
            t = torch.as_tensor(arr[idxs], dtype=dtype)
            if self.use_pin:
                t = t.pin_memory()
            return t.to(self.device, non_blocking=True)

        return (
            to_tensor(self.states),
            to_tensor(self.actions, torch.int64),
            to_tensor(self.rewards),
            to_tensor(self.next_states),
            to_tensor(self.dones),
        )

    def __len__(self):
        return self.size


# ==================== GPU-aware DQN Agent ====================

class GPUDQNAgent(DQNAgent):
    """
    继承 DQNAgent，替换 Buffer 为 PinnedReplayBuffer，
    支持 AMP 混合精度和 torch.compile。
    """

    def __init__(self, cfg: DQNConfig, device: torch.device,
                 double: bool = False, use_compile: bool = False):
        super().__init__(cfg)

        # 覆盖设备
        self.device = device
        self.double = double
        self.policy_net = self.policy_net.to(device)
        self.target_net = self.target_net.to(device)
        self.target_net.eval()

        # torch.compile（PyTorch 2.0+）
        if use_compile:
            try:
                self.policy_net = torch.compile(self.policy_net)
                self.target_net = torch.compile(self.target_net)
                print("[GPU] torch.compile 已启用")
            except Exception as e:
                print(f"[GPU] torch.compile 不可用: {e}")

        # 替换 Buffer
        self.memory = PinnedReplayBuffer(cfg.memory_size, cfg.feature_dim, device)

        # 重建 optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)

        # AMP 仅 CUDA 支持
        self.use_amp = (device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    @property
    def epsilon(self) -> float:
        ratio = min(self.total_steps / self.cfg.epsilon_decay, 1.0)
        return self.cfg.epsilon_start + ratio * (
            self.cfg.epsilon_end - self.cfg.epsilon_start)

    def act(self, state: np.ndarray, legal_actions: List[int],
            epsilon: Optional[float] = None) -> int:
        eps = epsilon if epsilon is not None else self.epsilon
        if np.random.random() < eps:
            return int(np.random.choice(legal_actions))
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device, non_blocking=True)
            qs = self.policy_net(s).squeeze(0).cpu().numpy()
        return max(legal_actions, key=lambda a: qs[a])

    def learn(self) -> Optional[float]:
        if len(self.memory) < self.cfg.min_replay_size:
            return None
        self.total_steps += 1

        states_t, actions_t, rewards_t, next_states_t, dones_t = \
            self.memory.sample(self.cfg.batch_size)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                loss = self._compute_loss(states_t, actions_t, rewards_t,
                                          next_states_t, dones_t)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self._compute_loss(states_t, actions_t, rewards_t,
                                      next_states_t, dones_t)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
            self.optimizer.step()

        if self.total_steps % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def _compute_loss(self, states_t, actions_t, rewards_t, next_states_t, dones_t):
        q_current = self.policy_net(states_t).gather(
            1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double:
                # Double DQN：policy 选动作，target 估值
                best_a = self.policy_net(next_states_t).argmax(1, keepdim=True)
                q_next = self.target_net(next_states_t).gather(1, best_a).squeeze(1)
            else:
                q_next = self.target_net(next_states_t).max(1)[0]
            q_target = rewards_t + self.cfg.gamma * q_next * (1 - dones_t)

        # Huber Loss 比 MSE 对异常值更鲁棒
        return F.smooth_l1_loss(q_current, q_target)


# ==================== 日志 ====================

class TrainingLogger:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.win_flags: List[int] = []
        self.epsilon_history: List[float] = []
        self.loss_history: List[float] = []
        self.card_play_counts: Dict[str, int] = {}

    def log_episode(self, reward, length, win, epsilon, loss=None):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.win_flags.append(int(win))
        self.epsilon_history.append(epsilon)
        if loss is not None:
            self.loss_history.append(loss)

    def log_card(self, name: str):
        self.card_play_counts[name] = self.card_play_counts.get(name, 0) + 1

    def save(self, prefix: str = "training"):
        data = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "win_flags": self.win_flags,
            "epsilon_history": self.epsilon_history,
            "loss_history": self.loss_history,
            "card_play_counts": self.card_play_counts,
        }
        path = self.save_dir / f"{prefix}_log.json"
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[Logger] 日志已保存到 {path}")
        return str(path)


# ==================== 训练循环 ====================

def train_linear(episodes: int, seed: int, save_dir: str):
    """Linear Q-Learning 纯 CPU，GPU 对线性代数运算收益极小。"""
    env = CardGameEnv(seed=seed)
    features = env.reset()
    cfg = QLearningConfig(feature_dim=features.shape[0])
    agent = LinearQLearningAgent(cfg)
    logger = TrainingLogger(save_dir)

    print(f"\n===== Linear Q-Learning 训练开始 =====")
    print(f"状态维度: {features.shape[0]}  |  总 episodes: {episodes}")
    print(f"[Device] CPU（线性近似不需要 GPU）")

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward, step_count = 0.0, 0
        ep_loss = []

        while True:
            legal = env.get_legal_actions()
            action = agent.act(state, legal)
            next_state, reward, done, info = env.step(action)
            if not info.get("illegal", False):
                td = agent.update(state, action, reward, next_state, done,
                                  env.get_legal_actions())
                ep_loss.append(td)
            total_reward += reward
            step_count += 1
            state = next_state
            if done:
                break

        win = env.enemy.hp <= 0
        logger.log_episode(total_reward, step_count, win, agent.epsilon,
                           float(np.mean(ep_loss)) if ep_loss else 0.0)

        if ep % 500 == 0:
            wr = np.mean(logger.win_flags[-500:])
            ar = np.mean(logger.episode_rewards[-500:])
            print(f"EP {ep:5d} | WinRate {wr:.2%} | AvgReward {ar:.2f} | ε {agent.epsilon:.3f}")

    agent.save(str(Path(save_dir) / "q_weights.npy"))
    return logger.save("linear")


def train_dqn_gpu(episodes: int, seed: int, save_dir: str,
                  double: bool, device: torch.device, use_compile: bool):
    """DQN / Double DQN GPU 加速训练主循环。"""
    env = CardGameEnv(seed=seed)
    features = env.reset()

    # GPU 显存充足时用更大 batch 和 buffer
    is_gpu = device.type in ("cuda", "mps")
    cfg = DQNConfig(
        feature_dim=features.shape[0],
        batch_size=128 if is_gpu else 64,
        memory_size=100_000 if is_gpu else 50_000,
    )

    algo_name = "Double DQN" if double else "DQN"
    agent = GPUDQNAgent(cfg, device, double=double, use_compile=use_compile)
    logger = TrainingLogger(save_dir)

    print(f"\n===== {algo_name} GPU 训练开始 =====")
    print(f"状态维度: {features.shape[0]}  |  总 episodes: {episodes}")
    print(f"Batch: {cfg.batch_size}  |  Buffer: {cfg.memory_size}  |  "
          f"AMP: {'ON' if agent.use_amp else 'OFF'}")
    if device.type == "cuda":
        print_gpu_memory(device)

    t_start = time.time()

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward, step_count = 0.0, 0
        ep_losses = []

        while True:
            legal = env.get_legal_actions()
            action = agent.act(state, legal)
            next_state, reward, done, info = env.step(action)

            if not info.get("illegal", False):
                agent.memory.push(state, action, reward, next_state, done)
                loss = agent.learn()
                if loss is not None:
                    ep_losses.append(loss)

            total_reward += reward
            step_count += 1
            state = next_state
            if done:
                break

        win = env.enemy.hp <= 0
        avg_loss = float(np.mean(ep_losses)) if ep_losses else None
        logger.log_episode(total_reward, step_count, win, agent.epsilon, avg_loss)

        if ep % 500 == 0:
            elapsed = time.time() - t_start
            wr  = np.mean(logger.win_flags[-500:])
            ar  = np.mean(logger.episode_rewards[-500:])
            eps_per_sec = ep / elapsed
            print(f"EP {ep:5d} | WinRate {wr:.2%} | AvgReward {ar:.2f} | "
                  f"ε {agent.epsilon:.3f} | {eps_per_sec:.0f} ep/s")
            if device.type == "cuda":
                print_gpu_memory(device)

    prefix = "double_dqn" if double else "dqn"
    agent.save(str(Path(save_dir) / f"{prefix}_weights.pth"))
    return logger.save(prefix)


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description="卡牌游戏 RL 训练（GPU 版）")
    parser.add_argument("--algo", choices=["linear", "dqn", "double"], default="dqn")
    parser.add_argument("--episodes", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="强制指定设备: cuda / cuda:0 / cuda:1 / mps / cpu")
    parser.add_argument("--compile", action="store_true",
                        help="启用 torch.compile（PyTorch 2.0+）")
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = f"logs/{args.algo}"

    # 全局随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True   # 自动优化卷积核，提升吞吐

    device = get_device(args.device)

    t0 = time.time()
    if args.algo == "linear":
        train_linear(args.episodes, args.seed, args.save_dir)
    elif args.algo == "dqn":
        train_dqn_gpu(args.episodes, args.seed, args.save_dir,
                      double=False, device=device, use_compile=args.compile)
    elif args.algo == "double":
        train_dqn_gpu(args.episodes, args.seed, args.save_dir,
                      double=True, device=device, use_compile=args.compile)

    total = time.time() - t0
    print(f"\n训练完成，总耗时 {total:.1f}s（{total / 60:.1f} min）")


if __name__ == "__main__":
    main()