# Reinforcement Learning Card Battle Game

A turn-based strategic card battle game trained using Q-Learning, Deep
Q-Network (DQN), and Double DQN.

Inspired by strategic deck-building games, this project explores how
reinforcement learning agents can learn long-term combat strategies
involving offense, defense, resource management, and delayed rewards.

------------------------------------------------------------------------

## 🎮 Project Overview

This project was developed for CDS524 -- Reinforcement Learning Game
Design.

The goal is to design a complete game environment and apply Q-learning
algorithms to train an AI agent that learns to:

-   Manage limited energy
-   Choose optimal cards
-   Respond to enemy intent
-   Survive and defeat the opponent

The project includes:

-   Linear Q-Learning
-   Deep Q-Network (DQN)
-   Double DQN
-   Potential-based reward shaping
-   GPU-accelerated training
-   Pygame visualization

------------------------------------------------------------------------

## 🧠 Game Design

### Objective

The agent must defeat the enemy before its HP reaches zero.

### Core Mechanics

-   Turn-based combat
-   Limited energy per turn
-   Hand size limit
-   Enemy intent system (attack / defend / buff)
-   Status effects:
    -   Poison (damage over time)
    -   Weak (reduce attack damage)
    -   Vulnerable (increase damage taken)
    -   Rage (attack bonus)
-   Synergy between cards

### State Space

The state vector includes:

-   Player HP, block, energy
-   Enemy HP, block, status effects
-   Enemy intent (one-hot)
-   Planned damage
-   Hand composition summary
-   Potential damage and block this turn
-   Lethal prediction feature
-   Survival pressure features

The final state is a numerical feature vector compatible with linear and
deep models.

### Action Space

Discrete actions:

-   0--4: Play one of up to 5 cards
-   5: End turn

Illegal actions receive penalties.

------------------------------------------------------------------------

## 🏆 Reward Design

The reward function combines:

-   +10 for winning
-   -10 for losing
-   Small step penalty

To improve learning stability, this project uses potential-based reward
shaping:

R = R_base + gamma \* Phi(s') - Phi(s)

The potential function evaluates:

-   HP advantage
-   Incoming threat
-   Block efficiency
-   Poison future value
-   Buff accumulation
-   Card synergy

This ensures dense learning signals while preserving optimal policy.

------------------------------------------------------------------------

## 🤖 Reinforcement Learning Algorithms

### 1️⃣ Linear Q-Learning

-   Linear function approximation
-   TD update
-   Epsilon-greedy exploration

Efficient but limited in handling complex non-linear interactions.

### 2️⃣ Deep Q-Network (DQN)

Implemented using PyTorch:

-   Fully connected neural network
-   Experience replay buffer
-   Target network
-   Huber loss
-   Gradient clipping

### 3️⃣ Double DQN

Reduces overestimation bias by:

-   Using policy network to select action
-   Using target network to evaluate action

Results in more stable training.

------------------------------------------------------------------------

## ⚡ GPU Acceleration

Training supports:

-   CUDA (NVIDIA GPUs)
-   Apple MPS
-   CPU fallback
-   AMP mixed precision
-   torch.compile (PyTorch 2.0+)
-   Pinned replay buffer

------------------------------------------------------------------------

## 🖥 UI and Visualization

The game interface is built using Pygame.

It displays:

-   Player and enemy HP
-   Block and status effects
-   Enemy intent
-   Current hand cards
-   Agent's last action
-   Last reward received

------------------------------------------------------------------------

## 📂 Project Structure

    .
    ├── env.py
    ├── train.py
    ├── play_agent.py
    ├── game_config.py
    ├── agent.py
    ├── game_ui.py
    ├── logs/
    └── weights/

------------------------------------------------------------------------

## 🚀 How to Run

### Install Dependencies

pip install torch pygame numpy

### Train the Agent

Linear Q-Learning:

python train.py --algo linear --episodes 5000

DQN:

python train.py --algo dqn --episodes 8000

Double DQN:

python train.py --algo double --episodes 8000

Force GPU:

python train.py --algo dqn --device cuda

### Play with Trained Agent

python play_agent.py

------------------------------------------------------------------------

## 🔬 Future Improvements

-   Boss phase with dynamic difficulty
-   Curriculum learning
-   Prioritized experience replay
-   Actor-Critic methods (PPO)
-   Multi-enemy combat

------------------------------------------------------------------------

## 📜 License

Developed for academic purposes (CDS524 Assignment).
