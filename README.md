# RL Card Battle Game

A turn-based card battle game trained using Q-Learning, DQN, and Double
DQN (PyTorch + Pygame).

------------------------------------------------------------------------

## 🎯 Overview

The agent learns to defeat an enemy using strategic card play, energy
management, and status effects.

**Features:** - Linear Q-Learning - Deep Q-Network (DQN) - Double DQN -
Potential-based reward shaping - GPU training support - Pygame
visualization

------------------------------------------------------------------------

## 🧠 Game Mechanics

-   Turn-based combat
-   Limited energy per turn
-   5-card hand limit
-   Enemy intent system (attack / defend / buff)
-   Status effects: Poison, Weak, Vulnerable, Rage

**Win condition:** Reduce enemy HP to 0\
**Lose condition:** Player HP reaches 0

------------------------------------------------------------------------

## 🤖 RL Implementation

-   Epsilon-greedy exploration
-   Experience replay
-   Target network
-   Huber loss
-   GPU acceleration (CUDA / MPS)

------------------------------------------------------------------------

## 🚀 How to Run

Install:

    pip install torch pygame numpy

Train:

    python train.py --algo linear --episodes 5000

Play_agent:

    python play_agent.py

------------------------------------------------------------------------

## 📂 Structure

    env.py
    train.py
    play_agent.py
    agent.py
    game_config.py
    play_human.py
    logs

------------------------------------------------------------------------

Developed for CDS524 Reinforcement Learning Assignment.
