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

1. **Create a virtual environment** (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate     # macOS/Linux
    # venv\Scripts\activate    # Windows
    ```

2. **Install dependencies**:

    ```bash
    pip install --upgrade pip
    pip install torch pygame numpy
    ```

   - `torch` is required for the neural-network based agents (DQN/Double
     DQN).
   - `pygame` is used for the simple visualization interface in
     `game_ui.py`.
   - `numpy` is used by some utility code and logging.

3. **Training a model**

    ```bash
    python train.py --algo <linear|dqn|double_dqn> --episodes <N>
    ```
    ```bash
    python train.py --algo linear --episodes 8000
    ```
   - `--algo` chooses the learning algorithm (default `linear`).
   - `--episodes` sets how many training episodes to run.
   - Additional command-line options are printed if you run
     `python train.py --help`.
   - Trained weights and logs are written to `logs/<algo>/`.

4. **Play against the trained agent**

    ```bash
    python play_agent.py --algo linear
    ```

   - By default `play_agent.py` will load the latest weights from the
     corresponding logs folder.
   - Pass `--visual` to enable the Pygame GUI (`game_ui.py`).

5. **Play as a human**

    ```bash
    python play_human.py
    ```

   - The human interface also uses the Pygame UI by default.  Controls
     are printed to screen.

6. **Miscellaneous**

   - `agent.py` contains the agent implementations and model definitions.
   - `env.py` is the environment simulator that defines states, actions,
     and transitions.
   - `game_config.py` holds global configuration values used by the
     environment.
   - `game_ui.py` implements the Pygame-based user interface used by
     both play scripts.


----

## 📂 Structure

    agent.py           # agent/network definitions and Q‑learning logic
    env.py             # game environment (state, step, reset)
    game_config.py     # constants and configuration values
    game_ui.py         # pygame visualization / human controls
    train.py           # entry point for training agents
    play_agent.py      # run an agent against a default enemy
    play_human.py      # allow a human to play via the GUI
    logs/              # output directory for training logs & weights

    __pycache__/       # Python bytecode cache (ignored)

----



