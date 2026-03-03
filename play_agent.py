"""
AI 对战界面：从 game_ui 复用样式与绘制，与 play_human 改一处两处生效。
支持两类权重：
  - 线性 Q-learning: q_weights.npy
  - DQN / Double DQN: *_weights.pth
"""
import sys

import pygame

from agent import LinearQLearningAgent, QLearningConfig, DQNAgent, DQNConfig
from env import CardGameEnv
from game_config import MAX_HAND_SIZE
from game_ui import (
    WIDTH, HEIGHT, CARD_W, CARD_H, FPS,
    BG_COLOR, EMPTY_CARD_BG, EMPTY_CARD_BORDER,
    draw_text, card_rect, draw_card, draw_enemy,
    draw_player_info, draw_enemy_info,
)


def load_agent(weights_path: str, feature_dim: int):
    """
    根据权重文件后缀自动加载 Linear 或 DQN Agent。
    - *.npy  -> LinearQLearningAgent（与 train_linear 保存的权重兼容）
    - *.pth  -> DQNAgent（与 train_dqn_gpu / double 保存的权重兼容）
    """
    if weights_path.endswith(".npy"):
        cfg = QLearningConfig(feature_dim=feature_dim)
        agent = LinearQLearningAgent(cfg)
        agent.load(weights_path)
        print(f"[play_agent] 已加载 LinearQ 权重: {weights_path}")
        return agent

    if weights_path.endswith(".pth"):
        cfg = DQNConfig(feature_dim=feature_dim)
        agent = DQNAgent(cfg)
        agent.load(weights_path)
        print(f"[play_agent] 已加载 DQN 权重: {weights_path}")
        return agent

    raise ValueError(f"不支持的权重文件类型: {weights_path}（仅支持 .npy 或 .pth）")


def run_agent_game(weights_path: str = "q_weights.npy") -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Card Battle (AI)")
    clock = pygame.time.Clock()

    env = CardGameEnv(seed=42)
    features = env.reset()

    if not weights_path:
        print("需要先训练模型并生成权重文件（q_weights.npy 或 *_weights.pth）")
        pygame.quit()
        sys.exit(1)

    agent = load_agent(weights_path, feature_dim=features.shape[0])

    running = True
    step_delay_frames = 10
    frame_counter = 0
    epsilon_eval = 0.0

    # UI: 记录最近一次动作与奖励（正=奖励，负=惩罚）
    last_action_desc = "N/A"

    while running:
        clock.tick(FPS)
        frame_counter += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not env.done and frame_counter % step_delay_frames == 0:
            state = env._get_features()
            legal_actions = env.get_legal_actions()
            a = agent.act(state, legal_actions, epsilon_eval)
            next_state, reward, done, info = env.step(a)

            # 记录动作描述：出哪张牌 / 结束回合
            if a == MAX_HAND_SIZE:
                last_action_desc = "End Turn"
            elif 0 <= a < len(env.player.hand):
                last_action_desc = f"Play: {env.player.hand[a]}"
            else:
                last_action_desc = f"Action {a}"

        # ---------- 绘制（与 play_human 共用 game_ui，改 game_ui 即生效）----------
        screen.fill(BG_COLOR)
        draw_player_info(screen, env)
        draw_enemy_info(screen, env)
        draw_enemy(screen, env, shake=0.0)

        for i in range(MAX_HAND_SIZE):
            rect = card_rect(i)
            if i < len(env.player.hand):
                name = env.player.hand[i]
                draw_card(screen, name, rect, env, hovered=False)
            else:
                pygame.draw.rect(screen, EMPTY_CARD_BG, rect, border_radius=10)
                pygame.draw.rect(screen, EMPTY_CARD_BORDER, rect, 1, border_radius=10)

        # 显示 agent 最近一次动作 + 奖励/惩罚
        draw_text(screen, f"Last action: {last_action_desc}", 20, (200, 230, 255), (20, 108))
        draw_text(screen, f"Last reward: {env.last_reward:.2f}", 20, (200, 255, 200), (20, 136))

        if env.done:
            msg = "AI Win! Close window to exit." if env.enemy.hp <= 0 else "AI Lost. Close window to exit."
            draw_text(screen, msg, 26, (255, 230, 0), (WIDTH // 2 - 260, HEIGHT // 2 + 80))

        pygame.display.flip()

        if env.done:
            pygame.time.wait(1500)
            env.reset()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run_agent_game()
