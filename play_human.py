"""
人机对战界面：从 game_ui 复用样式与绘制，只保留人类操作与动画逻辑。
"""
import sys
import math
from typing import Dict, Any

import pygame

from env import CardGameEnv
from game_config import MAX_HAND_SIZE, CARD_DEFINITIONS
from game_ui import (
    WIDTH, HEIGHT, CARD_W, CARD_H, FPS, END_BTN_W, END_BTN_H,
    CARD_FLY_DURATION, ATTACK_EFFECT_DURATION, ENEMY_ATTACK_DURATION, DAMAGE_NUMBER_FLOAT_DURATION,
    BG_COLOR, EMPTY_CARD_BG, EMPTY_CARD_BORDER,
    AnimState,
    draw_text, card_rect, end_turn_rect, draw_card, draw_enemy,
    draw_player_info, draw_enemy_info, draw_end_turn_button,
    lerp, ease_out_quad,
)


def run_human_game() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Card Battle — 出牌与攻击动画")
    clock = pygame.time.Clock()

    env = CardGameEnv(seed=0)
    env.reset()

    anim_state = AnimState.IDLE
    anim_start_time = 0.0
    anim_card_index = -1
    anim_card_name = ""
    anim_card_start_rect = None
    anim_damage_to_enemy = 0
    anim_damage_to_player = 0
    last_step_info: Dict[str, Any] = {}

    running = True
    selected_idx = -1

    while running:
        clock.tick(FPS)
        t = pygame.time.get_ticks() / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and anim_state == AnimState.IDLE:
                mx, my = event.pos
                for i in range(MAX_HAND_SIZE):
                    if i < len(env.player.hand) and card_rect(i).collidepoint(mx, my):
                        cost = CARD_DEFINITIONS.get(env.player.hand[i], {}).get("cost", 1)
                        if env.player.energy >= cost:
                            selected_idx = i
                        break
                else:
                    selected_idx = -1
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                mx, my = event.pos
                if anim_state == AnimState.IDLE:
                    # 对局结束后：点击上半区重开，并吞掉本次点击，避免 reset 后立刻 End Turn
                    if env.done and my < HEIGHT - CARD_H - 80:
                        env.reset()
                        anim_state = AnimState.IDLE
                        selected_idx = -1
                        anim_damage_to_enemy = 0
                        anim_damage_to_player = 0
                        last_step_info = {}
                        continue

                    if selected_idx >= 0:
                        rect = card_rect(selected_idx)
                        if rect.collidepoint(mx, my):
                            anim_state = AnimState.CARD_FLYING
                            anim_start_time = t
                            anim_card_index = selected_idx
                            anim_card_name = env.player.hand[selected_idx]
                            anim_card_start_rect = rect.copy()
                            selected_idx = -1
                    else:
                        if end_turn_rect().collidepoint(mx, my) and not env.done:
                            _, _, _, last_step_info = env.step(MAX_HAND_SIZE)
                            anim_damage_to_player = last_step_info.get("damage_to_player", 0)
                            if anim_damage_to_player > 0:
                                anim_state = AnimState.ENEMY_ATTACK
                                anim_start_time = t
                            selected_idx = -1
                            continue
                        if my < HEIGHT - CARD_H - 80 and not env.done:
                            env.step(MAX_HAND_SIZE)
                    selected_idx = -1

        if anim_state == AnimState.CARD_FLYING:
            elapsed = t - anim_start_time
            if elapsed >= CARD_FLY_DURATION:
                _, _, _, last_step_info = env.step(anim_card_index)
                anim_damage_to_enemy = last_step_info.get("damage_to_enemy", 0)
                if anim_damage_to_enemy > 0:
                    anim_state = AnimState.ATTACK_EFFECT
                    anim_start_time = t
                else:
                    anim_state = AnimState.IDLE
                anim_card_index = -1

        if anim_state == AnimState.ATTACK_EFFECT and (t - anim_start_time) >= ATTACK_EFFECT_DURATION:
            anim_state = AnimState.IDLE
        if anim_state == AnimState.ENEMY_ATTACK and (t - anim_start_time) >= ENEMY_ATTACK_DURATION:
            anim_state = AnimState.IDLE

        # ---------- 绘制（全部来自 game_ui，改 game_ui 即生效）----------
        screen.fill(BG_COLOR)
        draw_player_info(screen, env)
        draw_enemy_info(screen, env)

        shake = 0.0
        if anim_state == AnimState.ATTACK_EFFECT:
            elapsed = t - anim_start_time
            shake = math.sin(elapsed * 40) * (1.0 - elapsed / ATTACK_EFFECT_DURATION)
        if anim_state == AnimState.ENEMY_ATTACK:
            elapsed = t - anim_start_time
            shake = math.sin(elapsed * 30) * 0.5 * (1.0 - elapsed / ENEMY_ATTACK_DURATION)
        draw_enemy(screen, env, shake)

        if anim_state == AnimState.ATTACK_EFFECT:
            elapsed = t - anim_start_time
            progress = min(1.0, elapsed / 0.15)
            alpha = 255 * (1.0 - elapsed / ATTACK_EFFECT_DURATION)
            slash_x = WIDTH // 2
            slash_w = int(120 * progress)
            s = pygame.Surface((slash_w * 2, 20))
            s.set_alpha(int(alpha))
            s.fill((255, 220, 100))
            screen.blit(s, (slash_x - slash_w, HEIGHT // 2 - 110))
            num_y = HEIGHT // 2 - 80 - (elapsed / DAMAGE_NUMBER_FLOAT_DURATION) * 40
            draw_text(screen, f"-{anim_damage_to_enemy}", 36, (255, 200, 80), (WIDTH // 2, num_y), center_x=True)

        if anim_state == AnimState.ENEMY_ATTACK:
            elapsed = t - anim_start_time
            progress = min(1.0, elapsed / 0.2)
            pygame.draw.line(
                screen, (255, 80, 60),
                (WIDTH // 2, HEIGHT // 2 - 100),
                (lerp(WIDTH // 2, WIDTH // 2 - 80, progress), lerp(HEIGHT // 2 - 100, HEIGHT - 180, progress)), 6,
            )
            num_y = HEIGHT - 200 - (elapsed / ENEMY_ATTACK_DURATION) * 30
            draw_text(screen, f"-{anim_damage_to_player}", 32, (255, 100, 100), (80, num_y))

        btn = end_turn_rect()
        mx, my = pygame.mouse.get_pos()
        hovered_btn = btn.collidepoint(mx, my) and anim_state == AnimState.IDLE
        draw_end_turn_button(screen, hovered_btn, disabled=False)

        mouse_x, mouse_y = pygame.mouse.get_pos()
        enemy_center = (WIDTH // 2, HEIGHT // 2 - 100)

        for i in range(MAX_HAND_SIZE):
            rect = card_rect(i)
            hovered = rect.collidepoint(mouse_x, mouse_y) and anim_state == AnimState.IDLE
            if i < len(env.player.hand):
                name = env.player.hand[i]
                if anim_state == AnimState.CARD_FLYING and i == anim_card_index and anim_card_start_rect:
                    elapsed = t - anim_start_time
                    progress = ease_out_quad(min(1.0, elapsed / CARD_FLY_DURATION))
                    fly_x = lerp(anim_card_start_rect.centerx, enemy_center[0], progress) - CARD_W // 2
                    fly_y = lerp(anim_card_start_rect.centery, enemy_center[1], progress) - CARD_H // 2
                    fly_scale = lerp(1.0, 0.5, progress)
                    fly_rect = pygame.Rect(int(fly_x), int(fly_y), CARD_W, CARD_H)
                    draw_card(screen, anim_card_name, fly_rect, env, hovered=False, scale=fly_scale)
                else:
                    draw_card(screen, name, rect, env, hovered=hovered)
            else:
                pygame.draw.rect(screen, EMPTY_CARD_BG, rect, border_radius=10)
                pygame.draw.rect(screen, EMPTY_CARD_BORDER, rect, 1, border_radius=10)

        if env.done:
            msg = "You Win! Click top area to restart." if env.enemy.hp <= 0 else "You Died! Click top area to restart."
            draw_text(screen, msg, 26, (255, 230, 0), (WIDTH // 2 - 260, HEIGHT // 2 + 80))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run_human_game()
