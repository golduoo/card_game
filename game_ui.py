"""
game_ui.py — 人机对战与 AI 对战共用的 UI 层。
改这里即可同时生效于 play_human.py 和 play_agent.py。
支持自定义卡面和 Boss 图：将图片放入 assets/cards/ 和 assets/enemy/ 即可替换。
"""
import math
import os
from typing import Tuple, Dict, Optional

import pygame

from env import CardGameEnv
from game_config import MAX_HAND_SIZE, CARD_DEFINITIONS, CARD_EFFECTS

# 资源目录
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_CARDS_DIR = os.path.join(_BASE_DIR, "assets", "cards")
ASSETS_ENEMY_DIR = os.path.join(_BASE_DIR, "assets", "enemy")
ASSETS_ENEMY_FILENAME = "boss.png"   
ENEMY_IMAGE_W, ENEMY_IMAGE_H = 200, 160   # Boss 图在游戏中的显示宽高


# ==================== 常量====================
WIDTH, HEIGHT = 960, 640
CARD_W, CARD_H = 124, 172
FPS = 60
END_BTN_W, END_BTN_H = 160, 48

# 卡牌类型配色（攻击/技能/能力）
CARD_TYPE_COLORS = {
    "attack": {"bg": (180, 50, 50), "border": (255, 100, 100), "top": (220, 80, 80), "cost": (255, 200, 80)},
    "skill":  {"bg": (50, 80, 140), "border": (100, 140, 220), "top": (80, 120, 200), "cost": (120, 200, 255)},
    "power":  {"bg": (100, 50, 120), "border": (160, 100, 200), "top": (140, 80, 180), "cost": (200, 150, 255)},
}

# 动画时长（秒）
CARD_FLY_DURATION = 0.35
ATTACK_EFFECT_DURATION = 0.5
ENEMY_ATTACK_DURATION = 0.6
DAMAGE_NUMBER_FLOAT_DURATION = 0.7

# 背景色等
BG_COLOR = (18, 18, 38)
EMPTY_CARD_BG = (35, 35, 55)
EMPTY_CARD_BORDER = (70, 70, 100)

# 自定义图片缓存
_card_images: Dict[str, pygame.Surface] = {}
_enemy_image: Optional[pygame.Surface] = None
_assets_loaded = False


def load_assets() -> None:
    """从 assets/cards/ 和 assets/enemy/ 加载自定义卡面与 Boss 图（仅加载一次）。"""
    global _card_images, _enemy_image, _assets_loaded
    if _assets_loaded:
        return
    _assets_loaded = True
    # 卡面：assets/cards/Strike.png, Defend.png 等，文件名 = 卡牌英文名
    if os.path.isdir(ASSETS_CARDS_DIR):
        for f in os.listdir(ASSETS_CARDS_DIR):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                name = os.path.splitext(f)[0]
                path = os.path.join(ASSETS_CARDS_DIR, f)
                try:
                    img = pygame.image.load(path).convert_alpha()
                    _card_images[name] = img
                except Exception:
                    pass
    # Boss：assets/enemy/boss.png 或 enemy.png
    for filename in (ASSETS_ENEMY_FILENAME, "enemy.png"):
        path = os.path.join(ASSETS_ENEMY_DIR, filename)
        if os.path.isfile(path):
            try:
                _enemy_image = pygame.image.load(path).convert_alpha()
                break
            except Exception:
                pass


# ==================== 动画状态 ====================
class AnimState:
    IDLE = "idle"
    CARD_FLYING = "card_flying"
    ATTACK_EFFECT = "attack_effect"
    ENEMY_ATTACK = "enemy_attack"


# ==================== 工具 ====================
def draw_text(
    surface: pygame.Surface,
    text: str,
    size: int,
    color: Tuple[int, int, int],
    pos: Tuple[float, float],
    center_x: bool = False,
) -> None:
    font = pygame.font.SysFont("arial", size)
    img = font.render(text, True, color)
    if center_x:
        pos = (pos[0] - img.get_width() // 2, pos[1])
    surface.blit(img, pos)


def card_rect(index: int) -> pygame.Rect:
    margin = 24
    spacing = (WIDTH - 2 * margin - CARD_W) // max(1, (MAX_HAND_SIZE - 1))
    x = margin + index * spacing
    y = HEIGHT - CARD_H - 36
    return pygame.Rect(x, y, CARD_W, CARD_H)


def end_turn_rect() -> pygame.Rect:
    x = WIDTH - END_BTN_W - 24
    y = HEIGHT - CARD_H - END_BTN_H - 64
    return pygame.Rect(x, y, END_BTN_W, END_BTN_H)


def get_card_type(name: str) -> str:
    return CARD_DEFINITIONS.get(name, {}).get("type", "skill")


def card_effect_lines(name: str, env: CardGameEnv) -> list:
    eff = CARD_EFFECTS.get(name, {})
    lines = []
    if "damage" in eff:
        dmg = int(eff["damage"])
        if "block_scale_damage" in eff:
            dmg += int(env.player.block * float(eff["block_scale_damage"]))
            lines.append(f"DMG {dmg} (blk)")
        else:
            lines.append(f"DMG {dmg}")
    if "block" in eff:
        lines.append(f"BLOCK {int(eff['block'])}")
    if "heal" in eff:
        lines.append(f"HEAL {int(eff['heal'])}")
    if "draw" in eff:
        lines.append(f"DRAW {int(eff['draw'])}")
    if "energy" in eff:
        lines.append(f"+ENERGY {int(eff['energy'])}")
    if "poison" in eff:
        lines.append(f"POISON {int(eff['poison'])}")
    if "weak" in eff:
        lines.append(f"WEAK {int(eff['weak'])}")
    if "vulnerable" in eff:
        lines.append(f"VULN {int(eff['vulnerable'])}")
    if "rage" in eff:
        lines.append(f"RAGE +{int(eff['rage'])}")
    return lines[:3]


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_out_quad(t: float) -> float:
    return 1.0 - (1.0 - t) * (1.0 - t)


# ==================== 绘制：卡牌====================
def draw_card(
    surface: pygame.Surface,
    name: str,
    rect: pygame.Rect,
    env: CardGameEnv,
    hovered: bool = False,
    scale: float = 1.0,
) -> None:
    """绘制单张卡牌。若 assets/cards/{name}.png 存在则用该图作卡面，否则用类型配色绘制。"""
    load_assets()
    if scale != 1.0:
        rect = pygame.Rect(rect.x, rect.y, int(rect.w * scale), int(rect.h * scale))
    r = rect
    card_type = get_card_type(name)
    colors = CARD_TYPE_COLORS.get(card_type, CARD_TYPE_COLORS["skill"])
    radius = 12

    if name in _card_images:
        # 自定义卡面：缩放后绘制，再叠一层半透明底 + 边框和文字
        img = _card_images[name]
        scaled = pygame.transform.smoothscale(img, (r.w, r.h))
        surface.blit(scaled, r)
        # 底部半透明条便于读文字（仅下半部分）
        bar_h = max(r.h // 2, 70)
        overlay = pygame.Surface((r.w, bar_h))
        overlay.set_alpha(120)
        overlay.fill((0, 0, 0))
        surface.blit(overlay, (r.x, r.y + r.h - bar_h))
    else:
        pygame.draw.rect(surface, colors["bg"], r, border_radius=radius)
        top_h = max(2, r.h // 4)
        top_rect = pygame.Rect(r.x, r.y, r.w, top_h)
        pygame.draw.rect(surface, colors["top"], top_rect, border_radius=radius)
        pygame.draw.rect(
            surface, (0, 0, 0), top_rect, 0,
            border_bottom_left_radius=radius, border_bottom_right_radius=radius,
        )
        pygame.draw.rect(surface, colors["top"], (r.x, r.y, r.w, radius))

    border_w = 3 if hovered else 2
    pygame.draw.rect(surface, colors["border"], r, border_w, border_radius=radius)
    cost = CARD_DEFINITIONS.get(name, {}).get("cost", 1)
    cx, cy = r.x + 28, r.y + 26
    pygame.draw.circle(surface, (30, 30, 40), (cx, cy), 14)
    pygame.draw.circle(surface, colors["cost"], (cx, cy), 12)
    draw_text(surface, str(cost), 20, (20, 20, 30), (cx - 5, cy - 10))
    draw_text(surface, name, 16, (255, 255, 255), (r.x + 10, r.y + 48))
    for j, line in enumerate(card_effect_lines(name, env)):
        draw_text(surface, line, 14, (230, 230, 230), (r.x + 10, r.y + 72 + j * 18))


# ==================== 绘制：怪物形象====================
def draw_enemy(surface: pygame.Surface, env: CardGameEnv, shake: float = 0) -> None:
    """绘制敌人。若 assets/enemy/boss.png 存在则用该图，否则用程序绘制的身体+眼睛+嘴。意图图标始终在头顶。"""
    load_assets()
    cx = WIDTH // 2 + int(shake * 8)
    cy = HEIGHT // 2 - 100
    intent = env.enemy.intent
    intent_y = (cy - ENEMY_IMAGE_H // 2 - 28) if _enemy_image else (cy - 75)

    if _enemy_image is not None:
        # 自定义 Boss 图：缩放到 ENEMY_IMAGE_W x ENEMY_IMAGE_H，居中
        img = pygame.transform.smoothscale(_enemy_image, (ENEMY_IMAGE_W, ENEMY_IMAGE_H))
        dest = (cx - ENEMY_IMAGE_W // 2, cy - ENEMY_IMAGE_H // 2)
        surface.blit(img, dest)
    else:
        body = pygame.Rect(cx - 70, cy - 50, 140, 100)
        pygame.draw.rect(surface, (100, 40, 50), body, border_radius=24)
        pygame.draw.rect(surface, (140, 70, 80), body.inflate(-12, -12), border_radius=18)
        eye_y = cy - 25
        pygame.draw.ellipse(surface, (255, 255, 255), (cx - 45, eye_y - 12, 24, 22))
        pygame.draw.ellipse(surface, (255, 255, 255), (cx + 20, eye_y - 12, 24, 22))
        pygame.draw.ellipse(surface, (40, 20, 30), (cx - 38, eye_y - 6, 12, 14))
        pygame.draw.ellipse(surface, (40, 20, 30), (cx + 27, eye_y - 6, 12, 14))
        mouth_y = cy + 20
        pygame.draw.arc(
            surface, (60, 30, 40), (cx - 25, mouth_y - 15, 50, 30),
            math.pi * 0.2, math.pi * 0.8, 3,
        )

    # 意图图标
    if intent == "attack":
        pygame.draw.polygon(
            surface, (255, 80, 60),
            [(cx, intent_y - 18), (cx - 14, intent_y + 8), (cx + 14, intent_y + 8)],
        )
        draw_text(surface, str(env.enemy.planned_damage), 18, (255, 255, 200), (cx + 22, intent_y - 10))
    elif intent == "defend":
        pygame.draw.rect(surface, (100, 180, 255), (cx - 12, intent_y - 12, 24, 24), border_radius=6)
        draw_text(surface, "B", 16, (255, 255, 255), (cx - 5, intent_y - 10))
    else:
        pygame.draw.rect(surface, (200, 150, 255), (cx - 12, intent_y - 12, 24, 24), border_radius=6)
        draw_text(surface, "+", 18, (255, 255, 255), (cx - 6, intent_y - 12))


# ==================== 绘制：玩家/敌人信息面板（共用）====================
def draw_player_info(surface: pygame.Surface, env: CardGameEnv) -> None:
    draw_text(surface, f"Player HP: {env.player.hp}", 24, (255, 255, 255), (20, 20))
    draw_text(surface, f"Block: {env.player.block}  Energy: {env.player.energy}", 24, (200, 200, 255), (20, 50))
    draw_text(surface, f"Poison {env.player.poison} | Rage {env.player.rage_bonus}", 20, (200, 255, 220), (20, 78))


def draw_enemy_info(surface: pygame.Surface, env: CardGameEnv) -> None:
    draw_text(surface, f"Enemy HP: {env.enemy.hp}", 24, (255, 200, 200), (WIDTH - 260, 20))
    draw_text(surface, f"Block: {env.enemy.block}", 24, (255, 200, 200), (WIDTH - 260, 50))
    draw_text(surface, f"Intent: {env.enemy.intent} ({env.enemy.planned_damage})", 24, (255, 220, 150), (WIDTH - 260, 80))
    draw_text(
        surface,
        f"Poison {env.enemy.poison} | Weak {env.enemy.weak} | Vuln {env.enemy.vulnerable}",
        20, (255, 220, 200), (WIDTH - 440, 110),
    )
    draw_text(surface, f"Enemy Buff: ATK+{env.enemy.attack_buff}", 20, (255, 240, 180), (WIDTH - 260, 135))


# ==================== 绘制：手牌行（空槽 + 单卡用 draw_card）====================
def draw_hand_slot(surface: pygame.Surface, index: int, filled: bool) -> None:
    """只画空槽或占位；有牌时由调用方用 draw_card 画。"""
    rect = card_rect(index)
    if not filled:
        pygame.draw.rect(surface, EMPTY_CARD_BG, rect, border_radius=10)
        pygame.draw.rect(surface, EMPTY_CARD_BORDER, rect, 1, border_radius=10)


# ==================== 绘制：结束回合按钮 ====================
def draw_end_turn_button(surface: pygame.Surface, hovered: bool, disabled: bool = False) -> None:
    btn = end_turn_rect()
    if disabled:
        btn_color = (50, 60, 50)
    else:
        btn_color = (60, 100, 60) if not hovered else (90, 140, 90)
    pygame.draw.rect(surface, btn_color, btn, border_radius=10)
    pygame.draw.rect(surface, (180, 255, 180), btn, 2, border_radius=10)
    draw_text(surface, "End Turn", 22, (10, 25, 10), (btn.x + 28, btn.y + 12))
