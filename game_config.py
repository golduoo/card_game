MAX_PLAYER_HP = 65
MAX_ENEMY_HP = 120
MAX_BLOCK = 40
MAX_ENERGY = 3
MAX_HAND_SIZE = 5

# 敌人基础伤害，用作难度旋钮（推荐 14-15 作为标准难度）
ENEMY_BASE_DAMAGE = 13

PLAYER_START_DECK = [
    "Strike",
    "Strike",
    "Strike",
    "Strike",
    "Defend",
    "Defend",
    "Defend",
    "HeavyBlow",
    "QuickJab",
    "PoisonDart",
    "Weaken",
    "Vulnerable",
]

CARD_DEFINITIONS = {
    "Strike": {"cost": 1, "type": "attack"},
    "Defend": {"cost": 1, "type": "skill"},
    "HeavyBlow": {"cost": 2, "type": "attack"},
    "QuickJab": {"cost": 0, "type": "attack"},
    "Draw2": {"cost": 1, "type": "skill"},
    "GainEnergy": {"cost": 0, "type": "skill"},
    "PoisonDart": {"cost": 1, "type": "attack"},
    "Weaken": {"cost": 1, "type": "skill"},
    "Vulnerable": {"cost": 1, "type": "skill"},
    "Rage": {"cost": 1, "type": "power"},
    "ShieldBash": {"cost": 1, "type": "attack"},
    "Heal": {"cost": 2, "type": "skill"},
}

CARD_EFFECTS = {
    # damage / block / draw / energy / poison / weak / vulnerable / heal / rage
    "Strike": {"damage": 6},
    "Defend": {"block": 5},
    "HeavyBlow": {"damage": 14},
    "QuickJab": {"damage": 3, "draw": 1},
    "Draw2": {"draw": 2},
    "GainEnergy": {"energy": 1},
    "PoisonDart": {"damage": 3, "poison": 3},
    "Weaken": {"weak": 2},
    "Vulnerable": {"vulnerable": 2},
    "Rage": {"rage": 2},
    "ShieldBash": {"damage": 6, "block_scale_damage": 0.5},
    "Heal": {"heal": 8},
}

