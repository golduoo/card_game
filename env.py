import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np

from game_config import (
    MAX_PLAYER_HP,
    MAX_ENEMY_HP,
    MAX_BLOCK,
    MAX_ENERGY,
    MAX_HAND_SIZE,
    PLAYER_START_DECK,
    CARD_DEFINITIONS,
    CARD_EFFECTS,
    ENEMY_BASE_DAMAGE,
)


@dataclass
class PlayerState:
    hp: int = MAX_PLAYER_HP
    block: int = 0
    energy: int = MAX_ENERGY
    poison: int = 0
    deck: List[str] = field(default_factory=list)
    discard: List[str] = field(default_factory=list)
    hand: List[str] = field(default_factory=list)
    rage_bonus: int = 0


@dataclass
class EnemyState:
    hp: int = MAX_ENEMY_HP
    block: int = 0
    poison: int = 0
    weak: int = 0
    vulnerable: int = 0
    attack_buff: int = 0
    intent: str = "attack"
    planned_damage: int = 12


class CardGameEnv:
    """
    简化版杀戮尖塔战斗环境。
    状态、动作空间设计为适合线性近似 Q-learning。
    """

    def __init__(self, seed: int = 0, enemy_base_damage: int | None = None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.enemy_base_damage = enemy_base_damage if enemy_base_damage is not None else ENEMY_BASE_DAMAGE
        self.player = PlayerState()
        self.enemy = EnemyState()
        self.turn = 0
        self.done = False
        self.last_reward = 0.0
        self.played_card_this_turn = False

    # =================== 基础流程 ===================
    def reset(self) -> np.ndarray:
        self.player = PlayerState()
        self.enemy = EnemyState()
        self.turn = 0
        self.done = False
        self.last_reward = 0.0
        self.played_card_this_turn = False

        # 初始化牌库
        self.player.deck = PLAYER_START_DECK.copy()
        self.rng.shuffle(self.player.deck)
        self.player.discard = []
        self.player.hand = []

        # 抽起始手牌
        self._draw_to_hand(MAX_HAND_SIZE)
        self._roll_enemy_intent()

        return self._get_features()

    # potential-based reward shaping 势能函数：
    # 只依赖状态（含手牌/意图），用于表达“局面优势”，不直接给某张牌硬编码奖励。
    def _potential(self) -> float:
        my_hp_ratio = max(self.player.hp, 0) / MAX_PLAYER_HP
        enemy_hp_ratio = max(self.enemy.hp, 0) / MAX_ENEMY_HP
        block_ratio = min(self.player.block, MAX_BLOCK) / MAX_BLOCK

        # 敌人 buff 累积威胁（越高越危险）
        buff_threat = min(self.enemy.attack_buff, 15) / 15.0

        # 相对生存压力：下一次计划伤害 / 当前血量（低血量时会非常大）
        incoming_threat = 0.0
        if self.enemy.intent == "attack" and self.player.hp > 0:
            planned_raw = self.enemy.planned_damage
            if self.enemy.weak > 0:
                planned_raw = int(planned_raw * 0.75)
            incoming_threat = planned_raw / float(self.player.hp)
        incoming_threat = min(incoming_threat, 2.0)  # 截断，防止数值爆炸

        # 下一个敌方回合的“威胁值”：如果敌人准备攻击，弱化会降低伤害，格挡会抵消伤害
        threat = 0.0
        if self.enemy.intent == "attack":
            planned = self.enemy.planned_damage
            if self.enemy.weak > 0:
                planned = int(planned * 0.75)
            threat = max(0, planned - self.player.block)
        threat_norm = min(threat, 25.0) / 25.0

        # 毒是延迟伤害，近似用“未来总毒伤（三角数）”估值，敌人血越多越值钱
        poison = min(self.enemy.poison, 20)
        poison_future = (poison * (poison + 1) / 2.0) / MAX_ENEMY_HP
        poison_future = min(poison_future, 1.5)
        poison_value = poison_future * (0.5 + 0.5 * enemy_hp_ratio)

        weak_norm = min(self.enemy.weak, 5) / 5.0
        vuln_norm = min(self.enemy.vulnerable, 5) / 5.0
        intent_attack = 1.0 if self.enemy.intent == "attack" else 0.0

        # 组合潜力：当 HeavyBlow 在手时，Vulnerable 的局面价值更大（但不要求同回合）
        heavy_in_hand = 1.0 if "HeavyBlow" in self.player.hand else 0.0
        vuln_synergy = vuln_norm * heavy_in_hand

        # Weaken 在敌人准备攻击时更有价值（鼓励“先弱化再挨打/再输出”）
        weaken_ready = 1.0 if "Weaken" in self.player.hand else 0.0
        weaken_timing = weaken_ready * intent_attack

        return (
            3.5 * (my_hp_ratio - enemy_hp_ratio)
            + 0.6 * block_ratio
            - 0.8 * threat_norm
            - 0.35 * buff_threat
            - 0.50 * incoming_threat
            + 1.0 * poison_value
            + 0.25 * weak_norm * (0.3 + 0.7 * intent_attack)
            + 0.20 * vuln_norm
            + 0.35 * vuln_synergy
            + 0.15 * weaken_timing
        )

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步玩家动作。
        action_index: 0..MAX_HAND_SIZE-1 -> 打出对应手牌；MAX_HAND_SIZE -> 结束回合
        """
        if self.done:
            return self._get_features(), 0.0, True, {}

        legal_actions = self.get_legal_actions()
        if action_index not in legal_actions:
            # 非法动作给小惩罚，但不结束 episode
            return self._get_features(), -0.1, False, {"illegal": True}

        # 势能 shaping 之前的状态（只对合法动作计算）
        phi_before = self._potential()

        info: Dict = {}
        extra_penalty = 0.0
        if action_index == MAX_HAND_SIZE:
            # 结束回合：敌人行动，然后新回合
            hp_before = self.player.hp
            dmg_to_player = self._end_player_turn_and_enemy_act()
            info["damage_to_player"] = dmg_to_player
            info["enemy_intent"] = getattr(self, "_last_enemy_intent", "attack")

            # 血量紧迫度：HP<50%时，受伤惩罚开始放大（逼迫学会“低血量要防守”）
            hp_ratio = (hp_before / MAX_PLAYER_HP) if MAX_PLAYER_HP > 0 else 0.0
            urgency = 1.0 + max(0.0, 0.5 - hp_ratio) * 2.0
            if dmg_to_player > 0:
                extra_penalty -= 0.15 * dmg_to_player * urgency
        else:
            # 打出手牌
            card_idx = action_index
            card_name = self.player.hand[card_idx]
            dmg_dealt = self._play_card(card_idx, card_name)
            info["card_played"] = card_name
            info["damage_to_enemy"] = getattr(self, "_last_damage_to_enemy", 0)

        # 终局奖励（真实目标，保持稀疏、清晰）
        base_reward = 0.0
        if self.enemy.hp <= 0:
            self.done = True
            base_reward += 10.0
        elif self.player.hp <= 0:
            self.done = True
            base_reward -= 10.0

        # potential-based reward shaping：把“延迟价值”（毒/弱化/易伤/挡伤）转成可学习的密集信号
        # 终局时让 phi_after = 0，避免终局势能被重复计分
        phi_after = 0.0 if self.done else self._potential()
        gamma = 0.95
        reward = base_reward + gamma * phi_after - phi_before + extra_penalty

        # 轻微时间惩罚：鼓励更快结束战斗，但不喧宾夺主
        reward += -0.01

        self.last_reward = reward
        return self._get_features(), reward, self.done, info

    # =================== 牌库 / 回合管理 ===================
    def _draw_to_hand(self, target_size: int) -> None:
        while len(self.player.hand) < target_size:
            if not self.player.deck and not self.player.discard:
                break
            if not self.player.deck:
                self.player.deck = self.player.discard
                self.rng.shuffle(self.player.deck)
                self.player.discard = []
            self.player.hand.append(self.player.deck.pop())

    def _start_player_turn(self) -> None:
        self.turn += 1
        self.player.energy = MAX_ENERGY
        self.player.block = 0
        self.player.rage_bonus = 0
        self.played_card_this_turn = False
        # 毒伤害在各自回合开始时结算
        if self.player.poison > 0:
            self.player.hp -= self.player.poison
        if self.enemy.poison > 0:
            self.enemy.hp -= self.enemy.poison
            self.enemy.poison = max(0, self.enemy.poison - 1)

    def _end_player_turn_and_enemy_act(self) -> int:
        """玩家回合结束，敌人根据 intent 行动，然后开始新玩家回合。"""
        played_any = self.played_card_this_turn
        self._last_enemy_intent = self.enemy.intent

        # 敌人回合开始：清掉上一轮留下的护盾（符合回合制逻辑）
        self.enemy.block = 0

        # 敌人行动
        dmg_to_player = 0
        if self.enemy.intent == "attack":
            dmg = self.enemy.planned_damage
            # 虚弱减少攻击（是否攻击都应衰减见下方“回合衰减”）
            if self.enemy.weak > 0:
                dmg = int(dmg * 0.75)
            dmg_to_player = max(0, dmg - self.player.block)
            self.player.block = max(0, self.player.block - dmg)
            self.player.hp -= dmg_to_player
        elif self.enemy.intent == "defend":
            self.enemy.block += 10
        elif self.enemy.intent == "buff":
            # 增加后续攻击力（可叠加）
            self.enemy.attack_buff += 3

        # 状态回合衰减：弱化/易伤按“敌方回合数”递减（符合“持续 2 回合”的直觉）
        if self.enemy.weak > 0:
            self.enemy.weak = max(0, self.enemy.weak - 1)
        if self.enemy.vulnerable > 0:
            self.enemy.vulnerable = max(0, self.enemy.vulnerable - 1)

        # 敌人毒伤害在其回合开始/结束时已经由 _start_player_turn 处理

        self._last_damage_to_player = dmg_to_player

        # 回合结束，重新 roll intent 并开始新回合
        self._roll_enemy_intent()
        self._start_player_turn()
        if played_any:
            # 正常回合：丢弃剩余手牌并补抽
            if self.player.hand:
                self.player.discard.extend(self.player.hand)
            self.player.hand = []
            self._draw_to_hand(MAX_HAND_SIZE)
        else:
            # 惩罚规则：本回合未出牌就结束 -> 不补抽新手牌（保留原手牌）
            pass

        return int(dmg_to_player)

    def _roll_enemy_intent(self) -> None:
        # 简化版本：三种意图
        r = self.rng.random()
        if r < 0.7:
            self.enemy.intent = "attack"
            self.enemy.planned_damage = self.enemy_base_damage + self.enemy.attack_buff
        elif r < 0.85:
            self.enemy.intent = "defend"
            self.enemy.planned_damage = 0
        else:
            self.enemy.intent = "buff"
            self.enemy.planned_damage = 0

    # =================== 打牌效果 ===================
    def _play_card(self, idx: int, name: str) -> int:
        self._last_damage_to_enemy = 0
        info = CARD_DEFINITIONS.get(name, {})
        cost = info.get("cost", 1)
        if self.player.energy < cost:
            return 0
        self.played_card_this_turn = True
        self.player.energy -= cost

        # 移动到弃牌堆
        card = self.player.hand.pop(idx)
        self.player.discard.append(card)

        dmg_to_enemy = 0

        eff = CARD_EFFECTS.get(name, {})
        if "block" in eff:
            self.player.block += int(eff["block"])
        if "energy" in eff:
            self.player.energy = min(MAX_ENERGY + 2, self.player.energy + int(eff["energy"]))
        if "draw" in eff:
            self._draw_to_hand(min(MAX_HAND_SIZE, len(self.player.hand) + int(eff["draw"])))
        if "poison" in eff:
            self.enemy.poison = min(30, self.enemy.poison + int(eff["poison"]))
        if "weak" in eff:
            self.enemy.weak += int(eff["weak"])
        if "vulnerable" in eff:
            self.enemy.vulnerable += int(eff["vulnerable"])
        if "rage" in eff:
            self.player.rage_bonus += int(eff["rage"])
        if "heal" in eff:
            self.player.hp = min(MAX_PLAYER_HP, self.player.hp + int(eff["heal"]))

        if "damage" in eff:
            dmg_to_enemy = int(eff["damage"])
            if "block_scale_damage" in eff:
                dmg_to_enemy += int(self.player.block * float(eff["block_scale_damage"]))

        # 攻击类牌计算易伤和格挡（保留少量按伤害比例的奖励） 
        dealt = 0
        if dmg_to_enemy > 0:
            if self.enemy.vulnerable > 0:
                dmg_to_enemy = int(dmg_to_enemy * 1.5)
            if self.player.rage_bonus > 0:
                dmg_to_enemy += self.player.rage_bonus
            dealt = max(0, dmg_to_enemy - self.enemy.block)
            self.enemy.block = max(0, self.enemy.block - dmg_to_enemy)
            self.enemy.hp -= dealt

        self._last_damage_to_enemy = dealt

        return int(dealt)

    # =================== 特征与动作 ===================
    def get_legal_actions(self) -> List[int]:
        actions: List[int] = []
        # 打出手牌
        for i in range(MAX_HAND_SIZE):
            if i >= len(self.player.hand):
                continue
            name = self.player.hand[i]
            cost = CARD_DEFINITIONS.get(name, {}).get("cost", 1)
            if self.player.energy >= cost:
                actions.append(i)
        # 结束回合永远合法
        actions.append(MAX_HAND_SIZE)
        return actions

    def _get_features(self) -> np.ndarray:
        """
        构造线性近似用的特征向量 φ(s)。
        维度适中，包含：血量/格挡/能量、状态、敌人意图、手牌摘要。
        """
        features: List[float] = []

        # 玩家特征（归一化）
        features.append(self.player.hp / MAX_PLAYER_HP)
        features.append(self.player.block / MAX_BLOCK)
        features.append(self.player.energy / float(MAX_ENERGY))
        features.append(min(self.player.poison, 10) / 10.0)

        # 敌人特征
        features.append(max(self.enemy.hp, 0) / MAX_ENEMY_HP)
        features.append(self.enemy.block / MAX_BLOCK)
        features.append(min(self.enemy.poison, 10) / 10.0)
        features.append(min(self.enemy.weak, 5) / 5.0)
        features.append(min(self.enemy.vulnerable, 5) / 5.0)

        # 敌人意图 one-hot
        intents = ["attack", "defend", "buff"]
        for it in intents:
            features.append(1.0 if self.enemy.intent == it else 0.0)
        features.append(self.enemy.planned_damage / 20.0)

        # 手牌摘要：简单统计几类牌
        counts = {
            "attack": 0,
            "skill": 0,
            "power": 0,
        }
        special_flags = {
            "HeavyBlow": 0.0,
            "Heal": 0.0,
            "PoisonDart": 0.0,
        }
        for name in self.player.hand:
            info = CARD_DEFINITIONS.get(name, {})
            t = info.get("type", "skill")
            if t in counts:
                counts[t] += 1
            if name in special_flags:
                special_flags[name] = 1.0

        hand_size = max(1, len(self.player.hand))
        features.append(counts["attack"] / hand_size)
        features.append(counts["skill"] / hand_size)
        features.append(counts["power"] / hand_size)
        features.extend(list(special_flags.values()))

        # Enhanced features: hand potential analysis
        total_damage = 0.0
        total_block = 0.0
        affordable_damage = 0.0
        affordable_block = 0.0

        for name in self.player.hand:
            eff = CARD_EFFECTS.get(name, {})
            cost = CARD_DEFINITIONS.get(name, {}).get("cost", 1)
            
            # Calculate potential damage
            if "damage" in eff:
                dmg = int(eff["damage"])
                # Account for vulnerable
                if self.enemy.vulnerable > 0:
                    dmg = int(dmg * 1.5)
                # Account for rage
                if self.player.rage_bonus > 0:
                    dmg += self.player.rage_bonus
                total_damage += dmg
                if cost <= self.player.energy:
                    affordable_damage += dmg
            
            # Calculate potential block
            if "block" in eff:
                blk = int(eff["block"])
                total_block += blk
                if cost <= self.player.energy:
                    affordable_block += blk

        # Lethal prediction: can we kill enemy this turn?
        can_lethal = 1.0 if affordable_damage >= self.enemy.hp else 0.0

        # Normalize and add features
        features.append(total_damage / 50.0)  # Total potential damage
        features.append(total_block / 30.0)   # Total potential block
        features.append(affordable_damage / 50.0)  # Affordable damage this turn
        features.append(affordable_block / 30.0)   # Affordable block this turn
        features.append(can_lethal)  # Can kill enemy this turn (binary)

        # =================== 生存压力特征（防守学习关键） ===================
        effective_planned = self.enemy.planned_damage
        if self.enemy.intent == "attack" and self.enemy.weak > 0:
            effective_planned = int(effective_planned * 0.75)

        can_block_next = 1.0 if (
            self.enemy.intent == "attack" and self.player.block >= effective_planned
        ) else 0.0

        danger_ratio = (
            (effective_planned / float(self.player.hp))
            if (self.enemy.intent == "attack" and self.player.hp > 0)
            else 0.0
        )
        features.append(min(danger_ratio, 2.0) / 2.0)  # 归一化到[0,1]
        features.append(can_block_next)

        poison_turns_to_kill = (
            (self.enemy.hp / float(self.enemy.poison))
            if self.enemy.poison > 0
            else 99.0
        )
        poison_will_win = 1.0 if poison_turns_to_kill < (self.player.hp / 10.0) else 0.0
        features.append(poison_will_win)

        return np.array(features, dtype=np.float32)

