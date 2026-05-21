from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

Tile = int


@dataclass
class PlayerState:
    hand: List[Tile] = field(default_factory=list)
    discards: List[Tile] = field(default_factory=list)


@dataclass
class GameState:
    wall_remaining: int
    turn: int
    players: List[PlayerState]


def encode_state(state: GameState, player_index: int) -> list[float]:
    """Create a dense feature vector from a simplified game state.

    This baseline encoding is intentionally compact and easy to replace later.
    """
    player = state.players[player_index]
    hand_counts = [0] * 34
    discard_counts = [0] * 34

    for t in player.hand:
        hand_counts[t] += 1
    for t in player.discards:
        discard_counts[t] += 1

    return [
        state.wall_remaining / 70.0,
        state.turn / 100.0,
        *[c / 4.0 for c in hand_counts],
        *[c / 4.0 for c in discard_counts],
    ]
