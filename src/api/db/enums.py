from __future__ import annotations

from enum import Enum


class BotKind(str, Enum):
    MODEL = "model"
    HEURISTIC = "heuristic"
    SCRIPTED = "scripted"


class AgentType(str, Enum):
    HUMAN = "human"
    HEURISTIC = "heuristic"
    MODEL = "model"


class QueueType(str, Enum):
    RANKED = "ranked"
    CASUAL = "casual"
    VS_AI = "vs_ai"
    CUSTOM = "custom"


class GameStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    FINISHED = "finished"
    ABORTED = "aborted"


class WinnerSide(str, Enum):
    P1 = "p1"
    P2 = "p2"
    DRAW = "draw"


class TerminationReason(str, Enum):
    NORMAL = "normal"
    RESIGN = "resign"
    TIMEOUT = "timeout"
    DISCONNECT = "disconnect"


class GameSource(str, Enum):
    SELF_PLAY = "self_play"
    HUMAN = "human"
    MIXED = "mixed"


class PlayerSide(str, Enum):
    P1 = "p1"
    P2 = "p2"


class SampleSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class QueueEntryStatus(str, Enum):
    WAITING = "waiting"
    MATCHED = "matched"
    CANCELED = "canceled"
